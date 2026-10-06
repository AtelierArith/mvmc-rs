//! Lock-step multi-walker sampler with a device-resident Pfaffian backend (issue #434).
//!
//! `W` walkers run the real normal-mode sampler `vmc_make_sample_real_staged` side by side.
//! Everything the sampler does except the Pfaffian algebra stays on the host, in the walker's own
//! thread and in the existing code: candidate generation, projection counters, the acceptance
//! weight, the SFMT draws. The five Pfaffian operations go through [`WalkerStage`] to a
//! [`DeviceService`] that keeps every walker's inverse tables resident on the accelerator:
//!
//! | sampler step | request | crosses the host-device link |
//! | --- | --- | --- |
//! | initial tables | `Begin` | configuration, initial Pfaffians |
//! | proposal ratios | `Hop` / `Exchange` (blocking) | move: slot, site; back: `NQP` ratios |
//! | accepted move | `Accept` (not blocking) | one walker index |
//! | `CalculateMAll` recompute | `Recompute` (blocking) | configuration; back: `NQP` Pfaffians |
//!
//! The Slater planes are built on the device from the configuration and the resident orbital
//! table (never uploaded per step), the inverses are updated in place on the device.
//!
//! # Lock-step
//!
//! A walker that needs a blocking answer parks until the service replied. The service thread
//! (the caller of [`run_lockstep_real`], which owns the device context) collects requests until
//! **every live walker is parked**, then runs one batched pass: begins, accepts, recomputes, then
//! all proposals. Walkers that reject a candidate on the host (`continue` in the sampler) simply
//! advance to their next blocking request. A walker's results never depend on which walkers share
//! a pass, so every output is independent of thread timing and of the batching; per-walker order
//! is kept because a walker issues at most one blocking request at a time and its `Accept` of a
//! proposal precedes its next request.
//!
//! # RNG
//!
//! Draws happen only in the host code around the stage calls, which is the unchanged sampler
//! driver, so draw order and count equal the serial run by construction; the tests compare the
//! full SFMT state. A teacher-forced run ([`Teacher`]) makes every accept/reject follow a
//! reference run's decision, records the flips (the decision the device weights would have
//! made) and classifies them like `accel_validation`.
//!
//! # Convention
//!
//! The device holds mVMC's `invM = -X^-1` (the batched kernel's true inverse with the sign flip
//! of `calc_m_all_real` applied on the device) so the ratio and update kernels are the CPU
//! formulas of `sampling/updates.rs` verbatim.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::c_timer::CTimer;
use crate::pfaffian::calc_m_all_real;
use crate::reducer::SingleProcessReducer;
use crate::run::SamplingWalker;
use crate::sampling::driver::{trace, vmc_make_sample_real_staged, SampleStats};
use crate::sampling::metropolis::MetropolisDecision;
use crate::sampling::stage::{RealPfStage, StageGeom, StageResult, StageTables};
use crate::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};

// ---------------------------------------------------------------------------------------------
// Requests, batches and the service trait
// ---------------------------------------------------------------------------------------------

/// Sizes of one run (identical for every walker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// `Nsite`.
    pub n_site: usize,
    /// Electrons per spin.
    pub n_elec: usize,
    /// QP planes (`NQP`).
    pub n_qp: usize,
}

impl Geometry {
    /// Matrix side `2 * n_elec`.
    pub fn n_size(&self) -> usize {
        2 * self.n_elec
    }
}

/// One walker's initialization request.
#[derive(Debug, Clone)]
pub struct BeginReq {
    /// Walker index.
    pub walker: usize,
    /// Key of the (shared) Slater table in the registry.
    pub slater: u64,
    /// Working configuration `[n_size]`.
    pub ele_idx: Vec<i64>,
    /// Initial Pfaffians `[NQP]` (the host's `calc_m_all_real_native_info` result).
    pub pf: Vec<f64>,
}

/// One slot change of a proposal: electron slot `slot` of `spin` moves to `site`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotMove {
    /// Electron slot `ma + spin * n_elec`.
    pub slot: u32,
    /// Spin of the slot (0 or 1).
    pub spin: u8,
    /// New site of the electron.
    pub site: u32,
}

/// A proposal: one (hop) or two (exchange) slot moves relative to the walker's accepted
/// configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProposeReq {
    /// Walker index.
    pub walker: usize,
    /// First move.
    pub a: SlotMove,
    /// Second move for an exchange.
    pub b: Option<SlotMove>,
}

/// A recomputation request.
#[derive(Debug, Clone)]
pub struct RecomputeReq {
    /// Walker index.
    pub walker: usize,
    /// Configuration `[n_size]`.
    pub ele_idx: Vec<i64>,
}

/// Fast-path requests of one round: accepted moves and proposals. A round answers every
/// proposal before the next round starts.
#[derive(Debug, Default)]
pub struct FastBatch {
    /// Walkers whose pending proposal was accepted (applied before anything else of the round).
    pub accepts: Vec<usize>,
    /// Proposals.
    pub proposes: Vec<ProposeReq>,
}

/// Slow-path requests: initial table construction and the periodic recomputation (a batched
/// Pfaffian/inverse, hundreds of microseconds of device latency). They run asynchronously on a
/// second lane so that they do not stall the proposals of the other walkers.
#[derive(Debug, Default)]
pub struct SlowBatch {
    /// Walkers to initialize.
    pub begins: Vec<BeginReq>,
    /// Recomputations.
    pub recomputes: Vec<RecomputeReq>,
}

impl SlowBatch {
    /// Number of walkers waiting for this batch.
    pub fn walkers(&self) -> usize {
        self.begins.len() + self.recomputes.len()
    }
}

/// Completed slow batch.
#[derive(Debug, Default)]
pub struct SlowReply {
    /// Walkers whose `Begin` completed.
    pub begins: Vec<usize>,
    /// `(walker, failed, pf)` per recompute; `pf` is empty on failure.
    pub recomputes: Vec<(usize, bool, Vec<f64>)>,
}

impl SlowReply {
    /// Number of walkers answered by this reply.
    pub fn walkers(&self) -> usize {
        self.begins.len() + self.recomputes.len()
    }
}

/// A backend that holds the walkers' inverse tables and executes the Pfaffian stages in batches.
///
/// Per-walker program order is guaranteed by the runner (a walker has at most one blocking request
/// outstanding and its accept precedes its next request); the backend must apply the accepts of a
/// round before it reads or recomputes the accepted walkers' tables.
pub trait DeviceService {
    /// Register the shared Slater table `key` (copied or uploaded once). Called before the first
    /// `Begin` that uses it.
    fn register_slater(
        &mut self,
        key: u64,
        geom: Geometry,
        slater: &SlaterElmFlat<f64>,
    ) -> Result<(), String>;

    /// Prepare for `walkers` walkers.
    fn prepare(&mut self, geom: Geometry, walkers: usize) -> Result<(), String>;

    /// One round: apply `fast.accepts`, start `slow` (asynchronously where the backend can),
    /// run `fast.proposes` and return their ratios, `proposes.len() * n_qp` values in order.
    fn round(&mut self, fast: &FastBatch, slow: &SlowBatch) -> Result<Vec<f64>, String>;

    /// Completed slow batches, oldest first. With `block`, waits until at least one completed
    /// when any is in flight.
    fn poll_slow(&mut self, block: bool) -> Result<Vec<SlowReply>, String>;
}

// ---------------------------------------------------------------------------------------------
// Walker-side stage
// ---------------------------------------------------------------------------------------------

enum Msg {
    Begin(BeginReq),
    Propose(ProposeReq),
    Accept(usize),
    Recompute(RecomputeReq),
    Done,
}

/// One-shot reply slot of a walker: the walker spins for `spin` and then parks, the service sets
/// the value and unparks. A futex wake per walker per round from the single service thread costs
/// several microseconds each (hundreds of microseconds at 64 walkers), so replies of one round are
/// delivered through a wake tree ([`deliver`]): the service unparks [`WAKE_FANOUT`] walkers and
/// every woken walker unparks its children before it continues.
struct ReplySlot {
    ready: AtomicBool,
    data: Mutex<Option<Reply>>,
    thread: Mutex<Option<std::thread::Thread>>,
    children: Mutex<Vec<std::thread::Thread>>,
}

const WAKE_FANOUT: usize = 8;

impl ReplySlot {
    fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
            data: Mutex::new(None),
            thread: Mutex::new(None),
            children: Mutex::new(Vec::new()),
        }
    }

    fn put(&self, reply: Reply) {
        *self.data.lock().expect("reply slot") = Some(reply);
        self.ready.store(true, Ordering::Release);
        if let Some(t) = self.thread.lock().expect("reply thread").as_ref() {
            t.unpark();
        }
    }

    fn take(&self, spin: Duration) -> Option<Reply> {
        let t0 = Instant::now();
        loop {
            if self.ready.load(Ordering::Acquire) {
                self.ready.store(false, Ordering::Relaxed);
                let children: Vec<_> = self.children.lock().expect("children").drain(..).collect();
                for c in children {
                    c.unpark();
                }
                return self.data.lock().expect("reply slot").take();
            }
            if t0.elapsed() < spin {
                std::hint::spin_loop();
            } else {
                std::thread::park_timeout(Duration::from_micros(200));
            }
        }
    }
}

/// Deliver the replies of one round: data and children first, then the ready flags, then the
/// wake tree (service wakes the first `WAKE_FANOUT`; node `i` wakes `WAKE_FANOUT*(i+1)..`).
fn deliver(slots: &[Arc<ReplySlot>], replies: Vec<(usize, Reply)>) {
    let order: Vec<usize> = replies.iter().map(|(w, _)| *w).collect();
    let threads: Vec<Option<std::thread::Thread>> = order
        .iter()
        .map(|&w| slots[w].thread.lock().expect("reply thread").clone())
        .collect();
    for (i, &w) in order.iter().enumerate() {
        let lo = WAKE_FANOUT * (i + 1);
        let kids: Vec<std::thread::Thread> = (lo..(lo + WAKE_FANOUT).min(order.len()))
            .filter_map(|j| threads[j].clone())
            .collect();
        *slots[w].children.lock().expect("children") = kids;
    }
    for (w, reply) in replies {
        *slots[w].data.lock().expect("reply slot") = Some(reply);
        slots[w].ready.store(true, Ordering::Release);
    }
    for t in threads.iter().take(WAKE_FANOUT).flatten() {
        t.unpark();
    }
}

enum Reply {
    Unit,
    Pf(Vec<f64>),
    Recompute(bool, Vec<f64>),
    Err(String),
}

/// Shared registry of distinct Slater tables (exact content comparison).
#[derive(Default)]
struct SlaterRegistry {
    tables: Mutex<HashMap<u64, Arc<SlaterElmFlat<f64>>>>,
}

fn content_hash(data: &[f64]) -> u64 {
    // FNV-1a over the bit patterns, 8 bytes at a time.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in data {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

impl SlaterRegistry {
    /// Key of `slater`; inserts a copy when its content is new. Exact (collisions probe on).
    fn key_of(&self, slater: &SlaterElmFlat<f64>, n_site: usize) -> u64 {
        let mut key = content_hash(slater.as_slice());
        let mut map = self.tables.lock().expect("slater registry");
        loop {
            match map.get(&key) {
                Some(existing) if existing.as_slice() == slater.as_slice() => return key,
                Some(_) => key = key.wrapping_add(1),
                None => {
                    let mut copy = SlaterElmFlat::<f64>::zeros(slater.n_qp_full(), n_site);
                    copy.as_mut_slice().copy_from_slice(slater.as_slice());
                    map.insert(key, Arc::new(copy));
                    return key;
                }
            }
        }
    }
}

/// Teacher forcing: every decision follows a reference run; flips are recorded.
#[derive(Debug, Clone, Default)]
pub struct Teacher {
    /// Reference `(weight, draw)` sequence (from `trace::finish_decisions` of the CPU run).
    pub reference: Vec<(f64, f64)>,
    /// Absolute weight tolerance of the flip classification.
    pub weight_abs: f64,
    /// Relative weight tolerance.
    pub weight_rel: f64,
}

/// Outcome of a teacher-forced walker.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TeacherReport {
    /// Decisions compared.
    pub compared: usize,
    /// The device-weight decision differed from the reference decision.
    pub flips: usize,
    /// Flips whose margin exceeds the weight error (not explained by rounding).
    pub defects: usize,
    /// Draws that differ from the reference draw (must be 0: RNG parity).
    pub draw_mismatches: usize,
    /// Largest `|w_device - w_reference|`.
    pub max_weight_abs: f64,
    /// Largest `|w_device - w_reference| / max(|w_reference|, tiny)`.
    pub max_weight_rel: f64,
    /// Smallest margin `|w_reference - draw|`.
    pub min_margin: f64,
    /// More decisions than the reference holds.
    pub overrun: usize,
}

struct TeacherState {
    t: Teacher,
    pos: usize,
    report: TeacherReport,
}

/// Per-walker stage: forwards the Pfaffian operations to the service thread.
pub struct WalkerStage {
    walker: usize,
    tx: Sender<Msg>,
    slot: Arc<ReplySlot>,
    spin: Duration,
    registry: Arc<SlaterRegistry>,
    teacher: Option<TeacherState>,
    geom: Geometry,
    /// Host copy of the accepted configuration is the sampler's own `tmp_ele_idx`; the stage only
    /// remembers the pending proposal's moves (for the accepted-move bookkeeping).
    pending: Option<ProposeReq>,
}

impl WalkerStage {
    fn call(&mut self, msg: Msg) -> StageResult<Reply> {
        self.tx
            .send(msg)
            .map_err(|_| "device service stopped".to_string())?;
        match self.slot.take(self.spin) {
            Some(Reply::Err(e)) => Err(e),
            Some(r) => Ok(r),
            None => Err("device service stopped".to_string()),
        }
    }

    fn slot_move(ele_idx: &[i64], slot: usize, spin: u8) -> SlotMove {
        SlotMove {
            slot: slot as u32,
            spin,
            site: ele_idx[slot] as u32,
        }
    }
}

impl RealPfStage for WalkerStage {
    fn begin(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<()> {
        if g.qp_start != 0 || g.qp_end != self.geom.n_qp {
            return Err("device sampler needs the full QP range (no MPI QP split)".to_string());
        }
        let key = self.registry.key_of(t.slater_elm, g.n_site);
        self.call(Msg::Begin(BeginReq {
            walker: self.walker,
            slater: key,
            ele_idx: ele_idx.to_vec(),
            pf: t.pf_m.to_vec(),
        }))?;
        Ok(())
    }

    fn propose_hop(
        &mut self,
        _g: &StageGeom,
        _t: &mut StageTables<'_>,
        ele_idx: &[i64],
        ma: usize,
        spin: u8,
        pf_new: &mut [f64],
    ) -> StageResult<()> {
        let slot = ma + spin as usize * self.geom.n_elec;
        let req = ProposeReq {
            walker: self.walker,
            a: Self::slot_move(ele_idx, slot, spin),
            b: None,
        };
        self.pending = Some(req);
        match self.call(Msg::Propose(req))? {
            Reply::Pf(v) => {
                pf_new.copy_from_slice(&v);
                Ok(())
            }
            _ => Err("unexpected reply to a hop proposal".to_string()),
        }
    }

    fn accept_hop(
        &mut self,
        _g: &StageGeom,
        t: &mut StageTables<'_>,
        _ele_idx: &[i64],
        _ma: usize,
        _spin: u8,
        pf_new: &[f64],
    ) -> StageResult<()> {
        self.pending = None;
        self.tx
            .send(Msg::Accept(self.walker))
            .map_err(|_| "device service stopped".to_string())?;
        // The host keeps its own copy of the current Pfaffians for the Metropolis bookkeeping.
        t.pf_m.copy_from_slice(pf_new);
        Ok(())
    }

    fn propose_exchange(
        &mut self,
        _g: &StageGeom,
        _t: &mut StageTables<'_>,
        ele_idx: &[i64],
        slots: [(usize, u8); 2],
        pf_new: &mut [f64],
    ) -> StageResult<()> {
        let ne = self.geom.n_elec;
        let req = ProposeReq {
            walker: self.walker,
            a: Self::slot_move(ele_idx, slots[0].0 + slots[0].1 as usize * ne, slots[0].1),
            b: Some(Self::slot_move(
                ele_idx,
                slots[1].0 + slots[1].1 as usize * ne,
                slots[1].1,
            )),
        };
        self.pending = Some(req);
        match self.call(Msg::Propose(req))? {
            Reply::Pf(v) => {
                pf_new.copy_from_slice(&v);
                Ok(())
            }
            _ => Err("unexpected reply to an exchange proposal".to_string()),
        }
    }

    fn accept_exchange(
        &mut self,
        _g: &StageGeom,
        _t: &mut StageTables<'_>,
        _ele_idx: &[i64],
        _slots: [(usize, u8); 2],
        _old_sites: [usize; 2],
    ) -> StageResult<()> {
        self.pending = None;
        self.tx
            .send(Msg::Accept(self.walker))
            .map_err(|_| "device service stopped".to_string())
    }

    fn recompute(
        &mut self,
        _g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<bool> {
        match self.call(Msg::Recompute(RecomputeReq {
            walker: self.walker,
            ele_idx: ele_idx.to_vec(),
        }))? {
            Reply::Recompute(failed, pf) => {
                if !failed {
                    t.pf_m.copy_from_slice(&pf);
                }
                Ok(failed)
            }
            _ => Err("unexpected reply to a recompute".to_string()),
        }
    }

    fn decide(&mut self, d: &MetropolisDecision) -> bool {
        let Some(ts) = self.teacher.as_mut() else {
            return d.accepted;
        };
        let Some(&(w_ref, u_ref)) = ts.t.reference.get(ts.pos) else {
            ts.report.overrun += 1;
            return d.accepted;
        };
        ts.pos += 1;
        let r = &mut ts.report;
        r.compared += 1;
        if d.draw.to_bits() != u_ref.to_bits() {
            r.draw_mismatches += 1;
        }
        let accept_ref = w_ref > u_ref;
        let err = (d.weight - w_ref).abs();
        r.max_weight_abs = r.max_weight_abs.max(err);
        r.max_weight_rel = r
            .max_weight_rel
            .max(err / w_ref.abs().max(f64::MIN_POSITIVE));
        let margin = (w_ref - u_ref).abs();
        r.min_margin = r.min_margin.min(margin);
        if d.accepted != accept_ref {
            r.flips += 1;
            let allowed = ts.t.weight_abs + ts.t.weight_rel * w_ref.abs();
            if margin > err || margin > allowed {
                r.defects += 1;
            }
        }
        accept_ref
    }
}

// ---------------------------------------------------------------------------------------------
// The lock-step runner
// ---------------------------------------------------------------------------------------------

/// Result of one walker of a lock-step run.
#[derive(Debug)]
pub struct WalkerRun {
    /// Sampler statistics, or the error.
    pub stats: Result<SampleStats, String>,
    /// Recorded Metropolis `(weight, draw)` of every decision, in order.
    pub decisions: Vec<(f64, f64)>,
    /// Teacher-forcing report when a [`Teacher`] was supplied.
    pub teacher: Option<TeacherReport>,
}

/// Options of [`run_lockstep_real`].
#[derive(Debug, Clone, Default)]
pub struct LockstepOptions {
    /// One teacher per walker (same length as the walkers), or empty.
    pub teachers: Vec<Teacher>,
    /// Number of lock-step passes executed (output).
    pub passes: usize,
    /// How long a parked walker spins for the reply before parking, in microseconds
    /// (`0` parks immediately); `None` spins 50 us when the walkers do not outnumber the cores
    /// and parks immediately otherwise (spinning walkers would starve the service thread).
    pub spin_us: Option<u64>,
}

/// Statistics of the service loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct LockstepStats {
    /// Passes executed.
    pub passes: usize,
    /// Proposals processed.
    pub proposals: usize,
    /// Accepted moves applied.
    pub accepts: usize,
    /// Recomputes executed.
    pub recomputes: usize,
    /// Largest batch (blocking requests in one pass).
    pub max_batch: usize,
    /// Seconds the service thread spent waiting for walker requests.
    pub recv_s: f64,
    /// Seconds inside `DeviceService::round`.
    pub round_s: f64,
    /// Seconds delivering replies (wake tree).
    pub deliver_s: f64,
}

/// Geometry of a prepared walker.
pub fn geometry_of(w: &SamplingWalker) -> Geometry {
    Geometry {
        n_site: w.data.modpara.nsite as usize,
        n_elec: w.data.modpara.nelec as usize,
        n_qp: w.state.slater_matrix.slater_elm_real.n_qp_full(),
    }
}

/// Sync the real Slater table from the complex master (the driver does the same at its start;
/// repeating it is idempotent). Used to compare tables before the run.
fn sync_slater_real(state: &mut crate::state::VmcOptimizationState) {
    let n = state.slater_matrix.slater_elm.as_slice().len();
    let src = state.slater_matrix.slater_elm.as_slice();
    let dst = state.slater_matrix.slater_elm_real.as_mut_slice();
    assert_eq!(n, dst.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.re;
    }
}

/// Run `walkers` in lock-step against `service`. Returns the per-walker results and the loop
/// statistics. Walker threads are scoped; the calling thread owns the service.
///
/// # Errors
///
/// A service failure aborts every walker (their `stats` carry the error) and is returned.
pub fn run_lockstep_real<S: DeviceService + ?Sized>(
    walkers: &mut [SamplingWalker],
    service: &mut S,
    mut opts: LockstepOptions,
) -> Result<(Vec<WalkerRun>, LockstepStats), String> {
    if walkers.is_empty() {
        return Err("lock-step runner needs at least one walker".to_string());
    }
    if !opts.teachers.is_empty() && opts.teachers.len() != walkers.len() {
        return Err("one teacher per walker is required".to_string());
    }
    crate::serial_blas::initialize();
    let geom = geometry_of(&walkers[0]);
    for w in walkers.iter() {
        if geometry_of(w) != geom {
            return Err("all walkers must share one geometry".to_string());
        }
    }
    let registry = Arc::new(SlaterRegistry::default());
    service.prepare(geom, walkers.len())?;

    let count = walkers.len();
    let (tx, rx) = channel::<Msg>();
    let slots: Vec<Arc<ReplySlot>> = (0..count).map(|_| Arc::new(ReplySlot::new())).collect();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let spin = Duration::from_micros(opts.spin_us.unwrap_or(if count < cores { 50 } else { 0 }));
    let teachers = std::mem::take(&mut opts.teachers);
    let mut loop_stats = LockstepStats::default();
    let mut service_error: Option<String> = None;
    let mut known_slater: Vec<u64> = Vec::new();

    let runs: Vec<WalkerRun> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count);
        for (w, walker) in walkers.iter_mut().enumerate() {
            let stage = WalkerStage {
                walker: w,
                tx: tx.clone(),
                slot: slots[w].clone(),
                spin,
                registry: registry.clone(),
                teacher: teachers.get(w).map(|t| TeacherState {
                    t: t.clone(),
                    pos: 0,
                    report: TeacherReport {
                        min_margin: f64::INFINITY,
                        ..TeacherReport::default()
                    },
                }),
                geom,
                pending: None,
            };
            let done_tx = tx.clone();
            handles.push(scope.spawn(move || {
                struct DoneGuard(Sender<Msg>);
                impl Drop for DoneGuard {
                    fn drop(&mut self) {
                        let _ = self.0.send(Msg::Done);
                    }
                }
                let _guard = DoneGuard(done_tx);
                let mut stage = stage;
                *stage.slot.thread.lock().expect("reply thread") = Some(std::thread::current());
                trace::start_decisions();
                sync_slater_real(&mut walker.state);
                let result = vmc_make_sample_real_staged(
                    &walker.data,
                    &mut walker.state,
                    &mut walker.rng,
                    &mut CTimer::<false>::new(),
                    &SingleProcessReducer,
                    &mut stage,
                );
                let decisions = trace::finish_decisions();
                WalkerRun {
                    stats: result.map_err(|e| e.to_string()),
                    decisions,
                    teacher: stage.teacher.map(|t| t.report),
                }
            }));
        }
        drop(tx);

        // Service loop. A round fires when every live walker is parked on the fast path or is
        // waiting for a slow batch (pending or in flight): a recompute that is in flight does not
        // hold up the proposals of the other walkers.
        let mut live = count;
        let mut fast = FastBatch::default();
        let mut slow = SlowBatch::default();
        let mut inflight: Vec<usize> = Vec::new();
        let mut pending_walkers: Vec<usize> = Vec::new();
        let mut disconnected = false;
        let nq = geom.n_qp;
        let fail = |e: &str, who: &[usize]| {
            for &w in who {
                slots[w].put(Reply::Err(e.to_string()));
            }
        };
        'service: loop {
            if live == 0 {
                break;
            }
            // 1. receive; block only when nothing else can progress
            let t_recv = Instant::now();
            let mut got = false;
            loop {
                let msg = if got || !inflight.is_empty() {
                    rx.try_recv().ok()
                } else {
                    match rx.recv() {
                        Ok(m) => Some(m),
                        Err(_) => {
                            disconnected = true;
                            None
                        }
                    }
                };
                let Some(msg) = msg else { break };
                got = true;
                match msg {
                    Msg::Done => live -= 1,
                    Msg::Accept(w) => fast.accepts.push(w),
                    Msg::Begin(b) => {
                        if let Some(e) = &service_error {
                            slots[b.walker].put(Reply::Err(e.clone()));
                        } else {
                            pending_walkers.push(b.walker);
                            slow.begins.push(b);
                        }
                    }
                    Msg::Recompute(r) => {
                        if let Some(e) = &service_error {
                            slots[r.walker].put(Reply::Err(e.clone()));
                        } else {
                            pending_walkers.push(r.walker);
                            slow.recomputes.push(r);
                        }
                    }
                    Msg::Propose(p) => {
                        if let Some(e) = &service_error {
                            slots[p.walker].put(Reply::Err(e.clone()));
                        } else {
                            fast.proposes.push(p);
                        }
                    }
                }
            }
            loop_stats.recv_s += t_recv.elapsed().as_secs_f64();
            if disconnected {
                break;
            }
            // 2. completed slow batches
            if !inflight.is_empty() {
                match service.poll_slow(false) {
                    Ok(done) => {
                        let mut out = Vec::new();
                        for reply in done {
                            for &w in &reply.begins {
                                out.push((w, Reply::Unit));
                                inflight.retain(|&x| x != w);
                            }
                            for (w, failed, pf) in reply.recomputes {
                                out.push((w, Reply::Recompute(failed, pf)));
                                inflight.retain(|&x| x != w);
                            }
                        }
                        deliver(&slots, out);
                    }
                    Err(e) => {
                        service_error = Some(e.clone());
                        fail(&e, &inflight);
                        inflight.clear();
                    }
                }
            }
            // 3. fire a round
            let waiting = fast.proposes.len() + pending_walkers.len() + inflight.len();
            if service_error.is_some() {
                // everything still parked was answered with the error; walkers are unwinding
                let who: Vec<usize> = fast.proposes.iter().map(|p| p.walker).collect();
                fail(service_error.as_deref().unwrap_or(""), &who);
                fast = FastBatch::default();
                fail(service_error.as_deref().unwrap_or(""), &pending_walkers);
                pending_walkers.clear();
                slow = SlowBatch::default();
                continue;
            }
            if waiting == live && (!fast.proposes.is_empty() || !pending_walkers.is_empty()) {
                for b in &slow.begins {
                    if !known_slater.contains(&b.slater) {
                        let table = registry
                            .tables
                            .lock()
                            .expect("registry")
                            .get(&b.slater)
                            .cloned();
                        let r = table
                            .ok_or_else(|| "unregistered Slater table".to_string())
                            .and_then(|t| service.register_slater(b.slater, geom, &t));
                        match r {
                            Ok(()) => known_slater.push(b.slater),
                            Err(e) => {
                                service_error = Some(e);
                                break;
                            }
                        }
                    }
                }
                let t_round = Instant::now();
                let outcome = if let Some(e) = &service_error {
                    Err(e.clone())
                } else {
                    // A panicking service must not leave parked walkers waiting forever.
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        service.round(&fast, &slow)
                    }))
                    .unwrap_or_else(|_| Err("device service panicked".to_string()))
                };
                loop_stats.round_s += t_round.elapsed().as_secs_f64();
                loop_stats.passes += 1;
                loop_stats.proposals += fast.proposes.len();
                loop_stats.accepts += fast.accepts.len();
                loop_stats.recomputes += slow.recomputes.len();
                loop_stats.max_batch = loop_stats.max_batch.max(fast.proposes.len());
                match outcome {
                    Ok(pf_new) => {
                        let out: Vec<(usize, Reply)> = fast
                            .proposes
                            .iter()
                            .enumerate()
                            .map(|(i, p)| {
                                (p.walker, Reply::Pf(pf_new[i * nq..(i + 1) * nq].to_vec()))
                            })
                            .collect();
                        let t_deliver = Instant::now();
                        deliver(&slots, out);
                        loop_stats.deliver_s += t_deliver.elapsed().as_secs_f64();
                        inflight.append(&mut pending_walkers);
                    }
                    Err(e) => {
                        service_error = Some(e.clone());
                        let who: Vec<usize> = fast
                            .proposes
                            .iter()
                            .map(|p| p.walker)
                            .chain(pending_walkers.iter().copied())
                            .chain(inflight.iter().copied())
                            .collect();
                        fail(&e, &who);
                        pending_walkers.clear();
                        inflight.clear();
                    }
                }
                fast = FastBatch::default();
                slow = SlowBatch::default();
                continue 'service;
            }
            // everyone is waiting on slow work: block for it instead of spinning
            if waiting == live && !inflight.is_empty() {
                match service.poll_slow(true) {
                    Ok(done) => {
                        let mut out = Vec::new();
                        for reply in done {
                            for &w in &reply.begins {
                                out.push((w, Reply::Unit));
                                inflight.retain(|&x| x != w);
                            }
                            for (w, failed, pf) in reply.recomputes {
                                out.push((w, Reply::Recompute(failed, pf)));
                                inflight.retain(|&x| x != w);
                            }
                        }
                        deliver(&slots, out);
                    }
                    Err(e) => {
                        service_error = Some(e.clone());
                        fail(&e, &inflight);
                        inflight.clear();
                    }
                }
            } else if !inflight.is_empty() {
                std::hint::spin_loop();
            }
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("walker thread panicked"))
            .collect()
    });
    opts.passes = loop_stats.passes;
    match service_error {
        Some(e) => Err(e),
        None => Ok((runs, loop_stats)),
    }
}

// ---------------------------------------------------------------------------------------------
// Host service: the CPU kernels behind the device protocol
// ---------------------------------------------------------------------------------------------

struct HostWalker {
    ele_idx: Vec<i64>,
    inv: InvMColMajor<f64>,
    pf: Vec<f64>,
    pending: Option<(ProposeReq, Vec<i64>, Vec<f64>)>,
    slater: u64,
}

/// [`DeviceService`] on the host: per-walker resident tables updated by the same flat CPU
/// kernels as [`CpuStage`](crate::sampling::stage::CpuStage). It runs in normal CI, tests the
/// request protocol and the lock-step plumbing, and is the reference of the CUDA service.
pub struct HostService {
    geom: Option<Geometry>,
    walkers: Vec<Option<HostWalker>>,
    slater: HashMap<u64, SlaterElmFlat<f64>>,
    pool: Option<ThreadedPfaPackWorkspace>,
    done: std::collections::VecDeque<SlowReply>,
}

impl Default for HostService {
    fn default() -> Self {
        Self::new()
    }
}

impl HostService {
    /// Empty service.
    pub fn new() -> Self {
        Self {
            geom: None,
            walkers: Vec::new(),
            slater: HashMap::new(),
            pool: None,
            done: std::collections::VecDeque::new(),
        }
    }
}

fn candidate_of(cur: &[i64], p: &ProposeReq) -> Vec<i64> {
    let mut c = cur.to_vec();
    c[p.a.slot as usize] = i64::from(p.a.site);
    if let Some(b) = p.b {
        c[b.slot as usize] = i64::from(b.site);
    }
    c
}

impl DeviceService for HostService {
    fn register_slater(
        &mut self,
        key: u64,
        geom: Geometry,
        slater: &SlaterElmFlat<f64>,
    ) -> Result<(), String> {
        let mut copy = SlaterElmFlat::<f64>::zeros(geom.n_qp, geom.n_site);
        copy.as_mut_slice().copy_from_slice(slater.as_slice());
        self.slater.insert(key, copy);
        Ok(())
    }

    fn prepare(&mut self, geom: Geometry, walkers: usize) -> Result<(), String> {
        self.geom = Some(geom);
        self.walkers = (0..walkers).map(|_| None).collect();
        self.pool = Some(ThreadedPfaPackWorkspace::new(geom.n_size(), 1));
        Ok(())
    }

    fn round(&mut self, fast: &FastBatch, slow: &SlowBatch) -> Result<Vec<f64>, String> {
        let geom = self.geom.ok_or("service not prepared")?;
        let pool = self.pool.as_ref().ok_or("service not prepared")?;
        let n_size = geom.n_size();
        let inv_stride = n_size * n_size + 1;
        for &w in &fast.accepts {
            let hw = self.walkers[w].as_mut().ok_or("walker not begun")?;
            let (req, cand, pf_new) = hw.pending.take().ok_or("accept without proposal")?;
            let slater = &self.slater[&hw.slater];
            match req.b {
                None => {
                    let ma = req.a.slot as usize % geom.n_elec;
                    crate::sampling::updates::update_m_all_real_flat(
                        ma,
                        req.a.spin,
                        &cand,
                        slater,
                        hw.inv.as_mut_slice(),
                        inv_stride,
                        &mut hw.pf,
                        0,
                        geom.n_qp,
                        geom.n_site,
                        geom.n_elec,
                    );
                    hw.pf.copy_from_slice(&pf_new);
                }
                Some(b) => {
                    let ne = geom.n_elec;
                    let (ma, mb) = (req.a.slot as usize % ne, b.slot as usize % ne);
                    let ra_old = hw.ele_idx[req.a.slot as usize] as usize;
                    let rb_old = hw.ele_idx[b.slot as usize] as usize;
                    crate::sampling::updates::update_m_all_two_real_flat(
                        ma,
                        req.a.spin,
                        mb,
                        b.spin,
                        ra_old,
                        rb_old,
                        &cand,
                        slater,
                        hw.inv.as_mut_slice(),
                        inv_stride,
                        &mut hw.pf,
                        0,
                        geom.n_qp,
                        geom.n_site,
                        geom.n_elec,
                    );
                }
            }
            hw.ele_idx = cand;
        }
        // slow batch: the host executes it immediately and queues the completed reply
        if slow.walkers() > 0 {
            let mut reply = SlowReply::default();
            for b in &slow.begins {
                let slater = self.slater.get(&b.slater).ok_or("unknown Slater table")?;
                let mut inv = InvMColMajor::<f64>::zeros(geom.n_qp, geom.n_elec);
                let mut pf = vec![0.0; geom.n_qp];
                calc_m_all_real(
                    &b.ele_idx,
                    slater,
                    &mut inv,
                    &mut pf,
                    0,
                    geom.n_qp,
                    geom.n_site,
                    geom.n_elec,
                    pool,
                )
                .map_err(|e| format!("initial recompute failed: {e:?}"))?;
                self.walkers[b.walker] = Some(HostWalker {
                    ele_idx: b.ele_idx.clone(),
                    inv,
                    pf: b.pf.clone(),
                    pending: None,
                    slater: b.slater,
                });
                reply.begins.push(b.walker);
            }
            for r in &slow.recomputes {
                let hw = self.walkers[r.walker].as_mut().ok_or("walker not begun")?;
                let slater = &self.slater[&hw.slater];
                let failed = calc_m_all_real(
                    &r.ele_idx,
                    slater,
                    &mut hw.inv,
                    &mut hw.pf,
                    0,
                    geom.n_qp,
                    geom.n_site,
                    geom.n_elec,
                    pool,
                )
                .is_err();
                hw.ele_idx = r.ele_idx.clone();
                reply.recomputes.push((
                    r.walker,
                    failed,
                    if failed { Vec::new() } else { hw.pf.clone() },
                ));
            }
            self.done.push_back(reply);
        }
        let mut pf_out = Vec::with_capacity(fast.proposes.len() * geom.n_qp);
        for p in &fast.proposes {
            let hw = self.walkers[p.walker].as_mut().ok_or("walker not begun")?;
            let slater = &self.slater[&hw.slater];
            let cand = candidate_of(&hw.ele_idx, p);
            let mut pf_new = vec![0.0; geom.n_qp];
            let ne = geom.n_elec;
            match p.b {
                None => crate::sampling::updates::calculate_new_pf_m2_real_flat(
                    p.a.slot as usize % ne,
                    p.a.spin,
                    &mut pf_new,
                    &cand,
                    slater,
                    hw.inv.as_slice(),
                    inv_stride,
                    &hw.pf,
                    0,
                    geom.n_qp,
                    geom.n_site,
                    geom.n_elec,
                ),
                Some(b) => crate::sampling::updates::calculate_new_pf_m_two2_real_flat::<false>(
                    p.a.slot as usize % ne,
                    p.a.spin,
                    b.slot as usize % ne,
                    b.spin,
                    &mut pf_new,
                    &cand,
                    slater,
                    hw.inv.as_slice(),
                    inv_stride,
                    &hw.pf,
                    0,
                    geom.n_qp,
                    geom.n_site,
                    geom.n_elec,
                ),
            }
            pf_out.extend_from_slice(&pf_new);
            hw.pending = Some((*p, cand, pf_new));
        }
        Ok(pf_out)
    }

    fn poll_slow(&mut self, _block: bool) -> Result<Vec<SlowReply>, String> {
        Ok(self.done.drain(..).collect())
    }
}
