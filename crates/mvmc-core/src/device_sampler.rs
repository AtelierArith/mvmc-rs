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
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

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

/// Requests collected for one lock-step pass, in the order they must be processed.
#[derive(Debug, Default)]
pub struct Batch {
    /// Walkers to initialize.
    pub begins: Vec<BeginReq>,
    /// Walkers whose pending proposal was accepted.
    pub accepts: Vec<usize>,
    /// Recomputations.
    pub recomputes: Vec<RecomputeReq>,
    /// Proposals.
    pub proposes: Vec<ProposeReq>,
}

/// Answers of one pass.
#[derive(Debug, Default)]
pub struct BatchReply {
    /// For each recompute: `(failed, pf)`; `pf` is empty on failure.
    pub recomputes: Vec<(bool, Vec<f64>)>,
    /// Proposal ratios, `proposes.len() * n_qp` values in request order.
    pub pf_new: Vec<f64>,
}

/// A backend that holds the walkers' inverse tables and executes the Pfaffian stages in batches.
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

    /// Run one pass. Order: `begins`, `accepts`, `recomputes`, `proposes`.
    fn process(&mut self, batch: &Batch) -> Result<BatchReply, String>;
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
    rx: Receiver<Reply>,
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
        match self.rx.recv() {
            Ok(Reply::Err(e)) => Err(e),
            Ok(r) => Ok(r),
            Err(_) => Err("device service stopped".to_string()),
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
    let mut reply_tx: Vec<Sender<Reply>> = Vec::with_capacity(count);
    let mut reply_rx: Vec<Option<Receiver<Reply>>> = Vec::with_capacity(count);
    for _ in 0..count {
        let (t, r) = channel::<Reply>();
        reply_tx.push(t);
        reply_rx.push(Some(r));
    }
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
                rx: reply_rx[w].take().expect("reply receiver"),
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

        // Service loop.
        let mut live = count;
        let mut waiting: Vec<usize> = Vec::new();
        let mut batch = Batch::default();
        while live > 0 {
            let Ok(msg) = rx.recv() else { break };
            match msg {
                Msg::Done => live -= 1,
                Msg::Accept(w) => batch.accepts.push(w),
                Msg::Begin(b) => {
                    waiting.push(b.walker);
                    batch.begins.push(b);
                }
                Msg::Recompute(r) => {
                    waiting.push(r.walker);
                    batch.recomputes.push(r);
                }
                Msg::Propose(p) => {
                    waiting.push(p.walker);
                    batch.proposes.push(p);
                }
            }
            if live == 0 || waiting.len() < live {
                continue;
            }
            // every live walker is parked: one pass
            if service_error.is_none() {
                for b in &batch.begins {
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
            }
            let outcome = if let Some(e) = &service_error {
                Err(e.clone())
            } else {
                // A panicking service must not leave parked walkers waiting forever.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| service.process(&batch)))
                    .unwrap_or_else(|_| Err("device service panicked".to_string()))
            };
            loop_stats.passes += 1;
            loop_stats.proposals += batch.proposes.len();
            loop_stats.accepts += batch.accepts.len();
            loop_stats.recomputes += batch.recomputes.len();
            loop_stats.max_batch = loop_stats.max_batch.max(waiting.len());
            match outcome {
                Ok(reply) => {
                    for b in &batch.begins {
                        let _ = reply_tx[b.walker].send(Reply::Unit);
                    }
                    for (r, (failed, pf)) in batch.recomputes.iter().zip(reply.recomputes) {
                        let _ = reply_tx[r.walker].send(Reply::Recompute(failed, pf));
                    }
                    for (i, p) in batch.proposes.iter().enumerate() {
                        let nq = geom.n_qp;
                        let _ = reply_tx[p.walker]
                            .send(Reply::Pf(reply.pf_new[i * nq..(i + 1) * nq].to_vec()));
                    }
                }
                Err(e) => {
                    service_error = Some(e.clone());
                    for &w in &waiting {
                        let _ = reply_tx[w].send(Reply::Err(e.clone()));
                    }
                }
            }
            batch = Batch::default();
            waiting.clear();
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

    fn process(&mut self, batch: &Batch) -> Result<BatchReply, String> {
        let geom = self.geom.ok_or("service not prepared")?;
        let pool = self.pool.as_ref().ok_or("service not prepared")?;
        let n_size = geom.n_size();
        let inv_stride = n_size * n_size + 1;
        for b in &batch.begins {
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
        }
        for &w in &batch.accepts {
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
        let mut reply = BatchReply::default();
        for r in &batch.recomputes {
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
            reply
                .recomputes
                .push((failed, if failed { Vec::new() } else { hw.pf.clone() }));
        }
        for p in &batch.proposes {
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
            reply.pf_new.extend_from_slice(&pf_new);
            hw.pending = Some((*p, cand, pf_new));
        }
        Ok(reply)
    }
}
