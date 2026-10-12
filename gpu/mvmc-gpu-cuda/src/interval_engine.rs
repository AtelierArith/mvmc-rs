//! Group coordinator and `SamplingEngine` for the device-resident interval sampler (issue #547).
//!
//! One [`IntervalGroup`] is shared by every walker of a run (`Arc`). Each walker has a
//! per-walker [`WalkerState`] slot (host staging) and the group owns the single resident
//! [`IntervalSampler`] sized for `walkers` blocks. A block is advanced by every walker thread
//! calling [`IntervalEngine::run_block`] once: each stages its block tape, rendezvous through
//! `Reducer::barrier()` outside any lock, rank 0 runs `k_interval_hop` for all walkers (with
//! device rebuilds), and each walker reads its slot back. Only rank 0 touches the device; its
//! `CudaBackend` is a thread-local (a tenferro context is thread-affine) while the resident
//! `DeviceBuffer`s, which are `Send`, live in the shared group.
//!
//! The RNG contract: each walker generates its own block tape from its own stream (5 words per
//! hop, the Julia GPU layout of `interval_kernels.cu`), so batching never merges streams.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use mvmc_core::reducer::Reducer;
use mvmc_core::sampling::engine::{BlockCtx, SamplingEngine};
use mvmc_core::sampling::rng::SamplingRng;
use mvmc_core::sampling::stage::{StageGeom, StageResult, StageTables};
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_tensor::BackendSessionHost;

use crate::interval::{IntervalGeometry, IntervalHost, IntervalSampler};

thread_local! {
    /// The device backend of the rank-0 thread. A tenferro context is thread-affine, and only
    /// rank 0 issues device work, so the group keeps it here rather than behind the shared lock.
    static RANK0_BACKEND: RefCell<Option<CudaBackend>> = const { RefCell::new(None) };
}

/// Draw a block tape of `5 * n_hops` words from `rng` (source, spin, destination rank,
/// exchange threshold, accept - the Julia GPU layout).
pub fn tape_words<G: SamplingRng + ?Sized>(rng: &mut G, n_hops: usize) -> Vec<u32> {
    (0..5 * n_hops).map(|_| rng.gen_rand32()).collect()
}

/// Host staging of one walker: the state the group uploads (at `begin`) and the results it
/// downloads after each block.
#[derive(Clone, Debug)]
pub struct WalkerState {
    /// Configuration `[n]` (site index per electron slot).
    pub ele: Vec<i32>,
    /// Inverse `nqp*n*n` (mVMC convention).
    pub inv: Vec<f64>,
    /// Pfaffians `[nqp]`.
    pub pf: Vec<f64>,
    /// Cached `log|IP|` after the block.
    pub logip: f64,
    /// Accepted moves since the last rebuild (after the block).
    pub aw: i32,
    /// Accepted moves in the last block.
    pub accepts: i32,
}

/// Shared coordinator of one group of walkers.
pub struct IntervalGroup {
    geom: IntervalGeometry,
    slater: Vec<f64>,
    weights: Vec<f64>,
    gutz: Vec<i32>,
    jast: Vec<i32>,
    params: Vec<f64>,
    states: Vec<Mutex<WalkerState>>,
    /// Per-block tapes `[5][walker][step]` (linear `5*(walker + walkers*step) + word`).
    tapes: Mutex<Vec<u32>>,
    /// Number of hops per block (set on first use).
    n_in_step: Mutex<usize>,
    sampler: Mutex<Option<IntervalSampler>>,
}

impl IntervalGroup {
    /// Build a group of `geom.walkers` walkers sharing the Slater/weight/projection tables.
    pub fn new(
        geom: IntervalGeometry,
        slater: Vec<f64>,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
    ) -> Arc<Self> {
        let states = (0..geom.walkers)
            .map(|_| {
                Mutex::new(WalkerState {
                    ele: vec![0; geom.n],
                    inv: vec![0.0; geom.nqp * geom.n * geom.n],
                    pf: vec![0.0; geom.nqp],
                    logip: 0.0,
                    aw: 0,
                    accepts: 0,
                })
            })
            .collect();
        Arc::new(Self {
            geom,
            slater,
            weights,
            gutz,
            jast,
            params,
            states,
            tapes: Mutex::new(Vec::new()),
            n_in_step: Mutex::new(0),
            sampler: Mutex::new(None),
        })
    }

    /// Geometry of the group.
    pub fn geometry(&self) -> IntervalGeometry {
        self.geom
    }

    /// Store walker `w`'s initial state (called once per series, before any block).
    pub fn stage_initial(&self, w: usize, ele: Vec<i32>, inv: Vec<f64>, pf: Vec<f64>) {
        let mut st = self.states[w].lock().unwrap();
        st.ele = ele;
        st.inv = inv;
        st.pf = pf;
        st.accepts = 0;
    }

    /// Read walker `w`'s state (results after a block).
    pub fn walker_state(&self, w: usize) -> WalkerState {
        self.states[w].lock().unwrap().clone()
    }

    /// Advance one block for walker `w` with its tape and rendezvous. Every walker of the run
    /// must call this once per block; `n_in_step` is the block length.
    pub fn run_block(
        &self,
        w: usize,
        n_in_step: usize,
        tape: &[u32],
        barrier: &dyn Fn(),
    ) -> Result<(), String> {
        debug_assert_eq!(tape.len(), 5 * n_in_step);
        let world = self.geom.walkers;
        {
            let mut n = self.n_in_step.lock().unwrap();
            if *n == 0 {
                *n = n_in_step;
                *self.tapes.lock().unwrap() = vec![0u32; world * n_in_step * 5];
            }
            if *n != n_in_step {
                return Err(format!(
                    "interval group block size changed from {} to {n_in_step}",
                    *n
                ));
            }
        }
        {
            let mut tapes = self.tapes.lock().unwrap();
            for step in 0..n_in_step {
                for word in 0..5 {
                    tapes[5 * (w + world * step) + word] = tape[5 * step + word];
                }
            }
        }
        barrier();
        if w == 0 {
            self.run_device(n_in_step)?;
        }
        barrier();
        Ok(())
    }

    /// Rank-0 device pass: (re)build the resident sampler, upload the tapes, run rounds with
    /// rebuilds, download every walker's result into its slot.
    fn run_device(&self, n_in_step: usize) -> Result<(), String> {
        RANK0_BACKEND.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let devices = cuda_devices().map_err(|e| e.to_string())?;
                *slot = Some(CudaBackend::new(devices[0].id()).map_err(|e| e.to_string())?);
            }
            let backend = slot.as_mut().unwrap();
            backend.with_backend_session(|s| -> Result<(), String> {
                let mut guard = self.sampler.lock().unwrap();
                if guard.is_none() {
                    // Build the concatenated host state from the slots of every walker.
                    let mut ele_all = Vec::with_capacity(self.geom.walkers * self.geom.n);
                    let mut inv_all = Vec::new();
                    let mut pf_all = Vec::new();
                    for w in 0..self.geom.walkers {
                        let st = self.states[w].lock().unwrap();
                        ele_all.extend_from_slice(&st.ele);
                        inv_all.extend_from_slice(&st.inv);
                        pf_all.extend_from_slice(&st.pf);
                    }
                    let host = IntervalHost {
                        ele: &ele_all,
                        inv: &inv_all,
                        pf: &pf_all,
                        slater: &self.slater,
                        weights: &self.weights,
                        gutz: &self.gutz,
                        jast: &self.jast,
                        params: &self.params,
                    };
                    *guard = Some(IntervalSampler::prepare(
                        s,
                        self.geom,
                        &host,
                        n_in_step,
                        0.0,
                    )?);
                }
                let sampler = guard.as_ref().unwrap();
                sampler.reset_offsets(s)?;
                let tapes = self.tapes.lock().unwrap().clone();
                sampler.upload_tape(s, &tapes)?;
                let mut accepts = vec![0i32; self.geom.walkers];
                let mut hops = vec![0i32; self.geom.walkers];
                loop {
                    let results = sampler.run(s)?;
                    let mut need = Vec::new();
                    for (w, &(d, r, a)) in results.iter().enumerate() {
                        accepts[w] += a;
                        hops[w] += d;
                        if r != 0 {
                            need.push(w);
                        }
                    }
                    if need.is_empty() {
                        break;
                    }
                    sampler.rebuild(s, &need)?;
                }
                debug_assert!(hops.iter().all(|&h| h as usize == n_in_step));
                for (w, (st_mutex, &acc)) in self.states.iter().zip(accepts.iter()).enumerate()
                {
                    let (ele, pf, aw, logip) = sampler.download_walker(s, w)?;
                    let mut st = st_mutex.lock().unwrap();
                    st.ele = ele;
                    st.pf = pf;
                    st.aw = aw;
                    st.logip = logip;
                    st.accepts = acc;
                }
                Ok(())
            })
        })
    }
}

/// Per-walker [`SamplingEngine`] on a shared [`IntervalGroup`].
pub struct IntervalEngine {
    group: Arc<IntervalGroup>,
    walker: usize,
}

impl IntervalEngine {
    /// Engine for walker `walker` of `group`.
    pub fn new(group: Arc<IntervalGroup>, walker: usize) -> Self {
        Self { group, walker }
    }
}

impl SamplingEngine for IntervalEngine {
    fn begin(
        &mut self,
        _geom: &StageGeom,
        tables: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<()> {
        let g = self.group.geometry();
        let mut inv = Vec::with_capacity(g.nqp * g.n * g.n);
        for q in 0..g.nqp {
            inv.extend_from_slice(tables.inv_m.qp_matrix_slice(q));
        }
        self.group.stage_initial(
            self.walker,
            ele_idx.iter().map(|&v| v as i32).collect(),
            inv,
            tables.pf_m.clone(),
        );
        Ok(())
    }

    fn run_block<const TIMED: bool, G: SamplingRng, R: Reducer + ?Sized>(
        &mut self,
        ctx: &mut BlockCtx<'_, G, R>,
        _timer: &mut mvmc_core::c_timer::CTimer<TIMED>,
        n_in_step: usize,
    ) -> StageResult<()> {
        let tape = tape_words(ctx.rng, n_in_step);
        let reducer = ctx.reducer;
        self.group
            .run_block(self.walker, n_in_step, &tape, &|| reducer.barrier())?;
        let st = self.group.walker_state(self.walker);
        for (dst, src) in ctx.ele_idx.iter_mut().zip(&st.ele) {
            *dst = *src as i64;
        }
        ctx.pf_m.copy_from_slice(&st.pf);
        *ctx.log_ip_old = st.logip;
        ctx.counter[0] += n_in_step as i64;
        ctx.counter[1] += st.accepts as i64;
        *ctx.n_accept_window = st.aw.max(0) as usize;
        *ctx.accepted_total += st.accepts.max(0) as usize;
        ctx.inv_m.mark_stale("interval device sampling");
        Ok(())
    }
}
