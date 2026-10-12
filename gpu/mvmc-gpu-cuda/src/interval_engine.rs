//! Group coordinator, `SamplingEngine` and the production `SamplerStages` of the
//! device-resident interval sampler (issue #547).
//!
//! One [`IntervalGroup`] is shared by every walker of a run (`Arc<IntervalGroup>`, kept in a
//! per-device registry). Each walker has a [`WalkerState`] slot and the group owns the single
//! resident [`IntervalSampler`] sized for `walkers` blocks. A block is advanced by every walker
//! thread calling [`IntervalEngine::run_block`] once: each stages its block tape, rendezvous
//! through `Reducer::barrier()` outside any lock, rank 0 runs `k_interval_hop` for all walkers
//! (with device rebuilds), and each walker reads its slot back. Only rank 0 touches the device;
//! its `CudaBackend` is a thread-local (a tenferro context is thread-affine) while the resident
//! `DeviceBuffer`s, which are `Send`, live in the shared group.
//!
//! The RNG contract: each walker generates its own block tape from its own stream (5 words per
//! hop, the Julia GPU layout of `interval_kernels.cu`), so batching never merges streams.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use mvmc_core::reducer::Reducer;
use mvmc_core::sampler_backend::{SamplerStages, SamplerTiming};
use mvmc_core::sampling::driver::SampleStats;
use mvmc_core::sampling::engine::{BlockCtx, SamplingEngine};
use mvmc_core::sampling::projection::projection_tables;
use mvmc_core::sampling::rng::SamplingRng;
use mvmc_core::sampling::stage::{StageGeom, StageResult, StageTables};
use mvmc_core::stage_backend::StageError;
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::ExpertModeData;
use sfmt19937::Sfmt19937Rng;
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

/// Host staging of one walker.
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

struct Inner {
    geom: IntervalGeometry,
    slater: Vec<f64>,
    weights: Vec<f64>,
    gutz: Vec<i32>,
    jast: Vec<i32>,
    params: Vec<f64>,
    states: Vec<WalkerState>,
    tapes: Vec<u32>,
    n_in_step: usize,
    sampler: Option<IntervalSampler>,
}

/// Shared coordinator of one group of walkers; reconfigured once per sample series.
pub struct IntervalGroup {
    inner: Mutex<Inner>,
}

fn empty_state(geom: &IntervalGeometry) -> WalkerState {
    WalkerState {
        ele: vec![0; geom.n],
        inv: vec![0.0; geom.nqp * geom.n * geom.n],
        pf: vec![0.0; geom.nqp],
        logip: 0.0,
        aw: 0,
        accepts: 0,
    }
}

impl IntervalGroup {
    /// Build a configured group (used by tests and by `begin_series`).
    pub fn new(
        geom: IntervalGeometry,
        slater: Vec<f64>,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
    ) -> Arc<Self> {
        let states = (0..geom.walkers).map(|_| empty_state(&geom)).collect();
        Arc::new(Self {
            inner: Mutex::new(Inner {
                geom,
                slater,
                weights,
                gutz,
                jast,
                params,
                states,
                tapes: Vec::new(),
                n_in_step: 0,
                sampler: None,
            }),
        })
    }

    /// Reconfigure for a new sample series: geometry (reallocating when the walker count or
    /// shape changed), tables, and a cleared resident sampler (the Slater table changed).
    pub fn begin_series(
        &self,
        geom: IntervalGeometry,
        slater: Vec<f64>,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        if inner.geom.walkers != geom.walkers
            || inner.geom.n != geom.n
            || inner.geom.nqp != geom.nqp
        {
            inner.geom = geom;
            inner.states = (0..geom.walkers).map(|_| empty_state(&geom)).collect();
            inner.tapes = Vec::new();
            inner.n_in_step = 0;
        } else {
            inner.geom = geom;
        }
        inner.slater = slater;
        inner.weights = weights;
        inner.gutz = gutz;
        inner.jast = jast;
        inner.params = params;
        inner.sampler = None;
    }

    /// Store walker `w`'s initial state (called once per series, before any block).
    pub fn stage_initial(&self, w: usize, ele: Vec<i32>, inv: Vec<f64>, pf: Vec<f64>) {
        let mut inner = self.inner.lock().unwrap();
        let st = &mut inner.states[w];
        st.ele = ele;
        st.inv = inv;
        st.pf = pf;
        st.accepts = 0;
    }

    /// Read walker `w`'s state (results after a block).
    pub fn walker_state(&self, w: usize) -> WalkerState {
        self.inner.lock().unwrap().states[w].clone()
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
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.n_in_step == 0 {
                let world = inner.geom.walkers;
                inner.n_in_step = n_in_step;
                inner.tapes = vec![0u32; world * n_in_step * 5];
            }
            if inner.n_in_step != n_in_step {
                return Err(format!(
                    "interval group block size changed from {} to {n_in_step}",
                    inner.n_in_step
                ));
            }
            let world = inner.geom.walkers;
            for step in 0..n_in_step {
                for word in 0..5 {
                    inner.tapes[5 * (w + world * step) + word] = tape[5 * step + word];
                }
            }
        }
        barrier();
        if w == 0 {
            self.run_device()?;
        }
        barrier();
        Ok(())
    }

    /// Rank-0 device pass.
    fn run_device(&self) -> Result<(), String> {
        RANK0_BACKEND.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let devices = cuda_devices().map_err(|e| e.to_string())?;
                *slot = Some(CudaBackend::new(devices[0].id()).map_err(|e| e.to_string())?);
            }
            let backend = slot.as_mut().unwrap();
            backend.with_backend_session(|s| -> Result<(), String> {
                let mut inner = self.inner.lock().unwrap();
                if inner.sampler.is_none() {
                    let mut ele_all = Vec::new();
                    let mut inv_all = Vec::new();
                    let mut pf_all = Vec::new();
                    for st in &inner.states {
                        ele_all.extend_from_slice(&st.ele);
                        inv_all.extend_from_slice(&st.inv);
                        pf_all.extend_from_slice(&st.pf);
                    }
                    let host = IntervalHost {
                        ele: &ele_all,
                        inv: &inv_all,
                        pf: &pf_all,
                        slater: &inner.slater,
                        weights: &inner.weights,
                        gutz: &inner.gutz,
                        jast: &inner.jast,
                        params: &inner.params,
                    };
                    inner.sampler =
                        Some(IntervalSampler::prepare(s, inner.geom, &host, inner.n_in_step, 0.0)?);
                }
                let geom = inner.geom;
                let mut downloads = Vec::with_capacity(geom.walkers);
                {
                    let sampler = inner.sampler.as_ref().unwrap();
                    sampler.reset_offsets(s)?;
                    let tapes = inner.tapes.clone();
                    sampler.upload_tape(s, &tapes)?;
                    let mut accepts = vec![0i32; geom.walkers];
                    loop {
                        let results = sampler.run(s)?;
                        let mut need = Vec::new();
                        for (w, &(_, r, a)) in results.iter().enumerate() {
                            accepts[w] += a;
                            if r != 0 {
                                need.push(w);
                            }
                        }
                        if need.is_empty() {
                            break;
                        }
                        sampler.rebuild(s, &need)?;
                    }
                    for (w, &acc) in accepts.iter().enumerate() {
                        let (ele, pf, aw, logip) = sampler.download_walker(s, w)?;
                        downloads.push((ele, pf, aw, logip, acc));
                    }
                }
                for (w, (ele, pf, aw, logip, acc)) in downloads.into_iter().enumerate() {
                    let st = &mut inner.states[w];
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
    geom: IntervalGeometry,
    weights: Vec<f64>,
    gutz: Vec<i32>,
    jast: Vec<i32>,
    params: Vec<f64>,
}

impl IntervalEngine {
    /// Engine for walker `walker` of `group`, with the series geometry and projection tables.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group: Arc<IntervalGroup>,
        walker: usize,
        geom: IntervalGeometry,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
    ) -> Self {
        Self {
            group,
            walker,
            geom,
            weights,
            gutz,
            jast,
            params,
        }
    }
}

impl SamplingEngine for IntervalEngine {
    fn begin(
        &mut self,
        _geom: &StageGeom,
        tables: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<()> {
        self.group.begin_series(
            self.geom,
            tables.slater_elm.as_slice().to_vec(),
            self.weights.clone(),
            self.gutz.clone(),
            self.jast.clone(),
            self.params.clone(),
        );
        let g = self.geom;
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
        // Rebuild the host occupancy/counters consistent with the device configuration: the
        // driver does not run its own host move loop for this engine, so these working arrays
        // would otherwise stay at their pre-block values.
        let (ne, nsite) = (ctx.n_elec, ctx.n_site);
        for x in ctx.ele_cfg.iter_mut() {
            *x = -1;
        }
        for x in ctx.ele_num.iter_mut() {
            *x = 0;
        }
        for spin in 0..2usize {
            for m in 0..ne {
                let site = st.ele[m + spin * ne] as usize;
                ctx.ele_cfg[site + spin * nsite] = m as i64;
                ctx.ele_num[site + spin * nsite] = 1;
            }
        }
        mvmc_core::sampling::projection::make_proj_cnt(ctx.ele_proj_cnt, ctx.ele_num, ctx.data);
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

// ---------------------------------------------------------------------------------------------
// Production SamplerStages: one per-walker proxy over the shared per-device group.
// ---------------------------------------------------------------------------------------------

static GROUPS: Mutex<Vec<(usize, Arc<IntervalGroup>)>> = Mutex::new(Vec::new());

fn group_for(device: usize) -> Arc<IntervalGroup> {
    let mut groups = GROUPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, group)) = groups.iter().find(|(d, _)| *d == device) {
        return group.clone();
    }
    // A placeholder geometry; `begin_series` reconfigures it to the run's shape.
    let placeholder = IntervalGeometry {
        walkers: 1,
        n: 0,
        ne: 0,
        nsite: 0,
        nqp: 0,
        ngutz: 0,
        nproj: 0,
    };
    let group = IntervalGroup::new(
        placeholder,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    groups.push((device, group.clone()));
    group
}

/// [`SamplerStages`] of the device-resident interval sampler (`MVMC_RS_SAMPLER_BACKEND=
/// cuda-interval[:N]`). Group-capable: every walker of the run batches into one device call.
pub struct CudaIntervalSamplerStages {
    device: usize,
}

impl CudaIntervalSamplerStages {
    /// Proxy for `device`.
    pub fn new(device: usize) -> Self {
        Self { device }
    }
}

impl SamplerStages for CudaIntervalSamplerStages {
    fn label(&self) -> String {
        format!("cuda-interval:{}", self.device)
    }

    fn provider(&self) -> String {
        "device-resident interval sampler (issue #547)".to_string()
    }

    fn group_capable(&self) -> bool {
        true
    }

    fn sample_real(
        &mut self,
        _data: &ExpertModeData,
        _state: &mut VmcOptimizationState,
        _rng: &mut Sfmt19937Rng,
        _timing: &mut dyn SamplerTiming,
        _reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        Err(StageError::Unsupported(
            "the interval sampler runs through sample_real_group; sample_real is not used"
                .to_string(),
        ))
    }

    fn sample_real_group(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        if reducer.world_size() != reducer.reduction_size() {
            return Err(StageError::Unsupported(
                "the interval sampler supports ungrouped walker chains (world size == reduction size)"
                    .to_string(),
            ));
        }
        let nsite = data.modpara.nsite as usize;
        let ne = data.modpara.nelec as usize;
        let nqp = state.slater_matrix.slater_elm_real.n_qp_full();
        let tables = projection_tables(data).map_err(StageError::Unsupported)?;
        let weights: Vec<f64> = data
            .qp_weights
            .as_ref()
            .map(|w| w.qp_full_weight.iter().map(|v| v.re).collect())
            .unwrap_or_else(|| vec![1.0; nqp]);
        let geom = IntervalGeometry {
            walkers: reducer.world_size(),
            n: 2 * ne,
            ne,
            nsite,
            nqp,
            ngutz: tables.ngutz,
            nproj: tables.nproj,
        };
        let group = group_for(self.device);
        let mut engine = IntervalEngine::new(
            group,
            reducer.rank(),
            geom,
            weights,
            tables.gutz.clone(),
            tables.jast.clone(),
            tables.params.clone(),
        );
        let mut timer = mvmc_core::c_timer::CTimer::<false>::new();
        timer.diagnostics = timing.diagnostics();
        let t0 = std::time::Instant::now();
        let out = mvmc_core::sampling::driver::vmc_make_sample_real_engine(
            data, state, rng, &mut timer, reducer, &mut engine,
        );
        if reducer.rank() == 0 && std::env::var_os("MVMC_INTERVAL_TIMING").is_some() {
            eprintln!("[interval] sample_real_group {:.4}s", t0.elapsed().as_secs_f64());
        }
        out.map_err(|e| StageError::Failed(e.to_string()))
    }
}
