//! Group coordinator, `SamplingEngine` and the production `SamplerStages` of the
//! device-resident interval sampler (issue #547).
//!
//! One [`IntervalGroup`] is shared by every walker of a run (`Arc<IntervalGroup>`, kept in a
//! per-device registry). Each walker has a [`WalkerState`] slot and the group owns the single
//! resident [`IntervalSampler`] sized for `walkers` blocks. A block is advanced by every walker
//! thread calling [`IntervalEngine::run_block`] once: each writes its block tape into its own
//! slot (a per-walker lock, no cross-walker contention), rendezvous through `Reducer::barrier()`
//! outside any lock, rank 0 runs `k_interval_hop` for all walkers (with device rebuilds), and
//! each walker reads its slot back (again a per-walker lock). Only rank 0 touches the device;
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
    static RANK0_BACKEND: RefCell<Option<(usize, CudaBackend)>> = const { RefCell::new(None) };
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

struct Tables {
    slater: Vec<f64>,
    weights: Vec<f64>,
    gutz: Vec<i32>,
    jast: Vec<i32>,
    params: Vec<f64>,
}

fn new_state(geom: &IntervalGeometry) -> WalkerState {
    WalkerState {
        ele: vec![0; geom.n],
        inv: vec![0.0; geom.nqp * geom.n * geom.n],
        pf: vec![0.0; geom.nqp],
        logip: 0.0,
        aw: 0,
        accepts: 0,
    }
}

type States = Arc<Vec<Mutex<WalkerState>>>;
type WalkerTapes = Arc<Vec<Mutex<Vec<u32>>>>;

/// Shared coordinator of one group of walkers; reconfigured once per sample series.
///
/// The hot per-block path (staging a walker's tape and reading its state) touches only that
/// walker's own lock; the shared locks are held briefly to clone the per-walker buffers or by
/// rank 0 during the device pass.
pub struct IntervalGroup {
    device: usize,
    geom: Mutex<IntervalGeometry>,
    tables: Mutex<Tables>,
    states: Mutex<States>,
    walker_tapes: Mutex<WalkerTapes>,
    /// Interleaved device tape `[word][walker][step]` (rank 0 only).
    tapes: Mutex<Vec<u32>>,
    n_in_step: Mutex<usize>,
    sampler: Mutex<Option<IntervalSampler>>,
    /// Serializes `begin_series_walker` so a geometry reallocation never races a walker.
    series_lock: Mutex<()>,
    device_result: Mutex<Result<bool, String>>,
}

impl IntervalGroup {
    /// Build a configured group (used by tests and by `begin_series_walker`).
    pub fn new(
        geom: IntervalGeometry,
        slater: Vec<f64>,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
    ) -> Arc<Self> {
        Self::new_on_device(0, geom, slater, weights, gutz, jast, params)
    }

    fn new_on_device(
        device: usize,
        geom: IntervalGeometry,
        slater: Vec<f64>,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            geom: Mutex::new(geom),
            tables: Mutex::new(Tables {
                slater,
                weights,
                gutz,
                jast,
                params,
            }),
            states: Mutex::new(make_states(&geom)),
            walker_tapes: Mutex::new(make_tapes(geom.walkers)),
            tapes: Mutex::new(Vec::new()),
            n_in_step: Mutex::new(0),
            sampler: Mutex::new(None),
            series_lock: Mutex::new(()),
            device_result: Mutex::new(Ok(false)),
        })
    }

    fn states(&self) -> States {
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn walker_tapes(&self) -> WalkerTapes {
        self.walker_tapes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Reconfigure for a new sample series and stage walker `w`'s initial state. Both happen
    /// under `series_lock`, so the walkers of a run cannot race a geometry reallocation.
    /// Walker 0 publishes the common model tables; the other walkers pass empty table vectors.
    /// All walkers rendezvous in `run_block` before the shared tables are consumed.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_series_walker(
        &self,
        w: usize,
        geom: IntervalGeometry,
        slater: Vec<f64>,
        weights: Vec<f64>,
        gutz: Vec<i32>,
        jast: Vec<i32>,
        params: Vec<f64>,
        ele: Vec<i32>,
        inv: Vec<f64>,
        pf: Vec<f64>,
    ) {
        let _series = self.series_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut current = self.geom.lock().unwrap_or_else(|e| e.into_inner());
        if current.walkers != geom.walkers || current.n != geom.n || current.nqp != geom.nqp {
            // Geometry changed (first series of a run): reallocate the per-walker buffers.
            *self.states.lock().unwrap_or_else(|e| e.into_inner()) = make_states(&geom);
            *self.walker_tapes.lock().unwrap_or_else(|e| e.into_inner()) = make_tapes(geom.walkers);
            *self.n_in_step.lock().unwrap_or_else(|e| e.into_inner()) = 0;
            *self.tapes.lock().unwrap_or_else(|e| e.into_inner()) = Vec::new();
        }
        *current = geom;
        drop(current);
        if w == 0 {
            *self.tables.lock().unwrap_or_else(|e| e.into_inner()) = Tables {
                slater,
                weights,
                gutz,
                jast,
                params,
            };
            *self.sampler.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        let states = self.states();
        let mut st = states[w].lock().unwrap_or_else(|e| e.into_inner());
        st.ele = ele;
        st.inv = inv;
        st.pf = pf;
        st.logip = 0.0;
        st.aw = 0;
        st.accepts = 0;
    }

    /// Store walker `w`'s initial state for an already-configured group (test convenience).
    pub fn stage_initial(&self, w: usize, ele: Vec<i32>, inv: Vec<f64>, pf: Vec<f64>) {
        let states = self.states();
        let mut st = states[w].lock().unwrap_or_else(|e| e.into_inner());
        st.ele = ele;
        st.inv = inv;
        st.pf = pf;
        st.logip = 0.0;
        st.aw = 0;
        st.accepts = 0;
    }

    /// Read walker `w`'s state (results after a block).
    pub fn walker_state(&self, w: usize) -> WalkerState {
        self.states()[w]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Borrow one result without copying the stale O(nqp*n*n) initialization inverse.
    fn with_walker_state<T>(&self, w: usize, f: impl FnOnce(&WalkerState) -> T) -> T {
        let states = self.states();
        let state = states[w].lock().unwrap_or_else(|e| e.into_inner());
        f(&state)
    }

    /// Advance one block for walker `w` with its tape and rendezvous.
    pub fn run_block(
        &self,
        w: usize,
        n_in_step: usize,
        tape: &[u32],
        barrier: &dyn Fn(),
    ) -> Result<(), String> {
        debug_assert_eq!(tape.len(), 5 * n_in_step);
        {
            let tapes = self.walker_tapes();
            let mut wt = tapes[w].lock().unwrap_or_else(|e| e.into_inner());
            wt.clear();
            wt.extend_from_slice(tape);
        }
        {
            let mut n = self.n_in_step.lock().unwrap_or_else(|e| e.into_inner());
            if *n == 0 {
                let world = self.geom.lock().unwrap_or_else(|e| e.into_inner()).walkers;
                *n = n_in_step;
                *self.tapes.lock().unwrap_or_else(|e| e.into_inner()) =
                    vec![0u32; world * n_in_step * 5];
            }
            if *n != n_in_step {
                return Err(format!(
                    "interval group block size changed from {} to {n_in_step}",
                    *n
                ));
            }
        }
        self.device_pass(w, barrier, || self.run_device().map(|()| true))
            .map(|_| ())
    }

    /// Every peer observes rank 0's result after the rendezvous, including CUDA errors.
    fn device_pass(
        &self,
        w: usize,
        barrier: &dyn Fn(),
        pass: impl FnOnce() -> Result<bool, String>,
    ) -> Result<bool, String> {
        barrier();
        if w == 0 {
            *self.device_result.lock().unwrap_or_else(|e| e.into_inner()) = pass();
        }
        barrier();
        self.device_result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn with_backend<T>(
        &self,
        f: impl FnOnce(&mut CudaBackend) -> Result<T, String>,
    ) -> Result<T, String> {
        RANK0_BACKEND.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot
                .as_ref()
                .is_none_or(|(device, _)| *device != self.device)
            {
                let devices = cuda_devices().map_err(|e| e.to_string())?;
                let device = devices
                    .get(self.device)
                    .ok_or_else(|| format!("CUDA device {} unavailable", self.device))?;
                *slot = Some((
                    self.device,
                    CudaBackend::new(device.id()).map_err(|e| e.to_string())?,
                ));
            }
            f(&mut slot.as_mut().unwrap().1)
        })
    }

    /// Batch the real setup for every walker. Numeric failures keep the existing C setup;
    /// CUDA execution errors are returned to every peer. No RNG or configuration is changed.
    fn initialize_device(&self) -> Result<bool, String> {
        let start = std::time::Instant::now();
        self.with_backend(|backend| {
            backend.with_backend_session(|s| {
                let geom = *self.geom.lock().unwrap_or_else(|e| e.into_inner());
                let n_in_step = *self.n_in_step.lock().unwrap_or_else(|e| e.into_inner());
                let states = self.states();
                let ele: Vec<i32> = states
                    .iter()
                    .flat_map(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).ele.clone())
                    .collect();
                let tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
                let host = IntervalHost {
                    ele: &ele,
                    inv: &[],
                    pf: &[],
                    slater: &tables.slater,
                    weights: &tables.weights,
                    gutz: &tables.gutz,
                    jast: &tables.jast,
                    params: &tables.params,
                };
                let sampler = IntervalSampler::prepare_for_rebuild(s, geom, &host, n_in_step)?;
                if !sampler.try_rebuild(s, &(0..geom.walkers).collect::<Vec<_>>())? {
                    return Ok(false);
                }
                let results = sampler.download_all(s)?;
                if results.pf.iter().any(|v| !v.is_finite())
                    || results.logip.iter().any(|v| !v.is_finite())
                {
                    if std::env::var_os("MVMC_INTERVAL_TIMING").is_some() {
                        eprintln!("[interval] initialization nonfinite overlap");
                    }
                    return Ok(false);
                }
                for (w, slot) in states.iter().enumerate() {
                    let mut st = slot.lock().unwrap_or_else(|e| e.into_inner());
                    st.pf = results.pf[w * geom.nqp..(w + 1) * geom.nqp].to_vec();
                    st.logip = results.logip[w];
                    st.aw = 0;
                    st.accepts = 0;
                }
                *self.sampler.lock().unwrap_or_else(|e| e.into_inner()) = Some(sampler);
                if std::env::var_os("MVMC_INTERVAL_TIMING").is_some() {
                    eprintln!(
                        "[interval] GPU initialization {:.4}s",
                        start.elapsed().as_secs_f64()
                    );
                }
                Ok(true)
            })
        })
    }

    /// Rank-0 device pass.
    fn run_device(&self) -> Result<(), String> {
        self.with_backend(|backend| {
            backend.with_backend_session(|s| -> Result<(), String> {
                let geom = *self.geom.lock().unwrap_or_else(|e| e.into_inner());
                let n_in_step = *self.n_in_step.lock().unwrap_or_else(|e| e.into_inner());
                let states = self.states();
                let walker_tapes = self.walker_tapes();
                let timing = std::env::var_os("MVMC_INTERVAL_TIMING").is_some();
                let t_block = std::time::Instant::now();
                let mut t_run = 0.0f64;
                let mut t_rb = 0.0f64;
                let mut rounds = 0usize;
                let mut n_rebuild = 0usize;

                let mut sampler_guard = self.sampler.lock().unwrap_or_else(|e| e.into_inner());
                if sampler_guard.is_none() {
                    let tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
                    let mut ele_all = Vec::with_capacity(geom.walkers * geom.n);
                    let mut inv_all = Vec::new();
                    let mut pf_all = Vec::new();
                    for mx in states.iter() {
                        let st = mx.lock().unwrap_or_else(|e| e.into_inner());
                        ele_all.extend_from_slice(&st.ele);
                        inv_all.extend_from_slice(&st.inv);
                        pf_all.extend_from_slice(&st.pf);
                    }
                    let host = IntervalHost {
                        ele: &ele_all,
                        inv: &inv_all,
                        pf: &pf_all,
                        slater: &tables.slater,
                        weights: &tables.weights,
                        gutz: &tables.gutz,
                        jast: &tables.jast,
                        params: &tables.params,
                    };
                    *sampler_guard =
                        Some(IntervalSampler::prepare(s, geom, &host, n_in_step, 0.0)?);
                }
                let sampler = sampler_guard.as_ref().unwrap();

                {
                    let mut tapes = self.tapes.lock().unwrap_or_else(|e| e.into_inner());
                    for (w, mx) in walker_tapes.iter().enumerate() {
                        let wt = mx.lock().unwrap_or_else(|e| e.into_inner());
                        for step in 0..n_in_step {
                            for word in 0..5 {
                                tapes[5 * (w + geom.walkers * step) + word] = wt[5 * step + word];
                            }
                        }
                    }
                    sampler.reset_offsets(s)?;
                    sampler.upload_tape(s, &tapes)?;
                }

                let mut accepts = vec![0i32; geom.walkers];
                loop {
                    let t = std::time::Instant::now();
                    let results = sampler.run(s)?;
                    t_run += t.elapsed().as_secs_f64();
                    rounds += 1;
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
                    n_rebuild += need.len();
                    let t = std::time::Instant::now();
                    sampler.rebuild(s, &need)?;
                    t_rb += t.elapsed().as_secs_f64();
                }
                let t = std::time::Instant::now();
                let results = sampler.download_all(s)?;
                for (w, &acc) in accepts.iter().enumerate() {
                    let mut st = states[w].lock().unwrap_or_else(|e| e.into_inner());
                    st.ele.copy_from_slice(&results.ele[w * geom.n..(w + 1) * geom.n]);
                    st.pf.copy_from_slice(&results.pf[w * geom.nqp..(w + 1) * geom.nqp]);
                    st.aw = results.aw[w];
                    st.logip = results.logip[w];
                    st.accepts = acc;
                }
                let t_dl = t.elapsed().as_secs_f64();
                if timing {
                    eprintln!(
                        "[dev] block_ms={:.2} run_ms={:.2} rb_ms={:.2} dl_ms={:.2} rounds={rounds} rebuilds={n_rebuild}",
                        t_block.elapsed().as_secs_f64() * 1e3,
                        t_run * 1e3,
                        t_rb * 1e3,
                        t_dl * 1e3,
                    );
                }
                Ok(())
            })
        })
    }
}

fn make_states(geom: &IntervalGeometry) -> States {
    Arc::new(
        (0..geom.walkers)
            .map(|_| Mutex::new(new_state(geom)))
            .collect(),
    )
}

fn make_tapes(walkers: usize) -> WalkerTapes {
    Arc::new((0..walkers).map(|_| Mutex::new(Vec::new())).collect())
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
    initialized: bool,
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
            initialized: false,
        }
    }
}

impl SamplingEngine for IntervalEngine {
    fn initialize_real<R: Reducer + ?Sized>(
        &mut self,
        _geom: &StageGeom,
        tables: &mut StageTables<'_>,
        ele_idx: &[i64],
        n_in_step: usize,
        reducer: &R,
    ) -> StageResult<bool> {
        self.group.begin_series_walker(
            self.walker,
            self.geom,
            if self.walker == 0 {
                tables.slater_elm.as_slice().to_vec()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.weights.clone()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.gutz.clone()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.jast.clone()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.params.clone()
            } else {
                Vec::new()
            },
            ele_idx.iter().map(|&x| x as i32).collect(),
            Vec::new(),
            Vec::new(),
        );
        if self.walker == 0 {
            *self
                .group
                .n_in_step
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = n_in_step;
            self.group
                .tapes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .resize(self.geom.walkers * n_in_step * 5, 0);
        }
        self.initialized = self
            .group
            .device_pass(self.walker, &|| reducer.barrier(), || {
                self.group.initialize_device()
            })?;
        if self.initialized {
            self.group
                .with_walker_state(self.walker, |st| tables.pf_m.copy_from_slice(&st.pf));
            tables.inv_m.mark_stale("interval device initialization");
        }
        Ok(self.initialized)
    }

    fn begin(
        &mut self,
        _geom: &StageGeom,
        tables: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<()> {
        if self.initialized {
            return Ok(());
        }
        let g = self.geom;
        let mut inv = Vec::with_capacity(g.nqp * g.n * g.n);
        for q in 0..g.nqp {
            inv.extend_from_slice(tables.inv_m.qp_matrix_slice(q));
        }
        self.group.begin_series_walker(
            self.walker,
            self.geom,
            if self.walker == 0 {
                tables.slater_elm.as_slice().to_vec()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.weights.clone()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.gutz.clone()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.jast.clone()
            } else {
                Vec::new()
            },
            if self.walker == 0 {
                self.params.clone()
            } else {
                Vec::new()
            },
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
        self.group.with_walker_state(self.walker, |st| {
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
        });
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
    let placeholder = IntervalGeometry {
        walkers: 1,
        n: 0,
        ne: 0,
        nsite: 0,
        nqp: 0,
        ngutz: 0,
        nproj: 0,
    };
    let group = IntervalGroup::new_on_device(
        device,
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

/// [`SamplerStages`] of the device-resident interval sampler
/// (`MVMC_RS_SAMPLER_BACKEND=cuda-interval[:N]`). Group-capable: every walker of the run
/// batches into one device call.
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
        let t0 = std::time::Instant::now();
        let out = if timing.enabled() {
            let mut timer = mvmc_core::c_timer::CTimer::<true>::new();
            timer.diagnostics = timing.diagnostics();
            let result = mvmc_core::sampling::driver::vmc_make_sample_real_engine(
                data,
                state,
                rng,
                &mut timer,
                reducer,
                &mut engine,
            );
            timing.merge_elapsed(&timer.elapsed_ns);
            result
        } else {
            let mut timer = mvmc_core::c_timer::CTimer::<false>::new();
            timer.diagnostics = timing.diagnostics();
            mvmc_core::sampling::driver::vmc_make_sample_real_engine(
                data,
                state,
                rng,
                &mut timer,
                reducer,
                &mut engine,
            )
        };
        if reducer.rank() == 0 && std::env::var_os("MVMC_INTERVAL_TIMING").is_some() {
            eprintln!(
                "[interval] sample_real_group {:.4}s",
                t0.elapsed().as_secs_f64()
            );
        }
        out.map_err(|e| StageError::Failed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA"]
    fn interval_backend_honors_device_selection_and_switches() {
        let devices = cuda_devices().unwrap();
        assert!(!devices.is_empty());
        let geom = IntervalGeometry {
            walkers: 1,
            n: 2,
            ne: 1,
            nsite: 2,
            nqp: 1,
            ngutz: 0,
            nproj: 0,
        };
        for ordinal in (0..devices.len()).chain(std::iter::once(0)) {
            let group =
                IntervalGroup::new_on_device(ordinal, geom, vec![], vec![], vec![], vec![], vec![]);
            let actual = group
                .with_backend(|backend| Ok(backend.device_id()))
                .unwrap();
            assert_eq!(actual, devices[ordinal].id());
        }
        let group = IntervalGroup::new_on_device(
            devices.len(),
            geom,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert!(group
            .with_backend(|_| Ok(()))
            .unwrap_err()
            .contains("unavailable"));
        eprintln!(
            "verified selection across {} visible CUDA devices",
            devices.len()
        );
    }

    #[test]
    fn failed_device_pass_releases_every_peer_and_allows_next_pass() {
        let geom = IntervalGeometry {
            walkers: 4,
            n: 2,
            ne: 1,
            nsite: 2,
            nqp: 1,
            ngutz: 0,
            nproj: 0,
        };
        let group = IntervalGroup::new(geom, vec![], vec![], vec![], vec![], vec![]);
        let barrier = Arc::new(std::sync::Barrier::new(geom.walkers));
        let (tx, rx) = std::sync::mpsc::channel();
        let handles: Vec<_> = (0..geom.walkers)
            .map(|w| {
                let (group, barrier, tx) = (group.clone(), barrier.clone(), tx.clone());
                std::thread::spawn(move || {
                    let rendezvous = || {
                        barrier.wait();
                    };
                    let first =
                        group.device_pass(w, &rendezvous, || Err("injected CUDA failure".into()));
                    let second = group.device_pass(w, &rendezvous, || Ok(true));
                    tx.send((first, second)).unwrap();
                })
            })
            .collect();
        for _ in 0..geom.walkers {
            let (first, second) = rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("a failed device pass stranded a walker");
            assert_eq!(first.unwrap_err(), "injected CUDA failure");
            assert_eq!(second, Ok(true));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}
