//! Multi-chain walker runner (issue #425, design `docs/design/gpu-readiness.md` 5.5).
//!
//! `W` independent Markov chains ("walkers") run the fixed-parameter PhysCal sampling and
//! measurement loop side by side. A single chain cannot amortize a device round trip, so the
//! walker is the unit that a later accelerated backend batches over.
//!
//! # Seeds and draw order (C-compatible)
//!
//! C seeds every MPI group's chain with `init_gen_rand(RndSeed + group1)`
//! (`vmcmain.c`; Rust `resolve_rnd_seed`). Walker `w` of a runner whose first walker has group
//! index `group_base` uses
//!
//! ```text
//! seed(w) = RndSeed + group_base + w
//! ```
//!
//! so `W` walkers on one process are the same chains as `W` groups of a grouped C run, and
//! walker `w` is byte-for-byte the serial run with seed offset `group_base + w`. Each walker
//! owns its own host SFMT stream (inside its [`PhysCalPreparation`]); no stream is shared
//! between walkers and no draw is added, removed or reordered: the per-walker draw count and
//! order equal the serial path's. A time-based seed (`RndSeed < 0`) is resolved once for all
//! walkers, as the C group broadcast does. With `W = 1` the runner is the serial run.
//!
//! # Execution and thread budget
//!
//! Walkers run under a dedicated rayon pool over walkers (one walker per worker, results kept
//! in walker order). Everything inside a walker stays single-threaded (the serial path), and
//! BLAS is pinned to one thread, so total threads equal the pool size and output bytes do not
//! depend on it. Every walker owns its working state (tables, scratch, RNG), the
//! tenferro-decision-rs workspace pattern: nothing is shared mutably across walkers.
//!
//! # Batched stages across walkers
//!
//! [`pack_walker_tables`] packs the Pfaffian/inverse tables of all walkers into the
//! `BatchedPlanes` layout `[n, n, NQP, W]` of the sample-batched measurement stage (#422),
//! the staging format for a batched Pfaffian/inverse backend (#423). The sampler of one walker
//! still recomputes its tables on its own thread (a lock-step batched recomputation across
//! walkers needs the device-resident planes this layout feeds; see the design document).
//!
//! # Multi-walker optimization (ParaOpt, issue #435)
//!
//! [`run_para_opt_multichain`] runs the whole optimization (`NSROptItrStep` SR steps) on `W`
//! walkers. C runs `W` independent chains as `W` MPI ranks of an ungrouped run
//! (`NSplitSize = 1`): every rank samples its own chain (seed `RndSeed + rank`), `HO`, `OO`,
//! the energy and the weights are summed over all ranks with `MPI_Allreduce(MPI_SUM)`, every
//! rank then solves the same SR system and applies the same parameter update (the update is
//! deterministic given the reduced operands; `SROptO` stays rank-local as in C). The
//! optimizer is already written against the [`Reducer`] trait, so the multi-walker run is the
//! unchanged optimizer with an in-process reducer, [`ThreadReducer`], that replaces the MPI
//! communicator by shared memory between `W` walker threads: no SR code is duplicated and the
//! collective sequence, seeds and draw order are exactly those of the MPI run.
//!
//! Reduction order: [`ThreadReducer`] sums the contributions in rank order, as a left fold
//! `((v0 + v1) + v2) + ...`, identical on every walker. `MPI_Allreduce` leaves the order to the
//! MPI library, so the Rust sums can differ from C by last-bit roundoff for `W > 2`
//! (`W = 1` and `W = 2` are exact: one sum of two terms). Tolerances against the C fixtures
//! follow `tests/fixtures/mpi_matrix_179/README.md`.
//!
//! Thread budget: the collectives need all walkers running at once, so the runner uses one OS
//! thread per walker (not a pool smaller than `W`); every walker is single-threaded inside.
//! Intra-group splitting (`NSplitSize > 1`: QP and sample split of one chain over several
//! ranks) is not part of the walker model; `W = ranks / NSplitSize` groups of such a C run
//! compute the same chains and operands to reduction-order roundoff.
//!
//! # Decision margins
//!
//! The Metropolis `(weight, draw)` pairs of every walker are recorded (observationally; no RNG
//! or numerical effect) and summarized as [`DecisionSummary`]. The margin of a proposal is
//! `|w - u|`; the smallest margins tell how close a run came to a decision flip under an
//! accelerated backend (see `accel_validation`).

use std::path::PathBuf;

use rayon::prelude::*;
use sfmt19937::Sfmt19937Rng;

use crate::measurement_batch::BatchedPlanes;
use crate::reducer::SingleProcessReducer;
use crate::run::{
    prepare_phys_cal_with_seed_offset, resolve_rnd_seed, vmc_phys_cal, PhysCalResult,
};

/// Seed of walker `w`: `base + group_base + w` (wrapping, as C `RndSeed + group1`).
///
/// # Errors
///
/// Returns an error when the offset does not fit `i64`.
pub fn walker_seed(base: i64, group_base: usize, w: usize) -> Result<i64, String> {
    let offset = group_base
        .checked_add(w)
        .and_then(|v| i64::try_from(v).ok())
        .ok_or_else(|| format!("walker seed offset {group_base} + {w} does not fit"))?;
    Ok(base.wrapping_add(offset))
}

/// Multi-chain PhysCal configuration.
#[derive(Debug, Clone)]
pub struct MultiChainConfig {
    /// `namelist.def`.
    pub namelist: PathBuf,
    /// Fixed parameter file (`zqp_opt.dat`), or `None` for the C no-parameter-file path.
    pub opt_para: Option<PathBuf>,
    /// `real`, `cmp` or `fsz`.
    pub mode: String,
    /// Explicit `RndSeed` override; `None` uses the namelist value (negative: time, resolved
    /// once).
    pub seed: Option<i64>,
    /// Group index of the first walker (C `group1` of the first chain).
    pub group_base: usize,
    /// Number of walkers `W >= 1`.
    pub walkers: usize,
    /// C-compatible OptTrans selection.
    pub enable_opt_trans: bool,
    /// Override of `NDataQtySmp` (measurement samples per walker); `None` keeps the input.
    pub samples: Option<i64>,
    /// Pool size over walkers (`0`: `min(walkers, available parallelism)`).
    pub threads: usize,
}

/// Summary of the recorded Metropolis decisions of one walker.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionSummary {
    /// Proposals decided.
    pub proposals: usize,
    /// Accepted (`w > u`).
    pub accepted: usize,
    /// Smallest margin `|w - u|` over all proposals (`INFINITY` when none).
    pub min_margin: f64,
    /// Proposals whose margin is below `near_flip_threshold`.
    pub near_flip: usize,
    /// Threshold used for `near_flip`.
    pub near_flip_threshold: f64,
}

/// Threshold below which a decision counts as near a flip in [`DecisionSummary`]: a relative
/// weight perturbation of 1e-10 (the default of `accel_validation::Tolerances`) changes a
/// decision only if `|w - u| <= 1e-10 * max(w, 1) + 1e-12`.
pub const NEAR_FLIP_THRESHOLD: f64 = 1e-10;

impl DecisionSummary {
    /// Summarize `(weight, draw)` pairs.
    pub fn from_pairs(pairs: &[(f64, f64)]) -> Self {
        let mut s = Self {
            min_margin: f64::INFINITY,
            near_flip_threshold: NEAR_FLIP_THRESHOLD,
            ..Self::default()
        };
        for &(w, u) in pairs {
            s.proposals += 1;
            if w > u {
                s.accepted += 1;
            }
            let margin = (w - u).abs();
            s.min_margin = s.min_margin.min(margin);
            if margin <= NEAR_FLIP_THRESHOLD * w.abs().max(1.0) + 1e-12 {
                s.near_flip += 1;
            }
        }
        s
    }
}

/// Outcome of one walker.
#[derive(Debug)]
pub struct WalkerOutcome {
    /// Walker index.
    pub walker: usize,
    /// Resolved seed `RndSeed + group_base + w`.
    pub seed: i64,
    /// The serial PhysCal result of this walker.
    pub result: PhysCalResult,
    /// Recorded `(weight, draw)` of every Metropolis decision, in order.
    pub decisions: Vec<(f64, f64)>,
    /// Decision-margin summary.
    pub summary: DecisionSummary,
    /// SFMT words consumed by this walker's stream.
    pub rng_words: u128,
}

/// Result of [`run_phys_cal_multichain`].
#[derive(Debug)]
pub struct MultiChainResult {
    /// Resolved base seed (before `group_base + w`).
    pub base_seed: i64,
    /// Walkers in index order.
    pub walkers: Vec<WalkerOutcome>,
}

/// Resolve the base seed once for all walkers (explicit override, namelist value or time).
fn resolve_base_seed(cfg: &MultiChainConfig) -> Result<i64, String> {
    let rnd_seed = if cfg.seed.is_some() {
        0
    } else {
        mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
            &cfg.namelist,
            cfg.enable_opt_trans,
        )
        .map_err(|e| e.to_string())?
        .modpara
        .rnd_seed
    };
    resolve_rnd_seed(rnd_seed, cfg.seed, 0, &SingleProcessReducer)
}

/// Run one walker: seeded `base + group_base + w`, own state and RNG.
pub fn run_walker(cfg: &MultiChainConfig, base: i64, w: usize) -> Result<WalkerOutcome, String> {
    let seed = walker_seed(base, cfg.group_base, w)?;
    // The explicit seed already contains the offset; passing offset 0 keeps the C contract
    // `RndSeed + group1` in one place (`walker_seed`).
    let mut prep = prepare_phys_cal_with_seed_offset(
        &cfg.namelist,
        cfg.opt_para.as_deref(),
        &cfg.mode,
        Some(seed),
        cfg.enable_opt_trans,
        0,
    )?;
    if let Some(samples) = cfg.samples {
        prep.data.modpara.n_data_qty_smp = samples;
    }
    crate::sampling::driver::trace::start_decisions();
    let result = vmc_phys_cal(prep);
    let decisions = crate::sampling::driver::trace::finish_decisions();
    let result = result?;
    let rng_words = result.final_rng.words_consumed();
    let summary = DecisionSummary::from_pairs(&decisions);
    Ok(WalkerOutcome {
        walker: w,
        seed,
        result,
        decisions,
        summary,
        rng_words,
    })
}

/// Run `cfg.walkers` independent PhysCal chains; walker `w` is seeded
/// `RndSeed + group_base + w`. Results are in walker order and independent of the pool size.
///
/// # Errors
///
/// Returns the first failing walker's error (lowest index), or a configuration error.
pub fn run_phys_cal_multichain(cfg: &MultiChainConfig) -> Result<MultiChainResult, String> {
    if cfg.walkers == 0 {
        return Err("multi-chain runner needs at least one walker".to_string());
    }
    crate::serial_blas::initialize();
    let base_seed = resolve_base_seed(cfg)?;
    let threads = if cfg.threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        cfg.threads
    }
    .min(cfg.walkers);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("mvmc-walker-{i}"))
        .build()
        .map_err(|e| e.to_string())?;
    let outcomes: Vec<Result<WalkerOutcome, String>> = pool.install(|| {
        (0..cfg.walkers)
            .into_par_iter()
            .map(|w| run_walker(cfg, base_seed, w))
            .collect()
    });
    let mut walkers = Vec::with_capacity(cfg.walkers);
    for outcome in outcomes {
        walkers.push(outcome?);
    }
    Ok(MultiChainResult { base_seed, walkers })
}

/// Pfaffian/inverse tables of all walkers in the batched layout `[n, n, NQP, W]`.
#[derive(Debug)]
pub enum WalkerPlanes {
    /// Real wavefunction.
    Real(BatchedPlanes<f64>),
    /// Complex wavefunction.
    Complex(BatchedPlanes<num_complex::Complex64>),
}

/// Pack the current Pfaffian/inverse tables of each walker into one `[n, n, NQP, W]` batch
/// (walker `w` in slot `w`). All walkers must share the real/complex mode and sizes.
pub fn pack_walker_tables(walkers: &[WalkerOutcome]) -> Option<WalkerPlanes> {
    let first = walkers.first()?;
    if first.result.state.all_complex {
        let mut planes = BatchedPlanes::<num_complex::Complex64>::default();
        planes.ensure(
            first.result.state.slater_matrix.inv_m.n_size(),
            first.result.state.slater_matrix.inv_m.n_qp_full(),
            walkers.len(),
        );
        for (slot, w) in walkers.iter().enumerate() {
            planes.store(
                slot,
                &w.result.state.slater_matrix.inv_m,
                &w.result.state.slater_matrix.pf_m,
            );
        }
        Some(WalkerPlanes::Complex(planes))
    } else {
        let mut planes = BatchedPlanes::<f64>::default();
        planes.ensure(
            first.result.state.slater_matrix.inv_m_real.n_size(),
            first.result.state.slater_matrix.inv_m_real.n_qp_full(),
            walkers.len(),
        );
        for (slot, w) in walkers.iter().enumerate() {
            planes.store(
                slot,
                &w.result.state.slater_matrix.inv_m_real,
                &w.result.state.slater_matrix.pf_m_real,
            );
        }
        Some(WalkerPlanes::Real(planes))
    }
}

/// Final SFMT state words of a walker's stream (for exact stream comparisons).
pub fn rng_state(rng: &Sfmt19937Rng) -> ([u32; 624], usize) {
    rng.state_snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walker_seed_follows_c_group_offset() {
        assert_eq!(walker_seed(11272, 0, 0).unwrap(), 11272);
        assert_eq!(walker_seed(11272, 3, 2).unwrap(), 11277);
        assert_eq!(walker_seed(i64::MAX, 0, 1).unwrap(), i64::MIN);
    }

    #[test]
    fn decision_summary_reports_margins() {
        let s = DecisionSummary::from_pairs(&[(0.5, 0.25), (2.0, 0.5), (0.3, 0.3 + 1e-14)]);
        assert_eq!(s.proposals, 3);
        assert_eq!(s.accepted, 2);
        assert!(s.min_margin <= 1e-13);
        assert_eq!(s.near_flip, 1);
    }
}

// ---------------------------------------------------------------------------------------------
// In-process collectives and multi-walker optimization (issue #435)
// ---------------------------------------------------------------------------------------------

use std::sync::{Arc, Condvar, Mutex};

use num_complex::Complex64;

use crate::reducer::Reducer;
use crate::run::{run_para_opt_from_namelist_observed, RunConfig, RunSummary};
use crate::state::VmcOptimizationState;

#[derive(Debug)]
struct CommState {
    /// Number of participants that arrived at the current barrier generation.
    arrived: usize,
    generation: u64,
    /// Per-rank contribution slots of the current collective.
    f64s: Vec<Vec<f64>>,
    c64s: Vec<Vec<Complex64>>,
    i64s: Vec<Vec<i64>>,
}

/// Shared state of `W` walker threads standing in for one MPI communicator.
#[derive(Debug)]
pub struct ThreadComm {
    size: usize,
    state: Mutex<CommState>,
    cv: Condvar,
}

impl ThreadComm {
    /// A communicator of `size` walkers.
    pub fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            size,
            state: Mutex::new(CommState {
                arrived: 0,
                generation: 0,
                f64s: vec![Vec::new(); size],
                c64s: vec![Vec::new(); size],
                i64s: vec![Vec::new(); size],
            }),
            cv: Condvar::new(),
        })
    }

    /// Block until all `size` walkers called `barrier` (reusable).
    fn barrier(&self) {
        let mut st = self.state.lock().expect("thread comm lock");
        let generation = st.generation;
        st.arrived += 1;
        if st.arrived == self.size {
            st.arrived = 0;
            st.generation += 1;
            self.cv.notify_all();
        } else {
            while st.generation == generation {
                st = self.cv.wait(st).expect("thread comm wait");
            }
        }
    }

    /// Deposit this rank's data, wait for all, read everything, wait again (so the slots can be
    /// reused by the next collective).
    fn gather<T: Clone>(
        &self,
        rank: usize,
        data: &[T],
        slots: impl Fn(&mut CommState) -> &mut Vec<Vec<T>>,
    ) -> Vec<Vec<T>> {
        {
            let mut st = self.state.lock().expect("thread comm lock");
            slots(&mut st)[rank] = data.to_vec();
        }
        self.barrier();
        let all = {
            let mut st = self.state.lock().expect("thread comm lock");
            slots(&mut st).clone()
        };
        self.barrier();
        all
    }
}

/// [`Reducer`] over `W` walker threads: the in-process equivalent of an ungrouped
/// `MpiContext` (`NSplitSize = 1`, every rank its own chain).
#[derive(Debug, Clone)]
pub struct ThreadReducer {
    comm: Arc<ThreadComm>,
    rank: usize,
    group_base: usize,
}

impl ThreadReducer {
    /// Handle of walker `rank` (0-based) in `comm`; walker `rank` is seeded with the offset
    /// `group_base + rank`.
    pub fn new(comm: Arc<ThreadComm>, rank: usize, group_base: usize) -> Self {
        assert!(rank < comm.size);
        Self {
            comm,
            rank,
            group_base,
        }
    }
}

impl Reducer for ThreadReducer {
    fn allreduce_sum_f64(&self, buf: &mut [f64]) {
        if self.comm.size == 1 {
            return;
        }
        let all = self.comm.gather(self.rank, buf, |s| &mut s.f64s);
        for (i, slot) in buf.iter_mut().enumerate() {
            let mut acc = all[0][i];
            for v in &all[1..] {
                acc += v[i];
            }
            *slot = acc;
        }
    }

    fn allreduce_sum_c64(&self, buf: &mut [Complex64]) {
        if self.comm.size == 1 {
            return;
        }
        let all = self.comm.gather(self.rank, buf, |s| &mut s.c64s);
        for (i, slot) in buf.iter_mut().enumerate() {
            // real and imaginary parts are reduced separately, as the MPI path does
            let (mut re, mut im) = (all[0][i].re, all[0][i].im);
            for v in &all[1..] {
                re += v[i].re;
                im += v[i].im;
            }
            *slot = Complex64::new(re, im);
        }
    }

    fn allreduce_sum_i64(&self, buf: &mut [i64]) {
        if self.comm.size == 1 {
            return;
        }
        let all = self.comm.gather(self.rank, buf, |s| &mut s.i64s);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = all.iter().map(|v| v[i]).sum();
        }
    }

    fn broadcast_f64(&self, root: usize, buf: &mut [f64]) -> Result<(), String> {
        if self.comm.size > 1 {
            let all = self.comm.gather(self.rank, buf, |s| &mut s.f64s);
            buf.copy_from_slice(&all[root]);
        }
        Ok(())
    }

    fn broadcast_c64(&self, root: usize, buf: &mut [Complex64]) {
        if self.comm.size > 1 {
            let all = self.comm.gather(self.rank, buf, |s| &mut s.c64s);
            buf.copy_from_slice(&all[root]);
        }
    }

    fn broadcast_i64(&self, root: usize, buf: &mut [i64]) -> Result<(), String> {
        if self.comm.size > 1 {
            let all = self.comm.gather(self.rank, buf, |s| &mut s.i64s);
            buf.copy_from_slice(&all[root]);
        }
        Ok(())
    }

    fn barrier(&self) {
        if self.comm.size > 1 {
            self.comm.barrier();
        }
    }

    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        if self.comm.size == 1 {
            return Ok(info);
        }
        let all = self
            .comm
            .gather(self.rank, &[i64::from(info)], |s| &mut s.i64s);
        Ok(all.iter().map(|v| v[0]).max().expect("nonempty") as i32)
    }

    fn world_size(&self) -> usize {
        self.comm.size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn seed_offset(&self) -> usize {
        self.group_base + self.rank
    }
}

/// Configuration of a multi-walker optimization.
#[derive(Debug, Clone)]
pub struct ParaOptMultiChainConfig {
    /// `namelist.def` (shared by all walkers).
    pub namelist: PathBuf,
    /// Optimization run configuration (steps, mode, seed override, output directory, ...).
    /// Only walker 0 (the output root) writes the output files.
    pub run: RunConfig,
    /// Group index of the first walker; walker `w` uses the seed offset `group_base + w`.
    pub group_base: usize,
    /// Number of walkers `W >= 1`.
    pub walkers: usize,
}

/// Outcome of one walker of a multi-walker optimization.
#[derive(Debug)]
pub struct ParaOptWalker {
    /// Walker index (MPI rank in the equivalent ungrouped C run).
    pub walker: usize,
    /// Run summary (the reduced optimization output; identical on every walker).
    pub summary: RunSummary,
    /// Final sampler/SR state of this walker (`SROptO` is walker-local, as in C).
    pub state: VmcOptimizationState,
    /// Final RNG stream of this walker.
    pub rng: Sfmt19937Rng,
}

/// Run the optimization on `cfg.walkers` walkers with C-compatible cross-walker reductions.
///
/// `W = 1` is the serial optimization. All walkers must run concurrently (they synchronize in
/// every reduction), so one thread per walker is used. If any walker fails, every walker
/// reports the failure through the collective failure agreement of the optimizer (no
/// deadlock); the first failing walker's error is returned.
///
/// # Errors
///
/// Returns the optimizer's error text or a configuration error.
pub fn run_para_opt_multichain(
    cfg: &ParaOptMultiChainConfig,
) -> Result<Vec<ParaOptWalker>, String> {
    if cfg.walkers == 0 {
        return Err("multi-walker optimization needs at least one walker".to_string());
    }
    crate::serial_blas::initialize();
    let comm = ThreadComm::new(cfg.walkers);
    let results: Vec<Result<ParaOptWalker, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..cfg.walkers)
            .map(|w| {
                let reducer = ThreadReducer::new(comm.clone(), w, cfg.group_base);
                let namelist = cfg.namelist.clone();
                let run = cfg.run.clone();
                std::thread::Builder::new()
                    .name(format!("mvmc-walker-{w}"))
                    .spawn_scoped(scope, move || {
                        let (summary, state, rng) =
                            run_para_opt_from_namelist_observed(namelist, run, &reducer)?;
                        Ok(ParaOptWalker {
                            walker: w,
                            summary,
                            state,
                            rng,
                        })
                    })
                    .expect("spawn walker thread")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("walker thread panicked".to_string()))
            })
            .collect()
    });
    let mut out = Vec::with_capacity(cfg.walkers);
    for r in results {
        out.push(r?);
    }
    Ok(out)
}
