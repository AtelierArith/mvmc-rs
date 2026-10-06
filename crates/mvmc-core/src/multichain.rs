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
