//! Top-level entry points.
//!
//! Port targets:
//! * `vmc_para_opt.jl` -> `vmc_para_opt`
//! * `vmc_phys_cal.jl` -> `vmc_phys_cal`
//! * `run_para_opt_from_namelist.jl` -> `run_para_opt_from_namelist`
//! * `initial_params.jl` -> `read_initial_def`
//!
//! BIT-PARITY CRITICAL: preserve the upstream phase order documented in
//! `run_para_opt_from_namelist.jl:65-100`:
//!     init_gen_rand -> InitParameter -> ReadInitParameter
//!         -> ReadInputParameters -> SyncModifiedParameter -> InitQPWeight.

#[cfg(test)]
#[path = "../../../tests/support/reference_slater.rs"]
mod reference_slater;

#[cfg(test)]
use std::fs;
use std::path::Path;

#[cfg(test)]
use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::utils::parameter_init::{init_parameter, n_slater};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::average::{weight_average_sr_opt, weight_average_sr_opt_real, weight_average_we};
use crate::c_timer::{CTimer, TimerEnv};
use crate::counter::reduce_counter;
use crate::initial_params::{read_initial_def, read_opt_para_file};
use crate::io::{output_data, output_opt_data, store_opt_data};
use crate::observables::clear_phys_quantity;
use crate::reducer::{Reducer, SingleProcessReducer};
use crate::slater_update::{update_slater_elm, update_slater_elm_fsz};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use crate::sync::sync_modified_parameter as sync_modified;
use crate::sync::sync_modified_parameter_local as sync_modified_parameter;

/// C-compatible parser default when `RndSeed` is omitted. Zero is a
/// valid seed, and negative ModPara seeds request the current Unix time.
pub const FALLBACK_SEED: i64 = 11272;

/// Per-step callback; errors propagate to the caller like Julia exceptions.
/// Arguments are zero-based step, post-sync parameters, measured energy and status.
pub type StepCallback<'a> =
    dyn FnMut(usize, &mut ExpertModeData, Complex64, i32) -> Result<(), String> + 'a;

/// Fixed-parameter PhysCal preparation, before the sampling loop owns the
/// internal initialization and QP-weight setup.
#[derive(Debug)]
pub struct PhysCalPreparation {
    /// Parsed and overlaid Expert data.
    pub data: ExpertModeData,
    /// RNG positioned immediately before PhysCal's internal initialization.
    pub rng: Sfmt19937Rng,
    /// Number of fixed parameter slots consumed by the record.
    pub n_para_consumed: usize,
}

/// Result of a serial PhysCal core run.
#[derive(Debug)]
pub struct PhysCalResult {
    /// Fixed parameters after the measurement loop; unchanged from loading.
    pub data: ExpertModeData,
    /// Sampling and observable state.
    pub state: VmcOptimizationState,
    /// Number of measurement iterations completed.
    pub iterations: usize,
}

/// Prepare a fixed-parameter PhysCal run using Julia's phase order.
///
/// The function deliberately does not call `init_parameter` or
/// `init_qp_weight`; the PhysCal sampling driver must own those calls so the
/// fixed values are restored after the single C-compatible initialization RNG
/// consumption.
pub fn prepare_phys_cal_from_namelist(
    namelist_path: impl AsRef<Path>,
    opt_para_path: impl AsRef<Path>,
    mode: &str,
    seed: Option<i64>,
) -> Result<PhysCalPreparation, String> {
    if !matches!(mode, "real" | "cmp" | "fsz") {
        return Err(format!("mode must be :real, :cmp, or :fsz; got :{mode}"));
    }
    let namelist_path = namelist_path.as_ref();
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(namelist_path)
        .map_err(|error| error.to_string())?;
    let actual_seed = resolve_seed_with_clock(data.modpara.rnd_seed, seed, 0, || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .map_err(|error| format!("cannot resolve time-based RndSeed: {error}"))
    })?;
    let rng = seeded_rng(actual_seed)?;
    let n_para_consumed = read_opt_para_file(&mut data, opt_para_path)?;
    read_input_parameters(&mut data, namelist_path)?;
    sync_modified_parameter(&mut data, false);
    crate::validation::validate_phys_cal(&data)?;
    Ok(PhysCalPreparation {
        data,
        rng,
        n_para_consumed,
    })
}

/// Run the serial PhysCal sampling and main-calculation loop.
///
/// Parameter optimization and SR are intentionally absent. The preparation's
/// cloned data consumes the one C-compatible initialization draw block, while
/// the returned `data` remains the fixed loaded parameter set.
pub fn vmc_phys_cal(preparation: PhysCalPreparation) -> Result<PhysCalResult, String> {
    vmc_phys_cal_inner(preparation, None)
}

/// Run PhysCal and write indexed Green files under `output_dir`.
pub fn vmc_phys_cal_to_dir(
    preparation: PhysCalPreparation,
    output_dir: impl AsRef<Path>,
) -> Result<PhysCalResult, String> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    vmc_phys_cal_inner(preparation, Some(output_dir))
}

fn vmc_phys_cal_inner(
    mut preparation: PhysCalPreparation,
    output_dir: Option<&Path>,
) -> Result<PhysCalResult, String> {
    let mut init_data = preparation.data.clone();
    init_parameter(&mut init_data, &mut preparation.rng);
    if preparation.data.modpara.nmp_trans == 0 {
        preparation.data.modpara.nmp_trans = 1;
    } else if preparation.data.modpara.nmp_trans < 0 {
        preparation.data.modpara.nmp_trans = preparation.data.modpara.nmp_trans.abs();
    }
    preparation.data.modpara.vmc_calc_mode = 1;
    init_qp_weight(&mut preparation.data);
    let all_complex = get_all_complex_flag(&preparation.data);
    let use_fsz = preparation.data.i_flg_orbital_general != 0;
    let mut state = state_from_data(&preparation.data);
    if use_fsz {
        update_slater_elm_fsz(&mut preparation.data, &mut state);
    } else {
        update_slater_elm(&mut preparation.data, &mut state);
    }
    let iterations = preparation.data.modpara.n_data_qty_smp.max(0) as usize;
    for sample in 0..iterations {
        if use_fsz {
            if all_complex {
                crate::sampling::driver::vmc_make_sample_fsz(
                    &preparation.data,
                    &mut state,
                    &mut preparation.rng,
                );
            } else {
                crate::sampling::vmc_make_sample_fsz_real(
                    &preparation.data,
                    &mut state,
                    &mut preparation.rng,
                )
                .map_err(|error| error.to_string())?;
                sync_real_fsz_shadow(&mut state);
            }
        } else if all_complex {
            crate::sampling::driver::vmc_make_sample(
                &preparation.data,
                &mut state,
                &mut preparation.rng,
            );
        } else {
            crate::sampling::driver::vmc_make_sample_real(
                &preparation.data,
                &mut state,
                &mut preparation.rng,
            );
        }
        clear_phys_quantity(&mut state);
        accumulate_observables(
            &preparation.data,
            &mut state,
            all_complex,
            use_fsz,
            &mut CTimer::<false>::new(),
        );
        if let Some(output_dir) = output_dir {
            crate::io::output_phys_data(&preparation.data, &state, sample, Some(output_dir))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(PhysCalResult {
        data: preparation.data,
        state,
        iterations,
    })
}

/// Optional controls for the direct optimization loop.
#[derive(Default)]
pub struct OptimizationOptions<'a> {
    /// Called after successful SR and synchronization, or after sampling-only output.
    pub callback: Option<&'a mut StepCallback<'a>>,
    /// Stop after the first sample/output step, before SR, sync and final output.
    pub skip_sr: bool,
}

/// Run `nsteps` SR steps starting from `data` and the seeded `rng`.
///
/// Real and complex sz-conserved models and complex AP/P FSZ models use
/// their corresponding drivers. Runtime validation rejects unported
/// features before sampling; a failed SR step returns an error before
/// parameter synchronization or any subsequent iteration.
pub fn vmc_para_opt<R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    options: OptimizationOptions<'_>,
) -> Result<(), String> {
    vmc_para_opt_timed(
        data,
        state,
        rng,
        output_dir,
        reducer,
        options,
        &mut CTimer::<false>::new(),
    )
}

/// Run the same numerical loop with a caller-owned, optionally enabled timer.
/// Disabled instantiations compile clock reads and slot operations out of kernels.
pub fn vmc_para_opt_timed<const TIMED: bool, R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    mut options: OptimizationOptions<'_>,
    timer: &mut CTimer<TIMED>,
) -> Result<(), String> {
    crate::validation::validate_para_opt(data)?;
    let world_size = reducer.world_size();
    let rank = reducer.rank();
    if world_size > 1 && data.modpara.nsplit_size != 1 {
        return Err("MPI sample-parallel execution requires NSplitSize = 1 (issue #35)".into());
    }
    if rank >= world_size {
        return Err(format!(
            "MPI rank {rank} is outside world size {world_size}"
        ));
    }
    let n_steps = data.modpara.nsr_opt_itr_step.max(0) as usize;
    let window_start = n_steps as i64 - data.modpara.nsr_opt_itr_smp;
    let n_para = data.count_variational_parameters();
    data.ensure_optimization_flags(n_para);
    let all_complex = get_all_complex_flag(data);
    let i_flg_general = data.i_flg_orbital_general;
    let use_fsz = i_flg_general != 0;
    if world_size > 1 {
        let total_samples = data.modpara.nvmc_sample.max(0) as usize;
        let local = crate::parallel::partition_range(total_samples, world_size, rank);
        data.modpara.nvmc_sample = (local.end - local.start) as i64;
    }
    timer.start(2);
    for step in 0..n_steps {
        timer.start(20);
        // 1. Slater table refresh.
        if use_fsz {
            update_slater_elm_fsz(data, state);
        } else {
            update_slater_elm(data, state);
        }
        crate::qp::update_qp_weight_for(data);
        timer.stop(20);
        timer.start(3);
        // 2. Sampler.
        let _sample_stats: Result<crate::sampling::SampleStats, String> = if use_fsz {
            if all_complex {
                Ok(crate::sampling::driver::vmc_make_sample_fsz_timed(
                    data, state, rng, timer,
                ))
            } else {
                crate::sampling::vmc_make_sample_fsz_real(data, state, rng)
                    .map_err(|error| error.to_string())?;
                sync_real_fsz_shadow(state);
                Ok(crate::sampling::SampleStats {
                    accepted: state.electron_config.counter[1] as usize,
                    saved: data.modpara.nvmc_sample.max(0) as usize,
                })
            }
        } else if !all_complex {
            Ok(crate::sampling::driver::vmc_make_sample_real_timed(
                data, state, rng, timer,
            ))
        } else {
            Ok(crate::sampling::driver::vmc_make_sample_timed(
                data, state, rng, timer,
            ))
        };
        timer.stop(3);

        // Julia proceeds after a void sampler early return, retaining the saved
        // configurations. Any nonfinite SR result then stops before mutation.
        // 3. Main accumulator.
        timer.start(4);
        timer.start(24);
        clear_phys_quantity(state);
        timer.stop(24);
        accumulate_observables(data, state, all_complex, use_fsz, timer);

        timer.stop(4);
        timer.start(21);
        reduce_accumulators(state, reducer);
        timer.start_diag(960, timer.diagnostics.weightavg);
        // 4. Weighted averages + counter reduction.
        timer.start_diag(962, timer.diagnostics.weightavg);
        weight_average_we(state);
        timer.stop_diag(962, timer.diagnostics.weightavg);
        timer.start(25);
        timer.start_diag(965, timer.diagnostics.weightavg);
        if all_complex {
            weight_average_sr_opt(state);
        } else {
            weight_average_sr_opt_real(state);
        }
        timer.stop_diag(965, timer.diagnostics.weightavg);
        timer.stop(25);
        timer.start_diag(966, timer.diagnostics.weightavg);
        reduce_counter(state, reducer);
        timer.stop_diag(966, timer.diagnostics.weightavg);
        timer.stop_diag(960, timer.diagnostics.weightavg);
        timer.stop(21);

        // 5. Output.
        timer.start(22);
        if rank == 0 {
            output_data(data, state, step, output_dir).map_err(|e| e.to_string())?;
        }
        timer.stop(22);

        if options.skip_sr {
            timer.stop(2);
            if let Some(callback) = options.callback.as_mut() {
                callback(step, data, state.energy.etot, 0)?;
            }
            return Ok(());
        }

        // 6. SR update.
        timer.start(5);
        let info = if data.modpara.nsrcg != 0 {
            crate::sr_cg::stochastic_opt_cg(data, state, output_dir).map_err(|e| e.to_string())?
        } else if all_complex {
            crate::sr::stochastic_opt_complex_timed(data, state, timer)
        } else {
            crate::sr::stochastic_opt_real_timed(data, state, timer)
        };
        timer.stop(5);
        if info != 0 {
            timer.stop(2);
            return Err(format!(
                "vmc_para_opt: {} SR failed at step {step} (status {info}); parameters were not updated",
                if data.modpara.nsrcg != 0 { "CG" } else { "direct" }
            ));
        }

        // 7. Sync modified parameters.
        timer.start(23);
        sync_modified(data, reducer);
        timer.stop(23);
        if step as i64 >= window_start {
            store_opt_data(data, state, (step as i64 - window_start) as usize);
        }
        if let Some(callback) = options.callback.as_mut() {
            callback(step, data, state.energy.etot, info)?;
        }
    }

    if rank == 0 {
        output_opt_data(data, output_dir).map_err(|e| e.to_string())?;
    }
    timer.stop(2);
    Ok(())
}

/// Initial parameter overlay policy, matching Julia's `:auto`, `:none` and path.
#[derive(Debug, Clone, Default)]
pub enum InitialDef {
    /// Load a neighboring initial.def if present; malformed records fail.
    #[default]
    Auto,
    /// Skip initial.def even when present.
    None,
    /// Require this file, resolved relative to the current working directory.
    Path(std::path::PathBuf),
}

/// Configuration for the namelist runner. Actual numerical mode comes from inputs.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Positive number of optimization steps.
    pub nsteps: i64,
    /// Sanity label: `real`, `cmp` or `fsz`; does not override input mode.
    pub mode: String,
    /// Final averaging window; None preserves NSROptItrSmp from modpara.def.
    pub nsmp: Option<i64>,
    /// Output directory; None creates a fresh directory in the system temp area.
    pub output_dir: Option<std::path::PathBuf>,
    /// Explicit seed override.
    pub seed: Option<i64>,
    /// Initial parameter file selection.
    pub initial_def: InitialDef,
    /// C-style OptTrans activation. `None` preserves the library's Julia
    /// definition-file default; `Some(false)` ignores OptTrans definitions.
    pub enable_opt_trans: Option<bool>,
}

impl RunConfig {
    /// Construct a configuration with Julia's optional argument defaults.
    pub fn new(nsteps: i64, mode: impl Into<String>) -> Self {
        Self {
            nsteps,
            mode: mode.into(),
            nsmp: None,
            output_dir: None,
            seed: None,
            initial_def: InitialDef::Auto,
            enable_opt_trans: None,
        }
    }
}

/// Run optimization with Julia's configuration, phase order and summary contract.
pub fn run_para_opt_from_namelist(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
) -> Result<RunSummary, String> {
    validate_run_options(&config)?;
    let flags = TimerEnv::from_env();
    if flags.legacy_warning() {
        eprintln!("warning: MVMC_TIMER is deprecated; use MVMC_C_TIMER=1 for the C-compatible zvo_CalcTimer.dat timer.");
    }
    if flags.enabled() {
        run_para_opt_timed(namelist_path, config, &mut CTimer::<true>::new(), flags)
    } else {
        run_para_opt_timed(namelist_path, config, &mut CTimer::<false>::new(), flags)
    }
}

fn validate_run_options(config: &RunConfig) -> Result<(), String> {
    if !matches!(config.mode.as_str(), "real" | "cmp" | "fsz") {
        return Err(format!(
            "mode must be :real, :cmp, or :fsz; got {}",
            config.mode
        ));
    }
    if config.nsteps <= 0 {
        return Err(format!("nsteps must be positive; got {}", config.nsteps));
    }
    if config.nsmp.is_some_and(|value| value <= 0) {
        return Err(format!(
            "nsmp must be positive when provided; got {}",
            config.nsmp.unwrap()
        ));
    }
    Ok(())
}

fn run_para_opt_timed<const TIMED: bool>(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
    timer: &mut CTimer<TIMED>,
    flags: TimerEnv,
) -> Result<RunSummary, String> {
    timer.reset();
    timer.diagnostics = flags;
    timer.start(0);
    timer.start(1);
    timer.start(11);
    let mut data = match config.enable_opt_trans {
        Some(enabled) => {
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist_path, enabled)
        }
        // The production runner follows C's FlagOptTrans contract: a
        // definition file alone does not activate optimized translation.
        None => {
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist_path, false)
        }
    }
    .map_err(|e| e.to_string())?;
    timer.stop(11);
    crate::validation::validate_para_opt(&data)?;
    let effective_nsmp = config.nsmp.unwrap_or(data.modpara.nsr_opt_itr_smp);
    if effective_nsmp <= 0 {
        return Err(format!(
            "effective nsmp must be positive; got {effective_nsmp}"
        ));
    }
    if config.nsteps < effective_nsmp {
        return Err(format!("nsteps ({}) must be >= nsmp ({effective_nsmp}); smaller nsteps would zero-pad optimisation averages", config.nsteps));
    }

    // Preserve Julia's init -> initial.def -> In*.def -> sync -> QP phase order.
    let actual_seed = resolve_seed_with_clock(data.modpara.rnd_seed, config.seed, 0, || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .map_err(|error| format!("cannot resolve time-based RndSeed: {error}"))
    })?;
    let mut rng = seeded_rng(actual_seed)?;
    timer.start(13);
    init_parameter(&mut data, &mut rng);
    let (initial_path, auto) = match config.initial_def {
        InitialDef::Auto => {
            let path = namelist_path
                .as_ref()
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("initial.def");
            (path.is_file().then_some(path), true)
        }
        InitialDef::None => (None, false),
        InitialDef::Path(path) => (Some(path), false),
    };
    if let Some(path) = initial_path {
        if !read_initial_def(&mut data, &path).map_err(|error| error.to_string())? {
            return Err(if auto {
                format!("read_initial_def! failed on auto-detected initial.def at {}; pass initial_def=None to skip explicitly", path.display())
            } else {
                format!(
                    "read_initial_def! failed for explicitly requested path: {}",
                    path.display()
                )
            });
        }
    }
    read_input_parameters(&mut data, &namelist_path)?;
    sync_modified_parameter(&mut data, true);
    timer.stop(13);
    init_qp_weight(&mut data);
    timer.stop(1);
    data.modpara.nsr_opt_itr_step = config.nsteps;
    data.modpara.nsr_opt_itr_smp = effective_nsmp;
    let mut state = state_from_data(&data);
    let output_dir = match config.output_dir {
        Some(path) => {
            std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
            path
        }
        None => fresh_output_directory()?,
    };
    vmc_para_opt_timed(
        &mut data,
        &mut state,
        &mut rng,
        Some(&output_dir),
        &SingleProcessReducer,
        OptimizationOptions::default(),
        timer,
    )?;
    timer.stop(0);
    if TIMED {
        timer
            .write_para_opt(&output_dir, "zvo")
            .map_err(|error| error.to_string())?;
        if flags.any_diag() {
            timer
                .write_diag(&output_dir, "zvo")
                .map_err(|error| error.to_string())?;
        }
    }
    read_run_summary(&data, &output_dir)
}

fn fresh_output_directory() -> Result<std::path::PathBuf, String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    loop {
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("mvmc-run-{}-{id}", std::process::id()));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.to_string()),
        }
    }
}

/// Sum the per-rank sample accumulators before normalization and SR.
///
/// Sampling buffers remain rank-local; only quantities that feed the global
/// weighted averages are reduced.  This mirrors Julia's `WeightAverage!`
/// boundary and keeps rank-local configurations available for diagnostics.
fn reduce_accumulators<R: Reducer + ?Sized>(state: &mut VmcOptimizationState, reducer: &R) {
    let mut energy = [
        state.energy.wc,
        state.energy.etot,
        state.energy.etot2,
        state.energy.sztot,
        state.energy.sztot2,
    ];
    reducer.allreduce_sum_c64(&mut energy);
    [
        &mut state.energy.wc,
        &mut state.energy.etot,
        &mut state.energy.etot2,
        &mut state.energy.sztot,
        &mut state.energy.sztot2,
    ]
    .into_iter()
    .zip(energy)
    .for_each(|(dst, value)| *dst = value);

    reducer.allreduce_sum_c64(&mut state.sr_opt.sr_opt_oo);
    reducer.allreduce_sum_c64(&mut state.sr_opt.sr_opt_ho);
    if !state.sr_opt.sr_opt_oo_real.is_empty() {
        reducer.allreduce_sum_f64(&mut state.sr_opt.sr_opt_oo_real);
        reducer.allreduce_sum_f64(&mut state.sr_opt.sr_opt_ho_real);
    }
    if let Some(phys) = state.phys_quantities.as_mut() {
        reducer.allreduce_sum_c64(&mut phys.phys_cis_ajs);
        reducer.allreduce_sum_c64(&mut phys.phys_cis_ajs_ckt_alt);
        reducer.allreduce_sum_c64(&mut phys.phys_cis_ajs_ckt_alt_dc);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qqqq);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qcisajsq);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qcisajscktaltq);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qcisajscktaltq_dc);
    }
}

#[cfg(test)]
mod mpi_tests {
    use super::*;

    #[derive(Debug)]
    struct ScalingReducer {
        world: usize,
        rank: usize,
    }

    impl Reducer for ScalingReducer {
        fn allreduce_sum_f64(&self, values: &mut [f64]) {
            for value in values {
                *value *= self.world as f64;
            }
        }

        fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
            for value in values {
                *value *= self.world as f64;
            }
        }

        fn allreduce_sum_i64(&self, values: &mut [i64]) {
            for value in values {
                *value *= self.world as i64;
            }
        }

        fn world_size(&self) -> usize {
            self.world
        }

        fn rank(&self) -> usize {
            self.rank
        }
    }

    #[test]
    fn sample_parallel_reduction_sums_energy_and_sr_buffers() {
        let mut state = VmcOptimizationState::zeros(2, 2, 0, 1, 1, 2, true, false);
        state.energy.wc = Complex64::new(3.0, 0.0);
        state.sr_opt.sr_opt_oo[0] = Complex64::new(4.0, -1.0);
        state.sr_opt.sr_opt_ho[0] = Complex64::new(-2.0, 0.5);
        reduce_accumulators(&mut state, &ScalingReducer { world: 2, rank: 1 });
        assert_eq!(state.energy.wc, Complex64::new(6.0, 0.0));
        assert_eq!(state.sr_opt.sr_opt_oo[0], Complex64::new(8.0, -2.0));
        assert_eq!(state.sr_opt.sr_opt_ho[0], Complex64::new(-4.0, 1.0));
    }
}

fn read_run_summary(data: &ExpertModeData, output_dir: &Path) -> Result<RunSummary, String> {
    let nsteps = data.modpara.nsr_opt_itr_step as usize;
    let nsmp = data.modpara.nsr_opt_itr_smp as usize;
    // Issue #48 requires configured-head readback; Julia v0.5.0 hardcodes
    // zvo here despite honoring CDataFileHead in its writer.
    let head = if data.modpara.c_data_file_head.is_empty() {
        "zvo"
    } else {
        &data.modpara.c_data_file_head
    };
    let path = output_dir.join(format!("{head}_out.dat"));
    let text = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
    let lines: Vec<_> = text.lines().map(|line| line.trim().to_owned()).collect();
    if lines.len() < nsteps {
        return Err(format!(
            "expected at least {nsteps} output rows, found {} at {}",
            lines.len(),
            path.display()
        ));
    }
    let zvo_first_n = lines[..nsteps].to_vec();
    let rows: Vec<Vec<f64>> = zvo_first_n
        .iter()
        .map(|line| {
            line.split_whitespace()
                .map(|token| token.parse::<f64>().map_err(|error| error.to_string()))
                .collect()
        })
        .collect::<Result<_, _>>()?;
    if rows.iter().any(|row| row.len() < 2) {
        return Err(format!(
            "output must contain at least two columns for C ctest comparison: {}",
            path.display()
        ));
    }
    if data.modpara.nsite <= 0 {
        return Err("modpara.nsite must be positive to compute energy per site".into());
    }
    let window = &rows[nsteps - nsmp..nsteps];
    let ctest_values = (0..2)
        .map(|column| window.iter().map(|row| row[column]).sum::<f64>() / nsmp as f64)
        .collect();
    Ok(RunSummary {
        status: 0,
        output_dir: std::fs::canonicalize(output_dir).map_err(|error| error.to_string())?,
        zvo_first_n,
        ctest_values,
        final_energy_per_site: rows[nsteps - 1][0] / data.modpara.nsite as f64,
        effective_nsteps: nsteps,
        effective_nsmp: nsmp,
    })
}

/// Infer the execution mode from explicit flags or current factor values.
///
/// Mirrors Julia's `get_all_complex_flag`. Parameter initialization has
/// its own declaration-based rule: inferring from loaded values there
/// would alter the RNG draw count before the overlays are applied.
pub fn get_all_complex_flag(data: &ExpertModeData) -> bool {
    if !data.complex_flags.is_empty() {
        return data.complex_flags.iter().any(|&flag| flag != 0);
    }
    mvmc_expert_parsers::utils::parameter_init::all_complex_flag(data)
}

fn resolve_seed_with_clock(
    rnd_seed: i64,
    seed_override: Option<i64>,
    group1: i64,
    unix_seconds: impl FnOnce() -> Result<i64, String>,
) -> Result<i64, String> {
    let base = if let Some(seed) = seed_override {
        seed
    } else if rnd_seed < 0 {
        unix_seconds()?
    } else {
        rnd_seed
    };
    Ok(base.wrapping_add(group1))
}

fn seeded_rng(seed: i64) -> Result<Sfmt19937Rng, String> {
    let seed = u32::try_from(seed)
        .map_err(|_| format!("resolved SFMT seed {seed} is outside the UInt32 range"))?;
    Ok(Sfmt19937Rng::new(seed))
}

#[cfg(test)]
mod seed_tests {
    use super::*;

    #[test]
    fn seed_conversion_rejects_values_outside_julia_uint32_range() {
        for seed in [-1, i64::from(u32::MAX) + 1] {
            assert!(seeded_rng(seed).is_err(), "SFMT.jl UInt32 rejects {seed}");
        }
        assert!(seeded_rng(0).is_ok());
        assert!(seeded_rng(i64::from(u32::MAX)).is_ok());
    }

    // Julia test_unit_parallel.jl: resolve_rnd_seed C parity.
    #[test]
    fn resolve_seed_matches_julia_serial_and_group_policy() {
        for (input, explicit, group, expected) in [
            (FALLBACK_SEED, None, 0, 11272),
            (0, None, 0, 0),
            (123, None, 0, 123),
            (-1, None, 0, 1700000000),
            (123, Some(777), 0, 777),
            (-1, Some(777), 0, 777),
            (100, None, 3, 103),
            (-1, None, 3, 1700000003),
        ] {
            assert_eq!(
                resolve_seed_with_clock(input, explicit, group, || Ok(1700000000)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn clock_is_only_read_for_negative_modpara_without_override() {
        assert_eq!(
            resolve_seed_with_clock(0, None, 0, || panic!("unneeded clock read")).unwrap(),
            0
        );
        assert_eq!(
            resolve_seed_with_clock(-1, Some(7), 0, || panic!("override reads clock")).unwrap(),
            7
        );
        assert!(resolve_seed_with_clock(-1, None, 0, || Err("clock failed".into())).is_err());
    }

    #[test]
    fn resolved_seed_stream_matches_julia_sfmt_full_block() {
        // SFMT.jl at the pinned Julia v0.5.0 revision. Hash every word
        // of a full 624-word SFMT block; do not compare sample averages.
        for (input, explicit, group, expected_hash) in [
            (11272, None, 0, 6236248514720008343_u64),
            (0, None, 0, 859227577111010836),
            (123, None, 0, 3014287863449549136),
            (-1, None, 0, 2984471086139650888),
            (-1, Some(777), 0, 18056312496568059157),
            (100, None, 3, 8720158099669028788),
            (-1, None, 3, 8638024564356521936),
        ] {
            let seed = resolve_seed_with_clock(input, explicit, group, || Ok(1700000000)).unwrap();
            let mut rng = seeded_rng(seed).unwrap();
            let mut hash = 0xcbf29ce484222325_u64;
            for _ in 0..624 {
                hash = (hash ^ u64::from(rng.gen_rand32())).wrapping_mul(0x100000001b3);
            }
            assert_eq!(hash, expected_hash, "resolved seed {seed}");
        }
    }
}

fn state_from_data(data: &ExpertModeData) -> VmcOptimizationState {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    let n_sp = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_opt = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_full = n_sp * n_mp * n_opt;
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let all_complex = get_all_complex_flag(data);
    let mut state = VmcOptimizationState::zeros(
        n_site,
        n_elec,
        n_proj,
        n_para,
        n_qp_full,
        n_vmc_sample,
        all_complex,
        data.i_flg_orbital_general != 0,
    );
    if data.modpara.vmc_calc_mode != 0 {
        state.phys_quantities = Some(crate::state::PhysicalQuantities::zeros(
            data.green_one_terms.len(),
            data.green_two_ex_terms.len(),
            data.green_two_terms.len(),
        ));
    }
    state
}

/// Make real-FSZ sampling results visible to the shared observable kernels.
fn sync_real_fsz_shadow(state: &mut VmcOptimizationState) {
    for (dst, src) in state
        .slater_matrix
        .slater_elm
        .as_mut_slice()
        .iter_mut()
        .zip(state.slater_matrix.slater_elm_real.as_slice())
    {
        *dst = Complex64::new(*src, 0.0);
    }
    for (dst, src) in state
        .slater_matrix
        .inv_m
        .as_mut_slice()
        .iter_mut()
        .zip(state.slater_matrix.inv_m_real.as_slice())
    {
        *dst = Complex64::new(*src, 0.0);
    }
    for (dst, src) in state
        .slater_matrix
        .pf_m
        .iter_mut()
        .zip(state.slater_matrix.pf_m_real.iter())
    {
        *dst = Complex64::new(*src, 0.0);
    }
}

#[cfg(test)]
mod mode_tests {
    use super::*;
    use mvmc_expert_parsers::{
        GreenOneTerm, GreenTwoExTerm, GreenTwoTerm, GutzwillerTerm, JastrowTerm, OrbitalTerm, Spin,
    };

    #[test]
    fn opttrans_state_reserves_active_width_after_projection_rbm_and_slater() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/opttrans/namelist_layout.def");
        let mut data = parse_expert_mode_files(path).unwrap();
        let base = data.projection_layout().n_proj + data.count_rbm_parameters() + n_slater(&data);
        for width in [0, 1, 2, 3] {
            data.opt_trans.resize(width, Complex64::new(1.0, 0.0));
            let state = state_from_data(&data);
            assert_eq!(state.sr_opt.sr_opt_size, 1 + base + width);
            assert_eq!(
                state.slater_matrix.pf_m.len(),
                data.modpara.nsp_gauss_leg.max(1) as usize
                    * data.modpara.nmp_trans.unsigned_abs().max(1) as usize
                    * data.n_qp_opt_trans.max(1) as usize
            );
        }
    }

    #[test]
    fn rbm_state_places_all_nine_blocks_between_projection_and_slater() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rbm/namelist_all.def");
        let data = crate::historical_orbital_model::historical_kernel_model(path).unwrap();
        assert_eq!(state_from_data(&data).sr_opt.sr_opt_size, 37);
    }

    fn data() -> ExpertModeData {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.modpara.nmp_trans = 1;
        data.modpara.nvmc_sample = 1;
        data.modpara.n_orbital_idx = 1;
        data.slater_params = vec![Complex64::new(1.0, 0.0)];
        data.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: 0,
            is_complex: false,
            sign: 1,
        });
        data
    }

    #[test]
    fn real_fsz_observation_refresh_populates_real_shadow() {
        let mut data = data();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, true);
        state
            .slater_matrix
            .slater_elm
            .set(0, 0, 3, Complex64::new(1.0, 0.0));
        state
            .slater_matrix
            .slater_elm
            .set(0, 3, 0, Complex64::new(-1.0, 0.0));
        let pool = ThreadedPfaPackWorkspace::new(2, 1);
        refresh_fsz_observation_matrix(&data, &mut state, false, &[0, 1], &[0, 1], &pool).unwrap();
        assert_ne!(state.slater_matrix.pf_m_real[0].to_bits(), 0);
        assert_eq!(state.slater_matrix.pf_m[0].im, 0.0);
        assert_eq!(
            state.slater_matrix.pf_m[0].re.to_bits(),
            state.slater_matrix.pf_m_real[0].to_bits()
        );
        assert!(state
            .slater_matrix
            .inv_m_real
            .qp_matrix_slice(0)
            .iter()
            .any(|value| value.to_bits() != 0));
    }

    // Julia test_unit_types.jl checks that complex SROptData has no real
    // buffers. Exercise the actual runner allocation for each factor family.
    #[test]
    fn complex_gutzwiller_with_real_orbitals_allocates_complex_state() {
        let mut data = data();
        data.gutzwiller_terms.push(GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.0, 0.0),
            is_complex: true,
        });
        let state = state_from_data(&data);
        assert!(state.sr_opt.sr_opt_oo_real.is_empty());
        assert_eq!(state.slater_matrix.slater_elm_real.n_qp_full(), 0);
    }

    #[test]
    fn complex_jastrow_with_real_orbitals_allocates_complex_state() {
        let mut data = data();
        data.jastrow_terms.push(JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(0.0, 0.0),
            is_complex: true,
        });
        assert!(state_from_data(&data).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn imaginary_loaded_parameter_does_not_change_c_header_mode() {
        let mut data = data();
        data.slater_params[data.orbital_terms[0].idx as usize].im = 0.5;
        let state = state_from_data(&data);
        // C's AllComplexFlag is decided from definition headers before
        // loading values; an imaginary overlay cannot change the buffers.
        assert!(!state.sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn all_real_parameters_allocate_real_state() {
        assert!(!state_from_data(&data()).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn physcal_state_allocates_green_measurement_buffers() {
        let mut data = data();
        data.modpara.vmc_calc_mode = 1;
        data.green_one_terms.push(GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
        });
        data.green_two_terms.push(GreenTwoTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 1,
            spin3: Spin::Down,
            site4: 0,
            spin4: Spin::Down,
        });
        data.green_two_ex_terms.push(GreenTwoExTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 1,
            spin3: Spin::Down,
            site4: 0,
            spin4: Spin::Down,
        });
        let state = state_from_data(&data);
        let phys = state.phys_quantities.expect("PhysCal buffers");
        assert_eq!(phys.local_cis_ajs.len(), 1);
        assert_eq!(phys.phys_cis_ajs_ckt_alt.len(), 1);
        assert_eq!(phys.local_cis_ajs_ckt_alt_dc.len(), 1);
    }

    #[test]
    fn physcal_preparation_loads_fixed_parameters_before_rng_consumption() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans");
        let parsed = parse_expert_mode_files(root.join("namelist_layout.def")).unwrap();
        let n_fields = 6 + 3 * parsed.count_variational_parameters();
        let opt_path =
            std::env::temp_dir().join(format!("mvmc-physcal-opt-{}", std::process::id()));
        fs::write(
            &opt_path,
            (0..n_fields).map(|_| "0").collect::<Vec<_>>().join(" "),
        )
        .unwrap();
        let prepared = prepare_phys_cal_from_namelist(
            root.join("namelist_layout.def"),
            &opt_path,
            "real",
            Some(11272),
        )
        .expect("fixed PhysCal preparation");
        fs::remove_file(opt_path).unwrap();
        assert!(prepared.n_para_consumed > 0);
        assert!(
            prepared.data.modpara.vmc_calc_mode == 0 || prepared.data.modpara.vmc_calc_mode == 1
        );
        let mut rng = prepared.rng;
        assert_ne!(rng.gen_rand32(), 0);
    }

    #[test]
    fn physcal_iteration_keeps_fixed_parameters_unchanged() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans");
        let parsed = parse_expert_mode_files(root.join("namelist_layout.def")).unwrap();
        let n_fields = 6 + 3 * parsed.count_variational_parameters();
        let opt_path =
            std::env::temp_dir().join(format!("mvmc-physcal-iteration-{}", std::process::id()));
        fs::write(
            &opt_path,
            (0..n_fields).map(|_| "0").collect::<Vec<_>>().join(" "),
        )
        .unwrap();
        let mut preparation = prepare_phys_cal_from_namelist(
            root.join("namelist_layout.def"),
            &opt_path,
            "real",
            Some(11272),
        )
        .unwrap();
        fs::remove_file(opt_path).unwrap();
        preparation.data.modpara.n_data_qty_smp = 1;
        let before = preparation.data.slater_params.clone();
        let result = vmc_phys_cal(preparation).unwrap();
        assert_eq!(result.iterations, 1);
        assert_eq!(result.data.slater_params, before);
    }

    #[test]
    fn zero_translation_count_is_rejected_before_mutation_rng_or_output() {
        let mut data = data();
        data.modpara.nmp_trans = 0;
        data.modpara.nsp_gauss_leg = 3;
        let mut state = state_from_data(&data);
        assert_eq!(state.slater_matrix.slater_elm.n_qp_full(), 0);
        assert_eq!(state.slater_matrix.slater_elm_real.n_qp_full(), 0);
        let parameters = data.slater_params.clone();
        let mut rng = Sfmt19937Rng::new(1);
        let mut expected_rng = rng.clone();
        let dir = std::env::temp_dir().join(format!("mvmc-c-zero-count-{}", std::process::id()));
        assert!(!dir.exists());
        let err = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap_err();
        assert!(err.contains("NMPTrans"), "{err}");
        assert_eq!(data.modpara.nmp_trans, 0);
        assert_eq!(data.slater_params, parameters);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), expected_rng.gen_rand32());
        }
        assert!(!dir.exists());
    }

    #[test]
    fn pure_general_and_ap_parallel_runs_have_identical_chain_and_updates() {
        let namelist = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_fsz/namelist.def");
        let template = parse_expert_mode_files(namelist).unwrap();
        let run = |pure_general| {
            let mut data = template.clone();
            data.modpara.nsr_opt_itr_step = 1;
            if pure_general {
                let nsite = data.modpara.nsite;
                let ap = data.n_orbital_anti_parallel;
                for term in &mut data.orbital_terms {
                    if term.idx < ap {
                        term.site2 += nsite;
                    } else if (term.idx - ap) % 2 == 1 {
                        term.site1 += nsite;
                        term.site2 += nsite;
                    }
                }
                data.i_flg_orbital_anti_parallel = 0;
                data.i_flg_orbital_parallel = 0;
                data.orbital_idx_matrix = None;
                data.orbital_sgn_matrix = None;
            }
            let mut rng = Sfmt19937Rng::new(1);
            init_parameter(&mut data, &mut rng);
            sync_modified_parameter(&mut data, true);
            init_qp_weight(&mut data);
            let mut state = state_from_data(&data);
            let output = std::env::temp_dir().join(format!(
                "mvmc-general-trajectory-{}-{pure_general}",
                std::process::id()
            ));
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&output),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            fs::remove_dir_all(output).unwrap();
            let words: Vec<_> = (0..624).map(|_| rng.gen_rand32()).collect();
            let values: Vec<_> = data
                .orbital_terms
                .iter()
                .map(|term| data.slater_params[term.idx as usize])
                .collect();
            (state, words, values)
        };
        let (general_state, general_words, general_values) = run(true);
        let (ap_state, ap_words, ap_values) = run(false);
        assert_eq!(general_values, ap_values);
        assert_eq!(
            general_state.electron_config.ele_idx,
            ap_state.electron_config.ele_idx
        );
        assert_eq!(
            general_state.electron_config.ele_spn,
            ap_state.electron_config.ele_spn
        );
        assert_eq!(general_state.energy.etot, ap_state.energy.etot);
        assert_eq!(general_words, ap_words);
    }

    #[test]
    fn explicit_zero_complex_flags_override_imaginary_values() {
        let mut data = data();
        data.slater_params[data.orbital_terms[0].idx as usize].im = 0.5;
        data.complex_flags = vec![0, 0];
        assert!(!state_from_data(&data).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn explicit_nonzero_complex_flags_select_complex_state() {
        let mut data = data();
        data.complex_flags = vec![0, -1];
        assert!(state_from_data(&data).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn modpara_flag_does_not_override_runtime_factor_inference() {
        let mut data = data();
        data.modpara.complex_flag = 1;
        assert!(!state_from_data(&data).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn imaginary_projection_values_do_not_change_c_header_mode() {
        for gutzwiller in [true, false] {
            let mut data = data();
            if gutzwiller {
                data.gutzwiller_terms.push(GutzwillerTerm {
                    site: 0,
                    value: Complex64::new(0.0, 0.5),
                    is_complex: false,
                });
            } else {
                data.jastrow_terms.push(JastrowTerm {
                    site1: 0,
                    site2: 1,
                    value: Complex64::new(0.0, 0.5),
                    is_complex: false,
                });
            }
            let state = state_from_data(&data);
            assert!(!state.sr_opt.sr_opt_oo_real.is_empty());
            assert!(!state.sr_opt.sr_opt_ho_real.is_empty());
            assert!(!state.sr_opt.sr_opt_o_real.is_empty());
            assert!(!state.sr_opt.sr_opt_o_store_real.is_empty());
        }
    }
}

/// Summary returned by [`run_para_opt_from_namelist`].
///
/// Mirrors the `NamedTuple` returned by Julia's `run_para_opt_from_namelist`.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Successful optimizer status (failures return an error).
    pub status: i32,
    /// Absolute output directory.
    pub output_dir: std::path::PathBuf,
    /// First effective_nsteps output rows, trimmed as in Julia.
    pub zvo_first_n: Vec<String>,
    /// Means of the first two columns over the final effective_nsmp rows.
    pub ctest_values: Vec<f64>,
    /// Last step's total energy divided by Nsite.
    pub final_energy_per_site: f64,
    /// NSROptItrStep used for this run.
    pub effective_nsteps: usize,
    /// NSROptItrSmp used for this run.
    pub effective_nsmp: usize,
}

fn accumulate_observables<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    all_complex: bool,
    use_fsz: bool,
    timer: &mut CTimer<TIMED>,
) {
    let diag = timer.diagnostics.maincal && !use_fsz;
    timer.start_diag(940, diag);
    timer.start_diag(941, diag);
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let n_proj = data.projection_layout().n_proj;
    let sr_opt_size = state.sr_opt.sr_opt_size;
    let use_store = data.modpara.nstore_o != 0 || data.modpara.nsrcg != 0;
    state.sr_opt.sr_opt_o_store.fill(Complex64::new(0.0, 0.0));
    state.sr_opt.sr_opt_o_store_real.fill(0.0);
    let n_rbm = data.count_rbm_parameters();
    let n_orb_total = n_slater(data);
    let pool = crate::state::ThreadedPfaPackWorkspace::new(n_size, 1);
    let mut slater_derivative_scratch = crate::slater_derivative::SlaterDerivativeScratch::new();

    timer.stop_diag(941, diag);
    timer.stop_diag(940, diag);
    for sample in 0..n_vmc_sample {
        timer.start_diag(940, diag);
        timer.start_diag(942, diag);
        let ele_idx = state.electron_config.ele_idx_slice(sample).to_vec();
        if ele_idx.iter().all(|&v| v == 0) || ele_idx.iter().all(|&v| v < 0) {
            timer.stop_diag(942, diag);
            timer.stop_diag(940, diag);
            continue;
        }
        let ele_cfg = state.electron_config.ele_cfg_slice(sample).to_vec();
        let ele_num = state.electron_config.ele_num_slice(sample).to_vec();
        let ele_spn = if use_fsz {
            state.electron_config.ele_spn_slice(sample).to_vec()
        } else {
            Vec::new()
        };
        let ele_proj_cnt = if n_proj > 0 {
            state.electron_config.ele_proj_cnt_slice(sample).to_vec()
        } else {
            Vec::new()
        };

        timer.stop_diag(942, diag);
        timer.stop_diag(940, diag);
        timer.start(40);
        // Refresh Pfaffian for the saved walker.
        let info = if use_fsz {
            refresh_fsz_observation_matrix(data, state, all_complex, &ele_idx, &ele_spn, &pool)
                .err()
        } else if all_complex {
            crate::pfaffian::calc_m_all_complex(
                &ele_idx,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        } else {
            crate::pfaffian::calc_m_all_real(
                &ele_idx,
                &state.slater_matrix.slater_elm_real,
                &mut state.slater_matrix.inv_m_real,
                &mut state.slater_matrix.pf_m_real,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        };
        timer.stop(40);
        timer.start_diag(940, diag);
        timer.start_diag(943, diag);
        if info.is_some() {
            timer.stop_diag(943, diag);
            timer.stop_diag(940, diag);
            continue;
        }
        if !all_complex {
            for qp in 0..n_qp_full {
                let real_plane = state.slater_matrix.inv_m_real.qp_matrix_slice(qp);
                for col in 0..n_size {
                    for row in 0..n_size {
                        let value = real_plane[row + col * n_size];
                        state
                            .slater_matrix
                            .inv_m
                            .set(qp, row, col, Complex64::new(value, 0.0));
                    }
                }
                state.slater_matrix.pf_m[qp] =
                    Complex64::new(state.slater_matrix.pf_m_real[qp], 0.0);
            }
        }
        timer.stop_diag(943, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(944, diag);
        let ip = if all_complex {
            crate::observables::calculate_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data)
        } else {
            Complex64::new(
                crate::observables::calculate_ip_real(
                    &state.slater_matrix.pf_m_real,
                    0,
                    n_qp_full,
                    data,
                ),
                0.0,
            )
        };
        timer.stop_diag(944, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(945, diag);
        if ip.norm() < 1.0e-100 {
            timer.stop_diag(945, diag);
            timer.stop_diag(940, diag);
            continue;
        }
        let w = 1.0;
        timer.stop_diag(945, diag);
        timer.stop_diag(940, diag);
        timer.start(41);
        let e = if use_fsz {
            crate::observables::calculate_local_energy_fsz_timed(
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                &ele_spn,
                timer,
            )
        } else {
            crate::observables::calculate_local_energy_timed(
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                timer,
            )
        };
        timer.stop(41);
        // Julia rejects the sum, including overflow of otherwise finite parts.
        if !(e.re + e.im).is_finite() {
            continue;
        }
        timer.start_diag(940, diag);
        timer.start_diag(946, diag);
        let sz = crate::observables::calculate_sz(&ele_num, n_site);

        state.energy.wc += Complex64::new(w, 0.0);
        state.energy.etot += Complex64::new(w, 0.0) * e;
        state.energy.etot2 += Complex64::new(w, 0.0) * e.conj() * e;
        state.energy.sztot += Complex64::new(w * sz, 0.0);
        state.energy.sztot2 += Complex64::new(w * sz * sz, 0.0);

        // The Lanczos path evaluates H on each moved configuration. Transfer
        // Transfer, PairHop and Exchange use the Julia operator order;
        // InterAll and FSZ remain gated until their operator moves are
        // ported.
        if data.modpara.lanczos_mode > 0 && data.inter_all_terms.is_empty() && !use_fsz {
            let h2 = crate::observables::calculate_lanczos_h2_transfer(
                e,
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                all_complex,
            );
            if let Some(phys) = state.phys_quantities.as_mut() {
                let _ = crate::lanczos::accumulate_lanczos_qqqq(
                    &mut phys.phys_lanczos_qqqq,
                    w,
                    e,
                    h2,
                    all_complex,
                );
            }
        }

        if state.phys_quantities.is_some() {
            let mut one_body = vec![Complex64::new(0.0, 0.0); data.green_one_terms.len()];
            for (index, term) in data.green_one_terms.iter().enumerate() {
                one_body[index] = if use_fsz {
                    crate::observables::green_func1_fsz(
                        term.site1 as usize,
                        term.site2 as usize,
                        crate::observables::spin_code(term.spin1),
                        crate::observables::spin_code(term.spin2),
                        ip,
                        data,
                        state,
                        &ele_idx,
                        &ele_cfg,
                        &ele_num,
                        &ele_proj_cnt,
                        &ele_spn,
                    )
                } else {
                    crate::observables::green_func1(
                        term.site1 as usize,
                        term.site2 as usize,
                        crate::observables::spin_code(term.spin1),
                        crate::observables::spin_code(term.spin2),
                        ip,
                        data,
                        state,
                        &ele_idx,
                        &ele_cfg,
                        &ele_num,
                        &ele_proj_cnt,
                    )
                };
            }
            let mut direct = vec![Complex64::new(0.0, 0.0); data.green_two_terms.len()];
            for (index, term) in data.green_two_terms.iter().enumerate() {
                direct[index] = if use_fsz {
                    crate::observables::green_func2_fsz(
                        term.site1 as usize,
                        term.site2 as usize,
                        term.site3 as usize,
                        term.site4 as usize,
                        crate::observables::spin_code(term.spin1),
                        crate::observables::spin_code(term.spin2),
                        crate::observables::spin_code(term.spin3),
                        crate::observables::spin_code(term.spin4),
                        ip,
                        data,
                        state,
                        &ele_idx,
                        &ele_cfg,
                        &ele_num,
                        &ele_proj_cnt,
                        &ele_spn,
                    )
                } else {
                    crate::observables::green_func2(
                        term.site1 as usize,
                        term.site2 as usize,
                        term.site3 as usize,
                        term.site4 as usize,
                        crate::observables::spin_code(term.spin1),
                        crate::observables::spin_code(term.spin3),
                        ip,
                        data,
                        state,
                        &ele_idx,
                        &ele_cfg,
                        &ele_num,
                        &ele_proj_cnt,
                    )
                };
            }
            let lanczos_green = if data.modpara.lanczos_mode > 1 {
                Some(crate::observables::calculate_lanczos_green(
                    e,
                    ip,
                    data,
                    state,
                    &ele_idx,
                    &ele_cfg,
                    &ele_num,
                    &ele_proj_cnt,
                    &one_body,
                    &direct,
                    all_complex,
                ))
            } else {
                None
            };
            let phys = state.phys_quantities.as_mut().expect("checked above");
            for (index, value) in one_body.iter().copied().enumerate() {
                phys.local_cis_ajs[index] = value;
                phys.phys_cis_ajs[index] += value;
            }
            crate::observables::accumulate_two_body_gex_sample(
                &mut phys.phys_cis_ajs_ckt_alt,
                &one_body,
                &data.green_two_ex_indices,
                Complex64::new(w, 0.0),
            );
            for (index, value) in direct.into_iter().enumerate() {
                phys.local_cis_ajs_ckt_alt_dc[index] = value;
                phys.phys_cis_ajs_ckt_alt_dc[index] += value;
            }
            if let Some(values) = lanczos_green {
                for (dst, src) in phys.phys_lanczos_qcisajsq.iter_mut().zip(values.one_body) {
                    *dst += src;
                }
                for (dst, src) in phys
                    .phys_lanczos_qcisajscktaltq
                    .iter_mut()
                    .zip(values.factored_two_body)
                {
                    *dst += src;
                }
                for (dst, src) in phys
                    .phys_lanczos_qcisajscktaltq_dc
                    .iter_mut()
                    .zip(values.direct_two_body)
                {
                    *dst += src;
                }
            }
        }

        timer.stop_diag(946, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(948, diag);
        // SR `O` vector — projection diff fills the leading block.
        for slot in state.sr_opt.sr_opt_o.iter_mut() {
            *slot = Complex64::new(0.0, 0.0);
        }
        crate::observables::set_projection_diff(&mut state.sr_opt.sr_opt_o, &ele_proj_cnt, n_proj);
        // Normal Julia main-calculation reserves all RBM derivative slots.
        // Its FSZ main-calculation places Slater immediately after projection.
        if !use_fsz && n_rbm > 0 {
            let cfg = crate::sampling::rbm::RbmConfig::from(data);
            let cnt = crate::sampling::rbm::make_rbm_cnt(&ele_num, &cfg);
            let offset = 2 * (1 + n_proj);
            crate::sampling::rbm::set_rbm_diff(
                &mut state.sr_opt.sr_opt_o[offset..offset + 2 * n_rbm],
                &cnt,
                &ele_num,
                &cfg,
            );
        }
        let slater_offset = 2 * (1 + n_proj + if use_fsz { 0 } else { n_rbm });
        timer.stop_diag(948, diag);
        timer.stop_diag(940, diag);
        if n_orb_total > 0 && slater_offset < state.sr_opt.sr_opt_o.len() {
            timer.start(42);
            let n_copy = (2 * n_orb_total).min(state.sr_opt.sr_opt_o.len() - slater_offset);
            let slater_o = &mut state.sr_opt.sr_opt_o[slater_offset..slater_offset + n_copy];
            if use_fsz {
                crate::slater_derivative::slater_elm_diff_fsz_with_scratch(
                    slater_o,
                    ip,
                    &ele_idx,
                    &ele_spn,
                    data,
                    &state.slater_matrix,
                    &mut slater_derivative_scratch,
                );
            } else {
                timer.start_diag(930, timer.diagnostics.slater);
                crate::slater_derivative::slater_elm_diff_with_scratch_timed(
                    slater_o,
                    ip,
                    &ele_idx,
                    data,
                    &state.slater_matrix,
                    &mut slater_derivative_scratch,
                    timer,
                );
                timer.stop_diag(930, timer.diagnostics.slater);
            }
            timer.stop(42);
        }
        let n_opt = data.count_opt_trans_parameters();
        let opt_offset = slater_offset + 2 * n_orb_total;
        let opt_end = opt_offset + 2 * n_opt;
        if n_opt > 0 && opt_end <= state.sr_opt.sr_opt_o.len() {
            let diag = !use_fsz && timer.diagnostics.maincal;
            timer.start_diag(940, diag);
            timer.start_diag(949, diag);
            crate::observables::opt_trans_diff(
                &mut state.sr_opt.sr_opt_o[opt_offset..opt_end],
                ip,
                data,
                &state.slater_matrix.pf_m,
            );
            timer.stop_diag(949, diag);
            timer.stop_diag(940, diag);
        }
        timer.start(43);
        if all_complex && use_store {
            crate::observables::calculate_oo_store(
                &mut state.sr_opt.sr_opt_ho,
                &mut state.sr_opt.sr_opt_o_store,
                &state.sr_opt.sr_opt_o,
                w,
                e,
                sample,
                sr_opt_size,
            );
        } else if all_complex {
            crate::observables::calculate_oo(
                &mut state.sr_opt.sr_opt_oo,
                &mut state.sr_opt.sr_opt_ho,
                &state.sr_opt.sr_opt_o,
                w,
                e,
                sr_opt_size,
            );
        } else {
            for i in 0..sr_opt_size {
                state.sr_opt.sr_opt_o_real[i] = state.sr_opt.sr_opt_o[2 * i].re;
            }
            if use_store {
                crate::observables::calculate_oo_store_real(
                    &mut state.sr_opt.sr_opt_ho_real,
                    &mut state.sr_opt.sr_opt_o_store_real,
                    &state.sr_opt.sr_opt_o_real,
                    w,
                    e.re,
                    sample,
                    sr_opt_size,
                );
            } else {
                crate::observables::calculate_oo_real(
                    &mut state.sr_opt.sr_opt_oo_real,
                    &mut state.sr_opt.sr_opt_ho_real,
                    &state.sr_opt.sr_opt_o_real,
                    w,
                    e.re,
                    sr_opt_size,
                );
            }
        }
        timer.stop(43);
    }
    if let Some(phys) = state.phys_quantities.as_mut() {
        let count = state.energy.wc.re;
        if count != 0.0 {
            let denominator = Complex64::new(count, 0.0);
            for value in &mut phys.phys_lanczos_qqqq {
                *value /= denominator;
            }
            for value in &mut phys.phys_lanczos_qcisajsq {
                *value /= denominator;
            }
            for value in &mut phys.phys_lanczos_qcisajscktaltq {
                *value /= denominator;
            }
            for value in &mut phys.phys_lanczos_qcisajscktaltq_dc {
                *value /= denominator;
            }
            for value in &mut phys.phys_cis_ajs {
                *value /= denominator;
            }
            for value in &mut phys.phys_cis_ajs_ckt_alt {
                *value /= denominator;
            }
            for value in &mut phys.phys_cis_ajs_ckt_alt_dc {
                *value /= denominator;
            }
        }
    }
    if use_store {
        timer.start(45);
        let options = crate::observables::StoreFinalization {
            sample_start: 0,
            diagonal_only: data.modpara.nsrcg != 0,
        };
        if all_complex {
            crate::observables::finalize_oo_store(
                &mut state.sr_opt.sr_opt_oo,
                &state.sr_opt.sr_opt_o_store,
                sr_opt_size,
                n_vmc_sample,
                options,
            );
        } else {
            crate::observables::finalize_oo_store_real(
                &mut state.sr_opt.sr_opt_oo_real,
                &state.sr_opt.sr_opt_o_store_real,
                sr_opt_size,
                n_vmc_sample,
                options,
            );
        }
        timer.stop(45);
    }
    // Julia merges the local SR accumulator into cleared global arrays.
    // Keep the addition: it turns negative zero into positive zero.
    for value in state
        .sr_opt
        .sr_opt_oo
        .iter_mut()
        .chain(&mut state.sr_opt.sr_opt_ho)
        .chain(&mut state.sr_opt.sr_opt_o_store)
    {
        *value = Complex64::new(0.0, 0.0) + *value;
    }
    for value in state
        .sr_opt
        .sr_opt_oo_real
        .iter_mut()
        .chain(&mut state.sr_opt.sr_opt_ho_real)
        .chain(&mut state.sr_opt.sr_opt_o_store_real)
    {
        *value += 0.0;
    }
}

/// Rebuild the saved-walker Pfaffian using the mode selected by C's
/// `AllComplexFlag`. Real-FSZ must refresh its real shadows before the shared
/// observable kernels consume them.
fn refresh_fsz_observation_matrix(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    all_complex: bool,
    ele_idx: &[i64],
    ele_spn: &[i64],
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), crate::pfaffian::CalcMAllError> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    if all_complex {
        crate::pfaffian::calc_m_all_fsz_complex(
            ele_idx,
            ele_spn,
            &state.slater_matrix.slater_elm,
            &mut state.slater_matrix.inv_m,
            &mut state.slater_matrix.pf_m,
            0,
            n_qp_full,
            n_site,
            n_elec,
            pool,
        )
    } else {
        crate::pfaffian::calc_m_all_fsz_real(
            ele_idx,
            ele_spn,
            &mut state.slater_matrix,
            0,
            n_qp_full,
            n_site,
            n_elec,
            pool,
        )?;
        sync_real_fsz_shadow(state);
        Ok(())
    }
}

#[cfg(test)]
mod callback_tests {
    use super::*;

    use super::reference_slater::{declared_output, declared_slater_rows};

    fn declared_history_bits(data: &ExpertModeData, historical: Vec<u64>) -> Vec<u64> {
        let prefix = 2 * (data.gutzwiller_terms.len() + data.jastrow_terms.len());
        let mapped: Vec<[u64; 2]> = historical[prefix..]
            .chunks_exact(2)
            .map(|row| [row[0], row[1]])
            .collect();
        historical[..prefix]
            .iter()
            .copied()
            .chain(declared_slater_rows(data, &mapped).into_iter().flatten())
            .collect()
    }

    fn prepared(steps: i64) -> (ExpertModeData, VmcOptimizationState, Sfmt19937Rng) {
        prepared_case(steps, "heisenberg_chain_real")
    }

    fn prepared_case(
        steps: i64,
        name: &str,
    ) -> (ExpertModeData, VmcOptimizationState, Sfmt19937Rng) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/examples/inputs")
            .join(name)
            .join("namelist.def");
        prepared_namelist(steps, &path)
    }

    fn prepared_namelist(
        steps: i64,
        path: &Path,
    ) -> (ExpertModeData, VmcOptimizationState, Sfmt19937Rng) {
        // These Julia mixed-DH models relied on global complex mode enabling
        // real-orbital imaginary flags. C uses only orbital headers; explicit
        // complex AP replacements preserve these historical SR/RNG workloads.
        // RBM legacy workloads also use explicit binary flags: C flag 2 is
        // fixed for SR, while Julia had converted it to true. These helpers
        // never change flags on a parsed production model.
        let input = path.to_string_lossy();
        let replacement = [
            ("dh2/production_cmp/namelist.def", "dh2_cmp"),
            ("dh4/production_dh4_cmp/namelist.def", "dh4_cmp"),
            ("dh4/production_dh24_cmp/namelist.def", "dh24_cmp"),
            ("rbm/run_rbm_real/namelist.def", "rbm_real"),
            ("rbm/run_rbm_cmp/namelist.def", "rbm_cmp"),
            ("rbm/run_rbm_general_cmp/namelist.def", "rbm_general_cmp"),
            ("rbm/run_rbm_dh24_cmp/namelist.def", "rbm_dh24_cmp"),
            ("rbm/run_rbm_fsz/namelist.def", "rbm_fsz"),
            (
                "opttrans/run_opt_dh24_rbm_cmp/namelist.def",
                "opt_dh24_rbm_cmp",
            ),
        ]
        .into_iter()
        .find(|(suffix, _)| input.ends_with(suffix))
        .map(|(_, name)| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../tests/fixtures/c_orbital_inputs/namelist_{name}.def"
            ))
        });
        let path = replacement.as_deref().unwrap_or(path);

        let mut data = parse_expert_mode_files(path).unwrap();
        data.modpara.nsr_opt_itr_step = steps;
        data.modpara.nsr_opt_itr_smp = steps;
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng);
        if !data.doublon_holon_2site_indices.is_empty()
            || !data.doublon_holon_4site_indices.is_empty()
            || data.has_rbm_terms()
        {
            read_input_parameters(&mut data, path).unwrap();
        }
        sync_modified_parameter(&mut data, true);
        init_qp_weight(&mut data);
        let state = state_from_data(&data);
        (data, state, rng)
    }

    #[test]
    fn callbacks_observe_post_sync_parameters_and_do_not_change_rng_trajectory() {
        let (mut baseline, mut base_state, mut base_rng) = prepared(3);
        let base_dir = fresh_output_directory().unwrap();
        vmc_para_opt(
            &mut baseline,
            &mut base_state,
            &mut base_rng,
            Some(&base_dir),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap();
        let (mut data, mut state, mut rng) = prepared(3);
        let dir = fresh_output_directory().unwrap();
        let mut records = Vec::new();
        let mut callback = |step, data: &mut ExpertModeData, energy, info| {
            records.push((step, data.slater_params.clone(), energy, info));
            Ok(())
        };
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: false,
            },
        )
        .unwrap();
        assert_eq!(
            records.iter().map(|record| record.0).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(records.iter().all(|record| record.3 == 0));
        assert_eq!(records[2].1, data.slater_params);
        assert_eq!(records[2].2, state.energy.etot);
        assert_eq!(data.orbital_terms, baseline.orbital_terms);
        assert_eq!(data.slater_params, baseline.slater_params);
        assert_eq!(
            state.electron_config.ele_idx,
            base_state.electron_config.ele_idx
        );
        assert_eq!(
            fs::read(dir.join("zvo_out.dat")).unwrap(),
            fs::read(base_dir.join("zvo_out.dat")).unwrap()
        );
        let mut hash = 0xcbf29ce484222325_u64;
        for _ in 0..624 {
            let word = rng.gen_rand32();
            assert_eq!(word, base_rng.gen_rand32());
            hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
        }
        assert_eq!(hash, 382483484918994011);
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(base_dir).unwrap();
    }

    #[test]
    fn sampling_only_stops_after_one_output_and_before_sr_or_final_parameters() {
        let (mut data, mut state, mut rng) = prepared(3);
        data.modpara.dsr_opt_step_dt = f64::NAN; // Prove the solver is skipped.
        let before = data.slater_params.clone();
        let dir = fresh_output_directory().unwrap();
        let mut calls = Vec::new();
        let mut callback = |step, _: &mut ExpertModeData, energy, info| {
            calls.push((step, energy, info));
            Ok(())
        };
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: true,
            },
        )
        .unwrap();
        assert_eq!(calls, [(0, state.energy.etot, 0)]);
        assert_eq!(before, data.slater_params);
        assert_eq!(
            fs::read_to_string(dir.join("zvo_out.dat"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        assert!(!dir.join("zqp_opt.dat").exists());
        let (mut control, mut control_state, mut control_rng) = prepared(1);
        control.modpara.dsr_opt_step_dt = f64::NAN;
        let control_dir = fresh_output_directory().unwrap();
        assert!(vmc_para_opt(
            &mut control,
            &mut control_state,
            &mut control_rng,
            Some(&control_dir),
            &SingleProcessReducer,
            OptimizationOptions::default()
        )
        .is_err());
        let mut hash = 0xcbf29ce484222325_u64;
        for _ in 0..624 {
            let word = rng.gen_rand32();
            assert_eq!(word, control_rng.gen_rand32());
            hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
        }
        assert_eq!(hash, 13510181319970448127);
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(control_dir).unwrap();
    }

    #[test]
    fn callback_errors_propagate_and_failed_sr_does_not_call_callback() {
        let (mut data, mut state, mut rng) = prepared(3);
        let dir = fresh_output_directory().unwrap();
        let mut calls = 0;
        let mut callback = |_, _: &mut ExpertModeData, _, _| {
            calls += 1;
            Err("callback failed".into())
        };
        let error = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: false,
            },
        )
        .unwrap_err();
        assert_eq!(error, "callback failed");
        assert_eq!(calls, 1);
        assert!(!dir.join("zqp_opt.dat").exists());
        let (mut data, mut state, mut rng) = prepared(3);
        data.modpara.dsr_opt_step_dt = f64::NAN;
        let mut callback = |_, _: &mut ExpertModeData, _, _| {
            calls += 1;
            Ok(())
        };
        assert!(vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: false
            }
        )
        .is_err());
        assert_eq!(calls, 1);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn final_window_history_is_stored_before_callback_and_uses_initial_window() {
        for window in [2, 5] {
            let (mut data, mut state, mut rng) = prepared(3);
            data.modpara.nsr_opt_itr_smp = window;
            let dir = fresh_output_directory().unwrap();
            let mut records = Vec::new();
            let mut callback = |step, data: &mut ExpertModeData, energy, _| {
                records.push((
                    step,
                    data.gutzwiller_terms
                        .iter()
                        .map(|t| t.value)
                        .chain(data.jastrow_terms.iter().map(|t| t.value))
                        .chain(data.slater_params.iter().copied())
                        .collect::<Vec<_>>(),
                    energy,
                ));
                data.modpara.nsr_opt_itr_smp = 100; // Julia captures n_smp before the loop.
                data.slater_params[data.orbital_terms[0].idx as usize].re += 0.001;
                Ok(())
            };
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    callback: Some(&mut callback),
                    skip_sr: false,
                },
            )
            .unwrap();
            assert_eq!(state.opt_data.len(), window as usize);
            for (index, point) in state.opt_data.iter().enumerate() {
                let step = index as i64 + 3 - window;
                if step < 0 {
                    assert!(point.parameters.is_empty());
                } else {
                    assert_eq!(point.parameters, records[step as usize].1);
                    assert_eq!(point.energy, records[step as usize].2);
                }
            }
            assert_ne!(state.opt_data.last().unwrap().parameters.last(), None);
            let snapshot = state.opt_data.last().unwrap().parameters.clone();
            data.slater_params[data.orbital_terms[0].idx as usize].re = 999.0;
            assert_eq!(state.opt_data.last().unwrap().parameters, snapshot);
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn enabled_sections_preserve_parameters_samples_energy_and_rng() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
            "hubbard_chain_real",
        ] {
            let (mut baseline, mut base_state, mut base_rng) = prepared_case(3, case);
            let base_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut baseline,
                &mut base_state,
                &mut base_rng,
                Some(&base_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let (mut data, mut state, mut rng) = prepared_case(3, case);
            let dir = fresh_output_directory().unwrap();
            let mut timer = crate::c_timer::CTimer::<true>::new();
            timer.diagnostics = TimerEnv {
                calham1: true,
                slater: true,
                maincal: true,
                weightavg: true,
                ..TimerEnv::default()
            };
            vmc_para_opt_timed(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
                &mut timer,
            )
            .unwrap();
            for id in [
                2, 20, 3, 4, 21, 24, 25, 22, 5, 23, 30, 31, 35, 40, 41, 42, 43, 50, 51, 52, 56, 57,
                70, 71, 72, 960, 962, 965, 966,
            ] {
                assert!(timer.elapsed_ns[id] > 0, "{case}: missing timer {id}");
            }
            if case == "hubbard_chain_real" {
                for id in [32, 60, 61, 62, 63, 920, 921, 922, 924, 927] {
                    assert!(timer.elapsed_ns[id] > 0, "{case}: missing timer {id}");
                }
                assert_eq!(timer.elapsed_ns[33], 0);
            } else {
                for id in [33, 65, 66, 67, 68] {
                    assert!(timer.elapsed_ns[id] > 0, "{case}: missing timer {id}");
                }
            }
            if case != "heisenberg_chain_fsz" {
                for id in [
                    930, 931, 932, 933, 934, 940, 941, 942, 943, 944, 945, 946, 948,
                ] {
                    assert!(timer.elapsed_ns[id] > 0, "{case}: missing diag {id}");
                }
            }
            if case == "heisenberg_chain_fsz" {
                for id in [
                    930, 931, 932, 933, 934, 940, 941, 942, 943, 944, 945, 946, 948,
                ] {
                    assert_eq!(timer.elapsed_ns[id], 0, "unexpected FSZ diagnostic {id}");
                }
            }
            assert!(
                timer.elapsed_ns[2]
                    >= [20, 3, 4, 21, 22, 5, 23]
                        .iter()
                        .map(|&id| timer.elapsed_ns[id])
                        .sum::<u64>()
            );
            assert!(
                timer.elapsed_ns[3]
                    >= [30, 31, 32, 33, 34, 35, 36]
                        .iter()
                        .map(|&id| timer.elapsed_ns[id])
                        .sum::<u64>()
            );
            assert!(
                timer.elapsed_ns[960]
                    >= [962, 965, 966]
                        .iter()
                        .map(|&id| timer.elapsed_ns[id])
                        .sum::<u64>()
            );
            assert_eq!(timer.elapsed_ns[12], 0);
            assert_eq!(timer.elapsed_ns[55], 0);
            assert_eq!(baseline.orbital_terms, data.orbital_terms);
            assert_eq!(baseline.slater_params, data.slater_params);
            assert_eq!(base_state.electron_config, state.electron_config);
            assert_eq!(base_state.energy, state.energy);
            assert_eq!(base_state.opt_data, state.opt_data);
            assert_eq!(
                fs::read(base_dir.join("zvo_out.dat")).unwrap(),
                fs::read(dir.join("zvo_out.dat")).unwrap()
            );
            for _ in 0..624 {
                assert_eq!(base_rng.gen_rand32(), rng.gen_rand32());
            }
            fs::remove_dir_all(base_dir).unwrap();
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn real_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("real", true, 0);
    }

    #[test]
    fn complex_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("cmp", true, 0);
    }

    #[test]
    fn fsz_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("fsz", true, 0);
    }

    #[test]
    fn general_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("general", true, 0);
    }

    #[test]
    fn interall_fsz_cg_prefixes_match_julia_parameters_spins_samples_energy_and_rng() {
        check_sr_prefixes("interall", true, 0);
    }

    #[test]
    fn dh2_loaded_values_and_rng_match_julia_with_history_in_c_declared_order() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
        let bits = |text: &str| -> Vec<u64> {
            text.split_whitespace()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect()
        };
        let serialize = |values: Vec<Complex64>| -> Vec<u64> {
            values
                .iter()
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect()
        };
        for mode in ["real", "cmp", "fsz"] {
            let (mut data, mut state, mut rng) =
                prepared_namelist(3, &root.join(format!("production_{mode}/namelist.def")));
            let input = fs::read_to_string(root.join(format!("loaded-{mode}.txt"))).unwrap();
            let mut lines = input.lines().skip(1);
            assert_eq!(
                data.optimization_flags,
                crate::historical_optimization_flags::c_orbital_representation(
                    &data,
                    lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|s| s.parse::<i64>().unwrap())
                        .collect()
                )
            );
            let values = data
                .projection_parameters()
                .into_iter()
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|t| data.slater_params[t.idx as usize]),
                )
                .collect();
            assert_eq!(
                serialize(values),
                bits(lines.next().unwrap()),
                "{mode} loaded values"
            );
            let mut probe = rng.clone();
            assert_eq!(
                (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect::<Vec<_>>(),
                "{mode} loaded RNG"
            );
            assert!(lines.next().is_none());
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let text = fs::read_to_string(root.join(format!("history-{mode}.txt"))).unwrap();
            let mut lines = text.lines().skip(1);
            assert_eq!(state.opt_data.len(), 3);
            for point in &state.opt_data {
                assert_eq!(
                    serialize(vec![point.energy]),
                    bits(lines.next().unwrap()),
                    "{mode} history energy"
                );
                assert_eq!(
                    serialize(point.parameters.clone()),
                    declared_history_bits(&data, bits(lines.next().unwrap())),
                    "{mode} history values"
                );
            }
            assert!(lines.next().is_none());
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn dh4_loaded_values_and_rng_match_julia_with_history_in_c_declared_order() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
        let bits = |text: &str| -> Vec<u64> {
            text.split_whitespace()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect()
        };
        let serialize = |values: Vec<Complex64>| -> Vec<u64> {
            values
                .iter()
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect()
        };
        for mode in [
            "dh4_real",
            "dh4_cmp",
            "dh4_fsz",
            "dh24_real",
            "dh24_cmp",
            "dh24_fsz",
        ] {
            let (mut data, mut state, mut rng) =
                prepared_namelist(3, &root.join(format!("production_{mode}/namelist.def")));
            let input = fs::read_to_string(root.join(format!("loaded-{mode}.txt"))).unwrap();
            let mut lines = input.lines().skip(1);
            assert_eq!(
                data.optimization_flags,
                crate::historical_optimization_flags::c_orbital_representation(
                    &data,
                    lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|s| s.parse::<i64>().unwrap())
                        .collect()
                )
            );
            let values = data
                .projection_parameters()
                .into_iter()
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|t| data.slater_params[t.idx as usize]),
                )
                .collect();
            assert_eq!(
                serialize(values),
                bits(lines.next().unwrap()),
                "{mode} loaded values"
            );
            let mut probe = rng.clone();
            assert_eq!(
                (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect::<Vec<_>>(),
                "{mode} loaded RNG"
            );
            assert!(lines.next().is_none());
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let text = fs::read_to_string(root.join(format!("history-{mode}.txt"))).unwrap();
            let mut lines = text.lines().skip(1);
            assert_eq!(state.opt_data.len(), 3);
            for point in &state.opt_data {
                assert_eq!(
                    serialize(vec![point.energy]),
                    bits(lines.next().unwrap()),
                    "{mode} history energy"
                );
                assert_eq!(
                    serialize(point.parameters.clone()),
                    declared_history_bits(&data, bits(lines.next().unwrap())),
                    "{mode} history values"
                );
            }
            assert!(lines.next().is_none());
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn dh4_and_dh24_real_complex_and_fsz_cg_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in [
            "dh4_real",
            "dh4_cmp",
            "dh4_fsz",
            "dh24_real",
            "dh24_cmp",
            "dh24_fsz",
        ] {
            check_sr_prefixes(case, true, 0);
        }
    }
    #[test]
    fn dh4_real_complex_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh4_real", "dh4_cmp", "dh4_fsz"] {
            for store in 0..=1 {
                check_sr_prefixes(case, false, store);
            }
        }
    }
    #[test]
    fn dh24_real_complex_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh24_real", "dh24_cmp", "dh24_fsz"] {
            for store in 0..=1 {
                check_sr_prefixes(case, false, store);
            }
        }
    }

    #[test]
    fn dh2_real_complex_and_fsz_cg_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh2_real", "dh2_cmp", "dh2_fsz"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn dh2_real_complex_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh2_real", "dh2_cmp", "dh2_fsz"] {
            for store in 0..=1 {
                check_sr_prefixes(case, false, store);
            }
        }
    }

    #[test]
    fn pairhop_real_and_fsz_cg_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["pairhop_real", "pairhop_fsz"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn pairhop_real_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["pairhop_real", "pairhop_fsz"] {
            for store in [0, 1] {
                check_sr_prefixes(case, false, store);
            }
        }
    }

    #[test]
    fn pairhop_real_and_fsz_initial_flags_parameters_and_rng_match_julia() {
        for case in ["pairhop_real", "pairhop_fsz"] {
            let input = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../extern/Julia-mVMC/test/integration/reference/hubbard_chain_{case}/inputs/namelist.def"));
            let (data, _, mut rng) = prepared_namelist(1, &input);
            assert_eq!(data.pair_hop_terms.len(), 2);
            assert_eq!(get_all_complex_flag(&data), case == "pairhop_fsz");
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../tests/fixtures/sr_cg/{case}_runner"));
            check_initial_boundary(&data, &mut rng, &root);
        }
    }

    #[test]
    fn interall_fsz_direct_prefixes_match_julia_parameters_spins_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("interall", false, store);
        }
    }

    #[test]
    fn interall_fsz_initial_flags_parameters_and_rng_match_julia_before_sampling() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/c_orbital_inputs/namelist_interall_fsz.def");
        let (data, _, mut rng) = prepared_namelist(1, &input);
        assert_eq!(data.inter_all_terms.len(), 26);
        assert!(get_all_complex_flag(&data));
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sr_cg/interall_runner");
        check_initial_boundary(&data, &mut rng, &root);
    }

    fn check_initial_boundary(
        data: &ExpertModeData,
        rng: &mut sfmt19937::Sfmt19937Rng,
        root: &Path,
    ) {
        let flags: Vec<i64> = fs::read_to_string(root.join("initial-flags.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse::<i64>().unwrap())
            .collect();
        let flags = crate::historical_optimization_flags::c_orbital_representation(data, flags);
        assert_eq!(data.optimization_flags, flags);
        let bits: Vec<u64> = fs::read_to_string(root.join("initial-parameters.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| u64::from_str_radix(v, 16).unwrap())
            .collect();
        let mut mapped = data.clone();
        let mut rbm_values = Vec::new();
        mapped.visit_rbm_terms_mut(|_, t| rbm_values.push(t.value()));
        let values = data
            .gutzwiller_terms
            .iter()
            .map(|t| t.value)
            .chain(data.jastrow_terms.iter().map(|t| t.value))
            .chain(data.doublon_holon_2site_params.iter().copied())
            .chain(data.doublon_holon_4site_params.iter().copied())
            .chain(rbm_values)
            .chain(
                data.orbital_terms
                    .iter()
                    .map(|t| data.slater_params[t.idx as usize]),
            )
            .chain(data.opt_trans.iter().copied());
        let actual: Vec<_> = values
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect();
        assert_eq!(actual, bits);
        let words: Vec<u32> = fs::read_to_string(root.join("initial-rng.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(words.len(), 624);
        for expected in words {
            assert_eq!(rng.gen_rand32(), expected);
        }
    }

    #[test]
    fn general_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("general", false, store);
        }
    }

    #[test]
    fn hubbard_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("hubbard", true, 0);
    }

    #[test]
    fn real_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("real", false, store);
        }
    }

    #[test]
    fn cmp_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("cmp", false, store);
        }
    }

    #[test]
    fn fsz_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("fsz", false, store);
        }
    }

    #[test]
    fn hubbard_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("hubbard", false, store);
        }
    }

    #[test]
    fn rbm_complex_direct_prefixes_match_source_parameters_samples_energy_and_rng() {
        check_sr_prefixes("rbm_cmp", false, 0);
    }

    #[test]
    fn opttrans_initial_flags_parameters_and_rng_match_julia_before_sampling() {
        for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            let input = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../tests/fixtures/opttrans/run_{case}/namelist.def"
            ));
            let (data, _, mut rng) = prepared_namelist(1, &input);
            assert_eq!(data.opt_trans.len(), 3);
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../tests/fixtures/sr_cg/{case}_runner"));
            check_initial_boundary(&data, &mut rng, &root);
        }
    }

    #[test]
    fn opttrans_real_direct_prefixes_match_source_parameters_samples_energy_and_rng() {
        check_sr_prefixes("opt_real", false, 0);
    }

    #[test]
    fn opttrans_complex_fsz_and_all_factor_direct_prefixes_match_source() {
        for case in ["opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            check_sr_prefixes(case, false, 0);
        }
    }

    #[test]
    fn opttrans_stored_direct_prefixes_match_source() {
        for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            check_sr_prefixes(case, false, 1);
        }
    }

    #[test]
    fn opttrans_cg_prefixes_match_source() {
        for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn rbm_real_and_general_complex_direct_prefixes_match_source() {
        for case in ["rbm_real", "rbm_general_cmp"] {
            check_sr_prefixes(case, false, 0);
        }
    }
    #[test]
    fn rbm_real_and_complex_cg_prefixes_match_source() {
        for case in ["rbm_real", "rbm_cmp", "rbm_general_cmp"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn rbm_dh24_direct_and_cg_prefixes_match_source() {
        check_sr_prefixes("rbm_dh24_cmp", false, 0);
        check_sr_prefixes("rbm_dh24_cmp", true, 0);
    }

    #[test]
    fn rbm_fsz_direct_prefixes_match_source_including_failure_state() {
        check_sr_prefixes("rbm_fsz", false, 0);
    }

    #[test]
    fn rbm_fsz_cg_prefixes_match_source_including_empty_weight_steps() {
        check_sr_prefixes("rbm_fsz", true, 0);
    }

    #[test]
    fn rbm_stored_direct_prefixes_match_source() {
        for case in [
            "rbm_real",
            "rbm_cmp",
            "rbm_general_cmp",
            "rbm_dh24_cmp",
            "rbm_fsz",
        ] {
            check_sr_prefixes(case, false, 1);
        }
    }

    #[test]
    fn canonical_general_rbm_complex_reference_direct_and_cg_match_source() {
        check_sr_prefixes("rbm_reference_cmp", false, 1);
        check_sr_prefixes("rbm_reference_cmp", true, 0);
    }

    fn check_sr_prefixes(case: &str, cg: bool, store: i64) {
        let reference_case = if case == "general" { "fsz" } else { case };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(if cg {
                "../../tests/fixtures/sr_cg"
            } else {
                "../../tests/fixtures/sr_direct"
            })
            .join(format!(
                "{reference_case}{}",
                if store == 0 {
                    "_runner"
                } else {
                    "_store_runner"
                }
            ));
        let prefixes = if !cg && case == "hubbard" {
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
        } else if !cg && store == 0 && case == "opt_real" {
            // Cover the last successful update, changed acceptance decisions,
            // and the following native sampler/SR failure separately.
            vec![1, 2, 3, 27, 28, 29, 50]
        } else {
            vec![1, 2, 3, 50]
        };
        for steps in prefixes {
            let name = if case == "hubbard" {
                "hubbard_chain_real".into()
            } else {
                format!("heisenberg_chain_{case}")
            };
            let (mut data, mut state, mut rng) = if case == "rbm_reference_cmp" {
                let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def");
                let mut data = parse_expert_mode_files(&path).unwrap();
                data.modpara.nsr_opt_itr_step = steps;
                data.modpara.nsr_opt_itr_smp = steps;
                let mut rng = Sfmt19937Rng::new(12395);
                init_parameter(&mut data, &mut rng);
                assert!(
                    read_initial_def(&mut data, path.parent().unwrap().join("initial.def"))
                        .unwrap()
                );
                read_input_parameters(&mut data, &path).unwrap();
                sync_modified_parameter(&mut data, true);
                init_qp_weight(&mut data);
                let state = state_from_data(&data);
                (data, state, rng)
            } else if case.starts_with("opt_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "../../tests/fixtures/opttrans/run_{case}/namelist.def"
                    )),
                )
            } else if case.starts_with("rbm_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join(format!("../../tests/fixtures/rbm/run_{case}/namelist.def")),
                )
            } else if let Some(mode) = case.strip_prefix("dh2_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "../../tests/fixtures/dh2/production_{mode}/namelist.def"
                    )),
                )
            } else if case.starts_with("dh4_") || case.starts_with("dh24_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "../../tests/fixtures/dh4/production_{case}/namelist.def"
                    )),
                )
            } else if case.starts_with("pairhop_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../extern/Julia-mVMC/test/integration/reference/hubbard_chain_{case}/inputs/namelist.def")),
                )
            } else if case == "interall" {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tests/fixtures/c_orbital_inputs/namelist_interall_fsz.def"),
                )
            } else if case == "general" {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(
                        "../../tests/fixtures/c_orbital_inputs/namelist_heisenberg_general.def",
                    ),
                )
            } else {
                prepared_case(steps, &name)
            };
            data.modpara.nsrcg = i64::from(cg);
            data.modpara.nstore_o = store;
            let dir = fresh_output_directory().unwrap();
            let result = vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            );
            let failed = if case == "rbm_fsz" || case.starts_with("opt_") {
                let status =
                    fs::read_to_string(root.join(format!("step-{steps}-status.txt"))).unwrap();
                let mut status = status.split_whitespace();
                let info: i32 = status.next().unwrap().parse().unwrap();
                let step: i32 = status.next().unwrap().parse().unwrap();
                if info != 0 {
                    assert_eq!(result.unwrap_err(), format!("vmc_para_opt: direct SR failed at step {step} (status {info}); parameters were not updated"));
                } else {
                    result.unwrap();
                }
                info != 0
            } else {
                result.unwrap();
                false
            };
            let read = |kind: &str| {
                fs::read_to_string(root.join(format!("step-{steps}-{kind}.txt"))).unwrap()
            };
            if case.starts_with("rbm_") || case.starts_with("opt_") {
                let mut probe = rng.clone();
                let expected: Vec<u32> = read("rng")
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(
                    (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
                    expected,
                    "{case} step {steps} RNG before numerical checks"
                );
                let configurations = read("configs");
                for ((name, actual), line) in [
                    ("indices", &state.electron_config.ele_idx),
                    ("configuration", &state.electron_config.ele_cfg),
                    ("occupation", &state.electron_config.ele_num),
                ]
                .into_iter()
                .zip(configurations.lines())
                {
                    let expected: Vec<i64> = line
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(
                        actual, &expected,
                        "{case} step {steps} {name} before numerical checks"
                    );
                }
            }
            let bits = |text: &str| -> Vec<u64> {
                text.split_whitespace()
                    .map(|v| u64::from_str_radix(v, 16).unwrap())
                    .collect()
            };
            let mut rbm_values = Vec::new();
            data.visit_rbm_terms_mut(|_, t| rbm_values.push(t.value()));
            let values = data
                .gutzwiller_terms
                .iter()
                .map(|t| t.value)
                .chain(data.jastrow_terms.iter().map(|t| t.value))
                .chain(data.doublon_holon_2site_params.iter().copied())
                .chain(data.doublon_holon_4site_params.iter().copied())
                .chain(rbm_values)
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|t| data.slater_params[t.idx as usize]),
                )
                .chain(data.opt_trans.iter().copied());
            let actual: Vec<u64> = values
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect();
            assert_eq!(actual, bits(&read("parameters")), "step {steps} parameters");
            assert_eq!(
                [
                    state.energy.etot.re.to_bits(),
                    state.energy.etot.im.to_bits()
                ]
                .as_slice(),
                bits(&read("energy")),
                "step {steps} energy"
            );
            if (case.starts_with("rbm_") || case.starts_with("opt_")) && steps == 1 && !cg {
                let fixture = fs::read_to_string(root.join("fixed-input.txt")).unwrap();
                let lines: Vec<&str> = fixture.lines().filter(|l| !l.starts_with('#')).collect();
                let complex_bits = |values: &[Complex64]| -> Vec<u64> {
                    values
                        .iter()
                        .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                        .collect()
                };
                let (oo, ho) = if get_all_complex_flag(&data) {
                    (
                        complex_bits(&state.sr_opt.sr_opt_oo),
                        complex_bits(&state.sr_opt.sr_opt_ho),
                    )
                } else {
                    (
                        state
                            .sr_opt
                            .sr_opt_oo_real
                            .iter()
                            .map(|v| v.to_bits())
                            .collect(),
                        state
                            .sr_opt
                            .sr_opt_ho_real
                            .iter()
                            .map(|v| v.to_bits())
                            .collect(),
                    )
                };
                assert_eq!(oo, bits(lines[2]), "{case} sampled SR OO");
                assert_eq!(ho, bits(lines[3]), "{case} sampled SR HO");
                if store == 1 {
                    let fixture = fs::read_to_string(root.join("gram.txt")).unwrap();
                    let expected = bits(fixture.lines().nth(1).unwrap());
                    let actual = if get_all_complex_flag(&data) {
                        complex_bits(&state.sr_opt.sr_opt_o_store)
                    } else {
                        state
                            .sr_opt
                            .sr_opt_o_store_real
                            .iter()
                            .map(|v| v.to_bits())
                            .collect()
                    };
                    assert_eq!(actual.len(), expected.len());
                    for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(actual, expected, "{case} sampled SR O store component {i}");
                    }
                }
            }
            let conf = read("configs");
            let mut lines = conf.lines();
            for (name, actual) in [
                ("indices", &state.electron_config.ele_idx),
                ("configuration", &state.electron_config.ele_cfg),
                ("occupancy", &state.electron_config.ele_num),
                ("projection", &state.electron_config.ele_proj_cnt),
            ] {
                let expected: Vec<i64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(actual, &expected, "step {steps} {name}");
            }
            if matches!(
                case,
                "interall"
                    | "pairhop_fsz"
                    | "dh2_fsz"
                    | "dh4_fsz"
                    | "dh24_fsz"
                    | "rbm_fsz"
                    | "opt_fsz"
            ) {
                for (name, actual) in [
                    ("spins", &state.electron_config.ele_spn),
                    ("burn", &state.electron_config.burn_ele_idx),
                    ("counters", &state.electron_config.counter.to_vec()),
                ] {
                    let expected: Vec<i64> = lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(actual, &expected, "step {steps} {name}");
                }
            }
            if (case.starts_with("rbm_") || case.starts_with("opt_"))
                && case != "rbm_fsz"
                && case != "opt_fsz"
            {
                for (name, actual) in [
                    ("burn", &state.electron_config.burn_ele_idx),
                    ("counters", &state.electron_config.counter.to_vec()),
                ] {
                    let expected: Vec<i64> = lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(actual, &expected, "{case} step {steps} {name}");
                }
            }
            assert!(lines.next().is_none());
            let expected: Vec<u32> = read("rng")
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let actual: Vec<u32> = (0..624).map(|_| rng.gen_rand32()).collect();
            assert_eq!(actual, expected, "step {steps} RNG block");
            if cg {
                assert_eq!(
                    fs::read_to_string(dir.join("zvo_SRinfo.dat")).unwrap(),
                    read("SRinfo")
                );
            }
            if case.starts_with("dh2_")
                || case.starts_with("dh4_")
                || case.starts_with("dh24_")
                || case.starts_with("rbm_")
                || case.starts_with("opt_")
            {
                for name in [
                    "zvo_out.dat",
                    "zvo_var.dat",
                    "zqp_opt.dat",
                    "zqp_gutzwiller_opt.dat",
                    "zqp_jastrow_opt.dat",
                    "zqp_orbital_opt.dat",
                ] {
                    if failed && name.starts_with("zqp_") {
                        assert!(!dir.join(name).exists(), "{case} {steps} {name}");
                        assert_eq!(
                            fs::read_to_string(root.join(format!("step-{steps}-{name}"))).unwrap(),
                            "# absent after source SR failure\n"
                        );
                    } else {
                        assert_eq!(
                            fs::read_to_string(dir.join(name)).unwrap(),
                            declared_output(
                                &data,
                                name,
                                fs::read_to_string(root.join(format!("step-{steps}-{name}")))
                                    .unwrap()
                            ),
                            "{case} {steps} {name}"
                        );
                    }
                }
                assert!(!dir.join("zqp_dh2_opt.dat").exists());
                assert!(!dir.join("zqp_dh4_opt.dat").exists());
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn standard_cg_runs_three_steps_and_writes_srinfo() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, mut rng) = prepared_case(3, case);
            data.modpara.nsrcg = 1;
            data.modpara.nstore_o = 0;
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let info = fs::read_to_string(dir.join("zvo_SRinfo.dat")).unwrap();
            assert_eq!(info.lines().count(), 4, "{case}");
            assert_eq!(state.opt_data.len(), 3);
            assert!(data
                .slater_params
                .iter()
                .all(|t| t.re.is_finite() && t.im.is_finite()));
            let mut hash = 0xcbf29ce484222325_u64;
            for _ in 0..624 {
                hash = (hash ^ u64::from(rng.gen_rand32())).wrapping_mul(0x100000001b3);
            }
            let julia_hash = match case {
                "heisenberg_chain_real" => 382483484918994011,
                "heisenberg_chain_cmp" => 5883921295860317420,
                "heisenberg_chain_fsz" => 6705941385670463079,
                _ => unreachable!(),
            };
            assert_eq!(hash, julia_hash, "{case}: Julia SFMT block mismatch");
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn cg_accumulation_forces_store_even_when_nstore_is_zero() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, mut rng) = prepared_case(1, case);
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            // Exercise the CG accumulator with the already sampled walkers.
            data.modpara.nstore_o = 0;
            data.modpara.nsrcg = 1;
            clear_phys_quantity(&mut state);
            let complex = get_all_complex_flag(&data);
            let fsz = data.i_flg_orbital_general != 0;
            let mut timer = CTimer::<true>::new();
            accumulate_observables(&data, &mut state, complex, fsz, &mut timer);
            assert!(timer.elapsed_ns[45] > 0);
            let samples = data.modpara.nvmc_sample as usize;
            let n = state.sr_opt.sr_opt_size;
            if complex {
                assert_eq!(
                    state.sr_opt.sr_opt_oo[0],
                    Complex64::new(samples as f64, 0.0)
                );
                let n = 2 * n;
                for i in 0..n {
                    let mut mean = Complex64::new(0.0, 0.0);
                    let mut diagonal = 0.0;
                    for s in 0..samples {
                        let o = state.sr_opt.sr_opt_o_store[i + s * n];
                        mean += o;
                        diagonal += o.norm_sqr();
                    }
                    assert_eq!(state.sr_opt.sr_opt_oo[i], mean);
                    assert_eq!(state.sr_opt.sr_opt_oo[i + n], Complex64::new(diagonal, 0.0));
                }
                assert!(state.sr_opt.sr_opt_oo[2 * n..]
                    .iter()
                    .all(|&z| z == Complex64::new(0.0, 0.0)));
            } else {
                assert_eq!(state.sr_opt.sr_opt_oo_real[0], samples as f64);
                for i in 0..n {
                    let mut mean = 0.0;
                    let mut diagonal = 0.0;
                    for s in 0..samples {
                        let o = state.sr_opt.sr_opt_o_store_real[i + s * n];
                        mean += o;
                        diagonal += o * o;
                    }
                    assert_eq!(state.sr_opt.sr_opt_oo_real[i], mean);
                    assert_eq!(state.sr_opt.sr_opt_oo_real[i + n], diagonal);
                }
                assert!(state.sr_opt.sr_opt_oo_real[2 * n..]
                    .iter()
                    .all(|&v| v == 0.0));
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn nonfinite_local_energy_skips_energy_sr_and_sample_store() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, mut rng) = prepared_case(1, case);
            data.modpara.nstore_o = 1;
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            assert!(state.energy.wc.re > 0.0);
            data.coulomb_intra_terms = vec![mvmc_expert_parsers::CoulombIntraTerm {
                site: 0,
                value: f64::NAN,
            }];
            clear_phys_quantity(&mut state);
            let complex = get_all_complex_flag(&data);
            let fsz = data.i_flg_orbital_general != 0;
            accumulate_observables(&data, &mut state, complex, fsz, &mut CTimer::<false>::new());
            assert_eq!(state.energy.wc, Complex64::new(0.0, 0.0), "{case}");
            assert_eq!(state.energy.etot, Complex64::new(0.0, 0.0), "{case}");
            assert!(state
                .sr_opt
                .sr_opt_oo
                .iter()
                .all(|&z| z == Complex64::new(0.0, 0.0)));
            assert!(state
                .sr_opt
                .sr_opt_ho
                .iter()
                .all(|&z| z == Complex64::new(0.0, 0.0)));
            assert!(state
                .sr_opt
                .sr_opt_o_store
                .iter()
                .all(|&z| z == Complex64::new(0.0, 0.0)));
            assert!(state.sr_opt.sr_opt_oo_real.iter().all(|&v| v == 0.0));
            assert!(state.sr_opt.sr_opt_ho_real.iter().all(|&v| v == 0.0));
            assert!(state.sr_opt.sr_opt_o_store_real.iter().all(|&v| v == 0.0));
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn invalid_saved_walkers_clear_previous_sample_store() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, _) = prepared_case(1, case);
            data.modpara.nstore_o = 1;
            state.electron_config.ele_idx.fill(-1);
            state
                .sr_opt
                .sr_opt_o_store
                .fill(Complex64::new(123.0, -456.0));
            state.sr_opt.sr_opt_o_store_real.fill(123.0);
            clear_phys_quantity(&mut state);
            let complex = get_all_complex_flag(&data);
            let fsz = data.i_flg_orbital_general != 0;
            let mut timer = CTimer::<true>::new();
            accumulate_observables(&data, &mut state, complex, fsz, &mut timer);
            assert!(timer.elapsed_ns[45] > 0);
            assert!(state
                .sr_opt
                .sr_opt_o_store
                .iter()
                .all(|z| *z == Complex64::new(0.0, 0.0)));
            assert!(state.sr_opt.sr_opt_o_store_real.iter().all(|&v| v == 0.0));
            assert!(state
                .sr_opt
                .sr_opt_oo
                .iter()
                .all(|z| *z == Complex64::new(0.0, 0.0)));
            assert!(state.sr_opt.sr_opt_oo_real.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn nstore_preserves_three_step_optimization_trajectory() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
            "hubbard_chain_real",
        ] {
            let (mut direct, mut direct_state, mut direct_rng) = prepared_case(3, case);
            direct.modpara.nstore_o = 0;
            let direct_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut direct,
                &mut direct_state,
                &mut direct_rng,
                Some(&direct_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let (mut stored, mut stored_state, mut stored_rng) = prepared_case(3, case);
            stored.modpara.nstore_o = 1;
            let stored_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut stored,
                &mut stored_state,
                &mut stored_rng,
                Some(&stored_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            // Julia also produces different optimized parameters for NStore=0
            // and 1 (different OO reduction paths). Compare the discrete
            // trajectory exactly rather than asserting equality of those APIs.
            assert_eq!(
                direct_state.electron_config, stored_state.electron_config,
                "{case}"
            );
            let mut hash = 0xcbf29ce484222325_u64;
            for _ in 0..624 {
                let word = direct_rng.gen_rand32();
                assert_eq!(word, stored_rng.gen_rand32(), "{case}");
                hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
            }
            // Julia v0.5.0, seed=1, three SR steps, both NStore settings.
            // Regenerate via scripts/check_sample_store_parity.jl.
            let julia_hash = match case {
                "heisenberg_chain_real" => 382483484918994011,
                "heisenberg_chain_cmp" => 5883921295860317420,
                "heisenberg_chain_fsz" => 6705941385670463079,
                "hubbard_chain_real" => 5863593240845525434,
                _ => unreachable!(),
            };
            assert_eq!(hash, julia_hash, "{case}: Julia SFMT block mismatch");
            fs::remove_dir_all(direct_dir).unwrap();
            fs::remove_dir_all(stored_dir).unwrap();
        }
    }

    #[test]
    fn nstore_controls_production_buffers_without_changing_sampling() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
            "hubbard_chain_real",
        ] {
            let (mut direct, mut direct_state, mut direct_rng) = prepared_case(1, case);
            direct.modpara.nstore_o = 0;
            let direct_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut direct,
                &mut direct_state,
                &mut direct_rng,
                Some(&direct_dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            let (mut stored, mut stored_state, mut stored_rng) = prepared_case(1, case);
            stored.modpara.nstore_o = 1;
            // Only current valid samples may survive a new main-calculation call.
            stored_state
                .sr_opt
                .sr_opt_o_store
                .fill(Complex64::new(123.0, -456.0));
            stored_state.sr_opt.sr_opt_o_store_real.fill(123.0);
            let stored_dir = fresh_output_directory().unwrap();
            let mut timer = CTimer::<true>::new();
            vmc_para_opt_timed(
                &mut stored,
                &mut stored_state,
                &mut stored_rng,
                Some(&stored_dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
                &mut timer,
            )
            .unwrap();
            assert!(timer.elapsed_ns[45] > 0, "{case}: no Gram finalization");
            if get_all_complex_flag(&stored) {
                assert_eq!(
                    stored_state.sr_opt.sr_opt_o_store[0],
                    Complex64::new(1.0, 0.0)
                );
                assert!(direct_state
                    .sr_opt
                    .sr_opt_o_store
                    .iter()
                    .all(|value| value.norm() == 0.0));
            } else {
                assert_eq!(stored_state.sr_opt.sr_opt_o_store_real[0], 1.0);
                assert!(direct_state
                    .sr_opt
                    .sr_opt_o_store_real
                    .iter()
                    .all(|&value| value == 0.0));
            }
            for (left, right) in direct_state
                .sr_opt
                .sr_opt_oo
                .iter()
                .zip(&stored_state.sr_opt.sr_opt_oo)
            {
                // Julia's direct complex OO uses conj(O_i)*O_j; the store
                // finalizer uses O_i*conj(O_j), including the mean row.
                assert!(
                    (left.conj() - *right).norm() < 1e-12,
                    "{case}: complex OO mismatch"
                );
            }
            for (left, right) in direct_state
                .sr_opt
                .sr_opt_oo_real
                .iter()
                .zip(&stored_state.sr_opt.sr_opt_oo_real)
            {
                assert!((left - right).abs() < 1e-12, "{case}: real OO mismatch");
            }
            assert_eq!(direct_state.sr_opt.sr_opt_ho, stored_state.sr_opt.sr_opt_ho);
            assert_eq!(
                direct_state.sr_opt.sr_opt_ho_real,
                stored_state.sr_opt.sr_opt_ho_real
            );
            assert_eq!(direct_state.electron_config, stored_state.electron_config);
            assert_eq!(direct_state.energy, stored_state.energy);
            assert_eq!(
                fs::read(direct_dir.join("zvo_out.dat")).unwrap(),
                fs::read(stored_dir.join("zvo_out.dat")).unwrap()
            );
            for _ in 0..624 {
                assert_eq!(direct_rng.gen_rand32(), stored_rng.gen_rand32());
            }
            fs::remove_dir_all(direct_dir).unwrap();
            fs::remove_dir_all(stored_dir).unwrap();
        }
    }
}
