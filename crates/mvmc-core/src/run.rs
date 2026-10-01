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
use std::fs;
use std::path::Path;

use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::utils::parameter_init::{
    init_parameter, n_slater, sync_modified_parameter,
};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::average::{weight_average_sr_opt, weight_average_sr_opt_real, weight_average_we};
use crate::c_timer::{CTimer, TimerEnv};
use crate::counter::reduce_counter;
use crate::initial_params::read_initial_def;
use crate::io::{output_data, output_opt_data, store_opt_data};
use crate::observables::clear_phys_quantity;
use crate::reducer::{Reducer, SingleProcessReducer};
use crate::slater_update::{update_slater_elm, update_slater_elm_fsz};
use crate::state::VmcOptimizationState;
use crate::sync::sync_modified_parameter as sync_modified;

/// C-compatible parser default when `RndSeed` is omitted. Zero is a
/// valid seed, and negative ModPara seeds request the current Unix time.
pub const FALLBACK_SEED: i64 = 11272;

/// Per-step callback; errors propagate to the caller like Julia exceptions.
/// Arguments are zero-based step, post-sync parameters, measured energy and status.
pub type StepCallback<'a> =
    dyn FnMut(usize, &mut ExpertModeData, Complex64, i32) -> Result<(), String> + 'a;

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
    if reducer.world_size() != 1 {
        return Err("MPI execution is not implemented yet (issue #35)".into());
    }
    data.normalize_projection_count();
    let n_steps = data.modpara.nsr_opt_itr_step.max(0) as usize;
    let window_start = n_steps as i64 - data.modpara.nsr_opt_itr_smp;
    let n_proj = data.projection_layout().n_proj;
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    data.ensure_optimization_flags(n_para);
    let all_complex = get_all_complex_flag(data);
    let i_flg_general = data.i_flg_orbital_general;
    let use_fsz = i_flg_general != 0;
    if use_fsz && !all_complex {
        return Err("real FSZ is not implemented yet (issue #43)".into());
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
        timer.stop(20);
        timer.start(3);
        // 2. Sampler.
        let stats = if use_fsz {
            crate::sampling::driver::vmc_make_sample_fsz_timed(data, state, rng, timer)
        } else if !all_complex {
            crate::sampling::driver::vmc_make_sample_real_timed(data, state, rng, timer)
        } else {
            crate::sampling::driver::vmc_make_sample_timed(data, state, rng, timer)
        };
        timer.stop(3);
        if stats.saved == 0 {
            return Err(format!("vmc_para_opt: no samples saved at step {step}"));
        }

        // 3. Main accumulator.
        timer.start(4);
        timer.start(24);
        clear_phys_quantity(state);
        timer.stop(24);
        accumulate_observables(data, state, all_complex, use_fsz, timer);

        timer.stop(4);
        timer.start(21);
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
        output_data(data, state, step, output_dir).map_err(|e| e.to_string())?;
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

    output_opt_data(data, output_dir).map_err(|e| e.to_string())?;
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
    let mut data = parse_expert_mode_files(&namelist_path).map_err(|e| e.to_string())?;
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
    sync_modified_parameter(&mut data);
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
    data.orbital_terms
        .iter()
        .any(|term| term.is_complex || term.value.im != 0.0)
        || data
            .gutzwiller_terms
            .iter()
            .any(|term| term.is_complex || term.value.im != 0.0)
        || data
            .jastrow_terms
            .iter()
            .any(|term| term.is_complex || term.value.im != 0.0)
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
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    let n_sp = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp = data.modpara.nmp_trans.unsigned_abs().max(1) as usize;
    let n_opt = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_full = n_sp * n_mp * n_opt;
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let all_complex = get_all_complex_flag(data);
    VmcOptimizationState::zeros(
        n_site,
        n_elec,
        n_proj,
        n_para,
        n_qp_full,
        n_vmc_sample,
        all_complex,
        data.i_flg_orbital_general != 0,
    )
}

#[cfg(test)]
mod mode_tests {
    use super::*;
    use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};

    fn data() -> ExpertModeData {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.modpara.nmp_trans = 1;
        data.modpara.nvmc_sample = 1;
        data.modpara.n_orbital_idx = 1;
        data.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: 0,
            value: Complex64::new(1.0, 0.0),
            is_complex: false,
            sign: 1,
        });
        data
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
    fn imaginary_loaded_parameter_selects_complex_execution() {
        let mut data = data();
        data.orbital_terms[0].value.im = 0.5;
        assert!(state_from_data(&data).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn all_real_parameters_allocate_real_state() {
        assert!(!state_from_data(&data()).sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn zero_translation_count_allocates_one_projection_sector() {
        let mut data = data();
        data.modpara.nmp_trans = 0;
        data.modpara.nsp_gauss_leg = 3;
        let state = state_from_data(&data);
        assert_eq!(state.slater_matrix.slater_elm.n_qp_full(), 3);
        assert_eq!(state.slater_matrix.slater_elm_real.n_qp_full(), 3);
    }

    #[test]
    fn zero_and_one_translation_runs_have_identical_chain_and_rng_state() {
        let namelist = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def");
        let template = parse_expert_mode_files(namelist).unwrap();
        let run = |nmp| {
            let mut data = template.clone();
            data.modpara.nmp_trans = nmp;
            data.modpara.nsr_opt_itr_step = 1;
            let mut rng = Sfmt19937Rng::new(1);
            init_parameter(&mut data, &mut rng);
            sync_modified_parameter(&mut data);
            init_qp_weight(&mut data);
            let mut state = state_from_data(&data);
            let output = std::env::temp_dir()
                .join(format!("mvmc-zero-trajectory-{}-{nmp}", std::process::id()));
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
            (data, state, words)
        };
        let (zero_data, zero_state, zero_words) = run(0);
        let (one_data, one_state, one_words) = run(1);
        assert_eq!(zero_data.modpara.nmp_trans, 1);
        assert_eq!(zero_data.orbital_terms, one_data.orbital_terms);
        assert_eq!(
            zero_state.electron_config.ele_idx,
            one_state.electron_config.ele_idx
        );
        assert_eq!(zero_state.energy.etot, one_state.energy.etot);
        assert_eq!(zero_words, one_words);
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
            sync_modified_parameter(&mut data);
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
            let values: Vec<_> = data.orbital_terms.iter().map(|term| term.value).collect();
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
        data.orbital_terms[0].value.im = 0.5;
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
    fn imaginary_projection_values_select_complex_execution() {
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
            // Live Julia v0.5.0: sr_size=3, complex OO=48, HO=6,
            // O=6, sample store=6, all four real buffers empty.
            assert_eq!(state.sr_opt.sr_opt_oo.len(), 48);
            assert_eq!(state.sr_opt.sr_opt_ho.len(), 6);
            assert_eq!(state.sr_opt.sr_opt_o.len(), 6);
            assert_eq!(state.sr_opt.sr_opt_o_store.len(), 6);
            assert!(state.sr_opt.sr_opt_oo_real.is_empty());
            assert!(state.sr_opt.sr_opt_ho_real.is_empty());
            assert!(state.sr_opt.sr_opt_o_real.is_empty());
            assert!(state.sr_opt.sr_opt_o_store_real.is_empty());
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
    let n_orb_total = sr_opt_size.saturating_sub(1 + n_proj);
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
            crate::pfaffian::calc_m_all_fsz_complex(
                &ele_idx,
                &ele_spn,
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
        timer.start_diag(940, diag);
        timer.start_diag(946, diag);
        let sz = crate::observables::calculate_sz(&ele_num, n_site);

        state.energy.wc += Complex64::new(w, 0.0);
        state.energy.etot += Complex64::new(w, 0.0) * e;
        state.energy.etot2 += Complex64::new(w, 0.0) * e.conj() * e;
        state.energy.sztot += Complex64::new(w * sz, 0.0);
        state.energy.sztot2 += Complex64::new(w * sz * sz, 0.0);

        timer.stop_diag(946, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(948, diag);
        // SR `O` vector — projection diff fills the leading block.
        for slot in state.sr_opt.sr_opt_o.iter_mut() {
            *slot = Complex64::new(0.0, 0.0);
        }
        crate::observables::set_projection_diff(&mut state.sr_opt.sr_opt_o, &ele_proj_cnt, n_proj);
        let slater_offset = 2 * (1 + n_proj);
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
}

#[cfg(test)]
mod callback_tests {
    use super::*;

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
        let mut data = parse_expert_mode_files(path).unwrap();
        data.modpara.nsr_opt_itr_step = steps;
        data.modpara.nsr_opt_itr_smp = steps;
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng);
        sync_modified_parameter(&mut data);
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
            records.push((step, data.orbital_terms.clone(), energy, info));
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
        assert_eq!(records[2].1, data.orbital_terms);
        assert_eq!(records[2].2, state.energy.etot);
        assert_eq!(data.orbital_terms, baseline.orbital_terms);
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
        let before = data.orbital_terms.clone();
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
        assert_eq!(before, data.orbital_terms);
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
                        .chain(data.orbital_terms.iter().map(|t| t.value))
                        .collect::<Vec<_>>(),
                    energy,
                ));
                data.modpara.nsr_opt_itr_smp = 100; // Julia captures n_smp before the loop.
                data.orbital_terms[0].value.re += 0.001;
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
            data.orbital_terms[0].value.re = 999.0;
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

    fn check_sr_prefixes(case: &str, cg: bool, store: i64) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(if cg {
                "../../tests/fixtures/sr_cg"
            } else {
                "../../tests/fixtures/sr_direct"
            })
            .join(format!(
                "{case}{}",
                if store == 0 {
                    "_runner"
                } else {
                    "_store_runner"
                }
            ));
        let prefixes = if !cg && case == "hubbard" {
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
        } else {
            vec![1, 2, 3, 50]
        };
        for steps in prefixes {
            let name = if case == "hubbard" {
                "hubbard_chain_real".into()
            } else {
                format!("heisenberg_chain_{case}")
            };
            let (mut data, mut state, mut rng) = prepared_case(steps, &name);
            data.modpara.nsrcg = i64::from(cg);
            data.modpara.nstore_o = store;
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
            let read = |kind: &str| {
                fs::read_to_string(root.join(format!("step-{steps}-{kind}.txt"))).unwrap()
            };
            let bits = |text: &str| -> Vec<u64> {
                text.split_whitespace()
                    .map(|v| u64::from_str_radix(v, 16).unwrap())
                    .collect()
            };
            let values = data
                .gutzwiller_terms
                .iter()
                .map(|t| t.value)
                .chain(data.jastrow_terms.iter().map(|t| t.value))
                .chain(data.orbital_terms.iter().map(|t| t.value));
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
                .orbital_terms
                .iter()
                .all(|t| t.value.re.is_finite() && t.value.im.is_finite()));
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
