//! Focused PhysCal callback contract tests.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::{ExpertModeData, Reducer};
use num_complex::Complex64;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
    )
}

fn preparation(samples: i64) -> mvmc_core::PhysCalPreparation {
    let root = fixture();
    assert!(root.is_dir(), "Julia-mVMC PhysCal fixture is required");
    let namelist = root.join("inputs/namelist.def");
    let parsed = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap();
    let opt =
        std::env::temp_dir().join(format!("mvmc-physcal-callback-{}.dat", std::process::id()));
    let fields = 6 + 3 * parsed.count_variational_parameters();
    fs::write(&opt, (0..fields).map(|_| "0").collect::<Vec<_>>().join(" ")).unwrap();
    let mut prepared =
        mvmc_core::prepare_phys_cal_from_namelist(namelist, &opt, "real", Some(1)).unwrap();
    fs::remove_file(opt).unwrap();
    prepared.data.modpara.n_data_qty_smp = samples;
    prepared
}

fn output_snapshot(dir: &Path) -> Vec<(String, String)> {
    let mut files = fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                fs::read_to_string(entry.path()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.0.cmp(&right.0));
    files
}

#[test]
fn in_place_callback_error_retains_actual_completed_sample_rng_and_state() {
    // Both sides execute the same public production core, not a sampler replay.
    // This local boundary check complements the independent trajectory fixtures.
    let baseline = mvmc_core::vmc_phys_cal_with_reducer(
        preparation(1),
        None,
        &mvmc_core::SingleProcessReducer,
    )
    .unwrap();
    let mut prepared = preparation(3);
    let mut state = mvmc_core::VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    let mut calls = Vec::new();
    let mut callback = |sample, _: &ExpertModeData, _: Complex64, _| {
        calls.push(sample);
        Err("stop after completed sample".to_string())
    };
    let error = mvmc_core::vmc_phys_cal_in_place(
        &mut prepared.data,
        &mut state,
        &mut prepared.rng,
        None,
        &mvmc_core::SingleProcessReducer,
        Some(&mut callback),
    )
    .unwrap_err();
    assert_eq!(error, "stop after completed sample");
    assert_eq!(calls, [0]);
    assert_eq!(state.electron_config, baseline.state.electron_config);
    assert_eq!(prepared.data.slater_params, baseline.data.slater_params);
    let mut expected_rng = baseline.final_rng;
    for word in 0..624 {
        assert_eq!(
            prepared.rng.gen_rand32(),
            expected_rng.gen_rand32(),
            "draw after callback error at word {word}"
        );
    }
}

#[test]
fn callback_is_called_for_each_sample_after_average_without_changing_run() {
    let baseline_dir =
        std::env::temp_dir().join(format!("mvmc-physcal-callback-base-{}", std::process::id()));
    let callback_dir = std::env::temp_dir().join(format!(
        "mvmc-physcal-callback-observed-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&baseline_dir);
    let _ = fs::remove_dir_all(&callback_dir);

    let baseline = mvmc_core::vmc_phys_cal_to_dir(preparation(3), &baseline_dir).unwrap();
    let fixed_parameters = preparation(3).data.slater_params.clone();
    let mut observations = Vec::new();
    let mut callback = |sample, data: &ExpertModeData, energy: Complex64, status| {
        observations.push((sample, energy, status));
        assert_eq!(data.slater_params, fixed_parameters);
        Ok(())
    };
    let observed =
        mvmc_core::vmc_phys_cal_to_dir_with_callback(preparation(3), &callback_dir, &mut callback)
            .unwrap();

    assert_eq!(observations.len(), 3);
    assert_eq!(
        observations.iter().map(|row| row.0).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(observations.iter().all(|row| row.2 == 0));
    assert_eq!(baseline.iterations, observed.iterations);
    let mut baseline_rng = baseline.final_rng.clone();
    let mut observed_rng = observed.final_rng.clone();
    for word in 0..624 {
        assert_eq!(
            baseline_rng.gen_rand32(),
            observed_rng.gen_rand32(),
            "callback changed actual runner RNG word {word}"
        );
    }
    assert_eq!(baseline.data.slater_params, observed.data.slater_params);
    assert_eq!(
        baseline.state.electron_config,
        observed.state.electron_config
    );
    // Three measurement frames use the merged policy's accumulation budget.
    let energy_values = |energy: &mvmc_core::EnergyData| {
        [
            energy.wc,
            energy.etot,
            energy.etot2,
            energy.sztot,
            energy.sztot2,
        ]
        .into_iter()
        .flat_map(|value| [value.re, value.im])
        .collect::<Vec<_>>()
    };
    numerical_comparison::assert_values_close(
        energy_values(&observed.state.energy),
        energy_values(&baseline.state.energy),
        1e-12,
        1e-12,
        "callback measurement energy",
    );
    assert_eq!(observations.last().unwrap().1, observed.state.energy.etot);
    let expected_weights = baseline.data.qp_weights.as_ref().unwrap();
    let actual_weights = observed.data.qp_weights.as_ref().unwrap();
    // QP initialization is a short coefficient/quadrature calculation.
    for (actual, expected) in [
        (
            &actual_weights.qp_full_weight,
            &expected_weights.qp_full_weight,
        ),
        (
            &actual_weights.qp_fix_weight,
            &expected_weights.qp_fix_weight,
        ),
        (&actual_weights.spgl_cos, &expected_weights.spgl_cos),
        (&actual_weights.spgl_sin, &expected_weights.spgl_sin),
        (&actual_weights.spgl_cos_sin, &expected_weights.spgl_cos_sin),
        (&actual_weights.spgl_cos_cos, &expected_weights.spgl_cos_cos),
        (&actual_weights.spgl_sin_sin, &expected_weights.spgl_sin_sin),
    ] {
        numerical_comparison::assert_values_close(
            actual.iter().flat_map(|value| [value.re, value.im]),
            expected.iter().flat_map(|value| [value.re, value.im]),
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "callback QP weights",
        );
    }
    let baseline_files = output_snapshot(&baseline_dir);
    let callback_files = output_snapshot(&callback_dir);
    assert_eq!(baseline_files.len(), callback_files.len());
    for ((name, expected), (actual_name, actual)) in baseline_files.iter().zip(&callback_files) {
        assert_eq!(name, actual_name);
        // Indexed Green output retains its coordinates exactly; scalar output
        // has no index columns. Formatting has separate writer tests.
        let indices: &[usize] = if name.contains("cisajscktaltex") {
            &[]
        } else if name.contains("cisajscktalt") {
            &[0, 1, 2, 3, 4, 5, 6, 7]
        } else if name.contains("cisajs") {
            &[0, 1, 2, 3]
        } else {
            &[]
        };
        numerical_comparison::assert_numeric_text(actual, expected, 1e-12, 1e-12, indices, name);
    }

    let _ = fs::remove_dir_all(baseline_dir);
    let _ = fs::remove_dir_all(callback_dir);
}

#[test]
fn callback_error_stops_after_the_failed_sample_and_is_returned() {
    let mut seen = Vec::new();
    let mut callback = |sample, _: &ExpertModeData, _: Complex64, _: i32| {
        seen.push(sample);
        if sample == 1 {
            Err("physcal callback failed".to_string())
        } else {
            Ok(())
        }
    };
    let error = mvmc_core::vmc_phys_cal_with_callback(preparation(3), &mut callback).unwrap_err();
    assert_eq!(error, "physcal callback failed");
    assert_eq!(seen, vec![0, 1]);
}

#[derive(Debug)]
struct CallbackReducer {
    rank: usize,
    remote_failure_at: usize,
    failure_checks: Cell<usize>,
}

impl Reducer for CallbackReducer {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {}
    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {}
    fn allreduce_sum_i64(&self, _: &mut [i64]) {}
    fn world_size(&self) -> usize {
        2
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn any_failure(&self, failed: bool) -> bool {
        let check = self.failure_checks.get();
        self.failure_checks.set(check + 1);
        failed || (check + 1 == self.remote_failure_at)
    }
}

#[test]
fn reducer_callback_runs_on_local_rank_and_reports_remote_failure() {
    let reducer = CallbackReducer {
        rank: 1,
        remote_failure_at: 5,
        failure_checks: Cell::new(0),
    };
    let mut callback = |sample, _: &ExpertModeData, _: Complex64, status| {
        assert_eq!(sample, 0);
        assert_eq!(status, 0);
        Ok(())
    };
    let error = mvmc_core::vmc_phys_cal_with_reducer_and_callback(
        preparation(1),
        None,
        &reducer,
        Some(&mut callback),
    )
    .unwrap_err();
    assert_eq!(error, "PhysCal callback failed on another rank");
}

#[test]
fn absent_callback_still_participates_in_remote_failure_agreement() {
    let reducer = CallbackReducer {
        rank: 1,
        remote_failure_at: 5,
        failure_checks: Cell::new(0),
    };
    let error =
        mvmc_core::vmc_phys_cal_with_reducer_and_callback(preparation(1), None, &reducer, None)
            .unwrap_err();
    assert_eq!(error, "PhysCal callback failed on another rank");
    assert_eq!(reducer.failure_checks.get(), 5);
}

#[test]
fn remote_validation_failure_stops_before_sampling_or_output() {
    let reducer = CallbackReducer {
        rank: 1,
        remote_failure_at: 1,
        failure_checks: Cell::new(0),
    };
    let error = mvmc_core::vmc_phys_cal_with_reducer(preparation(1), None, &reducer).unwrap_err();
    assert_eq!(error, "PhysCal validation failed on another MPI rank");
    assert_eq!(reducer.failure_checks.get(), 1);
}

#[test]
fn remote_output_failure_prevents_callback() {
    for check in [2, 4] {
        // Directory setup and per-sample output agreement.
        let reducer = CallbackReducer {
            rank: 1,
            remote_failure_at: check,
            failure_checks: Cell::new(0),
        };
        let mut callback = |_: usize, _: &ExpertModeData, _: Complex64, _: i32| {
            panic!("callback must not run after output failure")
        };
        let error = mvmc_core::vmc_phys_cal_with_reducer_and_callback(
            preparation(1),
            None,
            &reducer,
            Some(&mut callback),
        )
        .unwrap_err();
        assert!(error.contains("output"), "{error}");
        assert_eq!(reducer.failure_checks.get(), check);
    }
}

#[test]
fn serial_validation_failure_does_not_create_output_directory() {
    let directory = std::env::temp_dir().join(format!(
        "mvmc-physcal-invalid-output-{}",
        std::process::id()
    ));
    assert!(!directory.exists());
    let mut prepared = preparation(1);
    prepared.data.modpara.lanczos_mode = 3;
    let error = mvmc_core::vmc_phys_cal_to_dir(prepared, &directory).unwrap_err();
    assert!(error.contains("NLanczosMode"));
    assert!(!directory.exists());
}

#[test]
fn remote_parse_failure_stops_before_seed_collectives() {
    let reducer = CallbackReducer {
        rank: 1,
        remote_failure_at: 1,
        failure_checks: Cell::new(0),
    };
    let root = fixture();
    let error = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
        &reducer,
    )
    .unwrap_err();
    assert_eq!(
        error,
        "PhysCal parse/load/validation failed on another MPI rank"
    );
    assert_eq!(reducer.failure_checks.get(), 1);
}
