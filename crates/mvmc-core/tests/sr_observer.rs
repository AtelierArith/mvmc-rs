//! Observer invariance tests, not independent numerical reference generation.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::sr::{observer, stochastic_opt_complex, stochastic_opt_real};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use num_complex::Complex64;

fn problem(complex: bool, failure: bool) -> (ExpertModeData, VmcOptimizationState) {
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = 2;
    data.modpara.dsr_opt_red_cut = 0.0;
    data.modpara.dsr_opt_sta_del = 0.0;
    data.modpara.dsr_opt_step_dt = 0.5;
    data.slater_params = vec![Complex64::new(10.0, 0.0); 2];
    data.optimization_flags = vec![1, 0, 1, 0];
    let mut state = VmcOptimizationState::zeros(1, 1, 0, 2, 1, 2, complex, false);
    let size = state.sr_opt.sr_opt_size;
    if complex {
        let lda = 2 * size;
        state.sr_opt.sr_opt_oo[2 * lda + 2] = Complex64::new(2.0, 0.0);
        state.sr_opt.sr_opt_oo[4 * lda + 4] = Complex64::new(3.0, 0.0);
        state.sr_opt.sr_opt_ho[2] = Complex64::new(if failure { f64::NAN } else { 1.0 }, 0.0);
        state.sr_opt.sr_opt_ho[4] = Complex64::new(2.0, 0.0);
    } else {
        state.sr_opt.sr_opt_oo_real[size + 1] = 2.0;
        state.sr_opt.sr_opt_oo_real[2 * size + 2] = 3.0;
        state.sr_opt.sr_opt_ho_real[1] = if failure { f64::NAN } else { 1.0 };
        state.sr_opt.sr_opt_ho_real[2] = 2.0;
    }
    (data, state)
}

fn solve(complex: bool, data: &mut ExpertModeData, state: &mut VmcOptimizationState) -> i32 {
    if complex {
        stochastic_opt_complex(data, state)
    } else {
        stochastic_opt_real(data, state)
    }
}

#[test]
fn direct_capture_preserves_original_success_and_failure_outputs() {
    for complex in [false, true] {
        for failure in [false, true] {
            let (mut baseline, mut baseline_state) = problem(complex, failure);
            let baseline_status = solve(complex, &mut baseline, &mut baseline_state);
            let guard = observer::capture().unwrap();
            let (mut observed, mut observed_state) = problem(complex, failure);
            let observed_status = solve(complex, &mut observed, &mut observed_state);
            let records = guard.finish();
            assert_eq!(observed_status, baseline_status);
            assert_eq!(observed_status, i32::from(failure));
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert_eq!(record.dimension, 2);
            assert_eq!(record.active_indices, [0, 2]);
            assert_eq!(record.flags, [1, 0, 1, 0]);
            assert_eq!(record.status, Some(observed_status));
            assert_eq!(record.factor_info, Some(0));
            assert_eq!(record.solve_info, Some(0));
            numerical_comparison::assert_values_close(
                record.matrix.iter().copied(),
                [2.0, 0.0, 0.0, 3.0],
                16.0 * f64::EPSILON,
                16.0 * f64::EPSILON,
                "actual original matrix",
            );
            if failure {
                assert!(record.rhs[0].is_nan());
                assert!(record.increment.iter().any(|value| !value.is_finite()));
            } else {
                numerical_comparison::assert_values_close(
                    record.rhs.iter().copied(),
                    [-1.0, -2.0],
                    16.0 * f64::EPSILON,
                    16.0 * f64::EPSILON,
                    "actual original RHS",
                );
                numerical_comparison::assert_values_close(
                    record.increment.iter().copied(),
                    [-0.5, -2.0 / 3.0],
                    32.0 * f64::EPSILON,
                    32.0 * f64::EPSILON,
                    "actual original increment",
                );
            }
            numerical_comparison::assert_values_close(
                observed
                    .slater_params
                    .iter()
                    .flat_map(|value| [value.re, value.im]),
                baseline
                    .slater_params
                    .iter()
                    .flat_map(|value| [value.re, value.im]),
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                "observer on/off parameter invariance",
            );
            assert_eq!(
                observed_state.electron_config.ele_idx,
                baseline_state.electron_config.ele_idx
            );
            assert_eq!(
                observed_state.electron_config.counter,
                baseline_state.electron_config.counter
            );
            // Disabling capture does not leave a record in a later scope.
            solve(complex, &mut observed, &mut observed_state);
            assert!(observer::capture().unwrap().finish().is_empty());
        }
    }
}

#[test]
fn c_factor_failure_bypasses_substitution_and_preserves_parameters() {
    for complex in [false, true] {
        let run = |enabled: bool| {
            let (mut data, mut state) = problem(complex, false);
            if complex {
                let lda = 2 * state.sr_opt.sr_opt_size;
                state.sr_opt.sr_opt_oo[2 * lda + 2] = Complex64::new(0.0, 0.0);
            } else {
                let size = state.sr_opt.sr_opt_size;
                state.sr_opt.sr_opt_oo_real[size + 1] = 0.0;
            }
            // Zero cutoff retains the singular component. C's DPOSV bypasses
            // substitution on positive factor INFO. Historical Julia discarded
            // that status and produced a nonfinite increment; that defect is
            // not the authoritative failure contract.
            let guard = enabled.then(|| observer::capture().unwrap());
            let status = solve(complex, &mut data, &mut state);
            let records = guard
                .map(observer::CaptureGuard::finish)
                .unwrap_or_default();
            (status, data, records)
        };
        let (baseline_status, baseline, _) = run(false);
        let (status, observed, records) = run(true);
        assert_eq!(status, baseline_status);
        assert_eq!(status, 1);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].factor_info, Some(1));
        assert_eq!(records[0].solve_info, None);
        assert_eq!(records[0].status, Some(1));
        assert_eq!(records[0].increment, records[0].rhs);
        assert!(records[0].increment.iter().all(|value| value.is_finite()));
        numerical_comparison::assert_values_close(
            observed
                .slater_params
                .iter()
                .flat_map(|value| [value.re, value.im]),
            baseline
                .slater_params
                .iter()
                .flat_map(|value| [value.re, value.im]),
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "factor/substitution failure on/off invariance",
        );
    }
}

#[test]
fn capture_is_thread_local_and_nested_failure_preserves_outer_scope() {
    let guard = observer::capture().unwrap();
    assert!(observer::capture().is_err());
    let (mut data, mut state) = problem(false, false);
    assert_eq!(solve(false, &mut data, &mut state), 0);
    std::thread::spawn(|| {
        let (mut data, mut state) = problem(false, false);
        assert_eq!(solve(false, &mut data, &mut state), 0);
        let guard = observer::capture().unwrap();
        assert_eq!(solve(false, &mut data, &mut state), 0);
        assert_eq!(guard.finish().len(), 1);
    })
    .join()
    .unwrap();
    assert_eq!(guard.finish().len(), 1);
}

#[test]
fn dropping_or_unwinding_clears_capture_and_no_active_components_are_not_fake_solves() {
    let result = std::panic::catch_unwind(|| {
        let _guard = observer::capture().unwrap();
        let (mut data, mut state) = problem(false, false);
        solve(false, &mut data, &mut state);
        panic!("exercise capture cleanup");
    });
    assert!(result.is_err());
    let guard = observer::capture().unwrap();
    let (mut data, mut state) = problem(false, false);
    data.optimization_flags.fill(0);
    assert_eq!(solve(false, &mut data, &mut state), 0);
    let records = guard.finish();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].not_solved,
        Some(observer::NotSolvedReason::NoActiveComponents)
    );
    assert_eq!(records[0].status, Some(0));
    assert!(
        records[0].matrix.is_empty()
            && records[0].rhs.is_empty()
            && records[0].increment.is_empty()
    );
    assert_eq!(records[0].factor_info, None);
    assert_eq!(records[0].solve_info, None);
    drop(observer::capture().unwrap());
    assert!(observer::capture().unwrap().finish().is_empty());
}

#[test]
fn actual_sampling_outputs_configurations_and_rng_are_unchanged_by_capture() {
    struct Output(std::path::PathBuf);
    impl Drop for Output {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let run = |enabled: bool| {
        let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/inputs/namelist.def");
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(input).unwrap();
        let mut rng = sfmt19937::Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
        mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng);
        mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
        mvmc_core::qp::init_qp_weight(&mut data);
        data.modpara.nsr_opt_itr_step = 2;
        data.modpara.nsr_opt_itr_smp = 2;
        let mut state = VmcOptimizationState::zeros(
            data.modpara.nsite as usize,
            data.modpara.nelec as usize,
            data.projection_layout().n_proj,
            data.count_variational_parameters(),
            data.modpara.nsp_gauss_leg.max(1) as usize
                * data.modpara.nmp_trans.unsigned_abs() as usize
                * data.n_qp_opt_trans.max(1) as usize,
            data.modpara.nvmc_sample as usize,
            mvmc_core::get_all_complex_flag(&data),
            false,
        );
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output = Output(
            std::env::temp_dir().join(format!("mvmc-sr-observer-{}-{id}", std::process::id())),
        );
        std::fs::create_dir(&output.0).unwrap();
        let guard = enabled.then(|| observer::capture().unwrap());
        mvmc_core::vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output.0),
            &mvmc_core::SingleProcessReducer,
            mvmc_core::OptimizationOptions::default(),
        )
        .unwrap();
        let records = guard
            .map(observer::CaptureGuard::finish)
            .unwrap_or_default();
        let words: Vec<_> = (0..624).map(|_| rng.gen_rand32()).collect();
        let streams: Vec<_> = ["zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"]
            .iter()
            .map(|name| std::fs::read_to_string(output.0.join(name)).unwrap())
            .collect();
        (data, state, words, streams, records)
    };
    let (baseline, baseline_state, baseline_rng, baseline_output, baseline_records) = run(false);
    let (observed, observed_state, observed_rng, observed_output, records) = run(true);
    assert!(baseline_records.is_empty());
    assert_eq!(records.len(), 2);
    assert!(records
        .iter()
        .all(|record| record.status == Some(0) && record.not_solved.is_none()));
    assert_eq!(baseline_rng, observed_rng);
    assert_eq!(
        baseline_state.electron_config.ele_idx,
        observed_state.electron_config.ele_idx
    );
    assert_eq!(
        baseline_state.electron_config.ele_cfg,
        observed_state.electron_config.ele_cfg
    );
    assert_eq!(
        baseline_state.electron_config.ele_num,
        observed_state.electron_config.ele_num
    );
    assert_eq!(
        baseline_state.electron_config.ele_proj_cnt,
        observed_state.electron_config.ele_proj_cnt
    );
    assert_eq!(
        baseline_state.electron_config.burn_ele_idx,
        observed_state.electron_config.burn_ele_idx
    );
    assert_eq!(
        baseline_state.electron_config.counter,
        observed_state.electron_config.counter
    );
    numerical_comparison::assert_values_close(
        baseline
            .slater_params
            .iter()
            .flat_map(|value| [value.re, value.im]),
        observed
            .slater_params
            .iter()
            .flat_map(|value| [value.re, value.im]),
        32.0 * f64::EPSILON,
        32.0 * f64::EPSILON,
        "observer on/off sampled parameter invariance",
    );
    for (baseline, observed) in baseline_output.iter().zip(&observed_output) {
        numerical_comparison::assert_numeric_text(
            baseline,
            observed,
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            &[],
            "observer on/off output invariance",
        );
    }
}
