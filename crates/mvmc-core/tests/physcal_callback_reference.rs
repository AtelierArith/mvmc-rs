//! Independent Julia 1.13.1 measurement references, not a Rust sampler replay.
//! See physcal_181 provenance: OptTrans uses the labelled C phase-order harness,
//! and these trajectories are not full native-C executable evidence.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use mvmc_core::{ExpertModeData, SingleProcessReducer};
use num_complex::Complex64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    fs,
    path::{Path, PathBuf},
};

struct OutputDir(PathBuf);
impl OutputDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mvmc-callback-reference-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for OutputDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture(model: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/two-samples")
        .join(model)
}

fn values<T: std::str::FromStr>(path: &Path) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|word| word.parse().unwrap())
        .collect()
}

fn fixed(data: &ExpertModeData) -> Vec<f64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .flat_map(|z| [z.re, z.im])
        .collect()
}

fn check_samples(model: &str, mode: &str, opt_trans: bool, fail: bool, samples: usize) {
    let root = fixture(model);
    let mut prepared = mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        mode,
        Some(1),
        &SingleProcessReducer,
        opt_trans,
    )
    .unwrap();
    // A prefix execution ends at the first actual runner sample boundary. It
    // does not replay the sampler or alter warm-up/interval/saved-sample counts.
    prepared.data.modpara.n_data_qty_smp = samples as i64;
    let fixed_flags = prepared.data.optimization_flags.clone();
    let expected_fixed = values::<f64>(&root.join("fixed-parameters.txt"));
    let mut seen = Vec::new();
    let mut callback = |index: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
        assert_eq!(index, seen.len(), "zero-based callback index");
        assert_eq!(status, 0);
        assert_eq!(data.optimization_flags, fixed_flags, "callback fixed flags");
        numerical_comparison::assert_values_close(
            fixed(data),
            expected_fixed.iter().copied(),
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "fixed callback parameters",
        );
        let energy_reference = values::<f64>(&root.join(format!("averaged-{index}/energy.txt")));
        assert_eq!(energy_reference.len(), 10);
        // Existing Green-reference budget: accumulated sample means across BLAS
        // providers, not permission to alter the exact sampling trajectory.
        numerical_comparison::assert_values_close(
            [energy.re, energy.im],
            energy_reference[2..4].iter().copied(),
            1e-12,
            1e-10,
            "independent post-average callback energy",
        );
        seen.push(index);
        if fail {
            Err("reference callback stop".into())
        } else {
            Ok(())
        }
    };
    let result = if samples == 1 {
        // Exercise the explicit Reducer wrapper with independent first-sample
        // references. This is serial Reducer evidence, NOT an MPI simulation.
        mvmc_core::vmc_phys_cal_with_reducer_and_callback(
            prepared,
            None,
            &SingleProcessReducer,
            Some(&mut callback),
        )
    } else {
        mvmc_core::vmc_phys_cal_with_callback(prepared, &mut callback)
    };
    if fail {
        assert_eq!(result.unwrap_err(), "reference callback stop");
        assert_eq!(seen, [0]);
        return;
    }
    let result = result.unwrap();
    assert_eq!(seen, (0..samples).collect::<Vec<_>>());
    assert_eq!(result.iterations, samples);
    assert_eq!(
        result.data.optimization_flags, fixed_flags,
        "returned fixed flags"
    );
    numerical_comparison::assert_values_close(
        fixed(&result.data),
        expected_fixed.iter().copied(),
        32.0 * f64::EPSILON,
        32.0 * f64::EPSILON,
        "returned fixed parameters",
    );
    let stage = root.join(format!("sample-{}", samples - 1));
    let config = &result.state.electron_config;
    for (name, actual) in [
        ("ele_idx", config.ele_idx.as_slice()),
        ("ele_cfg", config.ele_cfg.as_slice()),
        ("ele_num", config.ele_num.as_slice()),
        ("ele_proj_cnt", config.ele_proj_cnt.as_slice()),
        ("ele_spn", config.ele_spn.as_slice()),
        ("counter", config.counter.as_slice()),
    ] {
        assert_eq!(
            actual,
            values::<i64>(&stage.join(format!("{name}.txt"))),
            "{model} {name}"
        );
    }
    let expected = values::<u32>(&stage.join("next624.txt"));
    assert_eq!(expected.len(), 624);
    let counts = values::<usize>(&stage.join("draw-count.txt"));
    assert_eq!(counts.len(), 1);
    assert_eq!(
        result.final_rng.words_consumed(),
        counts[0] as u128,
        "ACTUAL successful runner draw count, not reconstructed fixture position"
    );
    let mut counted = sfmt19937::Sfmt19937Rng::new(1);
    for _ in 0..counts[0] {
        counted.gen_rand32();
    }
    assert_eq!(
        (0..624).map(|_| counted.gen_rand32()).collect::<Vec<_>>(),
        expected,
        "independent fixture's documented draw count"
    );
    let mut actual = result.final_rng.clone();
    assert_eq!(
        (0..624).map(|_| actual.gen_rand32()).collect::<Vec<_>>(),
        expected,
        "actual runner final RNG, not reconstructed sampling"
    );
}

fn check(model: &str, mode: &str, opt_trans: bool, fail: bool) {
    check_samples(model, mode, opt_trans, fail, 2);
}

#[test]
fn original_six_first_callback_boundaries_match_actual_runner_trajectories() {
    for (model, mode) in [
        ("heisenberg_chain_real", "real"),
        ("heisenberg_chain_cmp", "cmp"),
        ("heisenberg_chain_fsz", "fsz"),
        ("hubbard_chain_real", "real"),
        ("hubbard_chain_dh_real", "real"),
        ("kondo_chain_real", "real"),
    ] {
        check_samples(model, mode, false, false, 1);
    }
}

#[test]
fn remaining_original_two_sample_callback_boundaries_match_reference() {
    for model in [
        "hubbard_chain_real",
        "hubbard_chain_dh_real",
        "kondo_chain_real",
    ] {
        check(model, "real", false, false);
    }
}

#[test]
fn real_callback_matches_independent_two_sample_reference() {
    check("heisenberg_chain_real", "real", false, false);
}
#[test]
fn complex_callback_matches_independent_two_sample_reference() {
    check("heisenberg_chain_cmp", "cmp", false, false);
}
#[test]
fn fsz_callback_matches_independent_two_sample_reference() {
    check("heisenberg_chain_fsz", "fsz", false, false);
}
#[test]
fn opttrans_callback_matches_labelled_phase_order_reference() {
    check("hubbard_chain_dh_opttrans", "real", true, false);
}
#[test]
fn callback_failure_propagates_after_independent_first_sample() {
    check("heisenberg_chain_real", "real", false, true);
}

fn assert_retained_boundary(
    root: &Path,
    index: usize,
    state: &mvmc_core::VmcOptimizationState,
    rng: &sfmt19937::Sfmt19937Rng,
) {
    let dir = root.join(format!("averaged-{index}"));
    let config = &state.electron_config;
    for (name, actual) in [
        ("ele_idx", config.ele_idx.as_slice()),
        ("ele_cfg", config.ele_cfg.as_slice()),
        ("ele_num", config.ele_num.as_slice()),
        ("ele_proj_cnt", config.ele_proj_cnt.as_slice()),
        ("ele_spn", config.ele_spn.as_slice()),
        ("counter", config.counter.as_slice()),
    ] {
        // #181 averaged snapshots contain energy/RNG only; their sample
        // snapshots carry the identical serial configuration/counters.
        let path = if dir.join(format!("{name}.txt")).is_file() {
            dir.clone()
        } else {
            root.join(format!("sample-{index}"))
        };
        assert_eq!(
            actual,
            values::<i64>(&path.join(format!("{name}.txt"))),
            "{name} boundary {index}"
        );
    }
    let energy = &state.energy;
    numerical_comparison::assert_values_close(
        [
            energy.wc,
            energy.etot,
            energy.etot2,
            energy.sztot,
            energy.sztot2,
        ]
        .into_iter()
        .flat_map(|z| [z.re, z.im]),
        values::<f64>(&dir.join("energy.txt")),
        1e-12,
        1e-10,
        "retained averaged state",
    );
    let expected = values::<u32>(&dir.join("next624.txt"));
    assert_eq!(expected.len(), 624);
    let counts = values::<usize>(&dir.join("draw-count.txt"));
    assert_eq!(counts.len(), 1);
    assert_eq!(
        rng.words_consumed(),
        counts[0] as u128,
        "ACTUAL retained production RNG draw count at callback boundary {index}"
    );
    let mut counted = sfmt19937::Sfmt19937Rng::new(1);
    for _ in 0..counts[0] {
        counted.gen_rand32();
    }
    assert_eq!(
        (0..624).map(|_| counted.gen_rand32()).collect::<Vec<_>>(),
        expected
    );
    let mut actual = rng.clone();
    assert_eq!(
        (0..624).map(|_| actual.gen_rand32()).collect::<Vec<_>>(),
        expected,
        "retained ACTUAL production-core RNG at callback boundary {index}"
    );
}

#[test]
fn callback_errors_retain_actual_rng_and_averaged_state_in_real_complex_fsz_and_opttrans() {
    for (model, mode, opt_trans) in [
        ("heisenberg_chain_real", "real", false),
        ("heisenberg_chain_cmp", "cmp", false),
        ("heisenberg_chain_fsz", "fsz", false),
        ("hubbard_chain_dh_opttrans", "real", true),
    ] {
        for fail_at in [0, 1] {
            let root = fixture(model);
            let mut prepared =
                mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
                    root.join("inputs/namelist.def"),
                    root.join("zqp_opt.dat"),
                    mode,
                    Some(1),
                    &SingleProcessReducer,
                    opt_trans,
                )
                .unwrap();
            let expected_fixed = values::<f64>(&root.join("fixed-parameters.txt"));
            let fixed_flags = prepared.data.optimization_flags.clone();
            // In-place API replaces initial state, as Julia's vmc_phys_cal! does.
            let mut state = mvmc_core::VmcOptimizationState::zeros(1, 1, 0, 0, 1, 1, false, false);
            let mut seen = Vec::new();
            let mut callback =
                |index: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
                    assert_eq!(index, seen.len());
                    assert_eq!(status, 0);
                    assert_eq!(
                        data.optimization_flags, fixed_flags,
                        "error callback fixed flags"
                    );
                    numerical_comparison::assert_values_close(
                        fixed(data),
                        expected_fixed.iter().copied(),
                        32.0 * f64::EPSILON,
                        32.0 * f64::EPSILON,
                        "error callback fixed parameters",
                    );
                    let reference =
                        values::<f64>(&root.join(format!("averaged-{index}/energy.txt")));
                    numerical_comparison::assert_values_close(
                        [energy.re, energy.im],
                        reference[2..4].iter().copied(),
                        1e-12,
                        1e-10,
                        "error callback independent energy",
                    );
                    seen.push(index);
                    if index == fail_at {
                        Err("independent callback boundary stop".into())
                    } else {
                        Ok(())
                    }
                };
            let error = mvmc_core::run::vmc_phys_cal_in_place(
                &mut prepared.data,
                &mut state,
                &mut prepared.rng,
                None,
                &SingleProcessReducer,
                Some(&mut callback),
            )
            .unwrap_err();
            assert_eq!(error, "independent callback boundary stop");
            assert_eq!(seen, (0..=fail_at).collect::<Vec<_>>());
            assert_eq!(
                prepared.data.optimization_flags, fixed_flags,
                "fixed flags retained on Err"
            );
            assert_retained_boundary(&root, fail_at, &state, &prepared.rng);
        }
    }
}

fn check_lanczos_callback(model: &str, mode: usize) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_callbacks_175")
        .join(format!("{model}-mode{mode}"));
    let prepared = mvmc_core::prepare_phys_cal_from_namelist(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap();
    assert!(prepared.data.inter_all_terms.is_empty());
    assert_eq!(prepared.data.modpara.lanczos_mode, mode as i64);
    assert_eq!(prepared.data.modpara.n_data_idx_start, 7);
    assert_eq!(
        values::<usize>(&root.join("consumed-count.txt")),
        [prepared.n_para_consumed]
    );
    if model.starts_with("hubbard") {
        assert!(!prepared.data.transfer_terms.is_empty());
        assert!(!prepared.data.coulomb_intra_terms.is_empty());
    } else {
        assert!(!prepared.data.exchange_terms.is_empty());
        assert!(
            !prepared.data.coulomb_inter_terms.is_empty() || !prepared.data.hund_terms.is_empty()
        );
    }
    let flags = values::<i64>(&root.join("optimization-flags.txt"));
    let fixed_flags = prepared.data.optimization_flags.clone();
    let written = values::<i64>(&root.join("optimization-flags-written.txt"));
    assert_eq!(flags.len(), prepared.data.optimization_flags.len());
    assert_eq!(written.len(), flags.len());
    let assert_flags = |data: &ExpertModeData| {
        assert_eq!(
            data.optimization_flags, fixed_flags,
            "all loaded flags unchanged"
        );
        for ((actual, expected), mask) in data.optimization_flags.iter().zip(&flags).zip(&written) {
            if *mask == 1 {
                assert_eq!(actual, expected);
            }
        }
    };
    assert_flags(&prepared.data);
    let expected_fixed = values::<f64>(&root.join("fixed-parameters.txt"));
    let records = fs::read_to_string(root.join("callbacks.txt")).unwrap();
    let records = records
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 2);
    // Observe the FIRST Lanczos boundary on Err through the SAME production
    // core. No sampler replay or inferred error-state snapshot is substituted.
    let mut error_data = prepared.data.clone();
    let mut error_rng = prepared.rng.clone();
    let mut error_state = mvmc_core::VmcOptimizationState::zeros(1, 1, 0, 0, 1, 1, false, false);
    let mut error_seen = Vec::new();
    let mut stop = |index: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
        assert_eq!(index, 0);
        assert_eq!(status, records[0][1].parse::<i32>().unwrap());
        assert_flags(data);
        numerical_comparison::assert_values_close(
            [energy.re, energy.im],
            records[0][2..4]
                .iter()
                .map(|word| word.parse::<f64>().unwrap()),
            1e-12,
            1e-10,
            "first Lanczos error-boundary callback energy",
        );
        error_seen.push(index);
        Err("stop after first Lanczos callback".into())
    };
    let error = mvmc_core::run::vmc_phys_cal_in_place(
        &mut error_data,
        &mut error_state,
        &mut error_rng,
        None,
        &SingleProcessReducer,
        Some(&mut stop),
    )
    .unwrap_err();
    assert_eq!(error, "stop after first Lanczos callback");
    assert_eq!(error_seen, [0]);
    assert_flags(&error_data);
    assert_retained_boundary(&root, 0, &error_state, &error_rng);
    let mut seen = Vec::new();
    let mut callback = |index: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
        assert_eq!(index, seen.len());
        assert_eq!(records[index].len(), 4);
        assert_eq!(index, records[index][0].parse::<usize>().unwrap());
        assert_eq!(status, records[index][1].parse::<i32>().unwrap());
        assert_flags(data);
        numerical_comparison::assert_values_close(
            fixed(data),
            expected_fixed.iter().copied(),
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "Lanczos callback fixed parameters",
        );
        numerical_comparison::assert_values_close(
            [energy.re, energy.im],
            records[index][2..4]
                .iter()
                .map(|word| word.parse::<f64>().unwrap()),
            1e-12,
            1e-10,
            "actual independent Lanczos callback energy",
        );
        seen.push(index);
        Ok(())
    };
    let result = mvmc_core::vmc_phys_cal_with_callback(prepared, &mut callback).unwrap();
    assert_eq!(seen, [0, 1]);
    assert_flags(&result.data);
    assert_retained_boundary(&root, 1, &result.state, &result.final_rng);
    assert_eq!(
        values::<u32>(&root.join("callback-1/next624.txt")),
        values::<u32>(&root.join("averaged-1/next624.txt")),
        "independent actual callback peek has the post-average RNG position"
    );
}

#[test]
fn hopping_intra_lanczos_callbacks_match_independent_records() {
    for mode in [1, 2] {
        check_lanczos_callback("hubbard_chain_lanczos", mode);
    }
}

#[test]
fn exchange_spin_lanczos_callbacks_match_independent_records() {
    for mode in [1, 2] {
        check_lanczos_callback("spin_chain_lanczos", mode);
    }
}

#[test]
fn failing_callback_observes_completed_first_output_and_prevents_second_output() {
    let root = fixture("heisenberg_chain_real");
    let mut prepared = mvmc_core::prepare_phys_cal_from_namelist(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap();
    let output = OutputDir::new();
    let fixed_flags = prepared.data.optimization_flags.clone();
    let mut seen = Vec::new();
    let mut expected_names = Vec::new();
    let mut callback = |index: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
        assert_eq!((index, status), (0, 0));
        assert_eq!(
            data.optimization_flags, fixed_flags,
            "completed-output callback fixed flags"
        );
        let reference = values::<f64>(&root.join("averaged-0/energy.txt"));
        numerical_comparison::assert_values_close(
            [energy.re, energy.im],
            reference[2..4].iter().copied(),
            1e-12,
            1e-10,
            "failed callback's independently averaged energy",
        );
        for entry in fs::read_dir(root.join("expected")).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            if !name.ends_with("_007.dat") {
                continue;
            }
            let actual = fs::read_to_string(output.0.join(&name)).unwrap();
            let expected = fs::read_to_string(entry.path()).unwrap();
            let indices: &[usize] = if name.contains("cisajscktaltex") {
                &[]
            } else if name.contains("cisajscktalt") {
                &[0, 1, 2, 3, 4, 5, 6, 7]
            } else if name.contains("cisajs") {
                &[0, 1, 2, 3]
            } else {
                &[]
            };
            numerical_comparison::assert_numeric_text(
                actual.trim_end(),
                expected.trim_end(),
                1e-12,
                1e-10,
                indices,
                &name,
            );
            expected_names.push(name);
        }
        seen.push(index);
        Err("stop after completed output".into())
    };
    let mut state = mvmc_core::VmcOptimizationState::zeros(1, 1, 0, 0, 1, 1, false, false);
    let error = mvmc_core::run::vmc_phys_cal_in_place(
        &mut prepared.data,
        &mut state,
        &mut prepared.rng,
        Some(&output.0),
        &SingleProcessReducer,
        Some(&mut callback),
    )
    .unwrap_err();
    assert_eq!(error, "stop after completed output");
    assert_eq!(seen, [0]);
    assert_eq!(
        prepared.data.optimization_flags, fixed_flags,
        "output-error fixed flags retained"
    );
    assert!(!expected_names.is_empty());
    let mut actual_names = fs::read_dir(&output.0)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    expected_names.sort();
    actual_names.sort();
    assert_eq!(
        actual_names, expected_names,
        "failure must not write sample 008"
    );
    assert!(!output.0.join("zvo_out_008.dat").exists());
    assert!(!output.0.join("zvo_var_008.dat").exists());
    assert_retained_boundary(&root, 0, &state, &prepared.rng);
}

#[test]
fn actual_output_error_retains_first_averaged_state_rng_flags_and_no_second_sample() {
    for (model, mode, opt_trans) in [
        ("heisenberg_chain_real", "real", false),
        ("heisenberg_chain_cmp", "cmp", false),
        ("heisenberg_chain_fsz", "fsz", false),
        ("hubbard_chain_dh_opttrans", "real", true),
    ] {
        let root = fixture(model);
        let mut prepared = mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
            root.join("inputs/namelist.def"),
            root.join("zqp_opt.dat"),
            mode,
            Some(1),
            &SingleProcessReducer,
            opt_trans,
        )
        .unwrap();
        let fixed_flags = prepared.data.optimization_flags.clone();
        let output = OutputDir::new();
        // A genuine filesystem write failure, not an injected runner/oracle
        // hook. Rust writes out/var before the first OneBody file. This tests
        // the Rust API error boundary, NOT native C's earlier fopen timing.
        fs::create_dir(output.0.join("zvo_cisajs_007.dat")).unwrap();
        let mut state = mvmc_core::VmcOptimizationState::zeros(1, 1, 0, 0, 1, 1, false, false);
        let mut seen = Vec::new();
        let mut callback = |index: usize, _: &ExpertModeData, _: Complex64, _: i32| {
            seen.push(index);
            Ok(())
        };
        let error = mvmc_core::run::vmc_phys_cal_in_place(
            &mut prepared.data,
            &mut state,
            &mut prepared.rng,
            Some(&output.0),
            &SingleProcessReducer,
            Some(&mut callback),
        )
        .unwrap_err();
        assert!(!error.is_empty());
        assert!(seen.is_empty(), "callback must not run after failed output");
        assert_eq!(prepared.data.optimization_flags, fixed_flags);
        assert_retained_boundary(&root, 0, &state, &prepared.rng);
        numerical_comparison::assert_values_close(
            fixed(&prepared.data),
            values::<f64>(&root.join("fixed-parameters.txt")),
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "output-error fixed parameters",
        );
        for name in ["zvo_out_007.dat", "zvo_var_007.dat"] {
            let actual = fs::read_to_string(output.0.join(name)).unwrap();
            let expected = fs::read_to_string(root.join("expected").join(name)).unwrap();
            numerical_comparison::assert_numeric_text(
                actual.trim_end(),
                expected.trim_end(),
                1e-12,
                1e-10,
                &[],
                name,
            );
        }
        let mut files = fs::read_dir(&output.0)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                let name = entry.file_name().into_string().unwrap();
                assert!(
                    !name.ends_with("_008.dat"),
                    "no second sample after output error"
                );
                name
            })
            .collect::<Vec<_>>();
        files.sort();
        assert_eq!(
            files,
            ["zvo_cisajs_007.dat", "zvo_out_007.dat", "zvo_var_007.dat"]
        );
        assert!(output.0.join("zvo_cisajs_007.dat").is_dir());
    }
}
