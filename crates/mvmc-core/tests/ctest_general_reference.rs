//! Corrected GeneralRBM-only reference gates; no historical52-case dependency.
use std::{
    fs,
    path::{Path, PathBuf},
};
#[path = "support/fixture_status.rs"]
mod fixture_status;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
#[path = "support/ctest_general_provenance.rs"]
mod provenance;
mod support;

struct Output(PathBuf);
impl Output {
    fn new() -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ctest-general-{}-{id}", std::process::id()));
        fs::create_dir(&dir).expect("exclusive owned output directory");
        Self(dir)
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only exclusively owned output");
    }
}

fn values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}
fn complex(case: &Path, file: &str, actual: &[num_complex::Complex64], bound: f64) {
    let expected = values(&case.join(file));
    assert_eq!(actual.len() * 2, expected.len());
    numerical_comparison::assert_values_close(
        actual.iter().flat_map(|z| [z.re, z.im]),
        expected,
        bound,
        bound,
        file,
    );
}

/// Offline bundle compatibility only: no sampler, solver, or reference runtime.
#[test]
fn corrected_general_classifier_accepts_checked_in_independent_bundle() {
    let root = provenance::root();
    fixture_status::verify_selected(&root, "archive.sha256", || provenance::verify(&root))
        .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
#[ignore = "corrected General1/2/3/20 gate: MVMC_RS_CTEST_GENERAL=1 required"]
fn corrected_general_all_four_prefixes_match_independent_reference() {
    support::require_gate("ctest-general", "MVMC_RS_CTEST_GENERAL");
    let root = provenance::root();
    fixture_status::verify_selected(&root, "archive.sha256", || provenance::verify(&root))
        .unwrap_or_else(|error| panic!("{error}"));
    let input = root.join("inputs/general_rbm_cmp/namelist.def");
    for steps in [1, 2, 3, 20] {
        let case = root.join(format!("general_rbm_cmp/step-{steps}"));
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false).unwrap();
        provenance::settings(&root, &case, steps, &data);
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        let mut rng = sfmt19937::Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
        mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng);
        assert!(mvmc_core::read_initial_def(
            &mut data,
            input.parent().unwrap().join("initial.def")
        )
        .unwrap());
        mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(&mut data, &input)
            .unwrap();
        mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
        mvmc_core::qp::init_qp_weight(&mut data);
        data.modpara.nsr_opt_itr_step = steps;
        data.modpara.nsr_opt_itr_smp = steps;
        let mut state = mvmc_core::VmcOptimizationState::zeros(
            data.modpara.nsite as usize,
            data.modpara.nelec as usize,
            data.projection_layout().n_proj,
            data.count_variational_parameters(),
            data.modpara.nsp_gauss_leg.max(1) as usize
                * data.modpara.nmp_trans.unsigned_abs() as usize
                * data.n_qp_opt_trans.max(1) as usize,
            data.modpara.nvmc_sample as usize,
            mvmc_core::get_all_complex_flag(&data),
            data.i_flg_orbital_general != 0,
        );
        let output = Output::new();
        mvmc_core::vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output.0),
            &mvmc_core::SingleProcessReducer,
            mvmc_core::OptimizationOptions::default(),
        )
        .unwrap();
        // Exact stream and configurations precede every floating-point check.
        provenance::rng(&case, &rng);
        let config = fs::read_to_string(case.join("configs.txt")).unwrap();
        let actual = [
            state.electron_config.ele_idx.as_slice(),
            state.electron_config.ele_cfg.as_slice(),
            state.electron_config.ele_num.as_slice(),
            state.electron_config.ele_proj_cnt.as_slice(),
            state.electron_config.burn_ele_idx.as_slice(),
            state.electron_config.counter.as_slice(),
        ];
        assert_eq!(config.lines().count(), actual.len());
        for (line, row) in config.lines().zip(actual) {
            let expected: Vec<i64> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(row, expected);
        }
        let parameters: Vec<_> = data
            .projection_parameters()
            .into_iter()
            .chain(data.rbm_params.iter().copied())
            .chain(data.slater_params.iter().copied())
            .chain(data.opt_trans.iter().copied())
            .collect();
        complex(&case, "parameters.txt", &parameters, 1e-11);
        complex(&case, "energy.txt", &[state.energy.etot], 1e-11);
        complex(&case, "sr_oo.txt", &state.sr_opt.sr_opt_oo, 1e-12);
        complex(&case, "sr_ho.txt", &state.sr_opt.sr_opt_ho, 1e-12);
        for (actual, reference, width) in [
            ("zvo_out.dat", "zvo_out.dat", 6),
            ("zvo_var.dat", "zvo_c_slots_var.dat", 312),
        ] {
            let text = fs::read_to_string(output.0.join(actual)).unwrap();
            let expected = fs::read_to_string(case.join(reference)).unwrap();
            assert_eq!(text.lines().count(), steps as usize);
            assert!(text
                .lines()
                .all(|row| row.split_whitespace().count() == width));
            if actual == "zvo_var.dat" {
                let fields = values(&output.0.join(actual));
                for record in fields.as_chunks::<3>().0 {
                    assert_eq!(record[2], 0.0, "C reserved preSR slot");
                }
            }
            numerical_comparison::assert_numeric_text(
                &text,
                &expected,
                1e-11,
                1e-11,
                &[],
                format!("corrected General{steps} {actual}"),
            );
        }
        let c_main = values(&case.join("zqp_c_window_opt.dat"));
        assert_eq!(c_main.len(), if steps == 1 { 208 } else { 312 });
        if steps == 1 {
            for pair in values(&output.0.join("zqp_opt.dat")).as_chunks::<2>().0 {
                assert_eq!(pair[1], 0.0, "C one-snapshot second field");
            }
        }
        let provenance = fs::read_to_string(case.join("c-window-provenance.txt")).unwrap();
        assert!(provenance.contains("actual C avevar.c bodies; no Rust results"));
        let mut compared = 0;
        for entry in fs::read_dir(&case).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            if let Some(suffix) = name.strip_prefix("zqp_c_window") {
                numerical_comparison::assert_numeric_text(
                    &fs::read_to_string(output.0.join(format!("zqp{suffix}"))).unwrap(),
                    &fs::read_to_string(entry.path()).unwrap(),
                    1e-11,
                    1e-11,
                    &[],
                    format!("corrected General{steps} C window {suffix}"),
                );
                compared += 1;
            }
        }
        assert_eq!(compared, if steps == 1 { 1 } else { 7 });
        eprintln!("EXECUTED corrected3d0f Generalprefix{steps}: exactRAW/cursor/drawcount/next624/config; parameters/energy/OOHO/outputdeclaredCslots/allCwindows; no historical models claimed");
    }
}

#[test]
#[ignore = "corrected General20 public repeatability: MVMC_RS_CTEST_GENERAL=1 required"]
fn corrected_general_twenty_step_public_runner_is_repeatable() {
    support::require_gate("ctest-general", "MVMC_RS_CTEST_GENERAL");
    let root = provenance::root();
    fixture_status::verify_selected(&root, "archive.sha256", || provenance::verify(&root))
        .unwrap_or_else(|error| panic!("{error}"));
    let input = root.join("inputs/general_rbm_cmp/namelist.def");
    let case = root.join("general_rbm_cmp/step-20");
    let output = Output::new();
    let repeat = Output::new();
    let config = mvmc_core::RunConfig {
        nsmp: Some(20),
        seed: None,
        output_dir: Some(output.0.clone()),
        ..mvmc_core::RunConfig::new(20, "cmp")
    };
    let result = mvmc_core::run_para_opt_from_namelist(&input, config.clone()).unwrap();
    assert_eq!(result.status, 0);
    let repeated = mvmc_core::run_para_opt_from_namelist(
        &input,
        mvmc_core::RunConfig {
            output_dir: Some(repeat.0.clone()),
            ..config
        },
    )
    .unwrap();
    assert_eq!(repeated.status, result.status);
    assert_eq!(repeated.zvo_first_n, result.zvo_first_n);
    assert_eq!(repeated.ctest_values, result.ctest_values);
    assert_eq!(repeated.final_energy_per_site, result.final_energy_per_site);
    let names = |dir: &Path| -> std::collections::BTreeSet<_> {
        fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect()
    };
    assert_eq!(names(&output.0), names(&repeat.0));
    for name in names(&output.0) {
        if !name.to_string_lossy().contains("Timer") {
            assert_eq!(
                fs::read(output.0.join(&name)).unwrap(),
                fs::read(repeat.0.join(&name)).unwrap()
            );
        }
    }
    numerical_comparison::assert_numeric_text(
        &fs::read_to_string(output.0.join("zvo_out.dat")).unwrap(),
        &fs::read_to_string(case.join("zvo_out.dat")).unwrap(),
        1e-11,
        1e-11,
        &[],
        "corrected General20 public output",
    );
}
