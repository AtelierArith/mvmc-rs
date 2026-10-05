//! Independent canonical ctest trajectory and numerical reference gates.
//! C-window expectations are native aggregation of independent reference histories.
use std::fs;
use std::path::Path;

#[path = "support/fixture_status.rs"]
mod fixture_status;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
mod support;

const PREFIX_MODELS: &[&str] = &[
    "heisenberg_chain_real",
    "heisenberg_chain_cmp",
    "hubbard_chain_real",
    "hubbard_chain_cmp",
    "heisenberg_chain_fsz",
    "hubbard_chain_fsz",
    "kondo_chain_real",
    "kondo_chain_cmp",
    "kondo_chain_stot1_cmp",
    "hubbard_tetragonal_real",
    "hubbard_tetragonal_momentum_projection_real",
    "kondo_chain_fsz",
    "general_rbm_cmp",
    "general_rbm_cmp_cg",
];

/// Unknown, empty or duplicate selections are Unsupported, never an implicit pass.
fn select_prefix_models(selected: &str) -> Vec<&str> {
    fixture_status::selection(selected, PREFIX_MODELS).unwrap_or_else(|error| {
        support::unsupported("ctest-oracles", error.to_string());
    })
}

struct OutputDirectory(std::path::PathBuf);

impl OutputDirectory {
    fn create() -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("mvmc-ctest-prefix-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

/// Explicit independent fixture gate. Generation never reads Rust output.
#[test]
#[ignore = "independent canonical model fixtures: MVMC_RS_CTEST_PREFIX_MODELS required"]
fn canonical_models_match_independent_prefix_oracles() {
    support::require_gate("ctest-oracles", "MVMC_RS_CTEST_PREFIX_MODELS");
    let references = if let Ok(directory) = std::env::var("MVMC_RS_CTEST_ORACLE_ROOT") {
        support::require_gate("ctest-oracles", "MVMC_RS_CTEST_ORACLE_ROOT");
        std::path::PathBuf::from(directory)
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctest_model_prefixes")
    };
    if !references.join("provenance.txt").is_file() {
        support::missing_fixture(
            "ctest-oracles",
            references.join("provenance.txt").display().to_string(),
        );
    }
    let root = support::julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("ctest-oracles", "Julia-mVMC checkout not found")
    });
    let selected = std::env::var("MVMC_RS_CTEST_PREFIX_MODELS").unwrap();
    let selected = select_prefix_models(&selected);
    let mut failures = Vec::new();
    for model in selected {
        let outcome = std::panic::catch_unwind(|| {
            let mut window_outputs = Vec::new();
            let input_model = if model == "general_rbm_cmp_cg" {
                "general_rbm_cmp"
            } else {
                model
            };
            for steps in [1, 2, 3, 20] {
                let expected = references.join(model).join(format!("step-{steps}"));
                assert!(
                    !expected.join("UNVERIFIED.txt").exists(),
                    "{model} step {steps}: independent oracle failed"
                );
                assert_eq!(
                    fs::read_to_string(expected.join("status.txt"))
                        .unwrap()
                        .trim(),
                    "0"
                );
                let input = root
                    .join("test/integration/reference")
                    .join(input_model)
                    .join("inputs/namelist.def");
                let mut data =
                    mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false)
                        .unwrap();
                mvmc_core::validation::validate_para_opt(&data).unwrap();
                if model == "general_rbm_cmp_cg" {
                    assert_eq!((data.modpara.nsrcg, data.modpara.nstore_o), (0, 1));
                    data.modpara.nsrcg = 1;
                    data.modpara.nstore_o = 0;
                }
                let settings = fs::read_to_string(expected.join("model-settings.txt")).unwrap();
                let setting = |key: &str| -> Option<&str> {
                    settings
                        .split_whitespace()
                        .filter_map(|word| word.split_once('='))
                        .find_map(|(name, value)| (name == key).then_some(value))
                };
                for (key, value) in [
                    ("canonical_NSROptItrStep", data.modpara.nsr_opt_itr_step),
                    ("canonical_NSROptItrSmp", data.modpara.nsr_opt_itr_smp),
                    ("effective_NSROptItrStep", steps),
                    ("effective_NSROptItrSmp", steps),
                ] {
                    assert_eq!(
                        setting(key),
                        Some(value.to_string().as_str()),
                        "{model} {key}"
                    );
                }
                assert_eq!(setting("override"), Some("both_no_clamp"));
                let history =
                    fs::read_to_string(expected.join("c-window-declared-input.txt")).unwrap();
                let mut lines = history.lines();
                let window: i64 = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(window, steps, "{model} native C effective window");
                assert_eq!(
                    lines.count() as i64,
                    steps,
                    "{model} independent C window snapshots"
                );
                assert!(settings.contains(&format!("model={input_model}")));
                assert!(
                    settings.contains(&format!("seed={}", data.modpara.rnd_seed))
                        || settings.contains(&format!("RndSeed={}", data.modpara.rnd_seed))
                );
                assert!(settings.contains(&format!("NSRCG={}", data.modpara.nsrcg)));
                assert!(settings.contains(&format!("NStore={}", data.modpara.nstore_o)));
                assert!(settings.contains(&format!(
                    "initial_overlay={}",
                    input.parent().unwrap().join("initial.def").is_file()
                )));
                if data.i_flg_orbital_general != 0 {
                    let provenance = fs::read_to_string(references.join("provenance.txt")).unwrap();
                    assert!(
                        provenance.contains("actual serial native C FSZ local energy"),
                        "FSZ requires the native C energy oracle provenance"
                    );
                }
                assert!(
                    data.modpara.rnd_seed >= 0,
                    "oracle seed must be deterministic"
                );
                let mut rng = sfmt19937::Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
                mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng)
                    .unwrap();
                let initial = input.parent().unwrap().join("initial.def");
                if initial.is_file() {
                    assert!(mvmc_core::read_initial_def(&mut data, initial).unwrap());
                }
                mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
                    &mut data, &input,
                )
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
                    mvmc_core::get_all_complex_flag(&data).unwrap(),
                    data.i_flg_orbital_general != 0,
                );
                let output = OutputDirectory::create();
                mvmc_core::vmc_para_opt(
                    &mut data,
                    &mut state,
                    &mut rng,
                    Some(&output.0),
                    &mvmc_core::SingleProcessReducer,
                    mvmc_core::OptimizationOptions::default(),
                )
                .unwrap();
                let configs = fs::read_to_string(expected.join("configs.txt")).unwrap();
                let mut actual = vec![
                    state.electron_config.ele_idx.as_slice(),
                    state.electron_config.ele_cfg.as_slice(),
                    state.electron_config.ele_num.as_slice(),
                    state.electron_config.ele_proj_cnt.as_slice(),
                ];
                if data.i_flg_orbital_general != 0 {
                    actual.push(state.electron_config.ele_spn.as_slice());
                }
                actual.push(state.electron_config.burn_ele_idx.as_slice());
                actual.push(state.electron_config.counter.as_slice());
                assert_eq!(configs.lines().count(), actual.len());
                for (index, (line, actual)) in configs.lines().zip(actual).enumerate() {
                    let expected: Vec<i64> = line
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "FIRST DISCRETE DIVERGENCE: {model} prefix {steps} configs row {index}"
                    );
                }
                let expected_rng: Vec<u32> = fs::read_to_string(expected.join("rng.txt"))
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(expected_rng.len(), 624);
                let actual_rng: Vec<u32> = (0..624).map(|_| rng.gen_rand32()).collect();
                assert_eq!(
                    actual_rng, expected_rng,
                    "FIRST RNG DIVERGENCE: {model} prefix {steps}"
                );
                let parameters: Vec<_> = data
                    .projection_parameters()
                    .into_iter()
                    .chain(data.rbm_params.iter().copied())
                    .chain(data.slater_params.iter().copied())
                    .chain(data.opt_trans.iter().copied())
                    .collect();
                compare_complex(&expected.join("parameters.txt"), &parameters, 1e-11);
                compare_complex(&expected.join("energy.txt"), &[state.energy.etot], 1e-11);
                let complex = mvmc_core::get_all_complex_flag(&data).unwrap();
                let oo: Vec<_> = if complex {
                    state.sr_opt.sr_opt_oo.clone()
                } else {
                    state
                        .sr_opt
                        .sr_opt_oo_real
                        .iter()
                        .map(|&v| num_complex::Complex64::new(v, 0.0))
                        .collect()
                };
                let ho: Vec<_> = if complex {
                    state.sr_opt.sr_opt_ho.clone()
                } else {
                    state
                        .sr_opt
                        .sr_opt_ho_real
                        .iter()
                        .map(|&v| num_complex::Complex64::new(v, 0.0))
                        .collect()
                };
                compare_complex(&expected.join("sr_oo.txt"), &oo, 1e-12);
                compare_complex(&expected.join("sr_ho.txt"), &ho, 1e-12);
                if model == "general_rbm_cmp_cg" {
                    // Existing #190 budget for five-decimal CG diagnostics.
                    // Dimensions, cuts, index and iterations remain exact.
                    numerical_comparison::assert_numeric_text(
                        &fs::read_to_string(output.0.join("zvo_SRinfo.dat")).unwrap(),
                        &fs::read_to_string(expected.join("zvo_SRinfo.dat")).unwrap(),
                        1e-12,
                        1e-5,
                        &[0, 1, 2, 3, 7, 8],
                        format!("GeneralRBM CG prefix {steps} SR diagnostics"),
                    );
                }
                for filename in ["zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"] {
                    if filename == "zqp_opt.dat" && !expected.join("zqp_c_window_opt.dat").is_file()
                    {
                        eprintln!("ctest UNVERIFIED {model} prefix {steps}: independent C optimization-window output fixture missing");
                        continue;
                    }
                    let actual = numeric_values(&output.0.join(filename));
                    // Independently observed pre-SR declared slots follow C's
                    // Para array, unlike Julia's duplicated mapped terms.
                    let reference_filename = if filename == "zvo_var.dat" {
                        "zvo_c_slots_var.dat"
                    } else if filename == "zqp_opt.dat" {
                        "zqp_c_window_opt.dat"
                    } else {
                        filename
                    };
                    let reference = numeric_values(&expected.join(reference_filename));
                    if filename == "zqp_opt.dat" {
                        let provenance =
                            fs::read_to_string(expected.join("c-window-provenance.txt")).unwrap();
                        assert!(provenance.contains("actual C avevar.c bodies; no Rust results"));
                        // Collect all independent prefix quantities before
                        // reporting the known production C-window gap.
                        window_outputs.push((
                            steps,
                            actual,
                            reference,
                            fs::read_to_string(output.0.join(filename)).unwrap(),
                            fs::read_to_string(expected.join(reference_filename)).unwrap(),
                        ));
                        continue;
                    }
                    assert_eq!(
                        actual.len(),
                        reference.len(),
                        "{model} {filename} dimensions"
                    );
                    for (index, (actual, reference)) in
                        actual.iter().zip(reference.iter()).enumerate()
                    {
                        numerical_comparison::assert_close(
                            *actual,
                            *reference,
                            1e-11,
                            1e-11,
                            format!("{model} prefix {steps} {filename} component {index}"),
                        );
                    }
                    numerical_comparison::assert_numeric_text(
                        &fs::read_to_string(output.0.join(filename)).unwrap(),
                        &fs::read_to_string(expected.join(reference_filename)).unwrap(),
                        1e-11,
                        1e-11,
                        &[],
                        format!("{model} prefix {steps} {filename} layout"),
                    );
                }
                eprintln!("ctest independent oracle EXECUTED {model} prefix {steps}: exact trajectory/RNG then numerical parameters/energy/SR/zvo output; optimization-window output tracked separately");
            }
            assert!(
                [1, 2, 3, 20].iter().all(|steps| references.join(model)
                    .join(format!("step-{steps}/zqp_c_window_opt.dat")).is_file()),
                "UNVERIFIED {model}: independent C optimization-window fixtures missing; Rust/Julia final-parameter writers are not C OutputOptData parity"
            );
            for (steps, actual, reference, actual_text, reference_text) in window_outputs {
                numerical_comparison::assert_close(actual[0], reference[0], 1e-11, 1e-11,
                    format!("FIRST C OUTPUT FIELD DIVERGENCE {model} prefix {steps}: zqp_opt field 0 must be Etot (C StoreOptData/OutputOptData), not final parameter 0"));
                assert_eq!(
                    actual.len(),
                    reference.len(),
                    "{model} prefix {steps} C window output dimensions"
                );
                for (index, (actual, reference)) in actual.iter().zip(reference).enumerate() {
                    numerical_comparison::assert_close(
                        *actual,
                        reference,
                        1e-11,
                        1e-11,
                        format!("{model} prefix {steps} C window output component {index}"),
                    );
                }
                numerical_comparison::assert_numeric_text(
                    &actual_text,
                    &reference_text,
                    1e-11,
                    1e-11,
                    &[],
                    format!("{model} prefix {steps} C window output layout"),
                );
            }
        });
        if outcome.is_err() {
            failures.push(model);
        }
    }
    assert!(
        failures.is_empty(),
        "UNVERIFIED canonical prefix models: {failures:?}"
    );
}

fn numeric_values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn compare_complex(path: &Path, actual: &[num_complex::Complex64], bound: f64) {
    let expected = numeric_values(path);
    assert_eq!(
        actual.len() * 2,
        expected.len(),
        "{} dimensions",
        path.display()
    );
    for (index, (actual, expected)) in actual
        .iter()
        .flat_map(|z| [z.re, z.im])
        .zip(expected)
        .enumerate()
    {
        numerical_comparison::assert_close(
            actual,
            expected,
            bound,
            bound,
            format!("{} component {index}", path.display()),
        );
    }
}

impl Drop for OutputDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "20-step independent reference gate; MVMC_RS_CTEST_PREFIXES=1 required"]
fn general_rbm_ctest_short_prefix_and_twenty_step_discrete_trajectory() {
    support::require_gate("ctest-prefixes", "MVMC_RS_CTEST_PREFIXES");
    let root = support::julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("ctest-prefixes", "Julia-mVMC checkout not found")
    });
    let input = root.join("test/integration/reference/general_rbm_cmp/inputs/namelist.def");
    let fixtures =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/c_kernel_order");
    for (cg, store, directory) in [
        (0, 1, "sr_direct/rbm_reference_cmp_store_runner"),
        (1, 0, "sr_cg/rbm_reference_cmp_runner"),
    ] {
        for steps in [1, 2, 3, 20] {
            let mut data =
                mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false)
                    .unwrap();
            mvmc_core::validation::validate_para_opt(&data).unwrap();
            assert_eq!(data.modpara.rnd_seed, 12395);
            let mut rng = sfmt19937::Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
            mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng)
                .unwrap();
            assert!(mvmc_core::read_initial_def(
                &mut data,
                input.parent().unwrap().join("initial.def"),
            )
            .unwrap());
            mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
                &mut data, &input,
            )
            .unwrap();
            mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
            mvmc_core::qp::init_qp_weight(&mut data);
            data.modpara.nsr_opt_itr_step = steps;
            data.modpara.nsr_opt_itr_smp = steps;
            data.modpara.nsrcg = cg;
            data.modpara.nstore_o = store;
            let mut state = mvmc_core::VmcOptimizationState::zeros(
                data.modpara.nsite as usize,
                data.modpara.nelec as usize,
                data.projection_layout().n_proj,
                data.count_variational_parameters(),
                data.modpara.nsp_gauss_leg.max(1) as usize
                    * data.modpara.nmp_trans.unsigned_abs() as usize
                    * data.n_qp_opt_trans.max(1) as usize,
                data.modpara.nvmc_sample as usize,
                mvmc_core::get_all_complex_flag(&data).unwrap(),
                data.i_flg_orbital_general != 0,
            );
            let output = OutputDirectory::create();
            mvmc_core::vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&output.0),
                &mvmc_core::SingleProcessReducer,
                mvmc_core::OptimizationOptions::default(),
            )
            .unwrap();
            let reference = fixtures.join(directory);
            let configurations =
                fs::read_to_string(reference.join(format!("step-{steps}-configs.txt"))).unwrap();
            let actual = [
                state.electron_config.ele_idx.as_slice(),
                state.electron_config.ele_cfg.as_slice(),
                state.electron_config.ele_num.as_slice(),
                state.electron_config.ele_proj_cnt.as_slice(),
                state.electron_config.burn_ele_idx.as_slice(),
                state.electron_config.counter.as_slice(),
            ];
            assert_eq!(configurations.lines().count(), actual.len());
            for (index, (line, actual)) in configurations.lines().zip(actual).enumerate() {
                let expected: Vec<i64> = line
                    .split_whitespace()
                    .map(|word| word.parse().unwrap())
                    .collect();
                assert_eq!(
                    actual, expected,
                    "{directory} step {steps} configuration {index}"
                );
            }
            let expected: Vec<u32> =
                fs::read_to_string(reference.join(format!("step-{steps}-rng.txt")))
                    .unwrap()
                    .split_whitespace()
                    .map(|word| word.parse().unwrap())
                    .collect();
            assert_eq!(expected.len(), 624);
            let actual: Vec<u32> = (0..624).map(|_| rng.gen_rand32()).collect();
            assert_eq!(actual, expected, "{directory} step {steps} full SFMT block");
            // Reuse the independent canonical mixed-reference numerical
            // goldens, not only their discrete checkpoints. Historical
            // parameters enumerate mapped terms; actual values use that same
            // mapping, while the new oracle gate checks declared slots.
            let hex_values = |text: &str| -> Vec<f64> {
                text.split_whitespace()
                    .map(|word| f64::from_bits(u64::from_str_radix(word, 16).unwrap()))
                    .collect()
            };
            let mut rbm_values = Vec::new();
            data.visit_rbm_terms_mut(|_, term| rbm_values.push(term.value()));
            let parameters: Vec<f64> = data
                .gutzwiller_terms
                .iter()
                .map(|term| term.value)
                .chain(data.jastrow_terms.iter().map(|term| term.value))
                .chain(rbm_values)
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|term| data.slater_params[term.idx as usize]),
                )
                .chain(data.opt_trans.iter().copied())
                .flat_map(|value| [value.re, value.im])
                .collect();
            for (name, actual) in [
                ("parameters", parameters),
                ("energy", vec![state.energy.etot.re, state.energy.etot.im]),
            ] {
                let expected = hex_values(
                    &fs::read_to_string(reference.join(format!("step-{steps}-{name}.txt")))
                        .unwrap(),
                );
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    numerical_comparison::assert_close(*actual, expected, 1e-11, 1e-11,
                        format!("GeneralRBM cg={cg} store={store} prefix={steps} {name} component={index}"));
                }
            }
            if steps == 1 && cg == 0 {
                let fixed = fs::read_to_string(reference.join("fixed-input.txt")).unwrap();
                let lines: Vec<_> = fixed
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .collect();
                for (name, values, line) in [
                    ("SR OO", &state.sr_opt.sr_opt_oo, lines[2]),
                    ("SR HO", &state.sr_opt.sr_opt_ho, lines[3]),
                ] {
                    let actual: Vec<_> = values
                        .iter()
                        .flat_map(|value| [value.re, value.im])
                        .collect();
                    let expected = hex_values(line);
                    assert_eq!(actual.len(), expected.len());
                    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        numerical_comparison::assert_close(
                            *actual,
                            expected,
                            1e-12,
                            1e-12,
                            format!("GeneralRBM sampled {name} component={index}"),
                        );
                    }
                }
                let gram = fs::read_to_string(reference.join("gram.txt")).unwrap();
                let actual: Vec<_> = state
                    .sr_opt
                    .sr_opt_o_store
                    .iter()
                    .flat_map(|value| [value.re, value.im])
                    .collect();
                numerical_comparison::assert_values_close(
                    actual,
                    hex_values(gram.lines().nth(1).unwrap()),
                    1e-12,
                    1e-12,
                    "GeneralRBM direct stored SR Gram prefix 1",
                );
            }
            let actual = numeric_values(&output.0.join("zvo_out.dat"));
            let expected = numeric_values(&reference.join(format!("step-{steps}-zvo_out.dat")));
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                numerical_comparison::assert_close(
                    *actual,
                    expected,
                    1e-11,
                    1e-11,
                    format!("GeneralRBM cg={cg} prefix={steps} zvo_out component={index}"),
                );
            }
            eprintln!("GeneralRBM EXECUTED cg={cg} NStore={store} prefix={steps}: exact configs/RNG; independent parameters/energy/zvo_out (sampled SR at direct prefix 1)");
        }
    }
}

#[test]
fn unsupported_prefix_model_selection_cannot_pass() {
    for selection in [
        "",
        "unknown_model",
        "hubbard_chain_real,hubbard_chain_real",
        "../x",
    ] {
        assert!(
            std::panic::catch_unwind(|| select_prefix_models(selection)).is_err(),
            "{selection:?}"
        );
    }
    assert_eq!(select_prefix_models("all").len(), PREFIX_MODELS.len());
    assert_eq!(select_prefix_models("kondo_chain_fsz"), ["kondo_chain_fsz"]);
}
