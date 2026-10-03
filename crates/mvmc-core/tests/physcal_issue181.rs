//! Issue #181: serial PhysCal and non-InterAll Lanczos verification.
//!
//! These are opt-in because they replay the reference sampling scenarios.  Run
//! with `MVMC_RS_PHYSCAL_181=1 cargo nextest run -p mvmc-core --test
//! physcal_issue181`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

mod support;
use support::{julia_mvmc_root, report_gate, require_gate, GateStatus};

#[derive(Clone, Copy)]
struct Model {
    name: &'static str,
    mode: &'static str,
}

const PHYSCAL_MODELS: &[Model] = &[
    Model {
        name: "heisenberg_chain_real",
        mode: "real",
    },
    Model {
        name: "heisenberg_chain_cmp",
        mode: "cmp",
    },
    Model {
        name: "heisenberg_chain_fsz",
        mode: "fsz",
    },
    Model {
        name: "hubbard_chain_real",
        mode: "real",
    },
    Model {
        name: "hubbard_chain_dh_real",
        mode: "real",
    },
    Model {
        name: "kondo_chain_real",
        mode: "real",
    },
];

fn fixture(root: &Path, model: Model) -> PathBuf {
    root.join("test/integration/reference")
        .join(model.name)
        .join("physcal_ref")
}

fn output_dir(test: &str, name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "mvmc-issue181-{test}-{name}-{}",
        std::process::id()
    ))
}

fn read_values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("{}: {value}: {error}", path.display()))
        })
        .collect()
}

fn assert_reference(actual: &Path, expected: &Path) {
    // Independent contracts: Julia reference tools/green_compare.jl and
    // lanczos_equivalent.jl. Julia's scalar isapprox uses max(atol, rtol*scale),
    // not their sum. Lanczos references specify absolute errors only.
    let name = expected.file_name().unwrap().to_str().unwrap();
    let (indices, width, atol, rtol) = if name.starts_with("zvo_ls_out_") {
        (0, Some(3), 1.0e-8, 0.0)
    } else if name.starts_with("zvo_ls_qqqq_") {
        (0, Some(16), 1.0e-10, 0.0)
    } else if name.starts_with("zvo_ls_cisajscktaltex_") {
        (0, None, 1.0e-8, 0.0)
    } else if name.starts_with("zvo_ls_cisajscktalt_") {
        (8, Some(10), 1.0e-8, 0.0)
    } else if name.starts_with("zvo_ls_cisajs_") {
        (4, Some(6), 1.0e-8, 0.0)
    } else if name.starts_with("zvo_cisajscktaltex_") {
        (0, None, 1.0e-12, 1.0e-9)
    } else if name.starts_with("zvo_cisajscktalt_") {
        (8, Some(10), 1.0e-12, 1.0e-9)
    } else if name.starts_with("zvo_cisajs_") {
        (4, Some(6), 1.0e-12, 1.0e-10)
    } else {
        panic!("no independent output contract for {name}");
    };
    let actual_text = fs::read_to_string(actual).unwrap();
    let expected_text = fs::read_to_string(expected).unwrap();
    let rows = |text: &str| {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let actual_rows = rows(&actual_text);
    let expected_rows = rows(&expected_text);
    assert_eq!(actual_rows.len(), expected_rows.len(), "{name} row count");
    if indices == 0 {
        assert_eq!(expected_rows.len(), 1, "{name} value-only output row count");
    }
    for (row, (a, e)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        let width = width.unwrap_or(e.len());
        assert_eq!(e.len(), width, "{name} reference row {row} columns");
        assert_eq!(a.len(), width, "{name} actual row {row} columns");
        if name.contains("cisajscktaltex_") {
            assert_eq!(width % 2, 0, "{name} real/imaginary pairs");
        }
        for column in 0..width {
            if column < indices {
                // Parse as integers as well as checking the reference tokens;
                // fractional or approximately equal indices must never pass.
                assert_eq!(
                    a[column].parse::<i64>().unwrap(),
                    e[column].parse::<i64>().unwrap(),
                    "{name} row {row} index column {column}"
                );
                assert_eq!(
                    a[column], e[column],
                    "{name} row {row} index column {column}"
                );
            } else {
                let a = a[column].parse::<f64>().unwrap();
                let e = e[column].parse::<f64>().unwrap();
                let scale = a.abs().max(e.abs());
                let tolerance = f64::max(atol, rtol * scale);
                // Retain the old gate's upper bound too: adopting an
                // independent contract must not loosen any existing check.
                let tolerance = tolerance.min(1.0e-10 + 1.0e-8 * scale);
                assert!(a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance,
                    "{name} row {row} column {column}: actual={a:.17e}, expected={e:.17e}, tolerance={tolerance:.3e}");
            }
        }
    }
}

fn snapshot(dir: &Path) -> BTreeSet<(String, String)> {
    fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                fs::read_to_string(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn prepare(root: &Path, model: Model, seed: i64) -> (mvmc_core::PhysCalPreparation, PathBuf) {
    let fixture = fixture(root, model);
    let namelist = fixture.join("inputs/namelist.def");
    let opt = fixture.join("zqp_opt.dat");
    assert!(namelist.is_file(), "{}", namelist.display());
    assert!(opt.is_file(), "{}", opt.display());
    let preparation =
        mvmc_core::prepare_phys_cal_from_namelist(&namelist, &opt, model.mode, Some(seed))
            .unwrap_or_else(|error| panic!("{}: {error}", model.name));
    (preparation, fixture)
}

fn fixed_values(data: &mvmc_core::ExpertModeData) -> Vec<num_complex::Complex64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn reference_fixed_values(fixture: &Path) -> (Vec<num_complex::Complex64>, usize) {
    // These fixtures have no In*.def overlays. In C, initial.def is not
    // implicitly read beside namelist.def: the explicit zqp argument supplies
    // ReadInitParameter's input. In particular FSZ's neighboring initial.def
    // must not replace these values.
    let inputs = fixture.join("inputs");
    let namelist = fs::read_to_string(inputs.join("namelist.def")).unwrap();
    let mut projection = 0;
    let mut slater = 0;
    for line in namelist
        .lines()
        .filter(|line| !line.trim().starts_with('#'))
    {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.is_empty() {
            continue;
        }
        assert!(
            !fields[0].starts_with("In"),
            "uncovered C overlay contract: {line}"
        );
        let multiplier = match fields[0] {
            "Gutzwiller" | "Jastrow" => 1,
            "DH2" => 6,
            "DH4" => 10,
            "OrbitalParallel" => 2, // C: separate up-up and down-down slots.
            "Orbital" | "OrbitalAntiParallel" | "OrbitalGeneral" => 1,
            kind if kind.contains("RBM") || kind == "OptTrans" => {
                panic!("independent fixed-value layout not covered for {kind}")
            }
            _ => continue,
        };
        let definition = fs::read_to_string(inputs.join(fields[1])).unwrap();
        let count = definition
            .lines()
            .nth(1)
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        if fields[0].starts_with("Orbital") {
            slater += multiplier * count;
        } else {
            projection += multiplier * count;
        }
    }
    let count = projection + slater;
    let values = read_values(&fixture.join("zqp_opt.dat"));
    let record_width = 6 + 3 * count;
    assert!(!values.is_empty());
    assert_eq!(
        values.len() % record_width,
        0,
        "complete C parameter records"
    );
    // C reads until EOF, so the final complete record wins.
    let record = &values[values.len() - record_width..];
    let mut expected = record[6..]
        .chunks_exact(3)
        .map(|triple| num_complex::Complex64::new(triple[0], triple[1]))
        .collect::<Vec<_>>();
    // C SyncModifiedParameter rescales the Slater block to D_AmpMax=4.
    // Correlation shifts are disabled in this PhysCal preparation contract.
    let maximum = expected[projection..]
        .iter()
        .map(|v| v.re.hypot(v.im))
        .fold(0.0_f64, f64::max);
    assert!(maximum > 0.0);
    for value in &mut expected[projection..] {
        *value *= 4.0 / maximum;
    }
    (expected, count)
}

fn assert_fixed_values(data: &mvmc_core::ExpertModeData, expected: &[num_complex::Complex64]) {
    let actual = fixed_values(data);
    assert_eq!(actual.len(), expected.len(), "fixed parameter width");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (a, e) in [(a.re, e.re), (a.im, e.im)] {
            // Independent C cabs vs Rust hypot normalization can differ by
            // rounding; this bound is tighter than every output contract.
            assert!(
                (a - e).abs() <= 1.0e-14 * e.abs().max(1.0),
                "fixed parameter {index}: actual={a:.17e}, expected={e:.17e}"
            );
        }
    }
}

fn trajectory_fixture(model: Model) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(model.name)
}

fn reference_integers(path: &Path) -> Vec<i64> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect()
}

fn assert_discrete(label: &str, actual: &[i64], expected: &[i64]) {
    assert_eq!(actual.len(), expected.len(), "{label} width");
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        panic!(
            "{label}: first discrete divergence at element {index}: actual={}, reference={}",
            actual[index], expected[index]
        );
    }
}

fn assert_saved_trajectory(model: Model, state: &mvmc_core::VmcOptimizationState) {
    let stage = trajectory_fixture(model).join("sample-0");
    let config = &state.electron_config;
    for (name, values) in [
        ("ele_idx", config.ele_idx.as_slice()),
        ("ele_cfg", config.ele_cfg.as_slice()),
        ("ele_num", config.ele_num.as_slice()),
        ("ele_proj_cnt", config.ele_proj_cnt.as_slice()),
        ("ele_spn", config.ele_spn.as_slice()),
        ("counter", config.counter.as_slice()),
    ] {
        assert_discrete(
            &format!("{} sample-0 {name}", model.name),
            values,
            &reference_integers(&stage.join(format!("{name}.txt"))),
        );
    }
}

fn assert_rng_checkpoint(model: Model, stage: &str, rng: &sfmt19937::Sfmt19937Rng) {
    let fixture = trajectory_fixture(model).join(stage);
    let expected = reference_integers(&fixture.join("next624.txt"));
    assert_eq!(
        expected.len(),
        624,
        "{} {stage} complete RNG block",
        model.name
    );
    let count = reference_integers(&fixture.join("draw-count.txt"));
    assert_eq!(count.len(), 1);
    assert!(count[0] >= 0);
    // Validate the independent reference's documented position from seed 1,
    // then check the actual sampler position without advancing its RNG.
    let mut position = sfmt19937::Sfmt19937Rng::new(1);
    for _ in 0..count[0] {
        position.gen_rand32();
    }
    let next = |rng: &sfmt19937::Sfmt19937Rng| {
        let mut peek = rng.clone();
        (0..624)
            .map(|_| i64::from(peek.gen_rand32()))
            .collect::<Vec<_>>()
    };
    assert_discrete(
        &format!("{} {stage} documented draw count", model.name),
        &next(&position),
        &expected,
    );
    assert_discrete(
        &format!("{} {stage} next624 RNG", model.name),
        &next(rng),
        &expected,
    );
}

fn check_independent_sampling_trajectory(model: Model) {
    use mvmc_expert_parsers::utils::{parameter_init::init_parameter, qp_weight::init_qp_weight};
    let fixture = trajectory_fixture(model);
    let mut preparation = mvmc_core::prepare_phys_cal_from_namelist(
        fixture.join("inputs/namelist.def"),
        fixture.join("zqp_opt.dat"),
        model.mode,
        Some(1),
    )
    .unwrap();
    assert_rng_checkpoint(model, "seeded", &preparation.rng);
    let expected = read_values(&fixture.join("fixed-parameters.txt"));
    assert_eq!(expected.len() % 2, 0);
    let expected = expected
        .chunks_exact(2)
        .map(|pair| num_complex::Complex64::new(pair[0], pair[1]))
        .collect::<Vec<_>>();
    assert_fixed_values(&preparation.data, &expected);
    // C's single InitParameter advances RNG; the fixed zqp/overlay values are
    // restored before sampling. No runtime reference program is called here.
    init_parameter(&mut preparation.data.clone(), &mut preparation.rng);
    assert_rng_checkpoint(model, "initialized", &preparation.rng);
    let data = &mut preparation.data;
    data.modpara.nmp_trans = data.modpara.nmp_trans.abs().max(1);
    data.modpara.vmc_calc_mode = 1;
    init_qp_weight(data);
    let complex = mvmc_core::get_all_complex_flag(data);
    let fsz = data.i_flg_orbital_general != 0;
    let n_qp = data.qp_weights.as_ref().unwrap().qp_full_weight.len();
    let mut state = mvmc_core::VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        n_qp,
        data.modpara.nvmc_sample as usize,
        complex,
        fsz,
    );
    if fsz {
        mvmc_core::slater_update::update_slater_elm_fsz(data, &mut state);
    } else {
        mvmc_core::slater_update::update_slater_elm(data, &mut state);
    }
    assert_eq!(
        data.modpara.n_data_qty_smp, 1,
        "reference checkpoint coverage"
    );
    let stats = match (fsz, complex) {
        (false, false) => mvmc_core::vmc_make_sample_real(data, &mut state, &mut preparation.rng),
        (false, true) => mvmc_core::vmc_make_sample(data, &mut state, &mut preparation.rng),
        (true, true) => {
            mvmc_core::sampling::driver::vmc_make_sample_fsz(data, &mut state, &mut preparation.rng)
        }
        (true, false) => {
            mvmc_core::sampling::vmc_make_sample_fsz_real(data, &mut state, &mut preparation.rng)
                .unwrap()
        }
    };
    assert_eq!(stats.saved, data.modpara.nvmc_sample as usize);
    assert_saved_trajectory(model, &state);
    assert_rng_checkpoint(model, "sample-0", &preparation.rng);
}

macro_rules! trajectory_test {
    ($name:ident, $index:expr) => {
        #[test]
        fn $name() {
            check_independent_sampling_trajectory(PHYSCAL_MODELS[$index]);
        }
    };
}

trajectory_test!(independent_trajectory_heisenberg_real, 0);
trajectory_test!(independent_trajectory_heisenberg_cmp, 1);
trajectory_test!(independent_trajectory_heisenberg_fsz, 2);
trajectory_test!(independent_trajectory_hubbard_real, 3);
trajectory_test!(independent_trajectory_hubbard_dh_real, 4);
trajectory_test!(independent_trajectory_kondo_real, 5);

#[test]
#[ignore = "optional PhysCal gate; set MVMC_RS_PHYSCAL_181=1 and explicitly run ignored tests"]
fn six_supported_physcal_models_preserve_fixed_values_flags_and_outputs() {
    require_gate("physcal-issue181", "MVMC_RS_PHYSCAL_181");
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("physcal-issue181", "Julia-mVMC checkout not found")
    });

    for model in PHYSCAL_MODELS {
        let (preparation, fixture) = prepare(&root, *model, 1);
        let before = preparation.data.clone();
        let (expected_values, expected_count) = reference_fixed_values(&fixture);
        assert_fixed_values(&before, &expected_values);
        assert_eq!(before.count_variational_parameters(), expected_count);
        assert_eq!(preparation.n_para_consumed, expected_count);
        assert!(before
            .optimization_flags
            .iter()
            .all(|flag| *flag == 0 || *flag == 1));
        assert!(
            before.inter_all_terms.is_empty(),
            "{} is InterAll",
            model.name
        );
        if model.mode == "fsz" {
            assert_ne!(before.i_flg_orbital_general, 0, "{}", model.name);
            assert_ne!(before.i_flg_orbital_parallel, 0, "{}", model.name);
            assert!(
                fixture.join("inputs/initial.def").is_file(),
                "FSZ overlay fixture missing"
            );
        }

        let out = output_dir("matrix", model.name);
        let _ = fs::remove_dir_all(&out);
        let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        assert_saved_trajectory(*model, &result.state);
        assert_fixed_values(&result.data, &expected_values);
        assert_eq!(
            fixed_values(&result.data),
            fixed_values(&before),
            "{} fixed-value preservation",
            model.name
        );
        assert_eq!(
            result.data.optimization_flags, before.optimization_flags,
            "{} optimization flags",
            model.name
        );
        assert_eq!(result.iterations, 1, "{} iterations", model.name);

        let expected_dir = fixture.join("expected");
        let expected_names = fs::read_dir(&expected_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>();
        let actual_names = fs::read_dir(&out)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual_names, expected_names,
            "{} output file set",
            model.name
        );
        for name in expected_names {
            assert_reference(&out.join(&name), &expected_dir.join(name));
        }
        let _ = fs::remove_dir_all(out);
        report_gate("physcal-issue181", GateStatus::Pass, model.name);
    }
}

#[test]
#[ignore = "optional PhysCal gate; set MVMC_RS_PHYSCAL_181=1 and explicitly run ignored tests"]
fn serial_physcal_indexes_multiple_samples_and_reruns_deterministically() {
    require_gate("physcal-issue181-contract", "MVMC_RS_PHYSCAL_181");
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("physcal-issue181-contract", "Julia-mVMC checkout not found")
    });
    let model = PHYSCAL_MODELS[0];
    let out = output_dir("rerun", model.name);
    let _ = fs::remove_dir_all(&out);

    let (mut preparation, _) = prepare(&root, model, 1);
    preparation.data.modpara.n_data_idx_start = 7;
    preparation.data.modpara.n_data_qty_smp = 2;
    let first = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
    assert_eq!(first.iterations, 2);
    let first_snapshot = snapshot(&out);
    assert!(first_snapshot
        .iter()
        .any(|(name, _)| name.contains("_007.")));
    assert!(first_snapshot
        .iter()
        .any(|(name, _)| name.contains("_008.")));

    let (mut rerun, _) = prepare(&root, model, 1);
    rerun.data.modpara.n_data_idx_start = 7;
    rerun.data.modpara.n_data_qty_smp = 2;
    let second = mvmc_core::vmc_phys_cal_to_dir(rerun, &out).unwrap();
    assert_eq!(second.state.electron_config, first.state.electron_config);
    assert_eq!(second.state.energy, first.state.energy);
    assert_eq!(snapshot(&out), first_snapshot);
    let _ = fs::remove_dir_all(out);
}

#[test]
#[ignore = "optional PhysCal gate; set MVMC_RS_PHYSCAL_181=1 and explicitly run ignored tests"]
fn non_interall_lanczos_modes_one_and_two_write_the_defined_files() {
    require_gate("physcal-issue181-lanczos", "MVMC_RS_PHYSCAL_181");
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("physcal-issue181-lanczos", "Julia-mVMC checkout not found")
    });

    for model_name in [
        "hubbard_chain_lanczos",
        "spin_chain_lanczos",
        "hubbard_chain_real",
    ] {
        let model = Model {
            name: model_name,
            mode: "real",
        };
        let (_, fixture) = prepare(&root, model, 1);
        for lanczos_mode in [1, 2] {
            let (mut preparation, _) = prepare(&root, model, 1);
            preparation.data.modpara.lanczos_mode = lanczos_mode;
            let (expected_values, expected_count) = reference_fixed_values(&fixture);
            assert_eq!(preparation.n_para_consumed, expected_count);
            assert_fixed_values(&preparation.data, &expected_values);
            assert!(preparation.data.inter_all_terms.is_empty());
            let out = output_dir(&format!("lanczos-{lanczos_mode}"), model.name);
            let _ = fs::remove_dir_all(&out);
            let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
            assert_fixed_values(&result.data, &expected_values);

            let expected_ls_names = fs::read_dir(fixture.join("expected"))
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| {
                    name.starts_with("zvo_ls_")
                        && (lanczos_mode == 2
                            || name.starts_with("zvo_ls_out_")
                            || name.starts_with("zvo_ls_qqqq_"))
                })
                .chain((lanczos_mode == 2).then_some("zvo_ls_cisajscktaltex_001.dat".to_owned()))
                .collect::<BTreeSet<_>>();
            let actual_ls_names = fs::read_dir(&out)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with("zvo_ls_"))
                .collect::<BTreeSet<_>>();
            assert_eq!(
                actual_ls_names, expected_ls_names,
                "{model_name} Lanczos mode {lanczos_mode} file set"
            );

            for name in ["zvo_ls_out_001.dat", "zvo_ls_qqqq_001.dat"] {
                assert_reference(&out.join(name), &fixture.join("expected").join(name));
            }
            for name in [
                "zvo_ls_cisajs_001.dat",
                "zvo_ls_cisajscktalt_001.dat",
                "zvo_ls_cisajscktaltex_001.dat",
            ] {
                let expected_path = fixture.join("expected").join(name);
                let should_exist = lanczos_mode == 2;
                assert_eq!(
                    out.join(name).is_file(),
                    should_exist,
                    "{model_name} {name}"
                );
                if should_exist {
                    if expected_path.is_file() {
                        assert_reference(&out.join(name), &expected_path);
                    } else {
                        let namelist =
                            fs::read_to_string(fixture.join("inputs/namelist.def")).unwrap();
                        assert!(
                            !namelist
                                .lines()
                                .any(|line| line.split_whitespace().next() == Some("TwoBodyGEx")),
                            "missing nonempty independent {name} reference for {model_name}"
                        );
                        assert!(
                            fs::read_to_string(out.join(name))
                                .unwrap()
                                .trim()
                                .is_empty(),
                            "{model_name} {name} should be empty when the fixture has no terms"
                        );
                    }
                }
            }
            let _ = fs::remove_dir_all(out);
        }
    }
}
