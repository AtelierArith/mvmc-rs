#[path = "../../../tests/support/c_orbital_rng.rs"]
mod c_orbital_rng;
#[path = "../../../tests/support/reference_slater.rs"]
mod reference_slater;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use mvmc_core::initial_params::{read_initial_def, read_opt_para_file};
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use num_complex::Complex64;
use reference_slater::declared_output;
use std::path::{Path, PathBuf};
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm")
}

#[test]
fn full_initial_and_strict_optimized_records_scatter_rbm_before_slater() {
    let mut data = parse_expert_mode_files(root().join("namelist_all.def")).unwrap();
    assert!(read_initial_def(&mut data, root().join("production/initial.def")).unwrap());
    assert_eq!(
        read_opt_para_file(&mut data, root().join("production/initial.def")).unwrap(),
        36
    );
    let mut offsets = [5; 9];
    for i in 1..9 {
        offsets[i] = offsets[i - 1] + 3;
    }
    data.visit_rbm_terms_mut(|s, t| {
        let idx = offsets[s] + t.idx() as usize;
        assert_eq!(
            t.value(),
            Complex64::new((idx + 1) as f64 / 8.0, -((idx + 1) as f64) / 16.0)
        );
    });
    for t in data.orbital_terms {
        let idx = 5 + 27 + t.idx as usize;
        assert_eq!(
            data.slater_params[t.idx as usize],
            Complex64::new((idx + 1) as f64 / 8.0, -((idx + 1) as f64) / 16.0)
        );
    }
}

#[test]
fn nine_optional_overlays_share_indices_follow_last_values_and_ignore_unknown_names() {
    let namelist = root().join("production/namelist_all.def");
    let mut data = parse_expert_mode_files(&namelist).unwrap();
    read_input_parameters(&mut data, &namelist).unwrap();
    let mut visits = 0;
    data.visit_rbm_terms_mut(|_, t| {
        visits += 1;
        let expected = match t.idx() {
            0 => Complex64::new(0.125, -0.0625),
            2 => Complex64::new(0.25, -0.125),
            _ => unreachable!(),
        };
        assert_eq!(t.value(), expected);
    });
    assert_eq!(visits, 18);
}

use mvmc_core::sampling::rbm::{
    log_rbm_ratio, log_rbm_val, make_rbm_cnt, set_rbm_diff, update_rbm_cnt_hopping, RbmConfig,
};
fn complex_line(line: &str) -> Vec<Complex64> {
    let words: Vec<u64> = line
        .split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    words
        .chunks_exact(2)
        .map(|w| Complex64::new(f64::from_bits(w[0]), f64::from_bits(w[1])))
        .collect()
}
fn exact(got: &[Complex64], expected: &[Complex64], context: &str) {
    let bits = |values: &[Complex64]| {
        values
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(got), bits(expected), "{context}");
}

#[test]
fn parsed_rbm_counter_incremental_ratios_and_all_derivatives_match_original_julia() {
    let fixture = std::fs::read_to_string(root().join("production/kernels.txt")).unwrap();
    let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
    while let Some(header) = lines.next() {
        let words: Vec<_> = header.split_whitespace().collect();
        let (case, mask) = (words[0], words[1].parse::<u32>().unwrap());
        let namelist = if case == "empty" {
            root().join("namelist_empty.def")
        } else {
            root().join(format!("production/namelist_{case}.def"))
        };
        let mut data = parse_expert_mode_files(&namelist).unwrap();
        if case != "empty" {
            read_input_parameters(&mut data, &namelist).unwrap();
        }
        let cfg = RbmConfig::from(&data);
        let occupation: Vec<_> = (0..6).map(|i| i64::from((mask >> i) & 1)).collect();
        let cnt = make_rbm_cnt(&occupation, &cfg);
        exact(&cnt, &complex_line(lines.next().unwrap()), header);
        let mut derivative = vec![Complex64::new(0.0, 0.0); 2 * data.count_rbm_parameters()];
        set_rbm_diff(&mut derivative, &cnt, &occupation, &cfg);
        exact(&derivative, &complex_line(lines.next().unwrap()), header);
        exact(
            &[log_rbm_val(&occupation, &cfg)],
            &complex_line(lines.next().unwrap()),
            header,
        );
        for spin in 0..2 {
            for ri in 0..3 {
                for rj in 0..3 {
                    let mut new = vec![Complex64::new(99.0, -99.0); cnt.len()];
                    update_rbm_cnt_hopping(&mut new, &cnt, ri, rj, spin, &cfg);
                    exact(
                        &new,
                        &complex_line(lines.next().unwrap()),
                        &format!("{header} hop {ri} {rj} {spin}"),
                    );
                    exact(
                        &[log_rbm_ratio(&new, &cnt, &cfg)],
                        &complex_line(lines.next().unwrap()),
                        header,
                    );
                }
            }
        }
    }
}

#[test]
fn rbm_initial_overlays_sync_and_rng_follow_source_phase_order() {
    use mvmc_expert_parsers::utils::parameter_init::{init_parameter, sync_modified_parameter};
    use sfmt19937::Sfmt19937Rng;
    let fixture = std::fs::read_to_string(root().join("production/loading.txt")).unwrap();
    let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
    while let Some(case) = lines.next() {
        let file = root().join(format!("production/namelist_{case}.def"));
        let mut data = parse_expert_mode_files(&file).unwrap();
        let mut rng = Sfmt19937Rng::new(11272);
        let snapshot = |d: &mut mvmc_expert_parsers::ExpertModeData| {
            let mut v = d.projection_parameters();
            d.visit_rbm_terms_mut(|_, t| v.push(t.value()));
            v.extend(
                d.orbital_terms
                    .iter()
                    .map(|t| d.slater_params[t.idx as usize]),
            );
            v
        };
        init_parameter(&mut data, &mut rng);
        exact(
            &snapshot(&mut data),
            &complex_line(lines.next().unwrap()),
            case,
        );
        if case == "all" {
            assert!(read_initial_def(&mut data, root().join("production/initial.def")).unwrap());
            assert_eq!(
                read_opt_para_file(&mut data, root().join("production/initial.def")).unwrap(),
                36
            );
        }
        exact(
            &snapshot(&mut data),
            &complex_line(lines.next().unwrap()),
            case,
        );
        read_input_parameters(&mut data, &file).unwrap();
        exact(
            &snapshot(&mut data),
            &complex_line(lines.next().unwrap()),
            case,
        );
        // The historical Julia record initializes only mapped Slater slots.
        // C initializes the complete declared array. Check its native SFMT
        // record with the same active RBM prefix and declared Slater flags.
        let historical_rng: Vec<u32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(historical_rng.len(), 624);
        assert_eq!(
            (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
            c_orbital_rng::declared_slater_rng(&data),
            "{case} C declared Slater initialization"
        );
        let c_slater = if case == "all" {
            include_str!("../../../tests/fixtures/orbital_general/c_declared_flags.txt")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect::<Vec<_>>()
                .chunks_exact(4)
                .find(|record| record[0] == "rbm_layout_loaded")
                .unwrap()
                .try_into()
                .unwrap()
        } else {
            c_orbital_rng::declared_slater_record(&data)
        };
        exact(&data.slater_params, &complex_line(c_slater[1]), case);
        sync_modified_parameter(&mut data, false);
        let normalized_slater = complex_line(c_slater[2]);
        exact(&data.slater_params, &normalized_slater, case);
        let mut expected = complex_line(lines.next().unwrap());
        expected.truncate(expected.len() - data.orbital_terms.len());
        expected.extend(
            data.orbital_terms
                .iter()
                .map(|term| normalized_slater[term.idx as usize]),
        );
        exact(&snapshot(&mut data), &expected, case);
        let before = snapshot(&mut data);
        let invalid = std::env::temp_dir().join(format!(
            "mvmc-rbm-invalid-{case}-{}.def",
            std::process::id()
        ));
        let original = std::fs::read_to_string(root().join("production/initial.def")).unwrap();
        std::fs::write(&invalid, original.replace("77", "broken")).unwrap();
        assert!(!read_initial_def(&mut data, &invalid).unwrap());
        assert!(read_opt_para_file(&mut data, &invalid).is_err());
        exact(&snapshot(&mut data), &before, case);
        std::fs::remove_file(invalid).unwrap();
    }
}

#[test]
fn public_rbm_namelist_runner_matches_source_outputs_and_parameter_order() {
    for case in [
        "rbm_real",
        "rbm_cmp",
        "rbm_general_cmp",
        "rbm_dh24_cmp",
        "rbm_fsz",
        "rbm_reference_cmp",
    ] {
        let reference = case == "rbm_reference_cmp";
        let path = if reference {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def")
        } else {
            root().join(format!("../c_orbital_inputs/namelist_{case}.def"))
        };
        for steps in [1, 3] {
            let mut config = mvmc_core::RunConfig::new(
                steps,
                if case == "rbm_real" {
                    "real"
                } else if case == "rbm_fsz" {
                    "fsz"
                } else {
                    "cmp"
                },
            );
            config.nsmp = Some(steps);
            if !reference {
                config.seed = Some(1);
            }
            let data = parse_expert_mode_files(&path).unwrap();
            let summary = mvmc_core::run_para_opt_from_namelist(&path, config).unwrap();
            assert_eq!(summary.status, 0);
            assert_eq!(summary.effective_nsteps, steps as usize);
            assert_eq!(summary.effective_nsmp, steps as usize);
            let fixture = root().join(format!("../sr_direct/{case}_store_runner"));
            for name in [
                "zvo_out.dat",
                "zvo_var.dat",
                "zqp_opt.dat",
                "zqp_gutzwiller_opt.dat",
                "zqp_jastrow_opt.dat",
                "zqp_orbital_opt.dat",
            ] {
                assert_eq!(
                    std::fs::read_to_string(summary.output_dir.join(name)).unwrap(),
                    declared_output(
                        &data,
                        name,
                        std::fs::read_to_string(fixture.join(format!("step-{steps}-{name}")))
                            .unwrap()
                    ),
                    "{case} {steps} {name}"
                );
            }
            assert!(!summary.output_dir.join("zqp_rbm_opt.dat").exists());
            std::fs::remove_dir_all(summary.output_dir).unwrap();
        }
    }
}
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
