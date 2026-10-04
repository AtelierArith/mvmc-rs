//! DH4 production support and canonical runtime-mode selection.
#[path = "../../mvmc-expert-parsers/tests/support/historical_component_sequence.rs"]
mod historical_component_sequence;
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;
#[path = "../../../tests/support/native_fsz_fixture.rs"]
mod native_fsz_fixture;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
#[path = "../../../tests/support/reference_slater.rs"]
mod reference_slater;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use mvmc_core::ExpertModeData;
use mvmc_expert_parsers::utils::parameter_init::all_complex_flag;
use num_complex::Complex64;
use reference_slater::declared_output;
use std::path::Path;

#[test]
fn parsed_and_programmatic_dh4_sections_pass_runtime_validation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
    for name in [
        "orbital_first",
        "dh_first",
        "alias",
        "real",
        "empty",
        "replacement",
        "reverse_replacement",
        "combined_real",
        "only_dh4_complex",
        "empty_last",
        "empty_real",
        "ap_parallel",
        "general",
    ] {
        // Runtime validity of the final historical model is separate from
        // rejection of its archived repeated filename keywords.
        let mut data = historical_component_sequence::model(
            &root.join(format!("namelist_{name}.def")),
            "DH4",
            |path| parse_expert_mode_files(path),
        );
        let final_definition = data.namelist.iter().rposition(|(kind, _)| kind == "DH4");
        data.namelist = data
            .namelist
            .iter()
            .enumerate()
            .filter(|(index, (kind, _))| kind != "DH4" || Some(*index) == final_definition)
            .map(|(_, entry)| entry.clone())
            .collect();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        data.namelist.clear();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

#[test]
fn dh4_runtime_mode_uses_declarations_and_ignores_loaded_imaginary_values() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
    let fixture = std::fs::read_to_string(root.join("mode.txt")).unwrap();
    for line in fixture.lines().filter(|l| !l.starts_with('#')) {
        let row: Vec<i64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut data = ExpertModeData::new();
        data.modpara.nmp_trans = 1;
        data.doublon_holon_4site_complex = row[0] != 0;
        data.doublon_holon_4site_params = vec![Complex64::new(0.125, row[1] as f64 / 4.0)];
        if row[2] != -1 {
            data.complex_flags = vec![row[2]];
        }
        assert_eq!(
            mvmc_core::run::get_all_complex_flag(&data).unwrap(),
            // C uses definition flags; the archived Julia result in row[3]
            // also inferred the mode from loaded imaginary coefficients.
            if row[2] == -1 {
                row[0] != 0
            } else {
                row[2] != 0
            },
            "{line}"
        );
        assert_eq!(all_complex_flag(&data).unwrap(), row[0] != 0, "{line}");
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

#[test]
fn public_dh4_and_combined_runners_load_overlays_and_match_original_direct_store_outputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for case in [
        "dh4_real",
        "dh4_cmp",
        "dh4_fsz",
        "dh24_real",
        "dh24_cmp",
        "dh24_fsz",
    ] {
        let namelist = if matches!(case, "dh4_cmp" | "dh24_cmp") {
            root.join(format!("c_orbital_inputs/namelist_{case}.def"))
        } else {
            root.join(format!("dh4/production_{case}/namelist.def"))
        };
        let data = parse_expert_mode_files(&namelist).unwrap();
        let result = mvmc_core::run_para_opt_from_namelist(
            &namelist,
            mvmc_core::RunConfig {
                nsmp: Some(3),
                ..mvmc_core::RunConfig::new(3, case.rsplit_once('_').unwrap().1)
            },
        )
        .unwrap();
        let reference = if native_fsz_fixture::directory(&root).is_some() && case.ends_with("fsz") {
            native_fsz_fixture::directory(&root)
                .unwrap()
                .join(format!("sr_direct/{case}_store_runner"))
        } else {
            root.join(format!("sr_direct/{case}_store_runner"))
        };
        {
            let name = "zvo_out.dat";
            // Three SR steps: bounded accumulated solve/kernel roundoff. The
            // unit runner checkpoint gates still verify exact configurations/RNG.
            numerical_comparison::assert_numeric_text(
                &(std::fs::read_to_string(result.output_dir.join(name)).unwrap()),
                &(declared_output(
                    &data,
                    name,
                    julia_fixture::read_text(native_fsz_fixture::resolve(
                        julia_fixture::fixture_path(
                            &root,
                            reference
                                .strip_prefix(&root)
                                .unwrap()
                                .join(format!("step-3-{name}")),
                        ),
                    ))
                    .unwrap(),
                )),
                1e-12,
                1e-12,
                if name.starts_with("zqp_") && name != "zqp_opt.dat" {
                    &[0]
                } else {
                    &[]
                },
                format!("{case} {name}"),
            );
        }
        let actual_var = std::fs::read_to_string(result.output_dir.join("zvo_var.dat")).unwrap();
        numerical_comparison::assert_numeric_text(
            &actual_var,
            &std::fs::read_to_string(
                root.join(format!("ctest_dh_windows/{case}/zvo_c_slots_var.dat")),
            )
            .unwrap(),
            1e-12,
            1e-12,
            &[],
            format!("{case} independent full C-declared pre-SR var"),
        );
        let layout = data.projection_layout();
        let start = 6 + 3 * (layout.n_gutzwiller + layout.n_jastrow);
        let end = start + 3 * (6 * layout.n_dh2 + 10 * layout.n_dh4);
        let historical_subset = actual_var
            .lines()
            .map(|line| {
                let fields: Vec<_> = line.split_whitespace().collect();
                assert_eq!(fields.len(), 6 + 3 * data.count_variational_parameters());
                assert!(end <= fields.len());
                fields[..start]
                    .iter()
                    .chain(&fields[end..])
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let historical =
            julia_fixture::read_text(native_fsz_fixture::resolve(julia_fixture::fixture_path(
                &root,
                reference
                    .strip_prefix(&root)
                    .unwrap()
                    .join("step-3-zvo_var.dat"),
            )))
            .unwrap();
        numerical_comparison::assert_numeric_text(
            &historical_subset,
            &declared_output(&data, "zvo_var.dat", historical),
            1e-12,
            1e-12,
            &[],
            format!("{case} historical Julia var subsequence only"),
        );
        // Native C aggregation/formatting of independent Julia/C-contract
        // histories; no mapped-row adaptation or Rust-generated expectation.
        let mut window_outputs = vec![
            "zqp_opt.dat",
            "zqp_gutzwiller_opt.dat",
            "zqp_jastrow_opt.dat",
            "zqp_doublonHolon4site_opt.dat",
        ];
        if case.starts_with("dh24_") {
            window_outputs.push("zqp_doublonHolon2site_opt.dat");
        }
        if data.i_flg_orbital_general == 0 {
            window_outputs.push("zqp_orbital_opt.dat");
        } else if data.i_flg_orbital_parallel != 0 {
            window_outputs.extend([
                "zqp_orbitalAntiParallel_opt.dat",
                "zqp_orbitalParallel_opt.dat",
            ]);
        } else {
            window_outputs.push("zqp_orbital_general_opt.dat");
        }
        for name in window_outputs {
            numerical_comparison::assert_numeric_text(
                &std::fs::read_to_string(result.output_dir.join(name)).unwrap(),
                &std::fs::read_to_string(
                    root.join(format!("ctest_dh_windows/{case}/native_{name}")),
                )
                .unwrap(),
                1e-12,
                1e-12,
                if name == "zqp_opt.dat" { &[] } else { &[0] },
                format!("{case} independent C window {name}"),
            );
        }
        assert!(!result.output_dir.join("zqp_dh2_opt.dat").exists());
        assert!(!result.output_dir.join("zqp_dh4_opt.dat").exists());
        std::fs::remove_dir_all(result.output_dir).unwrap();
    }
}
