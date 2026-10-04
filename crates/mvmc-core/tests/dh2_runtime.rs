//! DH2 production support and canonical complex-mode selection.
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
fn supported_dh2_sections_pass_runtime_validation_with_or_without_namelist() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
    for name in [
        "orbital_first",
        "dh_first",
        "alias",
        "real",
        "empty",
        "replacement",
    ] {
        // A repeated archived keyword is component-sequence metadata, not C
        // loader acceptance. The helper checks duplicate rejection, separately
        // reads the sections and builds a valid single-keyword final model.
        let mut data = historical_component_sequence::model(
            &root.join(format!("namelist_{name}.def")),
            "DH2",
            |path| parse_expert_mode_files(path),
        );
        // Use the same final single-keyword metadata as the valid loader stage,
        // not the archived component sequence preserved by the parser helper.
        let final_definition = data.namelist.iter().rposition(|(kind, _)| kind == "DH2");
        // Archived zero-count DH is a no-DH programmatic layout, not a valid
        // required C definition. The shared helper separately asserts rejection.
        let active = data.projection_layout().n_dh2 != 0;
        data.namelist = data
            .namelist
            .iter()
            .enumerate()
            .filter(|(index, (kind, _))| {
                kind != "DH2" || (active && Some(*index) == final_definition)
            })
            .map(|(_, entry)| entry.clone())
            .collect();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        data.namelist.clear();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

#[test]
fn dh2_runtime_mode_uses_declarations_and_ignores_loaded_imaginary_values() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
    let fixture = std::fs::read_to_string(root.join("mode.txt")).unwrap();
    for line in fixture.lines().filter(|l| !l.starts_with('#')) {
        let row: Vec<i64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut data = ExpertModeData::new();
        data.doublon_holon_2site_complex = row[0] != 0;
        data.doublon_holon_2site_params = vec![Complex64::new(0.125, row[1] as f64 / 4.0)];
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
        // Initializing Slater parameters uses declarations, never loaded values
        // or the optional runtime override; changing this would alter draw count.
        assert_eq!(all_complex_flag(&data).unwrap(), row[0] != 0, "{line}");
    }
}

#[test]
fn public_dh2_runners_load_nonzero_overlays_and_match_original_direct_store_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for mode in ["real", "cmp", "fsz"] {
        let namelist = if mode == "cmp" {
            root.join("c_orbital_inputs/namelist_dh2_cmp.def")
        } else {
            root.join(format!("dh2/production_{mode}/namelist.def"))
        };
        let data = parse_expert_mode_files(&namelist).unwrap();
        let result = mvmc_core::run_para_opt_from_namelist(
            &namelist,
            mvmc_core::RunConfig {
                nsmp: Some(3),
                ..mvmc_core::RunConfig::new(3, mode)
            },
        )
        .unwrap();
        let reference = if native_fsz_fixture::directory(&root).is_some() && mode == "fsz" {
            native_fsz_fixture::directory(&root)
                .unwrap()
                .join("sr_direct/dh2_fsz_store_runner")
        } else {
            root.join(format!("sr_direct/dh2_{mode}_store_runner"))
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
                format!("{mode} {name}"),
            );
        }
        let actual_var = std::fs::read_to_string(result.output_dir.join("zvo_var.dat")).unwrap();
        numerical_comparison::assert_numeric_text(
            &actual_var,
            &std::fs::read_to_string(
                root.join(format!("ctest_dh_windows/dh2_{mode}/zvo_c_slots_var.dat")),
            )
            .unwrap(),
            1e-12,
            1e-12,
            &[],
            format!("{mode} independent full C-declared pre-SR var"),
        );
        // Keep the historical Julia subsequence as separate evidence. Its
        // omitted DH coefficients are checked by the complete oracle above.
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
            format!("{mode} historical Julia var subsequence only"),
        );
        // Actual C avevar.c output from independently captured post-SR/sync
        // histories, not the archived Julia final-parameter writer extension.
        let mut window_outputs = vec![
            "zqp_opt.dat",
            "zqp_gutzwiller_opt.dat",
            "zqp_jastrow_opt.dat",
            "zqp_doublonHolon2site_opt.dat",
        ];
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
                    root.join(format!("ctest_dh_windows/dh2_{mode}/native_{name}")),
                )
                .unwrap(),
                1e-12,
                1e-12,
                if name == "zqp_opt.dat" { &[] } else { &[0] },
                format!("{mode} independent C window {name}"),
            );
        }
        assert!(!result.output_dir.join("zqp_dh2_opt.dat").exists());
        std::fs::remove_dir_all(result.output_dir).unwrap();
    }
}
