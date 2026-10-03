//! DH2 production support and canonical complex-mode selection.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;
#[path = "../../../tests/support/native_fsz_fixture.rs"]
mod native_fsz_fixture;
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
        let mut data = parse_expert_mode_files(root.join(format!("namelist_{name}.def"))).unwrap();
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
            mvmc_core::run::get_all_complex_flag(&data),
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
        assert_eq!(all_complex_flag(&data), row[0] != 0, "{line}");
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
        for name in [
            "zvo_out.dat",
            "zvo_var.dat",
            "zqp_opt.dat",
            "zqp_gutzwiller_opt.dat",
            "zqp_jastrow_opt.dat",
            "zqp_orbital_opt.dat",
        ] {
            assert_eq!(
                std::fs::read_to_string(result.output_dir.join(name)).unwrap(),
                declared_output(
                    &data,
                    name,
                    std::fs::read_to_string(native_fsz_fixture::resolve(
                        julia_fixture::fixture_path(
                            &root,
                            reference
                                .strip_prefix(&root)
                                .unwrap()
                                .join(format!("step-3-{name}"))
                        ),
                    ))
                    .unwrap()
                ),
                "{mode} {name}"
            );
        }
        assert!(!result.output_dir.join("zqp_dh2_opt.dat").exists());
        std::fs::remove_dir_all(result.output_dir).unwrap();
    }
}
