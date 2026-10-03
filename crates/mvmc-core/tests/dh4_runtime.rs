//! DH4 production support and canonical runtime-mode selection.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;
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
        let mut data = parse_expert_mode_files(root.join(format!("namelist_{name}.def"))).unwrap();
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
        assert_eq!(all_complex_flag(&data), row[0] != 0, "{line}");
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
        let reference = root.join(format!("sr_direct/{case}_store_runner"));
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
                    std::fs::read_to_string(julia_fixture::fixture_path(
                        &root,
                        reference
                            .strip_prefix(&root)
                            .unwrap()
                            .join(format!("step-3-{name}"))
                    ))
                    .unwrap()
                ),
                "{case} {name}"
            );
        }
        assert!(!result.output_dir.join("zqp_dh2_opt.dat").exists());
        assert!(!result.output_dir.join("zqp_dh4_opt.dat").exists());
        std::fs::remove_dir_all(result.output_dir).unwrap();
    }
}
