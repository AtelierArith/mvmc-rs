//! DH2 definitions cannot bypass the production gate through library-owned data.
use mvmc_core::{vmc_para_opt, ExpertModeData, SingleProcessReducer, VmcOptimizationState};
use mvmc_expert_parsers::{parse_expert_mode_files, utils::parameter_init::all_complex_flag};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::path::Path;

#[test]
fn parsed_and_programmatic_dh2_fail_before_mutation_rng_or_matrix_access() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
    for name in [
        "orbital_first",
        "dh_first",
        "alias",
        "real",
        "empty",
        "replacement",
    ] {
        for only_field in 0..=4 {
            let mut data =
                parse_expert_mode_files(root.join(format!("namelist_{name}.def"))).unwrap();
            if only_field > 0 {
                data.namelist.clear();
                if only_field != 1 {
                    data.doublon_holon_2site_indices.clear();
                }
                if only_field != 2 {
                    data.doublon_holon_2site_params.clear();
                }
                if only_field != 3 {
                    data.doublon_holon_2site_opt_flags.clear();
                }
                if only_field != 4 {
                    data.doublon_holon_2site_complex = false;
                }
            }
            // Empty or real definitions can lack the selected isolated field.
            if only_field > 0
                && data.doublon_holon_2site_indices.is_empty()
                && data.doublon_holon_2site_params.is_empty()
                && data.doublon_holon_2site_opt_flags.is_empty()
                && !data.doublon_holon_2site_complex
            {
                continue;
            }
            let before = data.clone();
            let mut rng = Sfmt19937Rng::new(11272);
            let mut probe = Sfmt19937Rng::new(11272);
            let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
            let error = vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                None,
                &SingleProcessReducer,
                mvmc_core::OptimizationOptions::default(),
            )
            .unwrap_err();
            assert!(
                error.contains("DH2") && error.contains("issue #24"),
                "{name} {only_field}: {error}"
            );
            assert_eq!(data.modpara, before.modpara);
            assert_eq!(data.optimization_flags, before.optimization_flags);
            assert_eq!(data.orbital_terms, before.orbital_terms);
            assert_eq!(
                data.doublon_holon_2site_indices,
                before.doublon_holon_2site_indices
            );
            assert_eq!(
                data.doublon_holon_2site_params,
                before.doublon_holon_2site_params
            );
            assert_eq!(
                data.doublon_holon_2site_opt_flags,
                before.doublon_holon_2site_opt_flags
            );
            assert_eq!(
                data.doublon_holon_2site_complex,
                before.doublon_holon_2site_complex
            );
            for _ in 0..624 {
                assert_eq!(rng.gen_rand32(), probe.gen_rand32());
            }
        }
    }
}

#[test]
fn dh2_runtime_mode_matches_original_flags_declarations_and_loaded_values() {
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
            row[3] != 0,
            "{line}"
        );
        // Initializing Slater parameters uses declarations, never loaded values
        // or the optional runtime override; changing this would alter draw count.
        assert_eq!(all_complex_flag(&data), row[0] != 0, "{line}");
    }
}
