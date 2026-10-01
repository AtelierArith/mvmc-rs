//! Reject parsed and programmatic DH4 until its full production path passes parity.
use mvmc_core::ExpertModeData;
use mvmc_expert_parsers::{parse_expert_mode_files, utils::parameter_init::all_complex_flag};
use num_complex::Complex64;
use std::path::Path;

#[test]
fn parsed_dh4_sections_remain_rejected_before_initialization_or_output() {
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
        assert!(
            mvmc_core::validation::validate_para_opt(&data)
                .unwrap_err()
                .contains("issue #25"),
            "{name}"
        );
        // Empty real definitions have no programmatic representation after clearing the namelist.
        if name != "empty_real" {
            data.namelist.clear();
            assert!(
                mvmc_core::validation::validate_para_opt(&data)
                    .unwrap_err()
                    .contains("issue #25"),
                "{name}"
            );
        }
    }
}

#[test]
fn dh4_runtime_mode_matches_original_flags_declarations_and_loaded_values() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
    let fixture = std::fs::read_to_string(root.join("mode.txt")).unwrap();
    for line in fixture.lines().filter(|l| !l.starts_with('#')) {
        let row: Vec<i64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut data = ExpertModeData::new();
        data.doublon_holon_4site_complex = row[0] != 0;
        data.doublon_holon_4site_params = vec![Complex64::new(0.125, row[1] as f64 / 4.0)];
        if row[2] != -1 {
            data.complex_flags = vec![row[2]];
        }
        assert_eq!(
            mvmc_core::run::get_all_complex_flag(&data),
            row[3] != 0,
            "{line}"
        );
        assert_eq!(all_complex_flag(&data), row[0] != 0, "{line}");
        assert!(mvmc_core::validation::validate_para_opt(&data)
            .unwrap_err()
            .contains("issue #25"));
    }
}

#[test]
fn dh4_runner_rejection_preserves_parameters_rng_and_creates_no_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
    for name in [
        "orbital_first",
        "real",
        "empty",
        "empty_real",
        "ap_parallel",
        "general",
    ] {
        let path = root.join(format!("namelist_{name}.def"));
        let mut data = parse_expert_mode_files(&path).unwrap();
        for value in &mut data.doublon_holon_4site_params {
            *value = Complex64::new(0.25, -0.125);
        }
        let before = data.clone();
        let mut rng = sfmt19937::Sfmt19937Rng::new(11272);
        let mut probe = sfmt19937::Sfmt19937Rng::new(11272);
        let mut state = mvmc_core::VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
        let output =
            std::env::temp_dir().join(format!("mvmc-dh4-gate-{}-{name}", std::process::id()));
        assert!(!output.exists());
        let error = mvmc_core::vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output),
            &mvmc_core::SingleProcessReducer,
            mvmc_core::OptimizationOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("issue #25"), "{name}: {error}");
        assert_eq!(data.modpara, before.modpara);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        assert_eq!(data.projection_parameters(), before.projection_parameters());
        assert_eq!(data.orbital_terms, before.orbital_terms);
        assert_eq!(
            data.doublon_holon_4site_indices,
            before.doublon_holon_4site_indices
        );
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), probe.gen_rand32());
        }
        assert!(!output.exists());
        let error = mvmc_core::run_para_opt_from_namelist(
            &path,
            mvmc_core::RunConfig {
                output_dir: Some(output.clone()),
                ..mvmc_core::RunConfig::new(1, "real")
            },
        )
        .unwrap_err();
        assert!(error.contains("issue #25"), "{name}: {error}");
        assert!(!output.exists());
    }
}
