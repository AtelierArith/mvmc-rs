//! Runtime rejection must precede initialization, RNG consumption, and output.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
use mvmc_core::{vmc_para_opt, ExpertModeData, SingleProcessReducer, VmcOptimizationState};
use mvmc_expert_parsers::parsers::modpara::parse_modpara_content;
use sfmt19937::Sfmt19937Rng;

#[test]
fn rejects_modpara_solver_controls_instead_of_discarding_them() {
    for (text, expected) in [
        ("NSplitSize 0", "NSplitSize must be >= 1"),
        ("NSplitSize -1", "NSplitSize must be >= 1"),
        ("NLanczosMode 3", "NLanczosMode must be"),
        ("NLanczosMode -1", "NLanczosMode must be"),
        ("NLanczosMode 1", "parameter optimization"),
        ("NSRCG 2", "NSRCG >= 2"),
        ("useDiagScale 1", "useDiagScale"),
        ("RescaleSmat 1", "RescaleSmat"),
        ("NVMCCalMode 1", "PhysCal"),
        ("NSplitSize 2", "issue #36"),
        ("NVMCSample -1", "NVMCSample must be positive"),
    ] {
        let mut data = ExpertModeData::new();
        data.modpara = parse_modpara_content(text);
        let before = data.clone();
        let mut rng = Sfmt19937Rng::new(1);
        let mut probe = Sfmt19937Rng::new(1);
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
        assert!(error.contains(expected), "{text}: {error}");
        assert_eq!(data.modpara, before.modpara);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        assert_eq!(data.orbital_terms, before.orbital_terms);
        assert_eq!(data.slater_params, before.slater_params);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), probe.gen_rand32());
        }
    }
}

#[test]
fn unported_sections_cannot_silently_change_the_model() {
    for kind in ["InterAll", "TwoBodyGEx", "SpinJastrow"] {
        let mut data = ExpertModeData::new();
        data.namelist.push((kind.into(), "missing.def".into()));
        let error = mvmc_core::validation::validate_para_opt(&data).unwrap_err();
        assert!(error.contains(kind), "{kind}: {error}");
    }
}

#[test]
fn active_serial_opttrans_passes_validation_including_single_sector_payloads() {
    for count in [1, 2] {
        let mut data = ExpertModeData::new();
        data.modpara.nmp_trans = 1;
        data.n_qp_opt_trans = count;
        data.opt_trans = vec![num_complex::Complex64::new(0.5, 0.25); count as usize];
        data.namelist
            .push(("OptTrans".into(), "opttrans.def".into()));
        data.namelist
            .push(("InOptTrans".into(), "optional.def".into()));
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

#[test]
fn retained_interall_payload_is_rejected_before_initialization_with_or_without_namelist() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/interall");
    for has_namelist in [true, false] {
        let mut data =
            historical_orbital_model::historical_kernel_model(root.join("namelist.def")).unwrap();
        if !has_namelist {
            data.namelist.clear();
        }
        assert!(!mvmc_core::run::get_all_complex_flag(&data));
        let before = data.clone();
        let mut probe = Sfmt19937Rng::new(1);
        let mut rng = Sfmt19937Rng::new(1);
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
            error.contains("InterAll") && error.contains("issue #23"),
            "{error}"
        );
        assert_eq!(data.modpara, before.modpara);
        assert_eq!(data.orbital_terms, before.orbital_terms);
        assert_eq!(data.slater_params, before.slater_params);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        assert_eq!(data.inter_all_terms, before.inter_all_terms);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), probe.gen_rand32());
        }
    }
}

#[test]
fn real_fsz_pairhop_is_not_rejected_by_issue_43_gate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/pairhop/namelist.def");
    for has_namelist in [true, false] {
        let mut data = historical_orbital_model::historical_kernel_model(&root).unwrap();
        data.i_flg_orbital_general = 1;
        if !has_namelist {
            data.namelist.clear();
        }
        let mut rng = Sfmt19937Rng::new(1);
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
            !error.contains("real FSZ") || !error.contains("issue #43"),
            "{error}"
        );
    }
}

#[test]
fn interall_mode_and_invalid_spin_failures_precede_rng_consumption_and_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/interall/spin_chain/namelist.def");
    for mode in ["real", "spin"] {
        let mut data = historical_orbital_model::historical_kernel_model(&root).unwrap();
        if mode == "real" {
            data.complex_flags = vec![0];
        } else {
            data.inter_all_terms[0].spin2 = 2;
        }
        let before = data.clone();
        let mut rng = Sfmt19937Rng::new(1);
        let mut probe = Sfmt19937Rng::new(1);
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
        let expected = if mode == "real" {
            "issue #43"
        } else {
            "spin2 must be 0 or 1"
        };
        assert!(error.contains(expected), "{error}");
        assert_eq!(data.modpara, before.modpara);
        assert_eq!(data.orbital_terms, before.orbital_terms);
        assert_eq!(data.slater_params, before.slater_params);
        assert_eq!(data.inter_all_terms, before.inter_all_terms);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), probe.gen_rand32());
        }
    }
}

#[test]
fn supported_overlay_sections_pass_runtime_validation_even_when_optional_files_are_absent() {
    let dir = std::env::temp_dir().join(format!("mvmc-optional-overlay-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("modpara.def"),
        "Nsite 2\nNElec 1\nNMPTrans 1\nNVMCSample 1\nNVMCInterval 1\n",
    )
    .unwrap();
    let mut namelist = "ModPara modpara.def\n".to_owned();
    for kind in [
        "InGutzwiller",
        "InJastrow",
        "InOrbital",
        "InOrbitalAntiParallel",
        "InOrbitalParallel",
        "InOrbitalGeneral",
        "InOptTrans",
    ] {
        namelist.push_str(&format!("{kind} absent.def\n"));
    }
    std::fs::write(dir.join("namelist.def"), namelist).unwrap();
    let data = historical_orbital_model::historical_kernel_model(dir.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    mvmc_core::validation::validate_para_opt(&data).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_opttrans_rejection_matches_canonical_support_matrix_without_rng_use() {
    let fixture = include_str!("../../../tests/fixtures/opttrans/grouped.txt");
    for line in fixture.lines().filter(|s| !s.starts_with('#')) {
        let fields = line
            .split_whitespace()
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let mut data = ExpertModeData::new();
        data.modpara.nsplit_size = 2;
        data.n_qp_opt_trans = fields[0] as i64;
        data.opt_trans = vec![num_complex::Complex64::new(0.5, 0.25); fields[1]];
        data.qp_opt_trans = vec![vec![]; fields[2]];
        let before = data.clone();
        let mut rng = Sfmt19937Rng::new(1);
        let mut probe = rng.clone();
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
        if fields[3] == 1 {
            assert!(
                error.contains("OptTrans") && error.contains("not supported"),
                "{line}: {error}"
            );
        } else {
            assert!(error.contains("issue #36"), "{line}: {error}");
        }
        assert_eq!(data.opt_trans, before.opt_trans);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), probe.gen_rand32());
        }
    }
}
