//! Runtime rejection must precede initialization, RNG consumption, and output.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
use mvmc_core::{
    vmc_para_opt, ExpertModeData, Reducer, SingleProcessReducer, VmcOptimizationState,
};
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
        // A valid unrelated projection setting prevents default zero from
        // masking the deliberately invalid option under test.
        data.modpara = parse_modpara_content(&format!("NMPTrans 1\n{text}"));
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
fn rejects_unsupported_lanczos_physcal_combinations_before_sampling() {
    let mut data = ExpertModeData::new();
    data.modpara.lanczos_mode = 1;
    data.modpara.nmp_trans = 1;
    data.i_flg_orbital_general = 1;
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(error.contains("FSZ/general") && error.contains("issue #31"));

    data.i_flg_orbital_general = 0;
    data.modpara.nsplit_size = 2;
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(error.contains("NSplitSize") && error.contains("issue #31"));
}

#[test]
fn rejects_spin_changing_lanczos_operators_before_sampling() {
    let mut data = ExpertModeData::new();
    data.modpara.lanczos_mode = 1;
    data.modpara.nmp_trans = 1;
    data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
        site1: 0,
        spin1: mvmc_expert_parsers::Spin::Up,
        site2: 1,
        spin2: mvmc_expert_parsers::Spin::Down,
        value: num_complex::Complex64::new(1.0, 0.0),
    });
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(
        error.contains("spin-changing") && error.contains("issue #31"),
        "{error}"
    );
}

#[test]
fn rejects_duplicate_mode2_one_body_entries_without_factored_green() {
    let mut data = ExpertModeData::new();
    data.modpara.lanczos_mode = 2;
    data.modpara.nmp_trans = 1;
    let term = mvmc_expert_parsers::GreenOneTerm {
        site1: 0,
        spin1: mvmc_expert_parsers::Spin::Up,
        site2: 1,
        spin2: mvmc_expert_parsers::Spin::Up,
    };
    data.green_one_terms = vec![term, term];
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(
        error.contains("duplicate") && error.contains("issue #32"),
        "{error}"
    );
}

#[test]
fn unported_sections_cannot_silently_change_the_model() {
    for kind in ["TwoBodyGEx", "SpinJastrow"] {
        let mut data = ExpertModeData::new();
        data.modpara.nmp_trans = 1;
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
fn normal_spin_changing_interall_is_rejected_before_initialization_with_or_without_namelist() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/interall");
    for has_namelist in [true, false] {
        let mut data =
            historical_orbital_model::historical_kernel_model(root.join("namelist.def")).unwrap();
        // The archived kernel fixture omits projection settings. Supply the
        // accepted C identity count, keeping its actual InterAll terms intact.
        data.modpara.nmp_trans = 1;
        if !has_namelist {
            data.namelist.clear();
        }
        assert!(!mvmc_core::run::get_all_complex_flag(&data).unwrap());
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
            error.contains("InterAll") && error.contains("spin-conserving"),
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
fn normal_interall_passes_runtime_validation_in_real_and_complex_modes() {
    for complex in [false, true] {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.modpara.nvmc_sample = 1;
        data.modpara.nvmc_interval = 1;
        data.modpara.nmp_trans = 1;
        data.complex_flags = vec![i64::from(complex)];
        data.inter_all_terms
            .push(mvmc_expert_parsers::InterAllTerm {
                site0: 0,
                spin0: 1,
                site1: 1,
                spin1: 1,
                site2: 1,
                spin2: 0,
                site3: 0,
                spin3: 0,
                value: num_complex::Complex64::new(0.25, -0.375),
                is_complex: true,
            });
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

#[test]
fn interall_rejects_invalid_sites_and_fixed_twosz_spin_changes_as_in_c() {
    for general in [0, 1] {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.modpara.nvmc_sample = 1;
        data.modpara.nvmc_interval = 1;
        data.modpara.nmp_trans = 1;
        data.i_flg_orbital_general = general;
        data.inter_all_terms
            .push(mvmc_expert_parsers::InterAllTerm {
                site0: 0,
                spin0: 0,
                site1: 1,
                spin1: 0,
                site2: 1,
                spin2: 1,
                site3: 0,
                spin3: 1,
                value: num_complex::Complex64::new(0.25, 0.0),
                is_complex: false,
            });
        for site in [-1, 2] {
            data.inter_all_terms[0].site0 = site;
            let error = mvmc_core::validation::validate_para_opt(&data).unwrap_err();
            assert!(
                error.contains("InterAll") && error.contains("site"),
                "{error}"
            );
        }
        data.inter_all_terms[0].site0 = 0;
        data.inter_all_terms[0].spin1 = 1;
        data.inter_all_terms[0].spin3 = 0;
        data.modpara.two_sz = 0;
        let error = mvmc_core::validation::validate_para_opt(&data).unwrap_err();
        assert!(error.contains("spin-conserving"), "{error}");
        data.modpara.two_sz = -1;
        assert_eq!(
            mvmc_core::validation::validate_para_opt(&data).is_ok(),
            general != 0
        );
    }
}

#[test]
fn complex_fixed_sz_interall_passes_runtime_validation() {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 1;
    data.modpara.nvmc_sample = 1;
    data.modpara.nvmc_interval = 1;
    data.modpara.nmp_trans = 1;
    data.i_flg_orbital_general = 1;
    data.complex_flags = vec![1];
    data.inter_all_terms
        .push(mvmc_expert_parsers::InterAllTerm {
            site0: 0,
            spin0: 0,
            site1: 1,
            spin1: 0,
            site2: 1,
            spin2: 1,
            site3: 0,
            spin3: 1,
            value: num_complex::Complex64::new(0.25, 0.0),
            is_complex: false,
        });
    let result = mvmc_core::validation::validate_para_opt(&data);
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn real_fixed_sz_interall_passes_runtime_validation() {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 1;
    data.modpara.nvmc_sample = 1;
    data.modpara.nvmc_interval = 1;
    data.modpara.nmp_trans = 1;
    data.i_flg_orbital_general = 1;
    data.complex_flags = vec![0];
    data.inter_all_terms
        .push(mvmc_expert_parsers::InterAllTerm {
            site0: 0,
            spin0: 0,
            site1: 1,
            spin1: 0,
            site2: 1,
            spin2: 1,
            site3: 0,
            spin3: 1,
            value: num_complex::Complex64::new(0.25, 0.0),
            is_complex: false,
        });
    assert!(mvmc_core::validation::validate_para_opt(&data).is_ok());
}

#[test]
fn real_fsz_pairhop_is_not_rejected_by_issue_43_gate() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pairhop");
    // The archived PairHop helper input declares one row but contains seven
    // permissive rows. It is not a supported C definition file. Parse the
    // unchanged wavefunction inputs separately, then compose its public helper
    // result explicitly; do not relax production parsing or shared helpers.
    let directory = (0..)
        .find_map(|attempt| {
            let path = std::env::temp_dir().join(format!(
                "issue264-pairhop-runtime-{}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("exclusive PairHop runtime input: {error}"),
            }
        })
        .unwrap();
    let namelist = directory.join("namelist.def");
    std::fs::write(
        &namelist,
        format!(
            "ModPara {}\nOrbital {}\n",
            root.join("../interall/modpara.def").display(),
            root.join("../interall/orbital.def").display()
        ),
    )
    .unwrap();
    let model = historical_orbital_model::historical_kernel_model(&namelist);
    std::fs::remove_dir_all(&directory).unwrap();
    let mut model = model.unwrap();
    assert!(model.pair_hop_terms.is_empty());
    let component =
        mvmc_expert_parsers::parsers::pairhop::parse_pairhop_def(root.join("parser_cases.def"))
            .unwrap();
    assert!(component.is_success());
    model.pair_hop_terms = component.terms;
    // Retain the original metadata branch as well as the archived payload;
    // this does not send its unsupported definition back through the loader.
    model.namelist = mvmc_expert_parsers::utils::file::parse_namelist_content(
        &std::fs::read_to_string(root.join("namelist.def")).unwrap(),
    );
    for has_namelist in [true, false] {
        let mut data = model.clone();
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
    for _ in ["spin"] {
        let mut data = historical_orbital_model::historical_kernel_model(&root).unwrap();
        data.inter_all_terms[0].spin2 = 2;
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
        assert!(error.contains("spin2 must be 0 or 1"), "{error}");
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

#[test]
fn grouped_runtime_matrix_accepts_normal_physcal_and_rejects_unsupported_scopes() {
    let mut data = ExpertModeData::new();
    data.modpara.nsplit_size = 2;
    data.modpara.nmp_trans = 1;

    // Julia scopes grouped PhysCal to sz-conserved normal Green paths.
    // SR-CG is an optimization-only restriction, unused by PhysCal.
    let before = data.clone();
    mvmc_core::validation::validate_phys_cal(&data).unwrap();

    data.modpara.nsrcg = 1;
    mvmc_core::validation::validate_phys_cal(&data).unwrap();
    let error = mvmc_core::validation::validate_para_opt(&data).unwrap_err();
    assert!(error.contains("SR-CG"), "{error}");

    data.modpara.nsrcg = 0;
    data.i_flg_orbital_general = 1;
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(error.contains("FSZ / general-orbital PhysCal"), "{error}");
    data.i_flg_orbital_general = 0;
    data.n_qp_opt_trans = 2;
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(error.contains("NQPOptTrans > 1"), "{error}");
    data.n_qp_opt_trans = before.n_qp_opt_trans;

    data.modpara.nsrcg = 0;
    data.modpara.lanczos_mode = 1;
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(
        error.contains("NSplitSize > 1 with NLanczosMode > 0"),
        "{error}"
    );
    let error = mvmc_core::validation::validate_para_opt(&data).unwrap_err();
    assert!(
        error.contains("Lanczos") || error.contains("NLanczosMode"),
        "{error}"
    );
}

#[test]
fn grouped_invalid_rank_rejects_before_rng_use() {
    #[derive(Debug)]
    struct InvalidRankReducer;
    impl Reducer for InvalidRankReducer {
        fn allreduce_sum_f64(&self, _: &mut [f64]) {}
        fn allreduce_sum_c64(&self, _: &mut [num_complex::Complex64]) {}
        fn allreduce_sum_i64(&self, _: &mut [i64]) {}
        fn world_size(&self) -> usize {
            2
        }
        fn rank(&self) -> usize {
            2
        }
        fn supports_grouped_sampling(&self) -> bool {
            true
        }
    }

    let mut data = ExpertModeData::new();
    data.modpara.nsplit_size = 2;
    data.modpara.nmp_trans = 1;
    let mut rng = Sfmt19937Rng::new(1);
    let mut probe = rng.clone();
    let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    let error = vmc_para_opt(
        &mut data,
        &mut state,
        &mut rng,
        None,
        &InvalidRankReducer,
        mvmc_core::OptimizationOptions::default(),
    )
    .unwrap_err();
    assert!(
        error.contains("rank 2") && error.contains("world size 2"),
        "{error}"
    );
    for _ in 0..624 {
        assert_eq!(rng.gen_rand32(), probe.gen_rand32());
    }
}
