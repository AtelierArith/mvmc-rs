//! Julia MVMCExpertModeParsers.jl/test/test_validation.jl counterparts.
use mvmc_expert_parsers::*;
use num_complex::Complex64;

fn valid_params() -> ModParaParameters {
    ModParaParameters {
        nsite: 4,
        nelec: 2,
        nlocspin: 0,
        ncond: -1,
        nblock_size_rbm_ratio: 202,
        ..Default::default()
    }
}

#[test]
fn valid_modpara_and_fixture_data_have_structured_results() {
    let result = validate_modpara_params(&valid_params());
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert_eq!(
        result.warnings,
        ["NBlockSize_RBMRatio should be multiple of 8"]
    );
    let mut data = ExpertModeData::new();
    data.modpara = valid_params();
    data.transfer_terms.push(TransferTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Up,
        value: Complex64::new(1.0, 0.0),
    });
    data.coulomb_intra_terms.push(CoulombIntraTerm {
        site: 0,
        value: 4.0,
    });
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 0,
        value: Complex64::new(0.5, 0.0),
        is_complex: false,
    });
    assert_eq!(validate_expert_mode_data(&data), result);
}

#[test]
fn invalid_counts_produce_ordered_errors_and_warnings() {
    let p = ModParaParameters {
        nsite: -1,
        nelec: -1,
        nlocspin: -1,
        ncond: -2,
        two_sz: 3,
        nvmc_sample: 0,
        nvmc_interval: 0,
        nvmc_warmup: -1,
        nsr_opt_itr_step: 0,
        nsr_opt_itr_smp: 0,
        dsr_opt_red_cut: -1.0,
        dsr_opt_step_dt: 0.0,
        nblock_size_rbm_ratio: 0,
        ..Default::default()
    };
    let result = validate_modpara_params(&p);
    assert!(!result.is_valid);
    assert_eq!(
        result.errors,
        [
            "NSite must be positive",
            "NElec must be non-negative",
            "NLocSpin must be non-negative",
            "NCond must be non-negative or -1",
            "2Sz must be even or -1",
            "NVMCSample must be positive",
            "NVMCInterval must be positive",
            "NVMCWarmUp must be non-negative",
            "NSROptItrStep must be positive",
            "NSROptItrSmp must be positive",
            "DSROptRedCut must be non-negative",
            "DSROptStepDt must be positive",
        ]
    );
    assert!(result.warnings.is_empty());
}

#[test]
fn odd_cond_and_inconsistent_electron_counts_keep_upstream_severity() {
    let mut p = valid_params();
    p.ncond = 3;
    let result = validate_modpara_params(&p);
    assert_eq!(result.errors, ["NCond must be even"]);
    assert_eq!(
        result.warnings[0],
        "NElec (2) differs from expected value (1) based on NCond"
    );
    p.ncond = 2;
    let result = validate_modpara_params(&p);
    assert!(result.is_valid, "electron inconsistency alone is a warning");
    assert_eq!(
        result.warnings[0],
        "NElec (2) differs from expected value (1) based on NCond"
    );
}

#[test]
fn local_spin_branch_priority_matches_julia() {
    for (spin, electrons, path, error) in [
        (
            4,
            2,
            1,
            Some("NExUpdatePath must be 2 when 2*Ne = NLocalSpin (spin system)"),
        ),
        (4, 2, 2, None),
        (2, 2, 0, Some("NExUpdatePath must be 1")),
        (5, 2, 0, Some("NExUpdatePath must be 1")),
        (5, 2, 1, Some("2*Ne must satisfy 2*Ne >= NLocalSpin")),
    ] {
        let mut p = valid_params();
        p.nlocspin = spin;
        p.nelec = electrons;
        p.nex_update_path = path;
        let result = validate_modpara_params(&p);
        assert_eq!(
            result.errors,
            error.into_iter().map(str::to_owned).collect::<Vec<_>>()
        );
    }
}

#[test]
fn transfer_bounds_diagonals_and_spin_changes_are_distinguished() {
    let terms = [
        TransferTerm {
            site1: -1,
            site2: 4,
            spin1: Spin::Up,
            spin2: Spin::Up,
            value: Complex64::new(1.0, 0.0),
        },
        TransferTerm {
            site1: 2,
            site2: 2,
            spin1: Spin::Up,
            spin2: Spin::Down,
            value: Complex64::new(1.0, 0.0),
        },
    ];
    let result = validate_transfer_terms(&terms, 4);
    assert_eq!(
        result.errors,
        [
            "Transfer term 1: site1 (-1) out of range [0, 3]",
            "Transfer term 1: site2 (4) out of range [0, 3]"
        ]
    );
    assert_eq!(
        result.warnings,
        [
            "Transfer term 2: site1 == site2 (diagonal term)",
            "Transfer term 2: spin indices differ (0 != 1)"
        ]
    );
}

#[test]
fn supported_factor_bounds_and_negative_value_warnings_match_julia() {
    let intra = validate_coulomb_intra_terms(
        &[CoulombIntraTerm {
            site: -1,
            value: -2.0,
        }],
        4,
    );
    assert_eq!(
        intra.errors,
        ["CoulombIntra term 1: site (-1) out of range [0, 3]"]
    );
    assert_eq!(intra.warnings, ["CoulombIntra term 1: negative value -2.0"]);
    let inter = validate_coulomb_inter_terms(
        &[CoulombInterTerm {
            site1: 4,
            site2: 4,
            value: -1.0,
        }],
        4,
    );
    assert_eq!(inter.errors.len(), 2);
    assert_eq!(
        inter.warnings,
        [
            "CoulombInter term 1: site1 == site2 (on-site term)",
            "CoulombInter term 1: negative value -1.0"
        ]
    );
    let g = validate_gutzwiller_terms(
        &[GutzwillerTerm {
            site: 4,
            value: Complex64::new(-0.5, 0.0),
            is_complex: false,
        }],
        4,
    );
    assert_eq!(
        g.errors,
        ["Gutzwiller term 1: site (4) out of range [0, 3]"]
    );
    assert_eq!(g.warnings, ["Gutzwiller term 1: negative real part -0.5"]);
    let j = validate_jastrow_terms(
        &[JastrowTerm {
            site1: 3,
            site2: 3,
            value: Complex64::new(0.1, 0.0),
            is_complex: false,
        }],
        4,
    );
    assert!(j.is_valid);
    assert_eq!(
        j.warnings,
        ["Jastrow term 1: site1 == site2 (diagonal term)"]
    );
    let o = validate_orbital_terms(
        &[OrbitalTerm {
            site1: -1,
            site2: 4,
            idx: 0,
            sign: 1,

            is_complex: false,
        }],
        4,
    );
    assert_eq!(
        o.errors,
        [
            "Orbital term 1: site1 (-1) out of range [0, 3]",
            "Orbital term 1: site2 (4) out of range [0, 3]"
        ]
    );
}

#[test]
fn duplicate_and_nonfinite_coefficients_follow_existing_julia_checks() {
    // Julia's structure validators do not reject these values or duplicate
    // transfer rows. Strict parameter-file loaders have separate checks.
    let term = TransferTerm {
        site1: 0,
        site2: 1,
        spin1: Spin::Up,
        spin2: Spin::Up,
        value: Complex64::new(f64::NAN, f64::INFINITY),
    };
    let result = validate_transfer_terms(&[term, term], 4);
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
    let mut p = valid_params();
    p.dsr_opt_step_dt = f64::NAN;
    assert!(validate_modpara_params(&p).is_valid);
}

#[test]
fn combined_validation_preserves_family_order_and_does_not_mutate_data() {
    let mut data = ExpertModeData::new();
    data.modpara = valid_params();
    data.modpara.nsite = 0;
    data.transfer_terms.push(TransferTerm {
        site1: 0,
        site2: 1,
        spin1: Spin::Up,
        spin2: Spin::Up,
        value: Complex64::new(1.0, 0.0),
    });
    data.coulomb_intra_terms.push(CoulombIntraTerm {
        site: 0,
        value: -2.0,
    });
    let before = data.clone();
    let result = validate_expert_mode_data(&data);
    assert_eq!(
        result.errors,
        [
            "NSite must be positive",
            "Transfer term 1: site1 (0) out of range [0, -1]",
            "Transfer term 1: site2 (1) out of range [0, -1]",
            "CoulombIntra term 1: site (0) out of range [0, -1]"
        ]
    );
    assert_eq!(
        result.warnings,
        [
            "NBlockSize_RBMRatio should be multiple of 8",
            "CoulombIntra term 1: negative value -2.0"
        ]
    );
    assert_eq!(data.modpara, before.modpara);
    assert_eq!(data.transfer_terms, before.transfer_terms);
}
