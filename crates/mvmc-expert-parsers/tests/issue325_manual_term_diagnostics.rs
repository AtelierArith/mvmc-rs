//! Original Julia 8bb validation.jl:303–572 and 725–861 manual contracts.
//! Literal expectations are independent of dense validation and C loading.
use mvmc_expert_parsers::*;
use num_complex::Complex64;

const FAMILIES: [&str; 9] = [
    "ChargeRBM_PhysLayer",
    "SpinRBM_PhysLayer",
    "GeneralRBM_PhysLayer",
    "ChargeRBM_HiddenLayer",
    "SpinRBM_HiddenLayer",
    "GeneralRBM_HiddenLayer",
    "ChargeRBM_PhysHidden",
    "SpinRBM_PhysHidden",
    "GeneralRBM_PhysHidden",
];

fn valid_modpara() -> ModParaParameters {
    ModParaParameters {
        nsite: 4,
        nelec: 2,
        nlocspin: 0,
        ncond: -1,
        nblock_size_rbm_ratio: 200,
        ..Default::default()
    }
}

fn shadows(site1: i64, site2: i64, value: Complex64) -> ExpertModeData {
    ExpertModeData {
        modpara: valid_modpara(),
        rbm_section_widths: [1; 9],
        rbm_params: vec![Complex64::new(0.0, 0.0); 9],
        charge_rbm_phys_layer_terms: vec![ChargeRBMPhysLayerTerm {
            site: site1,
            idx: 0,
            value,
            is_complex: false,
        }],
        spin_rbm_phys_layer_terms: vec![SpinRBMPhysLayerTerm {
            site: site1,
            idx: 0,
            value,
            is_complex: false,
        }],
        general_rbm_phys_layer_terms: vec![GeneralRBMPhysLayerTerm {
            site: site1,
            spin: 0,
            idx: 0,
            value,
            is_complex: false,
        }],
        charge_rbm_hidden_layer_terms: vec![ChargeRBMHiddenLayerTerm {
            site: site1,
            idx: 0,
            value,
            is_complex: false,
        }],
        spin_rbm_hidden_layer_terms: vec![SpinRBMHiddenLayerTerm {
            site: site1,
            idx: 0,
            value,
            is_complex: false,
        }],
        general_rbm_hidden_layer_terms: vec![GeneralRBMHiddenLayerTerm {
            site: site1,
            idx: 0,
            value,
            is_complex: false,
        }],
        charge_rbm_phys_hidden_terms: vec![ChargeRBMPhysHiddenTerm {
            site1,
            site2,
            idx: 0,
            value,
            is_complex: false,
        }],
        spin_rbm_phys_hidden_terms: vec![SpinRBMPhysHiddenTerm {
            site1,
            site2,
            idx: 0,
            value,
            is_complex: false,
        }],
        general_rbm_phys_hidden_terms: vec![GeneralRBMPhysHiddenTerm {
            site1,
            spin: 0,
            site2,
            idx: 0,
            value,
            is_complex: false,
        }],
        ..ExpertModeData::new()
    }
}

fn primitive_results(data: &ExpertModeData, nsite: i64) -> [ValidationResult; 9] {
    [
        validate_charge_rbm_phys_layer_terms(&data.charge_rbm_phys_layer_terms, nsite),
        validate_spin_rbm_phys_layer_terms(&data.spin_rbm_phys_layer_terms, nsite),
        validate_general_rbm_phys_layer_terms(&data.general_rbm_phys_layer_terms, nsite),
        validate_charge_rbm_hidden_layer_terms(&data.charge_rbm_hidden_layer_terms, nsite),
        validate_spin_rbm_hidden_layer_terms(&data.spin_rbm_hidden_layer_terms, nsite),
        validate_general_rbm_hidden_layer_terms(&data.general_rbm_hidden_layer_terms, nsite),
        validate_charge_rbm_phys_hidden_terms(&data.charge_rbm_phys_hidden_terms, nsite),
        validate_spin_rbm_phys_hidden_terms(&data.spin_rbm_phys_hidden_terms, nsite),
        validate_general_rbm_phys_hidden_terms(&data.general_rbm_phys_hidden_terms, nsite),
    ]
}

// Complete Debug-visible state plus raw coefficient bits, including NaN payloads
// and signed zero. No mutable reference or RNG is passed to any diagnostic API.
fn snapshot(data: &ExpertModeData) -> (String, Vec<(u64, u64)>) {
    let mut values = data.rbm_params.clone();
    values.extend(data.charge_rbm_phys_layer_terms.iter().map(|t| t.value));
    values.extend(data.spin_rbm_phys_layer_terms.iter().map(|t| t.value));
    values.extend(data.general_rbm_phys_layer_terms.iter().map(|t| t.value));
    values.extend(data.charge_rbm_hidden_layer_terms.iter().map(|t| t.value));
    values.extend(data.spin_rbm_hidden_layer_terms.iter().map(|t| t.value));
    values.extend(data.general_rbm_hidden_layer_terms.iter().map(|t| t.value));
    values.extend(data.charge_rbm_phys_hidden_terms.iter().map(|t| t.value));
    values.extend(data.spin_rbm_phys_hidden_terms.iter().map(|t| t.value));
    values.extend(data.general_rbm_phys_hidden_terms.iter().map(|t| t.value));
    (
        format!("{data:?}"),
        values
            .iter()
            .map(|z| (z.re.to_bits(), z.im.to_bits()))
            .collect(),
    )
}

#[test]
fn all_nine_primitives_accept_empty_and_valid_coordinate_boundaries() {
    for result in primitive_results(&ExpertModeData::new(), 0) {
        assert!(result.is_valid);
        assert!(result.errors.is_empty() && result.warnings.is_empty());
    }
    for (a, b) in [(0, 3), (3, 0)] {
        let data = shadows(a, b, Complex64::new(0.25, -0.125));
        for result in primitive_results(&data, 4) {
            assert!(result.is_valid);
            assert!(result.errors.is_empty() && result.warnings.is_empty());
        }
    }
}

#[test]
fn all_nine_site_errors_keep_row_and_field_order_without_spin_or_index_checks() {
    let mut data = shadows(-1, 4, Complex64::new(0.0, 0.0));
    data.general_rbm_phys_layer_terms[0].spin = -9;
    data.general_rbm_phys_layer_terms[0].idx = i64::MAX;
    data.general_rbm_phys_hidden_terms[0].spin = 7;
    data.general_rbm_phys_hidden_terms[0].idx = -8;
    for (i, result) in primitive_results(&data, 4).into_iter().enumerate() {
        let family = FAMILIES[i];
        let expected = if i < 6 {
            vec![format!("{family} term 1: site (-1) out of range [0, 3]")]
        } else {
            vec![
                format!("{family} term 1: site1 (-1) out of range [0, 3]"),
                format!("{family} term 1: site2 (4) out of range [0, 3]"),
            ]
        };
        assert!(!result.is_valid);
        assert_eq!(result.errors, expected);
        assert!(result.warnings.is_empty());
    }
    let valid_sites = ExpertModeData {
        general_rbm_phys_layer_terms: vec![GeneralRBMPhysLayerTerm {
            site: 0,
            spin: -9,
            idx: i64::MAX,
            value: Complex64::new(0.0, 0.0),
            is_complex: false,
        }],
        general_rbm_phys_hidden_terms: vec![GeneralRBMPhysHiddenTerm {
            site1: 0,
            spin: 7,
            site2: 3,
            idx: -8,
            value: Complex64::new(0.0, 0.0),
            is_complex: false,
        }],
        ..ExpertModeData::new()
    };
    for result in primitive_results(&valid_sites, 4) {
        assert!(result.is_valid && result.errors.is_empty());
    }
    // Second row distinguishes traversal order from a single-row-only check.
    data.charge_rbm_phys_hidden_terms
        .push(ChargeRBMPhysHiddenTerm {
            site1: 4,
            site2: -2,
            idx: 0,
            value: Complex64::new(0.0, 0.0),
            is_complex: false,
        });
    assert_eq!(
        validate_charge_rbm_phys_hidden_terms(&data.charge_rbm_phys_hidden_terms, 4).errors,
        [
            "ChargeRBM_PhysHidden term 1: site1 (-1) out of range [0, 3]",
            "ChargeRBM_PhysHidden term 1: site2 (4) out of range [0, 3]",
            "ChargeRBM_PhysHidden term 2: site1 (4) out of range [0, 3]",
            "ChargeRBM_PhysHidden term 2: site2 (-2) out of range [0, 3]",
        ]
    );
}

#[test]
fn every_family_threshold_is_strict_and_warning_only() {
    for (value, count) in [(1e10, 0), (10000000001.0, 1)] {
        let data = shadows(0, 3, Complex64::new(value, 0.0));
        for result in primitive_results(&data, 4) {
            assert!(result.is_valid && result.errors.is_empty());
            assert_eq!(result.warnings.len(), count);
        }
        let combined = validate_expert_mode_term_diagnostics(&data);
        assert!(combined.is_valid && combined.errors.is_empty());
        assert_eq!(combined.warnings.len(), count * 9);
    }
}

#[test]
fn every_family_retains_independent_magnitude_and_inf_nan_decisions_without_mutation() {
    for (value, count) in [
        (Complex64::new(3e9, 4e9), 0),
        (Complex64::new(-3e10, 4e10), 1),
        (Complex64::new(f64::MAX / 2.0, f64::MAX / 2.0), 1),
        (Complex64::new(f64::MAX, f64::MAX), 1),
        (Complex64::new(f64::from_bits(1), -0.0), 0),
        (Complex64::new(f64::NAN, 0.0), 0),
        (Complex64::new(0.0, f64::NAN), 0),
        (Complex64::new(f64::NAN, f64::NAN), 0),
        (Complex64::new(f64::INFINITY, f64::NAN), 1),
        (Complex64::new(f64::NAN, f64::NEG_INFINITY), 1),
    ] {
        let data = shadows(0, 3, value);
        let before = snapshot(&data);
        for result in primitive_results(&data, 4) {
            assert!(result.is_valid && result.errors.is_empty());
            assert_eq!(result.warnings.len(), count);
        }
        let combined = validate_expert_mode_term_diagnostics(&data);
        assert!(combined.is_valid && combined.errors.is_empty());
        assert_eq!(combined.warnings.len(), count * 9);
        assert_eq!(snapshot(&data), before);
    }
}

#[test]
fn aggregate_has_literal_nine_family_errors_before_dh_and_literal_shadow_warnings() {
    let mut data = shadows(-1, 4, Complex64::new(10000000001.0, 0.0));
    data.doublon_holon_2site_indices = vec![DoublonHolon2SiteIndex {
        neighbors: vec![[0, 0]; 3],
    }];
    data.doublon_holon_4site_indices = vec![DoublonHolon4SiteIndex {
        neighbors: vec![[0; 4]; 3],
    }];
    let before = snapshot(&data);
    let result = validate_expert_mode_term_diagnostics(&data);
    assert!(!result.is_valid);
    assert_eq!(
        result.errors,
        [
            "ChargeRBM_PhysLayer term 1: site (-1) out of range [0, 3]",
            "SpinRBM_PhysLayer term 1: site (-1) out of range [0, 3]",
            "GeneralRBM_PhysLayer term 1: site (-1) out of range [0, 3]",
            "ChargeRBM_HiddenLayer term 1: site (-1) out of range [0, 3]",
            "SpinRBM_HiddenLayer term 1: site (-1) out of range [0, 3]",
            "GeneralRBM_HiddenLayer term 1: site (-1) out of range [0, 3]",
            "ChargeRBM_PhysHidden term 1: site1 (-1) out of range [0, 3]",
            "ChargeRBM_PhysHidden term 1: site2 (4) out of range [0, 3]",
            "SpinRBM_PhysHidden term 1: site1 (-1) out of range [0, 3]",
            "SpinRBM_PhysHidden term 1: site2 (4) out of range [0, 3]",
            "GeneralRBM_PhysHidden term 1: site1 (-1) out of range [0, 3]",
            "GeneralRBM_PhysHidden term 1: site2 (4) out of range [0, 3]",
            "DH2 index 0: neighbors must be 4 x 2",
            "DH4 index 0: neighbors must be 4 x 4",
        ]
    );
    assert_eq!(result.warnings, [
        "ChargeRBM_PhysLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "SpinRBM_PhysLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "GeneralRBM_PhysLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "ChargeRBM_HiddenLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "SpinRBM_HiddenLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "GeneralRBM_HiddenLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "ChargeRBM_PhysHidden term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "SpinRBM_PhysHidden term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
        "GeneralRBM_PhysHidden term 1: very large value Complex { re: 10000000001.0, im: 0.0 }",
    ]);
    assert_eq!(snapshot(&data), before);
}

#[test]
fn modpara_and_factor_diagnostics_precede_rbm_in_separate_vectors() {
    let mut data = shadows(0, 3, Complex64::new(10000000001.0, 0.0));
    data.modpara.nblock_size_rbm_ratio = 202;
    data.transfer_terms = vec![TransferTerm {
        site1: -1,
        site2: -1,
        spin1: Spin::Up,
        spin2: Spin::Down,
        value: Complex64::new(1.0, 0.0),
    }];
    data.coulomb_intra_terms = vec![CoulombIntraTerm {
        site: 4,
        value: -2.0,
    }];
    data.coulomb_inter_terms = vec![CoulombInterTerm {
        site1: -1,
        site2: -1,
        value: -1.0,
    }];
    data.gutzwiller_terms = vec![GutzwillerTerm {
        site: 4,
        value: Complex64::new(-0.5, 0.0),
        is_complex: false,
    }];
    data.jastrow_terms = vec![JastrowTerm {
        site1: 4,
        site2: 4,
        value: Complex64::new(0.1, 0.0),
        is_complex: false,
    }];
    data.orbital_terms = vec![OrbitalTerm {
        site1: -1,
        site2: 4,
        idx: 0,
        sign: 1,
        is_complex: false,
    }];
    let before = snapshot(&data);
    let result = validate_expert_mode_term_diagnostics(&data);
    assert_eq!(
        result.errors,
        [
            "Transfer term 1: site1 (-1) out of range [0, 3]",
            "Transfer term 1: site2 (-1) out of range [0, 3]",
            "CoulombIntra term 1: site (4) out of range [0, 3]",
            "CoulombInter term 1: site1 (-1) out of range [0, 3]",
            "CoulombInter term 1: site2 (-1) out of range [0, 3]",
            "Gutzwiller term 1: site (4) out of range [0, 3]",
            "Jastrow term 1: site1 (4) out of range [0, 3]",
            "Jastrow term 1: site2 (4) out of range [0, 3]",
            "Orbital term 1: site1 (-1) out of range [0, 3]",
            "Orbital term 1: site2 (4) out of range [0, 3]",
        ]
    );
    assert_eq!(
        &result.warnings[..8],
        [
            "NBlockSize_RBMRatio should be multiple of 8",
            "Transfer term 1: site1 == site2 (diagonal term)",
            "Transfer term 1: spin indices differ (0 != 1)",
            "CoulombIntra term 1: negative value -2.0",
            "CoulombInter term 1: site1 == site2 (on-site term)",
            "CoulombInter term 1: negative value -1.0",
            "Gutzwiller term 1: negative real part -0.5",
            "Jastrow term 1: site1 == site2 (diagonal term)",
        ]
    );
    assert_eq!(result.warnings.len(), 17);
    assert_eq!(
        result.warnings[8],
        "ChargeRBM_PhysLayer term 1: very large value Complex { re: 10000000001.0, im: 0.0 }"
    );
    assert_eq!(snapshot(&data), before);
}

#[test]
fn manual_large_shadows_and_dense_nonfinite_storage_are_distinct_observations() {
    let data = shadows(0, 3, Complex64::new(10000000001.0, 0.0));
    assert!(validate_rbm_parameters(&data).is_valid);
    assert!(validate_expert_mode_data(&data).warnings.is_empty());
    assert_eq!(
        validate_expert_mode_term_diagnostics(&data).warnings.len(),
        9
    );
    let mut small_shadows = shadows(0, 3, Complex64::new(0.25, -0.125));
    small_shadows.rbm_params[0] = Complex64::new(f64::NAN, f64::INFINITY);
    small_shadows.rbm_params.pop();
    let before = snapshot(&small_shadows);
    let manual = validate_expert_mode_term_diagnostics(&small_shadows);
    assert!(manual.is_valid && manual.errors.is_empty() && manual.warnings.is_empty());
    let dense = validate_rbm_parameters(&small_shadows);
    assert_eq!(
        dense.errors,
        [
            "RBM parameter storage has 8 values; declared sections require 9",
            "RBM parameter 0 must be finite",
        ]
    );
    assert_eq!(snapshot(&small_shadows), before);
}

#[test]
fn green_errors_remain_in_existing_dense_aggregate_not_manual_term_aggregate() {
    let data = ExpertModeData {
        modpara: valid_modpara(),
        green_one_terms: vec![GreenOneTerm {
            site1: -1,
            spin1: Spin::Up,
            site2: 4,
            spin2: Spin::Down,
        }],
        ..ExpertModeData::new()
    };
    let before = snapshot(&data);
    assert!(validate_expert_mode_term_diagnostics(&data).is_valid);
    assert_eq!(
        validate_expert_mode_data(&data).errors,
        [
            "OneBodyG term 1: site1 (-1) out of range [0, 3]",
            "OneBodyG term 1: site2 (4) out of range [0, 3]",
        ]
    );
    assert_eq!(snapshot(&data), before);
}

#[test]
fn original_valid_and_invalid_aggregate_examples_remain_literal_controls() {
    // Julia test_validation.jl:132–151 / M0473–M0476. This example is not
    // proof of every original aggregate overload or whole scenario completion.
    let data = ExpertModeData {
        modpara: valid_modpara(),
        transfer_terms: vec![TransferTerm {
            site1: 0,
            site2: 1,
            spin1: Spin::Up,
            spin2: Spin::Up,
            value: Complex64::new(1.0, 0.0),
        }],
        coulomb_intra_terms: vec![CoulombIntraTerm {
            site: 0,
            value: 4.0,
        }],
        gutzwiller_terms: vec![GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.5, 0.0),
            is_complex: false,
        }],
        ..ExpertModeData::new()
    };
    let result = validate_expert_mode_term_diagnostics(&data);
    assert!(result.is_valid && result.errors.is_empty());
    let invalid = ExpertModeData {
        modpara: ModParaParameters {
            nsite: -1,
            nelec: -1,
            nlocspin: -1,
            ..valid_modpara()
        },
        transfer_terms: vec![TransferTerm {
            site1: -1,
            site2: 1,
            spin1: Spin::Up,
            spin2: Spin::Up,
            value: Complex64::new(1.0, 0.0),
        }],
        ..ExpertModeData::new()
    };
    let result = validate_expert_mode_term_diagnostics(&invalid);
    assert!(!result.is_valid && !result.errors.is_empty());
    assert_eq!(
        &result.errors[..3],
        [
            "NSite must be positive",
            "NElec must be non-negative",
            "NLocSpin must be non-negative"
        ]
    );
}
