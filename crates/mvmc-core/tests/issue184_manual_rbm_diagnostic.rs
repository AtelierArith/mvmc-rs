//! Original M0587 manual shadow diagnostic, not a dense invariant or C input gate.

#[test]
fn charge_rbm_phys_layer_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::ChargeRBMPhysLayerTerm {
        site: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::ChargeRBMPhysLayerTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.charge_rbm_phys_layer_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::ChargePhysLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn spin_rbm_phys_layer_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::SpinRBMPhysLayerTerm {
        site: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::SpinRBMPhysLayerTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.spin_rbm_phys_layer_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::SpinPhysLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn general_rbm_phys_layer_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::GeneralRBMPhysLayerTerm {
        site: 0,
        spin: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::GeneralRBMPhysLayerTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.general_rbm_phys_layer_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::GeneralPhysLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn charge_rbm_hidden_layer_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::ChargeRBMHiddenLayerTerm {
        site: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::ChargeRBMHiddenLayerTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.charge_rbm_hidden_layer_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::ChargeHiddenLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn spin_rbm_hidden_layer_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::SpinRBMHiddenLayerTerm {
        site: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::SpinRBMHiddenLayerTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.spin_rbm_hidden_layer_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::SpinHiddenLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn general_rbm_hidden_layer_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::GeneralRBMHiddenLayerTerm {
        site: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::GeneralRBMHiddenLayerTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.general_rbm_hidden_layer_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::GeneralHiddenLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn charge_rbm_phys_hidden_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::ChargeRBMPhysHiddenTerm {
        site1: 0,
        site2: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::ChargeRBMPhysHiddenTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.charge_rbm_phys_hidden_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::ChargePhysHidden);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn spin_rbm_phys_hidden_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::SpinRBMPhysHiddenTerm {
        site1: 0,
        site2: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::SpinRBMPhysHiddenTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.spin_rbm_phys_hidden_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::SpinPhysHidden);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn general_rbm_phys_hidden_terms_detects_first_conflict_without_mutation() {
    let mut data = ExpertModeData::default();
    let first = mvmc_expert_parsers::GeneralRBMPhysHiddenTerm {
        site1: 0,
        spin: 0,
        site2: 0,
        idx: 0,
        value: Complex64::new(0.1, 0.0),
        is_complex: true,
    };
    let conflicting = mvmc_expert_parsers::GeneralRBMPhysHiddenTerm {
        value: Complex64::new(0.2, 0.0),
        ..first
    };
    data.general_rbm_phys_hidden_terms = vec![first, first, conflicting, conflicting];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::GeneralPhysHidden);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 2)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

use mvmc_core::parameter_diagnostics::{check_duplicate_consistency, RbmShadowSection};
use mvmc_expert_parsers::{ChargeRBMPhysLayerTerm, ExpertModeData};
use num_complex::Complex64;
use std::path::Path;

fn original_heisenberg() -> ExpertModeData {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/original_heisenberg_parser_184/namelist.def");
    let data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    data
}

fn term(site: i64, value: Complex64) -> ChargeRBMPhysLayerTerm {
    ChargeRBMPhysLayerTerm {
        site,
        idx: 0,
        value,
        is_complex: true,
    }
}

#[test]
fn original_m0587_unequal_rbm_shadows_report_without_repair() {
    let mut data = original_heisenberg();
    assert!(data.charge_rbm_phys_layer_terms.is_empty());
    data.charge_rbm_phys_layer_terms = vec![
        term(0, Complex64::new(0.1, 0.0)),
        term(1, Complex64::new(0.2, 0.0)),
    ];
    let before = format!("{data:?}");
    let error = check_duplicate_consistency(&data).unwrap_err();
    assert_eq!(error.section, RbmShadowSection::ChargePhysLayer);
    assert_eq!(
        (
            error.parameter_index,
            error.first_row,
            error.conflicting_row
        ),
        (0, 0, 1)
    );
    assert_eq!(error.first_value, Complex64::new(0.1, 0.0));
    assert_eq!(error.conflicting_value, Complex64::new(0.2, 0.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn equal_shadows_do_not_assert_dense_consistency() {
    let mut data = ExpertModeData {
        charge_rbm_phys_layer_terms: vec![
            term(0, Complex64::new(0.25, -0.5)),
            term(1, Complex64::new(0.25, -0.5)),
        ],
        ..ExpertModeData::default()
    };
    // Even missing dense allocation is outside the manual duplicate contract.
    assert!(data.rbm_params.is_empty());
    assert!(check_duplicate_consistency(&data).is_ok());
    data.rbm_params = vec![Complex64::new(11.0, 22.0)];
    let before = format!("{data:?}");
    assert!(check_duplicate_consistency(&data).is_ok());
    assert_eq!(data.rbm_params[0], Complex64::new(11.0, 22.0));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn signed_zero_and_nan_follow_numeric_duplicate_equality() {
    let mut data = ExpertModeData {
        charge_rbm_phys_layer_terms: vec![
            term(0, Complex64::new(0.0, 0.0)),
            term(1, Complex64::new(-0.0, -0.0)),
        ],
        ..ExpertModeData::default()
    };
    assert!(check_duplicate_consistency(&data).is_ok());
    data.charge_rbm_phys_layer_terms = vec![term(0, Complex64::new(f64::NAN, 0.0))];
    assert!(check_duplicate_consistency(&data).is_ok());
    data.charge_rbm_phys_layer_terms
        .push(term(1, Complex64::new(f64::NAN, 0.0)));
    assert!(check_duplicate_consistency(&data).is_err());
}

#[test]
fn different_indices_and_sections_are_not_duplicates() {
    let mut data = ExpertModeData::default();
    let first = term(0, Complex64::new(0.1, 0.0));
    let mut other = term(1, Complex64::new(0.2, 0.0));
    other.idx = 1;
    data.charge_rbm_phys_layer_terms = vec![first, other];
    data.spin_rbm_phys_layer_terms
        .push(mvmc_expert_parsers::SpinRBMPhysLayerTerm {
            site: 0,
            idx: 0,
            value: Complex64::new(0.3, 0.0),
            is_complex: true,
        });
    assert!(check_duplicate_consistency(&data).is_ok());
}
