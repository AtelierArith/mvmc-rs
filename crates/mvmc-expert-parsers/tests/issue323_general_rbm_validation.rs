//! Original A180 manual diagnostic conditions; not C loader/dense validation.
use mvmc_expert_parsers::{validate_general_rbm_phys_hidden_terms, GeneralRBMPhysHiddenTerm};
use num_complex::Complex64;

fn term(site1: i64, site2: i64, re: f64, im: f64) -> GeneralRBMPhysHiddenTerm {
    GeneralRBMPhysHiddenTerm {
        site1,
        spin: 0,
        site2,
        idx: 0,
        value: Complex64::new(re, im),
        is_complex: true,
    }
}

#[test]
fn empty_and_valid_site_boundaries_have_no_diagnostics() {
    let empty = validate_general_rbm_phys_hidden_terms(&[], 3);
    assert!(empty.is_valid);
    assert!(empty.errors.is_empty() && empty.warnings.is_empty());
    let result =
        validate_general_rbm_phys_hidden_terms(&[term(0, 2, 0.1, 0.0), term(2, 0, -0.1, 0.0)], 3);
    assert!(result.is_valid);
    assert!(result.errors.is_empty() && result.warnings.is_empty());
}

#[test]
fn site_errors_are_ordered_by_row_then_site1_site2() {
    let result =
        validate_general_rbm_phys_hidden_terms(&[term(-1, 3, 0.0, 0.0), term(3, -1, 0.0, 0.0)], 3);
    assert!(!result.is_valid);
    assert_eq!(
        result.errors,
        [
            "GeneralRBM_PhysHidden term 1: site1 (-1) out of range [0, 2]",
            "GeneralRBM_PhysHidden term 1: site2 (3) out of range [0, 2]",
            "GeneralRBM_PhysHidden term 2: site1 (3) out of range [0, 2]",
            "GeneralRBM_PhysHidden term 2: site2 (-1) out of range [0, 2]",
        ]
    );
    assert!(result.warnings.is_empty());
}

#[test]
fn exact_real_threshold_warns_only_above_and_remains_valid() {
    let result = validate_general_rbm_phys_hidden_terms(
        &[term(0, 0, 1e10, 0.0), term(0, 0, 1e10 + 1.0, 0.0)],
        1,
    );
    assert!(result.is_valid && result.errors.is_empty());
    assert_eq!(
        result.warnings,
        ["GeneralRBM_PhysHidden term 2: very large value Complex { re: 10000000001.0, im: 0.0 }"]
    );
}

#[test]
fn independent_three_four_five_magnitudes_select_warning_rows() {
    // Literal magnitudes 5e9 and 5e10, not expectations derived by norm().
    let result =
        validate_general_rbm_phys_hidden_terms(&[term(0, 0, 3e9, 4e9), term(0, 0, -3e10, 4e10)], 1);
    assert!(result.is_valid && result.errors.is_empty());
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].starts_with("GeneralRBM_PhysHidden term 2: very large value "));
}

#[test]
fn huge_finite_and_tiny_subnormal_values_preserve_warning_decisions() {
    let tiny = f64::from_bits(1);
    let result = validate_general_rbm_phys_hidden_terms(
        &[
            term(0, 0, f64::MAX / 2.0, f64::MAX / 2.0),
            term(0, 0, f64::MAX, f64::MAX),
            term(0, 0, tiny, -tiny),
        ],
        1,
    );
    assert!(result.is_valid && result.errors.is_empty());
    assert_eq!(result.warnings.len(), 2);
    assert!(result.warnings[0].starts_with("GeneralRBM_PhysHidden term 1: "));
    assert!(result.warnings[1].starts_with("GeneralRBM_PhysHidden term 2: "));
}

#[test]
fn infinity_dominates_nan_without_new_rejections_or_mutation() {
    let mut terms = [
        term(0, 0, f64::NAN, 0.0),
        term(0, 0, 0.0, f64::NAN),
        term(0, 0, f64::NAN, f64::NAN),
        term(0, 0, f64::INFINITY, f64::NAN),
        term(0, 0, f64::NAN, f64::NEG_INFINITY),
        term(-1, 1, f64::NEG_INFINITY, 0.0),
    ];
    // Manual architecture deliberately does not invent spin/index checks.
    terms[0].spin = -9;
    terms[0].idx = i64::MAX;
    let before: Vec<_> = terms
        .iter()
        .map(|t| {
            (
                t.site1,
                t.site2,
                t.spin,
                t.idx,
                t.value.re.to_bits(),
                t.value.im.to_bits(),
                t.is_complex,
            )
        })
        .collect();
    let result = validate_general_rbm_phys_hidden_terms(&terms, 1);
    assert!(!result.is_valid);
    assert_eq!(result.errors.len(), 2); // Only row6's two sites, not NaN/spin/idx.
    assert_eq!(result.warnings.len(), 3);
    for (warning, row) in result.warnings.iter().zip([4, 5, 6]) {
        assert!(warning.starts_with(&format!("GeneralRBM_PhysHidden term {row}: ")));
    }
    let after: Vec<_> = terms
        .iter()
        .map(|t| {
            (
                t.site1,
                t.site2,
                t.spin,
                t.idx,
                t.value.re.to_bits(),
                t.value.im.to_bits(),
                t.is_complex,
            )
        })
        .collect();
    assert_eq!(before, after);
}
