//! F077 / S188 constructor boundaries, not zero-sector C model validation.
//!
//! Julia-mVMC 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:
//! MVMCOptimizers.jl/src/types.jl:186-214 and
//! MVMCOptimizers.jl/test_unit/test_unit_types.jl:81-99 (M0971-M0980).
//! M0971's minimum-one QP assertion is deliberately NOT ported: C readdef.c
//! computes the actual QP product and setmemory.c allocates proportional
//! capacity. Literal sizes/zeros below are independent expectations; ordinary
//! tests do not execute or read C/Julia/toolbox programs.

use mvmc_core::state::{InvMColMajor, SlaterElmFlat, SlaterMatrixData};
use num_complex::Complex64;

#[test]
fn zero_qp_helper_stays_empty_in_both_modes_without_fabricating_a_plane() {
    for all_complex in [false, true] {
        for (sites, electrons) in [(0, 0), (3, 4)] {
            let data = SlaterMatrixData::zeros(0, sites, electrons, all_complex);
            assert_eq!(data.slater_elm.n_qp_full(), 0);
            assert_eq!(data.inv_m.n_qp_full(), 0);
            assert!(data.slater_elm.is_empty());
            assert!(data.inv_m.is_empty());
            assert!(data.pf_m.is_empty());
            // These are the exact original all-zero/all-complex empty-real
            // assertions M0972-M0974, plus the real-mode empty-storage boundary.
            assert!(data.slater_elm_real.is_empty());
            assert!(data.inv_m_real.is_empty());
            assert!(data.pf_m_real.is_empty());
        }
    }
}

#[test]
fn positive_qp_helper_normalizes_only_site_and_electron_dimensions() {
    for all_complex in [false, true] {
        let data = SlaterMatrixData::zeros(1, 0, 0, all_complex);
        assert_eq!(data.slater_elm.n_qp_full(), 1);
        assert_eq!(data.slater_elm.n_site2(), 2);
        assert_eq!(data.inv_m.n_size(), 2);
        // One normalized site gives a 2x2 plane; one normalized electron
        // gives a 2x2 inverse plus its single pad slot.
        assert_eq!(data.slater_elm.len(), 4);
        assert_eq!(data.inv_m.len(), 5);
        assert_eq!(data.pf_m, [Complex64::new(0.0, 0.0)]);
        if all_complex {
            assert!(data.slater_elm_real.is_empty());
            assert!(data.inv_m_real.is_empty());
            assert!(data.pf_m_real.is_empty());
        } else {
            assert_eq!(data.slater_elm_real.len(), 4);
            assert_eq!(data.inv_m_real.len(), 5);
            assert_eq!(data.pf_m_real, [0.0]);
        }
    }
}

#[test]
fn original_positive_real_constructor_has_all_six_independent_lengths() {
    let data = SlaterMatrixData::zeros(2, 3, 4, false);
    // Original M0975-M0980 input, with independently evaluated literal sizes:
    // 2*(2*3)^2=72; 2*((2*4)^2+1)=130; one Pfaffian per QP=2.
    assert_eq!(data.slater_elm.len(), 72);
    assert_eq!(data.inv_m.len(), 130);
    assert_eq!(data.pf_m.len(), 2);
    assert_eq!(data.slater_elm_real.len(), 72);
    assert_eq!(data.inv_m_real.len(), 130);
    assert_eq!(data.pf_m_real.len(), 2);
    let zero = Complex64::new(0.0, 0.0);
    assert!(data
        .slater_elm
        .as_slice()
        .iter()
        .all(|value| *value == zero));
    assert!(data.inv_m.as_slice().iter().all(|value| *value == zero));
    assert!(data.pf_m.iter().all(|value| *value == zero));
    assert!(data
        .slater_elm_real
        .as_slice()
        .iter()
        .all(|value| *value == 0.0));
    assert!(data.inv_m_real.as_slice().iter().all(|value| *value == 0.0));
    assert!(data.pf_m_real.iter().all(|value| *value == 0.0));
}

#[test]
fn all_complex_helper_keeps_unused_real_buffers_genuinely_empty() {
    for qp in [1, 2] {
        let data = SlaterMatrixData::zeros(qp, 3, 4, true);
        assert_eq!(data.pf_m.len(), qp);
        assert!(data.slater_elm_real.is_empty());
        assert!(data.inv_m_real.is_empty());
        assert!(data.pf_m_real.is_empty());
        assert_eq!(data.slater_elm_real.n_qp_full(), 0);
        assert_eq!(data.inv_m_real.n_qp_full(), 0);
    }
}

#[test]
fn raw_zero_plane_containers_remain_empty_without_helper_normalization() {
    let slater = SlaterElmFlat::<f64>::zeros(0, 0);
    let inverse = InvMColMajor::<f64>::zeros(0, 0);
    assert_eq!(slater.n_qp_full(), 0);
    assert_eq!(slater.n_site2(), 0);
    assert!(slater.is_empty());
    assert_eq!(inverse.n_qp_full(), 0);
    assert_eq!(inverse.n_size(), 0);
    assert!(inverse.is_empty());
}
