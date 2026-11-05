//! Julia 1.13.1 real FSZ setup fixtures, including full post-retry RNG blocks.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::pfaffian::{calc_m_all_fsz_real, CalcMAllError};
use mvmc_core::{SlaterMatrixData, ThreadedPfaPackWorkspace};
use num_complex::Complex64;

fn sentinels(s: &mut SlaterMatrixData) {
    s.pf_m.fill(Complex64::new(37.0, 11.0));
    s.inv_m.as_mut_slice().fill(Complex64::new(37.0, 11.0));
    s.pf_m_real.fill(23.0);
    s.inv_m_real.as_mut_slice().fill(23.0);
}

#[test]
fn real_fsz_later_qp_failure_publishes_no_partial_results() {
    let mut s = SlaterMatrixData::zeros(4, 3, 1, false);
    sentinels(&mut s);
    for qp in 0..4 {
        s.slater_elm.set(qp, 0, 5, Complex64::new(1.25, 0.0));
        s.slater_elm.set(qp, 5, 0, Complex64::new(-1.25, 0.0));
    }
    s.slater_elm
        .qp_slice_mut(2)
        .fill(Complex64::new(f64::NAN, 0.0));
    let before = s.clone();
    let pool = ThreadedPfaPackWorkspace::new(2, 1);
    assert_eq!(
        calc_m_all_fsz_real(&[0, 2], &[0, 1], &mut s, 1, 3, 3, 1, &pool),
        Err(CalcMAllError::NonFinitePfaffian { qp: 2 }),
    );
    assert_eq!(s.pf_m, before.pf_m);
    assert_eq!(s.inv_m, before.inv_m);
    assert_eq!(s.pf_m_real, before.pf_m_real);
    assert_eq!(s.inv_m_real, before.inv_m_real);
}
