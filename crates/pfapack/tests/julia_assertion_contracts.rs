//! Literal analytical cases from PfaPack.jl/test/runtests.jl, revision
//! 0dcf52c15caec63516d0703f36bfc8a4bc0e58d0, SHA256
//! 1f49686c051d49a93c7b56b4b3bd94fdbb55caadb0f69c93fff389f18dcc13a3.
//! No Julia/C/Fortran oracle runs. Rust's in-place API uses an explicit caller
//! clone to express Julia overwrite_a=false; no wrapper API is introduced.
use num_complex::Complex64;
use pfapack::{pfaffian_ltl_complex, pfaffian_ltl_real, SqMat};

fn original_four() -> [f64; 16] {
    // Column-major encoding of upper entries 1,2,3,4,5,6 and negative lower.
    [
        0.0, -1.0, -2.0, -3.0, 1.0, 0.0, -4.0, -5.0, 2.0, 4.0, 0.0, -6.0, 3.0, 5.0, 6.0, 0.0,
    ]
}

fn assert_pf(actual: f64, expected: f64) {
    // Tiny analytical case: at most one trailing 2x2 update plus products.
    // 1e-14 absolute covers rounding of the pivot division (1/3); zero rel.
    assert!((actual - expected).abs() <= 1e-14, "{actual} != {expected}");
}

#[test]
fn original_real_two_by_two_has_pfaffian_one() {
    let mut input = [0.0, -1.0, 1.0, 0.0];
    assert_pf(pfaffian_ltl_real(&mut SqMat::new(&mut input, 2)), 1.0);
}

#[test]
fn original_complex_two_by_two_has_pfaffian_one_plus_i() {
    let z = Complex64::new(0.0, 0.0);
    let mut input = [z, Complex64::new(-1.0, -1.0), Complex64::new(1.0, 1.0), z];
    let pf = pfaffian_ltl_complex(&mut SqMat::new(&mut input, 2));
    assert_pf(pf.re, 1.0);
    assert_pf(pf.im, 1.0);
}

#[test]
fn original_nonzero_odd_three_returns_zero() {
    let mut input = [0.0, -1.0, -2.0, 1.0, 0.0, -3.0, 2.0, 3.0, 0.0];
    assert_eq!(pfaffian_ltl_real(&mut SqMat::new(&mut input, 3)), 0.0);
}

#[test]
fn original_full_four_has_analytical_pfaffian_eight() {
    let mut input = original_four();
    let expected = 1.0 * 6.0 - 2.0 * 5.0 + 3.0 * 4.0;
    assert_pf(pfaffian_ltl_real(&mut SqMat::new(&mut input, 4)), expected);
}

#[test]
fn original_overwrite_case_mutates_work_but_caller_clone_preserves_source() {
    let source = original_four().to_vec();
    let expected = 1.0 * 6.0 - 2.0 * 5.0 + 3.0 * 4.0;
    let mut destructive = source.clone();
    let pf_destructive = pfaffian_ltl_real(&mut SqMat::new(&mut destructive, 4));
    assert_ne!(destructive, source);
    let preserved = original_four().to_vec();
    let before = preserved.clone();
    let mut work = preserved.clone();
    let pf_clone = pfaffian_ltl_real(&mut SqMat::new(&mut work, 4));
    assert_eq!(preserved, before);
    assert_ne!(work, preserved);
    assert_pf(pf_destructive, expected);
    assert_pf(pf_clone, expected);
    assert_pf(pf_clone, pf_destructive);
}

#[test]
fn original_zero_four_returns_zero() {
    let mut input = [0.0; 16];
    assert_eq!(pfaffian_ltl_real(&mut SqMat::new(&mut input, 4)), 0.0);
}
