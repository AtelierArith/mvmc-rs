//! Focused public IP contracts, not a live-MPI or full sampling parity gate.
//! Literal source/provenance and the optional-context API distinction are in
//! docs/reference/c-to-julia/verification/issue-179-qp-literals.md.
use mvmc_core::observables::{
    calculate_ip_complex, calculate_ip_real, calculate_log_ip_complex, calculate_log_ip_real,
};
use mvmc_core::sampling::driver::sampling_log_ip_complex;
use mvmc_core::{Reducer, SingleProcessReducer};
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
use num_complex::Complex64;

fn data(weights: &[Complex64]) -> ExpertModeData {
    ExpertModeData {
        qp_weights: Some(QuantumProjectionWeights {
            qp_full_weight: weights.to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn complex_fixture() -> (ExpertModeData, [Complex64; 3]) {
    (
        data(&[2.0, 3.0, 5.0].map(|x| Complex64::new(x, 0.0))),
        [7.0, 11.0, 13.0].map(|x| Complex64::new(x, 0.0)),
    )
}

fn real_fixture() -> (ExpertModeData, [f64; 2]) {
    (
        data(&[Complex64::new(2.0, 9.0), Complex64::new(3.0, 0.0)]),
        [7.0, 11.0],
    )
}

fn close(actual: f64, expected: f64) {
    // Three small products/sums or one logarithm. Eight epsilons at the
    // expected result's scale cover rounding/math-provider differences;
    // this is not a downstream SR/CG or trajectory error budget.
    let bound = 8.0 * f64::EPSILON * expected.abs().max(1.0);
    assert!(
        actual.is_finite() && (actual - expected).abs() <= bound,
        "actual={actual:.17e}, expected={expected:.17e}, bound={bound:.17e}"
    );
}

fn complex_close(actual: Complex64, expected: Complex64) {
    close(actual.re, expected.re);
    close(actual.im, expected.im);
}

#[test]
fn m1154_complex_full_range_is_112() {
    let (data, pf) = complex_fixture();
    complex_close(
        calculate_ip_complex(&pf, 0, 3, &data),
        Complex64::new(112.0, 0.0),
    );
}

#[test]
fn m1155_complex_middle_range_is_33() {
    let (data, pf) = complex_fixture();
    complex_close(
        calculate_ip_complex(&pf, 1, 2, &data),
        Complex64::new(33.0, 0.0),
    );
}

#[test]
fn m1156_complex_empty_range_is_zero() {
    let (data, pf) = complex_fixture();
    // Empty range performs no arithmetic. Equality accepts either zero sign.
    assert_eq!(
        calculate_ip_complex(&pf, 1, 1, &data),
        Complex64::new(0.0, 0.0)
    );
}

#[test]
fn m1157_complex_full_range_log_is_log_112() {
    let (data, pf) = complex_fixture();
    complex_close(
        calculate_log_ip_complex(&pf, 0, 3, &data),
        Complex64::new(112.0_f64.ln(), 0.0),
    );
}

#[test]
fn m1158_real_overlap_ignores_weight_imaginary_part_and_is_47() {
    let (data, pf) = real_fixture();
    close(calculate_ip_real(&pf, 0, 2, &data), 47.0);
}

#[test]
fn m1159_real_empty_range_is_zero() {
    let (data, pf) = real_fixture();
    assert_eq!(calculate_ip_real(&pf, 1, 1, &data), 0.0);
}

#[test]
fn m1160_real_log_is_log_abs_47() {
    let (data, pf) = real_fixture();
    close(
        calculate_log_ip_real(&pf, 0, 2, &data),
        (47.0_f64.abs() + 1e-100).ln(),
    );
}

#[test]
fn m1162_explicit_serial_reducer_preserves_overlap_2() {
    let data = data(&[Complex64::new(1.0, 0.0)]);
    let pf = [Complex64::new(2.0, 0.0)];
    let reducer = SingleProcessReducer;
    let range = reducer.sampling_qp_range(pf.len());
    let mut ip = [calculate_ip_complex(&pf, range.start, range.end, &data)];
    reducer.sampling_sum_c64(&mut ip);
    complex_close(ip[0], Complex64::new(2.0, 0.0));
    complex_close(
        sampling_log_ip_complex(&pf, &data, &reducer),
        Complex64::new(2.0_f64.ln(), 0.0),
    );
}
