#[path = "../../../tests/support/numerical_comparison.rs"]
mod comparison;

#[test]
fn arithmetic_nan_bits_are_portable_without_relaxing_literal_bits() {
    assert!(comparison::arithmetic_bits_match(
        0x7ff8000000000000,
        0xfff8000000000000
    ));
    assert!(comparison::arithmetic_bits_match(
        0x7ff8000000000042,
        0x7ff8000000000000
    ));
    assert!(!comparison::arithmetic_bits_match(
        0x7ff8000000000000,
        0x7ff0000000000000
    ));
    assert!(!comparison::arithmetic_bits_match(0, 1 << 63));
    assert!(!comparison::arithmetic_bits_match(
        1.0_f64.to_bits(),
        1.0000000000000002_f64.to_bits()
    ));
}

#[test]
fn portable_comparison_handles_scale_zero_and_nonfinite_values() {
    use comparison::within;
    assert!(within(
        1.0 + 4.0 * f64::EPSILON,
        1.0,
        0.0,
        8.0 * f64::EPSILON
    ));
    assert!(!within(
        1.0 + 32.0 * f64::EPSILON,
        1.0,
        0.0,
        8.0 * f64::EPSILON
    ));
    assert!(within(1e-15, 0.0, 2e-15, 0.0));
    assert!(!within(1e-12, 0.0, 2e-15, 0.0));
    assert!(within(-0.0, 0.0, 0.0, 0.0));
    assert!(within(f64::NAN, f64::NAN, 0.0, 0.0));
    assert!(!within(f64::NAN, 1.0, 1.0, 1.0));
    assert!(!within(1.0, f64::NAN, 1.0, 1.0));
    assert!(within(f64::INFINITY, f64::INFINITY, 0.0, 0.0));
    assert!(!within(f64::INFINITY, f64::NEG_INFINITY, 1.0, 1.0));
    assert!(!within(f64::INFINITY, f64::MAX, 1.0, 1.0));
    assert!(!within(2.0, -2.0, 1e-14, 1e-14));
    assert!(!within(f64::MAX, -f64::MAX, f64::MAX, 1e-10));
}

#[test]
#[should_panic(expected = "different vector lengths")]
fn portable_comparison_rejects_a_missing_component() {
    comparison::assert_values_close([1.0], [1.0, 2.0], 1e-14, 1e-14, "shape");
}

#[test]
#[should_panic]
fn portable_comparison_rejects_invalid_bounds() {
    comparison::within(1.0, 1.0, f64::NAN, 0.0);
}

#[test]
fn numerical_output_accepts_roundoff_and_checks_structure() {
    comparison::assert_numeric_text(
        "# step value\n3 1.0000000000000002e0\n",
        "# step value\n3 1.0\n",
        0.0,
        8.0 * f64::EPSILON,
        &[0],
        "output",
    );
}

#[test]
#[should_panic(expected = "discrete row")]
fn numerical_output_rejects_a_changed_discrete_field() {
    comparison::assert_numeric_text("4 1.0\n", "3 1.0\n", 1.0, 1.0, &[0], "step");
}

#[test]
#[should_panic(expected = "column count")]
fn numerical_output_rejects_a_missing_column() {
    comparison::assert_numeric_text("3\n", "3 1.0\n", 1.0, 1.0, &[0], "shape");
}

#[test]
#[should_panic(expected = "error=")]
fn numerical_output_rejects_a_wrong_sign() {
    comparison::assert_numeric_text("3 -1.0\n", "3 1.0\n", 1e-14, 1e-14, &[0], "sign");
}
