//! Floating-point parity only: RNG words, configurations and other discrete
//! contracts must use exact assertions. Callers choose bounds for their kernel.
#![allow(dead_code)]

use std::fmt::Display;

/// Componentwise absolute-plus-relative comparison. NaNs match only NaNs;
/// infinities must have the same sign. Signed zeros are numerically equivalent.
pub fn within(actual: f64, expected: f64, absolute: f64, relative: f64) -> bool {
    assert!(absolute.is_finite() && absolute >= 0.0);
    assert!(relative.is_finite() && relative >= 0.0);
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if !actual.is_finite() || !expected.is_finite() {
        return actual == expected;
    }
    let scale = actual.abs().max(expected.abs());
    let difference = (actual - expected).abs();
    if difference.is_infinite() {
        // Opposite finite endpoints can overflow subtraction. Compare their
        // normalized difference instead of accepting infinity <= infinity.
        (actual / scale - expected / scale).abs() <= absolute / scale + relative
    } else {
        difference <= absolute + relative * scale
    }
}

pub fn assert_close(
    actual: f64,
    expected: f64,
    absolute: f64,
    relative: f64,
    context: impl Display,
) {
    assert!(
        within(actual, expected, absolute, relative),
        "{context}: actual={actual:.17e}, expected={expected:.17e}, error={:.5e}, abs={absolute:.5e}, rel={relative:.5e}",
        (actual - expected).abs()
    );
}

pub fn assert_values_close(
    actual: impl IntoIterator<Item = f64>,
    expected: impl IntoIterator<Item = f64>,
    absolute: f64,
    relative: f64,
    context: impl Display,
) {
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut index = 0;
    loop {
        match (actual.next(), expected.next()) {
            (Some(a), Some(b)) => {
                assert_close(a, b, absolute, relative, format!("{context}[{index}]"));
                index += 1;
            }
            (None, None) => break,
            _ => panic!("{context}: different vector lengths at component {index}"),
        }
    }
}

/// Hexadecimal fixture encoding is lossless storage, not a comparison policy.
pub fn hex_values(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .map(|word| f64::from_bits(u64::from_str_radix(word, 16).unwrap()))
        .collect()
}

/// Numerical output parity, retaining row/column counts, headers, text fields
/// and explicitly declared integer columns. Writer-format tests remain exact.
pub fn assert_numeric_text(
    actual: &str,
    expected: &str,
    absolute: f64,
    relative: f64,
    integer_columns: &[usize],
    context: impl Display,
) {
    let actual: Vec<_> = actual.lines().collect();
    let expected: Vec<_> = expected.lines().collect();
    assert_eq!(actual.len(), expected.len(), "{context}: row count");
    for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
        if e.trim_start().starts_with('#') {
            assert_eq!(a, e, "{context}: header row {row}");
            continue;
        }
        let a: Vec<_> = a.split_whitespace().collect();
        let e: Vec<_> = e.split_whitespace().collect();
        assert_eq!(a.len(), e.len(), "{context}: row {row} column count");
        for (column, (a, e)) in a.iter().zip(&e).enumerate() {
            if integer_columns.contains(&column) || e.parse::<i64>().is_ok() {
                assert_eq!(a, e, "{context}: discrete row {row} column {column}");
            } else {
                match (a.parse::<f64>(), e.parse::<f64>()) {
                    (Ok(a), Ok(e)) => assert_close(
                        a,
                        e,
                        absolute,
                        relative,
                        format!("{context}: row {row} column {column}"),
                    ),
                    _ => assert_eq!(a, e, "{context}: text row {row} column {column}"),
                }
            }
        }
    }
}
