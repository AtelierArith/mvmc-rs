//! Complex number operations for quantum physics calculations
//!
//! This module provides utilities for complex number operations commonly used
//! in variational Monte Carlo calculations.

use num_complex::Complex64;

/// Calculate the absolute value squared (|z|²) of a complex number
///
/// This is more efficient than calculating |z| and then squaring it.
///
/// # Examples
///
/// ```
/// use mvmc_math::complex::abs_squared;
/// use num_complex::Complex64;
///
/// let z = Complex64::new(3.0, 4.0);
/// assert_eq!(abs_squared(z), 25.0);
/// ```
#[inline]
pub fn abs_squared(z: Complex64) -> f64 {
    z.re * z.re + z.im * z.im
}

/// Calculate the phase (argument) of a complex number
///
/// Returns the angle θ in the polar representation z = r·exp(iθ)
///
/// # Examples
///
/// ```
/// use mvmc_math::complex::phase;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// let z = Complex64::new(1.0, 1.0);
/// let theta = phase(z);
/// assert!((theta - PI / 4.0).abs() < 1e-10);
/// ```
#[inline]
pub fn phase(z: Complex64) -> f64 {
    z.im.atan2(z.re)
}

/// Calculate the logarithm of a complex number
///
/// Returns ln(z) = ln|z| + i·arg(z)
///
/// # Examples
///
/// ```
/// use mvmc_math::complex::log;
/// use num_complex::Complex64;
///
/// let z = Complex64::new(std::f64::consts::E, 0.0);
/// let result = log(z);
/// assert!((result.re - 1.0).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[inline]
pub fn log(z: Complex64) -> Complex64 {
    Complex64::new(z.norm().ln(), phase(z))
}

/// Calculate exp(z) for a complex number
///
/// Returns e^z = e^(re(z)) · (cos(im(z)) + i·sin(im(z)))
///
/// # Examples
///
/// ```
/// use mvmc_math::complex::exp;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// let z = Complex64::new(0.0, PI);
/// let result = exp(z);
/// assert!((result.re + 1.0).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
/// ```
#[inline]
pub fn exp(z: Complex64) -> Complex64 {
    let exp_re = z.re.exp();
    Complex64::new(exp_re * z.im.cos(), exp_re * z.im.sin())
}

/// Calculate the ratio of two complex numbers with numerical stability
///
/// This function handles the case where the denominator is very small
/// by checking for underflow.
///
/// # Examples
///
/// ```
/// use mvmc_math::complex::safe_div;
/// use num_complex::Complex64;
///
/// let a = Complex64::new(4.0, 2.0);
/// let b = Complex64::new(2.0, 0.0);
/// let result = safe_div(a, b).unwrap();
/// assert!((result.re - 2.0).abs() < 1e-10);
/// assert!((result.im - 1.0).abs() < 1e-10);
/// ```
pub fn safe_div(numerator: Complex64, denominator: Complex64) -> Option<Complex64> {
    let denom_abs_sq = abs_squared(denominator);

    if denom_abs_sq < f64::EPSILON {
        None
    } else {
        Some(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_abs_squared() {
        // Test with Pythagorean triple
        let z = Complex64::new(3.0, 4.0);
        assert_eq!(abs_squared(z), 25.0);

        // Test with zero
        let z = Complex64::new(0.0, 0.0);
        assert_eq!(abs_squared(z), 0.0);

        // Test with real number
        let z = Complex64::new(5.0, 0.0);
        assert_eq!(abs_squared(z), 25.0);

        // Test with imaginary number
        let z = Complex64::new(0.0, 5.0);
        assert_eq!(abs_squared(z), 25.0);
    }

    #[test]
    fn test_phase() {
        // Test with positive real number
        let z = Complex64::new(1.0, 0.0);
        assert_relative_eq!(phase(z), 0.0, epsilon = 1e-10);

        // Test with positive imaginary number
        let z = Complex64::new(0.0, 1.0);
        assert_relative_eq!(phase(z), PI / 2.0, epsilon = 1e-10);

        // Test with negative real number
        let z = Complex64::new(-1.0, 0.0);
        assert_relative_eq!(phase(z).abs(), PI, epsilon = 1e-10);

        // Test with 45 degree angle
        let z = Complex64::new(1.0, 1.0);
        assert_relative_eq!(phase(z), PI / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log() {
        // Test with e
        let z = Complex64::new(std::f64::consts::E, 0.0);
        let result = log(z);
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-10);

        // Test with i (imaginary unit)
        let z = Complex64::new(0.0, 1.0);
        let result = log(z);
        assert_relative_eq!(result.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.im, PI / 2.0, epsilon = 1e-10);

        // Test with -1
        let z = Complex64::new(-1.0, 0.0);
        let result = log(z);
        assert_relative_eq!(result.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.im.abs(), PI, epsilon = 1e-10);
    }

    #[test]
    fn test_exp() {
        // Test Euler's identity: e^(iπ) = -1
        let z = Complex64::new(0.0, PI);
        let result = exp(z);
        assert_relative_eq!(result.re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-10);

        // Test with real number
        let z = Complex64::new(1.0, 0.0);
        let result = exp(z);
        assert_relative_eq!(result.re, std::f64::consts::E, epsilon = 1e-10);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-10);

        // Test with imaginary number
        let z = Complex64::new(0.0, PI / 2.0);
        let result = exp(z);
        assert_relative_eq!(result.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.im, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_log_identity() {
        // Test that exp(log(z)) = z for various z
        let test_cases = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-2.0, 5.0),
        ];

        for z in test_cases {
            let result = exp(log(z));
            assert_relative_eq!(result.re, z.re, epsilon = 1e-10);
            assert_relative_eq!(result.im, z.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_safe_div() {
        // Test normal division
        let a = Complex64::new(4.0, 2.0);
        let b = Complex64::new(2.0, 0.0);
        let result = safe_div(a, b).unwrap();
        assert_relative_eq!(result.re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.im, 1.0, epsilon = 1e-10);

        // Test division by very small number
        let a = Complex64::new(1.0, 1.0);
        let b = Complex64::new(1e-20, 0.0);
        assert!(safe_div(a, b).is_none());

        // Test division by zero
        let a = Complex64::new(1.0, 1.0);
        let b = Complex64::new(0.0, 0.0);
        assert!(safe_div(a, b).is_none());
    }

    #[test]
    fn test_complex_division_properties() {
        // Test that (a/b) * b = a
        let a = Complex64::new(5.0, 3.0);
        let b = Complex64::new(2.0, 1.0);

        let quotient = safe_div(a, b).unwrap();
        let product = quotient * b;

        assert_relative_eq!(product.re, a.re, epsilon = 1e-10);
        assert_relative_eq!(product.im, a.im, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    fn complex_strategy() -> impl Strategy<Value = Complex64> {
        (-100.0..100.0, -100.0..100.0)
            .prop_map(|(re, im)| Complex64::new(re, im))
    }

    fn non_zero_complex_strategy() -> impl Strategy<Value = Complex64> {
        (-100.0..100.0, -100.0..100.0)
            .prop_filter("non-zero", |(re, im): &(f64, f64)| {
                re.abs() > 1e-6 || im.abs() > 1e-6
            })
            .prop_map(|(re, im)| Complex64::new(re, im))
    }

    proptest! {
        #[test]
        fn prop_abs_squared_non_negative(z in complex_strategy()) {
            let result = abs_squared(z);
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn prop_abs_squared_matches_norm_squared(z in complex_strategy()) {
            let result = abs_squared(z);
            let expected = z.norm().powi(2);
            prop_assert!((result - expected).abs() < 1e-10);
        }

        #[test]
        fn prop_phase_in_range(z in non_zero_complex_strategy()) {
            let theta = phase(z);
            prop_assert!(theta >= -PI && theta <= PI);
        }

        #[test]
        fn prop_exp_log_identity(z in non_zero_complex_strategy()) {
            let result = exp(log(z));
            prop_assert!((result.re - z.re).abs() < 1e-8);
            prop_assert!((result.im - z.im).abs() < 1e-8);
        }

        #[test]
        fn prop_log_exp_identity(z in complex_strategy()) {
            // Only test with bounded imaginary parts to avoid branch cuts
            if z.im.abs() < PI {
                let result = log(exp(z));
                prop_assert!((result.re - z.re).abs() < 1e-8);
                prop_assert!((result.im - z.im).abs() < 1e-8);
            }
        }

        #[test]
        fn prop_safe_div_multiplication(
            a in complex_strategy(),
            b in non_zero_complex_strategy()
        ) {
            if let Some(quotient) = safe_div(a, b) {
                let product = quotient * b;
                prop_assert!((product.re - a.re).abs() < 1e-8);
                prop_assert!((product.im - a.im).abs() < 1e-8);
            }
        }

        #[test]
        fn prop_exp_addition(
            a in -10.0..10.0,
            b in -10.0..10.0
        ) {
            // Test that exp(a + bi) = exp(a) * exp(bi)
            let z = Complex64::new(a, b);
            let result1 = exp(z);
            let result2 = a.exp() * exp(Complex64::new(0.0, b));
            prop_assert!((result1.re - result2.re).abs() < 1e-8);
            prop_assert!((result1.im - result2.im).abs() < 1e-8);
        }
    }
}

