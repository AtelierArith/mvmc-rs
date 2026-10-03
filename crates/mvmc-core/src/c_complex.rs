//! Pure-Rust quotient in the order used by clang's compiler-rt __divdc3.
//!
//! Derived from LLVM 17 compiler-rt/lib/builtins/divdc3.c.
//! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Full notice: LICENSE-llvm.txt. Native C oracle expectations live separately.
//! GNU/Linux selects the libgcc quotient; see c_complex_gnu.rs and its notice.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) use crate::c_complex_gnu::divide;

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
use num_complex::Complex64;

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn scale(mut value: f64, mut power: i32) -> f64 {
    // Intermediate downscales remain normal until the final multiplication,
    // avoiding double rounding when the result is subnormal.
    while power < -1022 {
        value *= f64::from_bits(54_u64 << 52); // 2^-969
        power += 969;
    }
    while power > 1023 {
        value *= f64::from_bits(2046_u64 << 52); // 2^1023
        power -= 1023;
    }
    value * f64::from_bits(((power + 1023) as u64) << 52)
}

/// C complex division, with binary exponent scaling and nonfinite recovery.
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn divide(z: Complex64, w: Complex64) -> Complex64 {
    let (mut a, mut b, mut c, mut d) = (z.re, z.im, w.re, w.im);
    let magnitude = c.abs().max(d.abs());
    let mut power = 0;
    if magnitude.is_finite() && magnitude != 0.0 {
        let bits = magnitude.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        power = if exponent == 0 {
            63 - bits.leading_zeros() as i32 - 1074
        } else {
            exponent - 1023
        };
        c = scale(c, -power);
        d = scale(d, -power);
    }
    let denominator = c * c + d * d;
    let mut result = Complex64::new(
        scale((a * c + b * d) / denominator, -power),
        scale((b * c - a * d) / denominator, -power),
    );
    if result.re.is_nan() && result.im.is_nan() {
        if denominator == 0.0 && (!a.is_nan() || !b.is_nan()) {
            let infinity = f64::INFINITY.copysign(c);
            result = Complex64::new(infinity * a, infinity * b);
        } else if (a.is_infinite() || b.is_infinite()) && c.is_finite() && d.is_finite() {
            a = f64::from(u8::from(a.is_infinite())).copysign(a);
            b = f64::from(u8::from(b.is_infinite())).copysign(b);
            result = Complex64::new(
                f64::INFINITY * (a * c + b * d),
                f64::INFINITY * (b * c - a * d),
            );
        } else if magnitude.is_infinite() && a.is_finite() && b.is_finite() {
            c = f64::from(u8::from(c.is_infinite())).copysign(c);
            d = f64::from(u8::from(d.is_infinite())).copysign(d);
            result = Complex64::new(0.0 * (a * c + b * d), 0.0 * (b * c - a * d));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn scaled_complex_quotients_and_range_recovery_match_native_c() {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let input =
            include_str!("../../../tests/fixtures/interall/c_complex_division_linux_gnu.txt");
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        let input = include_str!("../../../tests/fixtures/interall/c_complex_division.txt");
        for (case, line) in input
            .lines()
            .filter(|line| !line.starts_with('#'))
            .enumerate()
        {
            let values: Vec<_> = line
                .split_whitespace()
                .map(|value| f64::from_bits(u64::from_str_radix(value, 16).unwrap()))
                .collect();
            let actual = divide(
                Complex64::new(values[0], values[1]),
                Complex64::new(values[2], values[3]),
            );
            for (component, expected) in [(actual.re, values[4]), (actual.im, values[5])] {
                if expected.is_nan() {
                    // C does not specify a NaN payload/sign for invalid
                    // arithmetic. Recover finite/infinite values exactly.
                    assert!(component.is_nan(), "case {case}: {line}");
                } else {
                    assert_eq!(
                        component.to_bits(),
                        expected.to_bits(),
                        "case {case}: {line}"
                    );
                }
            }
        }
    }
}
