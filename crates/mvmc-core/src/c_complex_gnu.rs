//! Pure-Rust double-complex division in the GNU libgcc arithmetic order.
//!
//! Derived from GCC 13.3.0 `libgcc/libgcc2.c`, the `L_divdc3` specialization.
//! Copyright (C) 1989-2023 Free Software Foundation, Inc.
//! SPDX-License-Identifier: GPL-3.0-or-later WITH GCC-exception-3.1
//! Notices: repository LICENSE and this crate's LICENSE-gcc-runtime.txt.
//!
//! GNU/Linux C compilers use the scaled Smith quotient, while the archived
//! macOS compiler-rt contract uses exponent-scaled norm-squared division.
//! Keep individual products and sums separate; their rounding is observable.

use num_complex::Complex64;

const BIG: f64 = f64::MAX / 2.0;
const SMALL: f64 = f64::MIN_POSITIVE;
const SCALE_THRESHOLD: f64 = f64::EPSILON;
const SCALE_UP: f64 = 1.0 / f64::EPSILON;
const NUMERATOR_LIMIT: f64 = BIG * SCALE_THRESHOLD;

pub(crate) fn divide(z: Complex64, w: Complex64) -> Complex64 {
    let (mut a, mut b, mut c, mut d) = (z.re, z.im, w.re, w.im);
    let imaginary_dominant = c.abs() < d.abs();
    let largest = |c: f64, d: f64| if imaginary_dominant { d.abs() } else { c.abs() };
    if largest(c, d) >= BIG {
        a /= 2.0;
        b /= 2.0;
        c /= 2.0;
        d /= 2.0;
    }
    let largest = largest(c, d);
    if largest < SCALE_THRESHOLD
        || ((a.abs() < SMALL && b.abs() < NUMERATOR_LIMIT && largest < NUMERATOR_LIMIT)
            || (b.abs() < SMALL && a.abs() < NUMERATOR_LIMIT && largest < NUMERATOR_LIMIT))
    {
        a *= SCALE_UP;
        b *= SCALE_UP;
        c *= SCALE_UP;
        d *= SCALE_UP;
    }
    let mut result = if imaginary_dominant {
        let ratio = c / d;
        let denominator = c * ratio + d;
        if ratio.abs() > SMALL {
            Complex64::new((a * ratio + b) / denominator, (b * ratio - a) / denominator)
        } else {
            Complex64::new(
                (c * (a / d) + b) / denominator,
                (c * (b / d) - a) / denominator,
            )
        }
    } else {
        let ratio = d / c;
        let denominator = d * ratio + c;
        if ratio.abs() > SMALL {
            Complex64::new((b * ratio + a) / denominator, (b - a * ratio) / denominator)
        } else {
            Complex64::new(
                (a + d * (b / c)) / denominator,
                (b - d * (a / c)) / denominator,
            )
        }
    };
    if result.re.is_nan() && result.im.is_nan() {
        if c == 0.0 && d == 0.0 && (!a.is_nan() || !b.is_nan()) {
            let infinity = f64::INFINITY.copysign(c);
            result = Complex64::new(infinity * a, infinity * b);
        } else if (a.is_infinite() || b.is_infinite()) && c.is_finite() && d.is_finite() {
            a = f64::from(u8::from(a.is_infinite())).copysign(a);
            b = f64::from(u8::from(b.is_infinite())).copysign(b);
            result = Complex64::new(
                f64::INFINITY * (a * c + b * d),
                f64::INFINITY * (b * c - a * d),
            );
        } else if (c.is_infinite() || d.is_infinite()) && a.is_finite() && b.is_finite() {
            c = f64::from(u8::from(c.is_infinite())).copysign(c);
            d = f64::from(u8::from(d.is_infinite())).copysign(d);
            result = Complex64::new(0.0 * (a * c + b * d), 0.0 * (b * c - a * d));
        }
    }
    result
}
