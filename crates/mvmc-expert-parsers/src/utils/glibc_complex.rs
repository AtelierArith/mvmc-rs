//! Pure-Rust ports of the glibc 2.39 complex elementary functions `cexp`, `clog`, `ccosh` and
//! `ctanh` (issue #470), the functions the C RBM code calls (`rbm.c`: `cexp`, `clog(ccosh(z))`,
//! `ctanh`).
//!
//! Derived from the GNU C Library 2.39 `math/s_cexp_template.c`, `s_clog_template.c`,
//! `s_ccosh_template.c`, `s_ctanh_template.c` and `sysdeps/ieee754/dbl-64/x2y2m1.c`
//! (Copyright (C) 1997-2024 Free Software Foundation, Inc.; LGPL-2.1-or-later; this crate is
//! GPL-3.0-or-later). Source files, SHA-256 and the C probe that checks this port bitwise are
//! recorded in `c_toolbox/glibc_complex_470/README.md`.
//!
//! The scalar building blocks (`exp`, `log`, `log1p`, `hypot`, `atan2`, `sin`, `cos`, `sinh`,
//! `cosh`) are the platform libm through the Rust `f64` methods, exactly the functions the glibc
//! templates call; the operation order, branch structure and special-case handling of the
//! templates are kept line by line (exception-flag side effects have no Rust counterpart and are
//! omitted). On Linux x86_64 with glibc the results are bit-identical to the C library
//! (`tests` and the probe fixture); on macOS the system libm differs from glibc in the building
//! blocks, so only the algorithm, not the last bit, is shared.

// glibc writes `imag - imag` to produce NaN from an infinity and keeps explicit range tests;
// both are kept verbatim so the port can be compared with the C source line by line.
#![allow(clippy::eq_op, clippy::manual_range_contains)]

use num_complex::Complex64;

const MAX: f64 = f64::MAX;
const MIN: f64 = f64::MIN_POSITIVE;
const EPSILON: f64 = f64::EPSILON;
const MANT_DIG: i32 = 53;

/// `(int) ((DBL_MAX_EXP - 1) * M_LN2)` = 709, the overflow threshold of `exp`.
const T_EXP: f64 = 709.0;
/// `(int) ((DBL_MAX_EXP - 1) * M_LN2 / 2)` = 354 for `ctanh`.
const T_TANH: f64 = 354.0;

fn sincos_or_small(y: f64) -> (f64, f64) {
    // glibc: `if (fabs (imag) > DBL_MIN) sincos (imag, &sinix, &cosix); else { sinix = imag; cosix = 1; }`
    if y.abs() > MIN {
        (y.sin(), y.cos())
    } else {
        (y, 1.0)
    }
}

/// Exact `x*x + y*y - 1` rounded once, `glibc __x2y2m1` (`sysdeps/ieee754/dbl-64/x2y2m1.c`):
/// valid for `1 > x >= y >= eps/2` and `x*x + y*y >= 0.5`.
pub fn x2y2m1(x: f64, y: f64) -> f64 {
    fn mul_split(x: f64, y: f64) -> (f64, f64) {
        let hi = x * y;
        (hi, x.mul_add(y, -hi))
    }
    fn add_split(x: f64, y: f64) -> (f64, f64) {
        // Dekker's algorithm; |x| >= |y|.
        let hi = x + y;
        (hi, (x - hi) + y)
    }
    fn sort_abs(v: &mut [f64]) {
        // glibc `qsort` (merge sort, stable) with the `fabs` comparison.
        v.sort_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut vals = [0.0_f64; 5];
    (vals[1], vals[0]) = mul_split(x, x);
    (vals[3], vals[2]) = mul_split(y, y);
    vals[4] = -1.0;
    sort_abs(&mut vals);
    for i in 0..=3 {
        let (hi, lo) = add_split(vals[i + 1], vals[i]);
        vals[i + 1] = hi;
        vals[i] = lo;
        sort_abs(&mut vals[i + 1..]);
    }
    vals[4] + vals[3] + vals[2] + vals[1] + vals[0]
}

/// glibc `cexp`.
pub fn cexp(x: Complex64) -> Complex64 {
    let (re, im) = (x.re, x.im);
    if re.is_finite() {
        if im.is_finite() {
            let (mut sinix, mut cosix) = sincos_or_small(im);
            let mut re = re;
            if re > T_EXP {
                let exp_t = T_EXP.exp();
                re -= T_EXP;
                sinix *= exp_t;
                cosix *= exp_t;
                if re > T_EXP {
                    re -= T_EXP;
                    sinix *= exp_t;
                    cosix *= exp_t;
                }
            }
            if re > T_EXP {
                // Overflow (original real part of x > 3t).
                Complex64::new(MAX * cosix, MAX * sinix)
            } else {
                let exp_val = re.exp();
                Complex64::new(exp_val * cosix, exp_val * sinix)
            }
        } else {
            Complex64::new(f64::NAN, f64::NAN)
        }
    } else if re.is_infinite() {
        if im.is_finite() {
            let value = if re.is_sign_negative() {
                0.0
            } else {
                f64::INFINITY
            };
            if im == 0.0 {
                Complex64::new(value, im)
            } else {
                let (sinix, cosix) = sincos_or_small(im);
                Complex64::new(value.copysign(cosix), value.copysign(sinix))
            }
        } else if !re.is_sign_negative() {
            Complex64::new(f64::INFINITY, im - im)
        } else {
            Complex64::new(0.0, 0.0_f64.copysign(im))
        }
    } else {
        // Real part is NaN: NaN + iNaN unless the imaginary part is zero.
        if im == 0.0 {
            Complex64::new(f64::NAN, im)
        } else {
            Complex64::new(f64::NAN, f64::NAN)
        }
    }
}

/// glibc `clog`.
pub fn clog(x: Complex64) -> Complex64 {
    let (re, im) = (x.re, x.im);
    if re == 0.0 && im == 0.0 {
        // Real and imaginary part are 0.0.
        let imag = if re.is_sign_negative() {
            std::f64::consts::PI
        } else {
            0.0
        }
        .copysign(im);
        return Complex64::new(-1.0 / re.abs(), imag);
    }
    if !re.is_nan() && !im.is_nan() {
        let (mut absx, mut absy) = (re.abs(), im.abs());
        let mut scale: i32 = 0;
        if absx < absy {
            std::mem::swap(&mut absx, &mut absy);
        }
        if absx > MAX / 2.0 {
            scale = -1;
            absx *= 0.5;
            absy = if absy >= MIN * 2.0 { absy * 0.5 } else { 0.0 };
        } else if absx < MIN && absy < MIN {
            scale = MANT_DIG;
            let factor = (MANT_DIG as f64).exp2();
            absx *= factor;
            absy *= factor;
        }
        let real = if absx == 1.0 && scale == 0 {
            (absy * absy).ln_1p() / 2.0
        } else if absx > 1.0 && absx < 2.0 && absy < 1.0 && scale == 0 {
            let mut d2m1 = (absx - 1.0) * (absx + 1.0);
            if absy >= EPSILON {
                d2m1 += absy * absy;
            }
            d2m1.ln_1p() / 2.0
        } else if absx < 1.0 && absx >= 0.5 && absy < EPSILON / 2.0 && scale == 0 {
            let d2m1 = (absx - 1.0) * (absx + 1.0);
            d2m1.ln_1p() / 2.0
        } else if absx < 1.0 && absx >= 0.5 && scale == 0 && absx * absx + absy * absy >= 0.5 {
            x2y2m1(absx, absy).ln_1p() / 2.0
        } else {
            let d = absx.hypot(absy);
            d.ln() - f64::from(scale) * std::f64::consts::LN_2
        };
        Complex64::new(real, im.atan2(re))
    } else {
        let real = if re.is_infinite() || im.is_infinite() {
            f64::INFINITY
        } else {
            f64::NAN
        };
        Complex64::new(real, f64::NAN)
    }
}

/// glibc `ccosh`.
pub fn ccosh(x: Complex64) -> Complex64 {
    let (re, im) = (x.re, x.im);
    if re.is_finite() {
        if im.is_finite() {
            let (mut sinix, mut cosix) = sincos_or_small(im);
            if re.abs() > T_EXP {
                let exp_t = T_EXP.exp();
                let mut rx = re.abs();
                if re.is_sign_negative() {
                    sinix = -sinix;
                }
                rx -= T_EXP;
                sinix *= exp_t / 2.0;
                cosix *= exp_t / 2.0;
                if rx > T_EXP {
                    rx -= T_EXP;
                    sinix *= exp_t;
                    cosix *= exp_t;
                }
                if rx > T_EXP {
                    // Overflow (original real part of x > 3t).
                    Complex64::new(MAX * cosix, MAX * sinix)
                } else {
                    let exp_val = rx.exp();
                    Complex64::new(exp_val * cosix, exp_val * sinix)
                }
            } else {
                Complex64::new(re.cosh() * cosix, re.sinh() * sinix)
            }
        } else {
            Complex64::new(im - im, if re == 0.0 { 0.0 } else { f64::NAN })
        }
    } else if re.is_infinite() {
        // Real part is infinite; `icls > FP_ZERO` is "finite and nonzero".
        if im.is_finite() && im != 0.0 {
            let (sinix, cosix) = sincos_or_small(im);
            Complex64::new(
                f64::INFINITY.copysign(cosix),
                f64::INFINITY.copysign(sinix) * 1.0_f64.copysign(re),
            )
        } else if im == 0.0 {
            Complex64::new(f64::INFINITY, im * 1.0_f64.copysign(re))
        } else {
            Complex64::new(f64::INFINITY, im - im)
        }
    } else {
        Complex64::new(f64::NAN, if im == 0.0 { im } else { f64::NAN })
    }
}

/// glibc `ctanh`.
pub fn ctanh(x: Complex64) -> Complex64 {
    let (re, im) = (x.re, x.im);
    if !re.is_finite() || !im.is_finite() {
        if re.is_infinite() {
            let real = 1.0_f64.copysign(re);
            let imag = if im.is_finite() && im.abs() > 1.0 {
                let (sinix, cosix) = (im.sin(), im.cos());
                0.0_f64.copysign(sinix * cosix)
            } else {
                0.0_f64.copysign(im)
            };
            Complex64::new(real, imag)
        } else if im == 0.0 {
            x
        } else {
            let real = if re == 0.0 { re } else { f64::NAN };
            Complex64::new(real, f64::NAN)
        }
    } else {
        // tanh(x+iy) = (sinh(x)*cosh(x) + i*sin(y)*cos(y)) / (sinh(x)^2 + cos(y)^2)
        let (sinix, cosix) = sincos_or_small(im);
        if re.abs() > T_TANH {
            // Avoid intermediate overflow when the imaginary part may be subnormal.
            let exp_2t = (2.0 * T_TANH).exp();
            let real = 1.0_f64.copysign(re);
            let mut imag = 4.0 * sinix * cosix;
            let mut rx = re.abs();
            rx -= T_TANH;
            imag /= exp_2t;
            if rx > T_TANH {
                // Underflow (original real part of x has absolute value > 2t).
                imag /= exp_2t;
            } else {
                imag /= (2.0 * rx).exp();
            }
            Complex64::new(real, imag)
        } else {
            let (sinhrx, coshrx) = if re.abs() > MIN {
                (re.sinh(), re.cosh())
            } else {
                (re, 1.0)
            };
            let den = if sinhrx.abs() > cosix.abs() * EPSILON {
                sinhrx * sinhrx + cosix * cosix
            } else {
                cosix * cosix
            };
            Complex64::new(sinhrx * coshrx / den, sinix * cosix / den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Vec<f64> {
        line.split_whitespace()
            .map(|w| f64::from_bits(u64::from_str_radix(w, 16).unwrap()))
            .collect()
    }

    fn same(actual: f64, expected: f64) -> bool {
        if expected.is_nan() {
            return actual.is_nan();
        }
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            // glibc 2.39 on x86_64: the port calls the same libm building blocks in the same
            // order, so every result is bit-identical (sign of zero included).
            actual.to_bits() == expected.to_bits()
        } else {
            // Other platform libm (macOS): same algorithm, building blocks differ in the last
            // bits; compare to 16 eps of the component plus four subnormal quanta.
            actual == expected
                || (actual - expected).abs()
                    <= 16.0 * f64::EPSILON * actual.abs().max(expected.abs()) + 4.0 * 5e-324
        }
    }

    #[test]
    fn ports_match_the_glibc_probe_fixture() {
        let text = include_str!("../../../../tests/fixtures/glibc_complex_470/probe.txt");
        let mut cases = 0;
        let mut failures = Vec::new();
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let v = parse(line);
            assert_eq!(v.len(), 10, "{line}");
            let z = Complex64::new(v[0], v[1]);
            let results: [(&str, Complex64, (f64, f64)); 4] = [
                ("cexp", cexp(z), (v[2], v[3])),
                ("clog", clog(z), (v[4], v[5])),
                ("ccosh", ccosh(z), (v[6], v[7])),
                ("ctanh", ctanh(z), (v[8], v[9])),
            ];
            for (name, actual, (re, im)) in results {
                if !same(actual.re, re) || !same(actual.im, im) {
                    failures.push(format!(
                        "{name}({:e}, {:e}): Rust ({:e}, {:e}) vs C ({:e}, {:e})",
                        z.re, z.im, actual.re, actual.im, re, im
                    ));
                }
            }
            cases += 1;
        }
        assert!(cases >= 1000, "{cases}");
        assert!(
            failures.is_empty(),
            "{} mismatches of {cases} cases; first: {:#?}",
            failures.len(),
            &failures[..failures.len().min(8)]
        );
    }

    #[test]
    fn x2y2m1_is_the_exact_rounded_value() {
        // 0.8^2 + 0.7^2 - 1 = 0.13 computed without cancellation error
        let x = 0.8_f64;
        let y = 0.7_f64;
        let exact = 0.13;
        let v = x2y2m1(x, y);
        assert!((v - exact).abs() < 1e-16, "{v}");
        // the naive formula loses bits where the exact one does not (x^2 + y^2 close to 1)
        let (x, y) = (0.99999999, 1.5e-4);
        let naive = x * x + y * y - 1.0;
        let accurate = x2y2m1(x, y);
        let reference = (x - 1.0) * (x + 1.0) + y * y;
        assert!((accurate - reference).abs() <= (naive - reference).abs());
    }
}
