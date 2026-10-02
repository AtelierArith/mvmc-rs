//! Julia 1.13.1 complex operations used by RBM ratios and derivatives.
//! Julia portions: MIT; see mvmc-expert-parsers/LICENSE-julia-math.
use num_complex::Complex64;

use mvmc_expert_parsers::utils::{julia_exp, julia_log, julia_trig};

pub(crate) fn exp(z: Complex64) -> Complex64 {
    let (x, y) = (z.re, z.im);
    if x.is_nan() {
        return Complex64::new(x, if y == 0.0 { y } else { x });
    }
    if !y.is_finite() {
        return if x == f64::INFINITY {
            Complex64::new(-x, f64::NAN)
        } else if x == f64::NEG_INFINITY {
            Complex64::new(-0.0, 0.0f64.copysign(y))
        } else {
            Complex64::new(f64::NAN, f64::NAN)
        };
    }
    let e = julia_exp::exp(x);
    if y == 0.0 {
        Complex64::new(e, y)
    } else {
        Complex64::new(e * julia_trig::cos(y), e * julia_trig::sin(y))
    }
}
fn ldexp(mut x: f64, mut exponent: i32) -> f64 {
    while exponent > 1023 {
        x *= f64::from_bits(2046u64 << 52);
        exponent -= 1023;
    }
    while exponent < -1022 {
        x *= f64::from_bits(1u64 << 52);
        exponent += 1022;
    }
    x * f64::from_bits(((exponent + 1023) as u64) << 52)
}
pub(super) fn log(z: Complex64) -> Complex64 {
    let (x, y) = (z.re, z.im);
    let mut rho = x * x + y * y;
    let mut k = 0;
    if !rho.is_finite() && (x.is_infinite() || y.is_infinite()) {
        rho = f64::INFINITY;
    } else if rho.is_infinite()
        || (rho == 0.0 && (x != 0.0 || y != 0.0))
        || rho < f64::from_bits(1) / (2.0 * f64::EPSILON * f64::EPSILON)
    {
        let magnitude = x.abs().max(y.abs());
        if magnitude != 0.0 {
            let bits = magnitude.to_bits();
            let exponent = ((bits >> 52) & 0x7ff) as i32;
            k = if exponent == 0 {
                -1023 - ((bits & 0x000f_ffff_ffff_ffff).leading_zeros() as i32 - 12)
            } else {
                exponent - 1023
            };
        }
        let xk = ldexp(x, -k);
        let yk = ldexp(y, -k);
        rho = xk * xk + yk * yk;
    }
    let (theta, beta) = if x.abs() < y.abs() {
        (x.abs(), y.abs())
    } else {
        (y.abs(), x.abs())
    };
    let real = if k == 0 && 0.5 < beta * beta && (beta <= 1.25 || rho < 3.0) {
        julia_log::log1p((beta - 1.0) * (beta + 1.0) + theta * theta) / 2.0
    } else {
        julia_log::log(rho) / 2.0 + k as f64 * std::f64::consts::LN_2
    };
    Complex64::new(real, julia_trig::atan2(y, x))
}
pub(super) fn log1p(z: Complex64) -> Complex64 {
    if z.re.is_finite() {
        if z.im.is_infinite() {
            return log(z);
        }
        let u = Complex64::new(1.0 + z.re, z.im);
        if u == Complex64::new(1.0, 0.0) {
            z
        } else if u.re <= 0.0 {
            log(u)
        } else {
            log(u) * crate::julia_complex::divide(z, u - Complex64::new(1.0, 0.0))
        }
    } else if z.re.is_nan() {
        Complex64::new(z.re, z.re)
    } else if z.im.is_finite() {
        Complex64::new(
            f64::INFINITY,
            (if z.re > 0.0 {
                0.0
            } else {
                std::f64::consts::PI
            })
            .copysign(z.im),
        )
    } else {
        Complex64::new(f64::INFINITY, f64::NAN)
    }
}
pub(super) fn tanh(z: Complex64) -> Complex64 {
    let (x, y) = (z.re, z.im);
    if x.is_nan() && y == 0.0 {
        return z;
    }
    if 4.0 * x.abs() > julia_log::log(f64::MAX) + std::f64::consts::LN_2 {
        let sign = y * if y.is_finite() {
            julia_trig::sin(2.0 * y.abs())
        } else {
            1.0
        };
        return Complex64::new(1.0f64.copysign(x), 0.0f64.copysign(sign));
    }
    let t = julia_trig::tan(y);
    let beta = 1.0 + t * t;
    let s = julia_trig::sinh(x);
    let rho = (1.0 + s * s).sqrt();
    if t.is_infinite() {
        Complex64::new(rho / s, 1.0 / t)
    } else {
        Complex64::new(beta * rho * s, t) / (1.0 + beta * s * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rbm_complex_primitive_bits_match_julia_full_range_and_quadrant_boundaries() {
        let input = include_str!("../../../../tests/fixtures/rbm/production/math.txt");
        for (line_number, line) in input
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.starts_with('#'))
        {
            let words: Vec<u64> = line
                .split_whitespace()
                .map(|w| u64::from_str_radix(w, 16).unwrap())
                .collect();
            let z = Complex64::new(f64::from_bits(words[0]), f64::from_bits(words[1]));
            for (i, (name, actual)) in [
                ("exp", exp(z)),
                ("log", log(z)),
                ("log1p", log1p(z)),
                ("tanh", tanh(z)),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(
                    [actual.re.to_bits(), actual.im.to_bits()],
                    [words[2 + 2 * i], words[3 + 2 * i]],
                    "line {line_number}, {name}({z})"
                );
            }
        }
    }
}
