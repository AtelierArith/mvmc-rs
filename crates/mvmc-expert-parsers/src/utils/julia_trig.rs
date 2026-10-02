//! Julia 1.13.1 Float64 trigonometric and hyperbolic operations.
//!
//! Port of Base special/trig.jl and the small-angle Cody-Waite paths in
//! special/rem_pio2.jl, including Payne-Hanek large-argument reduction.
//! Explicit mul_add calls mirror Julia's @horner and
//! argument reduction. System libm may round differently by one ulp.
//!
//! Julia portions: MIT (see LICENSE-julia-math).
//! Kernel and Cody-Waite portions:
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this software is freely
//! granted, provided that this notice is preserved.

#![allow(clippy::excessive_precision)]

const DS1: f64 = -1.66666666666666324348e-01;
const DS2: f64 = 8.33333333332248946124e-03;
const DS3: f64 = -1.98412698298579493134e-04;
const DS4: f64 = 2.75573137070700676789e-06;
const DS5: f64 = -2.50507602534068634195e-08;
const DS6: f64 = 1.58969099521155010221e-10;
const DC1: f64 = 4.16666666666666019037e-02;
const DC2: f64 = -1.38888888888741095749e-03;
const DC3: f64 = 2.48015872894767294178e-05;
const DC4: f64 = -2.75573143513906633035e-07;
const DC5: f64 = 2.08757232129817482790e-09;
const DC6: f64 = -1.13596475577881948265e-11;
const PIO2_1: f64 = 1.57079632673412561417e+00;
const PIO2_1T: f64 = 6.07710050650619224932e-11;

fn sin_kernel(hi: f64, lo: Option<f64>) -> f64 {
    let y2 = hi * hi;
    let y4 = y2 * y2;
    let r = DS4.mul_add(y2, DS3).mul_add(y2, DS2) + y2 * y4 * DS6.mul_add(y2, DS5);
    let y3 = y2 * hi;
    match lo {
        Some(lo) => hi - ((y2 * (0.5 * lo - y3 * r) - lo) - y3 * DS1),
        None => hi + y3 * (DS1 + y2 * r),
    }
}

fn cos_kernel(hi: f64, lo: Option<f64>) -> f64 {
    let y2 = hi * hi;
    let y4 = y2 * y2;
    let r = y2 * DC3.mul_add(y2, DC2).mul_add(y2, DC1)
        + y4 * y4 * DC6.mul_add(y2, DC5).mul_add(y2, DC4);
    let half_y2 = 0.5 * y2;
    let w = 1.0 - half_y2;
    match lo {
        Some(lo) => w + (((1.0 - w) - half_y2) + (y2 * r - hi * lo)),
        None => w + (((1.0 - w) - half_y2) + y2 * r),
    }
}

fn reduce(x: f64) -> (i32, f64, f64) {
    let high = ((x.to_bits() & 0x7fff_ffff_ffff_ffff) >> 32) as u32;
    if high >= 0x413921fb {
        return payne_hanek(x);
    }
    let precise = if high <= 0x400f6a7a {
        high & 0xfffff == 0x921fb
    } else if high <= 0x4015fdbc {
        high == 0x4012d97c
    } else {
        high == 0x401921fb
    };
    if !precise && high <= 0x401c463b {
        let n = if high <= 0x4002d97c {
            1
        } else if high <= 0x400f6a7a {
            2
        } else if high <= 0x4015fdbc {
            3
        } else {
            4
        };
        let n = if x > 0.0 { n } else { -n };
        let nf = n as f64;
        let z = (-nf).mul_add(PIO2_1, x);
        let hi = (-nf).mul_add(PIO2_1T, z);
        let lo = (-nf).mul_add(PIO2_1T, z - hi);
        return (n, hi, lo);
    }
    // Precise reduction near pi/2 and pi, where cancellation needs more bits.
    let nf = (x * (2.0 / std::f64::consts::PI)).round_ties_even();
    let mut r = (-nf).mul_add(PIO2_1, x);
    let mut w = nf * PIO2_1T;
    let mut hi = r - w;
    let exponent = |y: f64| ((y.to_bits() >> 52) & 0x7ff) as i32;
    let j = (high >> 20) as i32;
    if j - exponent(hi) > 16 {
        let t = r;
        w = nf * 6.07710050630396597660e-11;
        r = t - w;
        w = nf.mul_add(2.02226624879595063154e-21, -((t - r) - w));
        hi = r - w;
        if j - exponent(hi) > 49 {
            let t = r;
            w = nf * 2.02226624871116645580e-21;
            r = t - w;
            w = nf.mul_add(8.47842766036889956997e-32, -((t - r) - w));
            hi = r - w;
        }
    }
    (nf as i32, hi, (r - hi) - w)
}

/// Julia Base's Float64 sine.
pub fn sin(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    assert!(x.is_finite());
    if x.abs() < std::f64::consts::FRAC_PI_4 {
        if x.abs() < f64::EPSILON.sqrt() {
            return x;
        }
        return sin_kernel(x, None);
    }
    let (n, hi, lo) = reduce(x);
    match n & 3 {
        0 => sin_kernel(hi, Some(lo)),
        1 => cos_kernel(hi, Some(lo)),
        2 => -sin_kernel(hi, Some(lo)),
        _ => -cos_kernel(hi, Some(lo)),
    }
}

/// Julia Base's Float64 cosine.
pub fn cos(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    assert!(x.is_finite());
    if x.abs() < std::f64::consts::FRAC_PI_4 {
        if x.abs() < (f64::EPSILON / 2.0).sqrt() {
            return 1.0;
        }
        return cos_kernel(x, None);
    }
    let (n, hi, lo) = reduce(x);
    match n & 3 {
        0 => cos_kernel(hi, Some(lo)),
        1 => -sin_kernel(hi, Some(lo)),
        2 => -cos_kernel(hi, Some(lo)),
        _ => sin_kernel(hi, Some(lo)),
    }
}

const INV_2PI: [u64; 19] = [
    0x28be_60db_9391_054a,
    0x7f09_d5f4_7d4d_3770,
    0x36d8_a566_4f10_e410,
    0x7f94_58ea_f7ae_f158,
    0x6dc9_1b8e_9093_74b8,
    0x0192_4bba_8274_6487,
    0x3f87_7ac7_2c4a_69cf,
    0xba20_8d7d_4bae_d121,
    0x3a67_1c09_ad17_df90,
    0x4e64_758e_60d4_ce7d,
    0x2721_17e2_ef7e_4a0e,
    0xc7fe_25ff_f781_6603,
    0xfbcb_c462_d682_9b47,
    0xdb4d_9fb3_c9f2_c26d,
    0xd3d1_8fd9_a797_fa8b,
    0x5d49_eeb1_faf9_7c5e,
    0xcf41_ce7d_e294_a4ba,
    0x9afe_d7ec_47e3_5742,
    0x1580_cc11_bf1e_daea,
];
fn fractional_parts(f: i128) -> (f64, f64) {
    if f == 0 {
        return (0.0, 0.0);
    }
    let sign = if f < 0 { 1u64 << 63 } else { 0 };
    let x = f.unsigned_abs();
    let shift = |x: u128, n: i32| if n >= 0 { x >> n } else { x << (-n) };
    let n = 128 - x.leading_zeros() as i32;
    let m = (shift(x, n - 26) as u64) << 27;
    let exponent = ((n - 128 + 1021) as u64) << 52;
    let hi = f64::from_bits(sign | (exponent + m));
    let remaining = x - shift(m as u128, 53 - n);
    if remaining == 0 {
        return (hi, 0.0);
    }
    let n = 128 - remaining.leading_zeros() as i32;
    let m = shift(remaining, n - 53) as u64;
    let exponent = ((n - 128 + 1021) as u64) << 52;
    (hi, f64::from_bits(sign | (exponent + m)))
}
fn payne_hanek(x: f64) -> (i32, f64, f64) {
    let bits = x.to_bits();
    let significand = (bits & 0x000f_ffff_ffff_ffff) | (1 << 52);
    let k = ((bits >> 52) & 0x7ff) as i32 - 1023 - 52;
    let index = k >> 6;
    let shift = (k - (index << 6)) as u32;
    let at = |offset: i32| INV_2PI[(index + offset) as usize];
    let (a1, a2, a3) = if shift == 0 {
        (at(0), at(1), at(2))
    } else {
        (
            (if index < 0 { 0 } else { at(0) << shift }) | (at(1) >> (64 - shift)),
            (at(1) << shift) | (at(2) >> (64 - shift)),
            (at(2) << shift) | (at(3) >> (64 - shift)),
        )
    };
    let w1 = (significand.wrapping_mul(a1) as u128) << 64;
    let w2 = (significand as u128) * (a2 as u128);
    let w3 = ((significand as u128) * (a3 as u128)) >> 64;
    let mut w = w1.wrapping_add(w2).wrapping_add(w3);
    if x.is_sign_negative() {
        w = 0u128.wrapping_sub(w);
    }
    let quadrant = ((w >> 125) as i32 + 1) >> 1;
    let (z_high, z_low) = fractional_parts((w << 2) as i128);
    let high = (z_high + z_low) * std::f64::consts::FRAC_PI_2;
    let low = (((z_high * 1.5707963407039642 - high) + z_high * (-1.3909067614167116e-8))
        + z_low * 1.5707963407039642)
        + z_low * (-1.3909067614167116e-8);
    (quadrant, high, low)
}
fn horner(x: f64, coefficients: &[f64]) -> f64 {
    let mut value = *coefficients.last().unwrap();
    for &coefficient in coefficients[..coefficients.len() - 1].iter().rev() {
        value = value.mul_add(x, coefficient);
    }
    value
}
fn tangent_kernel(mut hi: f64, mut lo: f64, k: f64) -> f64 {
    let original = hi;
    if hi.abs() >= 0.6744 {
        if hi < 0.0 {
            hi = -hi;
            lo = -lo;
        }
        hi = (std::f64::consts::FRAC_PI_4 - hi) + (3.06161699786838301793e-17 - lo);
        lo = 0.0;
    }
    let y2 = hi * hi;
    let y4 = y2 * y2;
    let r = horner(
        y4,
        &[
            1.33333333333201242699e-01,
            2.18694882948595424599e-02,
            3.59207910759131235356e-03,
            5.88041240820264096874e-04,
            7.81794442939557092300e-05,
            -1.85586374855275456654e-05,
        ],
    );
    let v = y2
        * horner(
            y4,
            &[
                5.39682539762260521377e-02,
                8.86323982359930005737e-03,
                1.45620945432529025516e-03,
                2.46463134818469906812e-04,
                7.14072491382608190305e-05,
                2.59073051863633712884e-05,
            ],
        );
    let y3 = y2 * hi;
    let mut r = lo + y2 * (y3 * (r + v) + lo);
    r += 3.33333333333334091986e-01 * y3;
    let p = hi + r;
    if original.abs() >= 0.6744 {
        return (if original.is_sign_negative() {
            -1.0
        } else {
            1.0
        }) * (k - 2.0 * (hi - (p * p / (k + p) - r)));
    }
    if k == 1.0 {
        return p;
    }
    let p0 = f64::from_bits(p.to_bits() & 0xffff_ffff_0000_0000);
    let v = r - (p0 - hi);
    let a = -1.0 / p;
    let t = f64::from_bits(a.to_bits() & 0xffff_ffff_0000_0000);
    let s = 1.0 + t * p0;
    t + a * (s + t * v)
}
/// Julia Base's Float64 tangent, including large-argument reduction.
pub fn tan(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    assert!(x.is_finite());
    if x.abs() < std::f64::consts::FRAC_PI_4 {
        if x.abs() < f64::EPSILON.sqrt() / 2.0 {
            return x;
        }
        return tangent_kernel(x, 0.0, 1.0);
    }
    let (n, hi, lo) = reduce(x);
    tangent_kernel(hi, lo, if n & 1 == 0 { 1.0 } else { -1.0 })
}
fn atan_polynomials(x: f64) -> (f64, f64) {
    let z = x * x;
    let w = z * z;
    (
        z * horner(
            w,
            &[
                3.33333333333329318027e-01,
                1.42857142725034663711e-01,
                9.09088713343650656196e-02,
                6.66107313738753120669e-02,
                4.97687799461593236017e-02,
                1.62858201153657823623e-02,
            ],
        ),
        w * horner(
            w,
            &[
                -1.99999999998764832476e-01,
                -1.11111104054623557880e-01,
                -7.69187620504482999495e-02,
                -5.83357013379057348645e-02,
                -3.65315727442169155270e-02,
            ],
        ),
    )
}
fn atan(x: f64) -> f64 {
    let absolute = x.abs();
    if absolute >= 73786976294838206464.0 {
        return std::f64::consts::FRAC_PI_2.copysign(x);
    }
    if absolute < 7.0 / 16.0 {
        if absolute < 7.450580596923828e-9 {
            return x;
        }
        let (p, q) = atan_polynomials(x);
        return x - x * (p + q);
    }
    let (hi, lo, argument) = if absolute < 11.0 / 16.0 {
        (
            4.63647609000806093515e-01,
            2.26987774529616870924e-17,
            (2.0 * absolute - 1.0) / (2.0 + absolute),
        )
    } else if absolute < 19.0 / 16.0 {
        (
            std::f64::consts::FRAC_PI_4,
            3.06161699786838301793e-17,
            (absolute - 1.0) / (absolute + 1.0),
        )
    } else if absolute < 39.0 / 16.0 {
        (
            9.82793723247329054082e-01,
            1.39033110312309984516e-17,
            (absolute - 1.5) / (1.0 + 1.5 * absolute),
        )
    } else {
        (
            std::f64::consts::FRAC_PI_2,
            6.12323399573676603587e-17,
            -1.0 / absolute,
        )
    };
    let (p, q) = atan_polynomials(argument);
    (hi - ((argument * (p + q) - lo) - argument)).copysign(x)
}
/// Julia Base's two-argument arctangent, including signed-zero branch cuts.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return if x.is_nan() { x } else { y };
    }
    if x == 1.0 {
        return atan(y);
    }
    let mut quadrant = 2 * u32::from(x.is_sign_negative()) + u32::from(y.is_sign_negative());
    if y == 0.0 {
        return if quadrant < 2 {
            y
        } else {
            std::f64::consts::PI.copysign(y)
        };
    }
    if x == 0.0 {
        return std::f64::consts::FRAC_PI_2.copysign(y);
    }
    if x.is_infinite() {
        return if y.is_infinite() {
            (if quadrant < 2 {
                std::f64::consts::FRAC_PI_4
            } else {
                3.0 * std::f64::consts::FRAC_PI_4
            })
            .copysign(y)
        } else if quadrant < 2 {
            0.0f64.copysign(y)
        } else {
            std::f64::consts::PI.copysign(y)
        };
    }
    if y.is_infinite() {
        return std::f64::consts::FRAC_PI_2.copysign(y);
    }
    let yh = ((y.to_bits() & 0x7fff_ffff_ffff_ffff) >> 32) as u32;
    let xh = ((x.to_bits() & 0x7fff_ffff_ffff_ffff) >> 32) as u32;
    let ratio_exponent = (yh.wrapping_sub(xh) as i32) >> 20;
    let low_pi = 1.2246467991473531772e-16;
    let z = if ratio_exponent > 60 {
        quadrant &= 1;
        std::f64::consts::FRAC_PI_2 + 0.5 * low_pi
    } else if x < 0.0 && ratio_exponent < -60 {
        0.0
    } else {
        atan((y / x).abs())
    };
    match quadrant {
        0 => z,
        1 => -z,
        2 => std::f64::consts::PI - (z - low_pi),
        _ => (z - low_pi) - std::f64::consts::PI,
    }
}
/// Julia Base's Float64 hyperbolic sine using its compensated polynomial.
pub fn sinh(x: f64) -> f64 {
    if x.abs() <= 2.1 {
        let x2 = x * x;
        let x2lo = x.mul_add(x, -x2);
        let higher = horner(
            x2,
            &[
                8.333333333336817e-3,
                1.9841269840165435e-4,
                2.7557319381151335e-6,
                2.5052096530035283e-8,
                1.6059550718903307e-10,
                7.634842144412119e-13,
                2.9696954760355812e-15,
            ],
        );
        let mut hi = higher;
        let mut lo = 0.0f64;
        for coefficient in [0.16666666666666635, 1.0] {
            let prod = hi * x2;
            let error = hi.mul_add(x2, -prod);
            hi = coefficient + prod;
            lo = lo.mul_add(x2, prod - (hi - coefficient) + error);
        }
        return x.mul_add(hi, x.mul_add(lo, x * x2lo * 0.16666666666666635));
    }
    if x.abs() >= 709.7822265633563 {
        let e = super::julia_exp::exp(0.5 * x.abs());
        return (0.5 * e * e).copysign(x);
    }
    let e = super::julia_exp::exp(x.abs());
    (0.5 * (e - 1.0 / e)).copysign(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rbm_phase_sine_cosine_bits_match_julia_near_quadrant_boundaries_and_seeded_draws() {
        let fixture = include_str!("../../../../tests/fixtures/rbm/phase.txt");
        for line in fixture.lines().filter(|l| !l.starts_with('#')) {
            let bits: Vec<_> = line
                .split_whitespace()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect();
            let phase = f64::from_bits(bits[0]);
            assert_eq!(super::sin(phase).to_bits(), bits[1], "sin({phase:?})");
            assert_eq!(super::cos(phase).to_bits(), bits[2], "cos({phase:?})");
        }
    }
}
