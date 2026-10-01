//! Julia 1.13.1 Float64 sine/cosine for projection angles in [-pi, pi].
//!
//! Port of Base special/trig.jl and the small-angle Cody-Waite paths in
//! special/rem_pio2.jl. Explicit mul_add calls mirror Julia's @horner and
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
    if high & 0xfffff != 0x921fb {
        let n = if high <= 0x4002d97c { 1 } else { 2 };
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

pub(super) fn sin(x: f64) -> f64 {
    assert!(x.is_finite() && x.abs() <= std::f64::consts::PI);
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

pub(super) fn cos(x: f64) -> f64 {
    assert!(x.is_finite() && x.abs() <= std::f64::consts::PI);
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
