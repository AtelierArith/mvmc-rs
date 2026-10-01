//! Float64 complex division following Julia 1.13.1 Base/complex.jl.
//!
//! Julia's scaled division and num_complex's norm-squared formula can round
//! differently even for a real denominator. Green ratios require the former.
//! MIT: Copyright (c) 2009-2025 Jeff Bezanson, Stefan Karpinski, Viral B. Shah,
//! and other Julia contributors. Full notice: crates/mvmc-expert-parsers/LICENSE-julia-math.
use num_complex::Complex64;

fn component(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        if br != 0.0 {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        }
    } else {
        (a + d * (b / c)) * t
    }
}

fn unscaled(a: f64, b: f64, c: f64, d: f64) -> Complex64 {
    let (a, b, c, d, swapped) = if d.abs() <= c.abs() {
        (a, b, c, d, false)
    } else {
        (b, a, d, c, true)
    };
    let r = d / c;
    let t = 1.0 / (c + d * r);
    let p = component(a, b, c, d, r, t);
    let q = component(b, -a, c, d, r, t);
    Complex64::new(p, if swapped { -q } else { q })
}

/// Julia's robust ComplexF64 / ComplexF64 operation, including scaling.
pub fn divide(z: Complex64, w: Complex64) -> Complex64 {
    let (mut a, mut b, mut c, mut d) = (z.re, z.im, w.re, w.im);
    let ab = if a.abs() >= b.abs() { a.abs() } else { b.abs() };
    let cd = if c.abs() >= d.abs() { c.abs() } else { d.abs() };
    if c.is_infinite() || d.is_infinite() {
        let sign = |x: f64| if x == 0.0 { x } else { x.signum() };
        return if a.is_finite() && b.is_finite() {
            Complex64::new(0.0 * sign(a) * sign(c), -0.0 * sign(b) * sign(d))
        } else {
            Complex64::new(f64::NAN, f64::NAN)
        };
    }
    let halfov = 0.5 * f64::MAX;
    let twouneps = f64::MIN_POSITIVE * 2.0 / f64::EPSILON;
    if ab >= halfov || ab <= twouneps || cd >= halfov || cd <= twouneps {
        let bs = 2.0 / (f64::EPSILON * f64::EPSILON);
        let mut s = 1.0;
        if ab >= halfov {
            a *= 0.5;
            b *= 0.5;
            s *= 2.0;
        } else if ab <= twouneps {
            a *= bs;
            b *= bs;
            s /= bs;
        }
        if cd >= halfov {
            c *= 0.5;
            d *= 0.5;
            s *= 0.5;
        } else if cd <= twouneps {
            c *= bs;
            d *= bs;
            s *= bs;
        }
        unscaled(a, b, c, d) * s
    } else {
        unscaled(a, b, c, d)
    }
}

/// Julia's ComplexF64 reciprocal, including its FMA norm and scaled branches.
pub fn reciprocal(w: Complex64) -> Complex64 {
    let (mut c, mut d) = (w.re, w.im);
    let (absc, absd) = (c.abs(), d.abs());
    let (cd, dc) = if absc > absd {
        (absc, absd)
    } else {
        (absd, absc)
    };
    if (f64::MIN_POSITIVE / 2.0).sqrt() <= cd && cd <= (f64::MAX / 2.0).sqrt() {
        let norm = cd.mul_add(cd, dc * dc);
        return Complex64::new(c / norm, -d / norm);
    }
    if c.is_infinite() || d.is_infinite() {
        return Complex64::new(0.0_f64.copysign(c), (-0.0_f64).copysign(-d));
    }
    let bs = 2.0 / (f64::EPSILON * f64::EPSILON);
    let mut scale = 1.0;
    if cd >= f64::MAX / 2.0 {
        c *= 0.5;
        d *= 0.5;
        scale = 0.5;
    } else if cd <= 2.0 * f64::MIN_POSITIVE / f64::EPSILON {
        c *= bs;
        d *= bs;
        scale = bs;
    }
    let robust = |c: f64, d: f64| {
        let r = d / c;
        let z = d.mul_add(r, c);
        (1.0 / z, -r / z)
    };
    let (p, q) = if absd <= absc {
        robust(c, d)
    } else {
        let (q, p) = robust(-d, -c);
        (p, q)
    };
    Complex64::new(p * scale, q * scale)
}
