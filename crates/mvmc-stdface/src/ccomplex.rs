//! Minimal C99 `double complex` emulation with the exact operation order of the C StdFace.
//!
//! Signed zeros are observable in the generated text (`-0.000000000000000` appears in
//! `trans.def` for `conj` of a real hopping), so complex products use the textbook
//! `(ac - bd, ad + bc)` formula exactly like compiled C, not any normalised library formula.
//! Mixed real/complex operations follow what GCC emits (see the `scale`, `plus_real`,
//! `real_plus`, `real_minus` helpers); `tests/complex_expressions.rs` checks them bitwise.
//! `cpow` follows glibc (`cexp(y * clog(x))`) including the `log1p` branches of `clog`.

use std::ops::{Add, Mul, Neg, Sub};

/// `double complex`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl C64 {
    /// The imaginary unit `I`.
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };

    /// Construct from parts.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Implicit `double` to `double complex` conversion (imaginary part `+0.0`).
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// `conj`.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// `real * complex` / `complex * real`: GCC scales both components (no `(x, +0)` product).
    pub fn scale(self, x: f64) -> Self {
        Self {
            re: self.re * x,
            im: self.im * x,
        }
    }

    /// `complex + real`: the imaginary part is untouched (no `+ 0.0`).
    pub fn plus_real(self, x: f64) -> Self {
        Self {
            re: self.re + x,
            im: self.im,
        }
    }

    /// `real + complex`: the imaginary part is copied (no `0.0 + im`).
    pub fn real_plus(x: f64, c: Self) -> Self {
        Self {
            re: x + c.re,
            im: c.im,
        }
    }

    /// `real - complex`: the imaginary part is negated (no `0.0 - im`).
    pub fn real_minus(x: f64, c: Self) -> Self {
        Self {
            re: x - c.re,
            im: -c.im,
        }
    }

    /// `cabs` (glibc `hypot`).
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, rhs: C64) -> C64 {
        C64::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for C64 {
    type Output = C64;
    fn sub(self, rhs: C64) -> C64 {
        C64::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for C64 {
    type Output = C64;
    fn mul(self, rhs: C64) -> C64 {
        C64::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64::new(-self.re, -self.im)
    }
}

/// Exact `x*x + y*y - 1` rounded once (glibc `__x2y2m1` contract), for `0.5 <= |x|,|y| < 1`.
fn x2y2m1(x: f64, y: f64) -> f64 {
    let x2 = x * x;
    let x2e = x.mul_add(x, -x2);
    let y2 = y * y;
    let y2e = y.mul_add(y, -y2);
    let mut terms = [-1.0, x2, x2e, y2, y2e];
    terms.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    // Exact accumulation with error-free transformations (Shewchuk expansion sum).
    let mut parts: Vec<f64> = Vec::new();
    for &t in &terms {
        let mut carry = t;
        let mut next = Vec::with_capacity(parts.len() + 1);
        for &p in &parts {
            let (hi, lo) = if carry.abs() >= p.abs() {
                (carry, p)
            } else {
                (p, carry)
            };
            let s = hi + lo;
            let err = lo - (s - hi);
            if err != 0.0 {
                next.push(err);
            }
            carry = s;
        }
        next.push(carry);
        parts = next;
    }
    parts.iter().rev().fold(0.0, |acc, &p| acc + p)
}

/// glibc `clog` for finite non-zero arguments.
fn clog(x: C64) -> C64 {
    if x.re == 0.0 && x.im == 0.0 {
        let im = if x.re.is_sign_negative() {
            std::f64::consts::PI.copysign(x.im)
        } else {
            x.im
        };
        return C64::new(f64::NEG_INFINITY, im);
    }
    let (mut absx, mut absy) = (x.re.abs(), x.im.abs());
    if absx < absy {
        std::mem::swap(&mut absx, &mut absy);
    }
    let real = if absx == 1.0 {
        (absy * absy).ln_1p() / 2.0
    } else if absx > 1.0 && absx < 2.0 && absy < 1.0 {
        let mut d2m1 = (absx - 1.0) * (absx + 1.0);
        if absy >= f64::EPSILON {
            d2m1 += absy * absy;
        }
        d2m1.ln_1p() / 2.0
    } else if (0.5..1.0).contains(&absx) && absy < f64::EPSILON / 2.0 {
        ((absx - 1.0) * (absx + 1.0)).ln_1p() / 2.0
    } else if (0.5..1.0).contains(&absx) && absx * absx + absy * absy >= 0.5 {
        x2y2m1(absx, absy).ln_1p() / 2.0
    } else {
        absx.hypot(absy).ln()
    };
    C64::new(real, x.im.atan2(x.re))
}

/// glibc `cexp` for finite arguments.
fn cexp(x: C64) -> C64 {
    let (sinix, cosix) = if x.im.abs() > f64::MIN_POSITIVE {
        (x.im.sin(), x.im.cos())
    } else {
        (x.im, 1.0)
    };
    let exp_val = x.re.exp();
    C64::new(exp_val * cosix, exp_val * sinix)
}

/// glibc `cpow(x, y)` = `cexp(y * clog(x))`.
pub fn cpow(x: C64, y: C64) -> C64 {
    cexp(y * clog(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conj_flips_the_sign_of_zero() {
        let c = C64::real(1.0).conj();
        assert_eq!(c.re, 1.0);
        assert!(c.im.is_sign_negative() && c.im == 0.0);
    }

    #[test]
    fn unit_power_zero_is_one() {
        let one = C64::new(1.0, 0.0);
        let r = cpow(one, C64::real(0.0));
        assert_eq!((r.re, r.im), (1.0, 0.0));
        let r = cpow(one, C64::real(-1.0));
        assert_eq!(r.re, 1.0);
    }

    #[test]
    fn antiperiodic_phase_has_unit_modulus() {
        let m1 = C64::new(-1.0, 180.0f64.to_radians().sin());
        let r = cpow(m1, C64::real(1.0));
        assert!((r.re + 1.0).abs() < 1e-15);
    }
}
