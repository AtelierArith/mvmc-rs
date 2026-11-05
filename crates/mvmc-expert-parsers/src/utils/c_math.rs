//! Scalar math functions as the C reference calls them (issue #457).
//!
//! C mVMC calls the platform libm for `exp`, `log`, `sin`, `cos`, `hypot`, `atan2`, ... Rust's
//! `f64` methods of the same names are the platform libm too (glibc on Linux, the system libm
//! on macOS; LLVM may constant-fold or lower some to intrinsics but never changes the rounding
//! of the library call), so these functions call them.
//!
//! Platform caveat: glibc and the macOS libm are different implementations; a result that
//! depends on the last bit of a transcendental function can differ between Linux and macOS.
//! Linux (glibc, the platform of the C reference outputs) is the numerical reference.

/// C `exp`.
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// C `log`.
#[inline]
pub fn log(x: f64) -> f64 {
    x.ln()
}

/// C `log1p`.
#[inline]
pub fn log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// C `sin`.
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// C `cos`.
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// C `tan`.
#[inline]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// C `atan2`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// C `sinh`.
#[inline]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// C `hypot`.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARGS: [f64; 9] = [-3.7, -0.5, 0.0, 1e-9, 0.31, 1.0, 2.5, 17.25, 650.0];

    #[test]
    fn functions_are_the_platform_libm_bit_for_bit() {
        for x in ARGS {
            let c = x.min(700.0);
            assert_eq!(exp(c).to_bits(), c.exp().to_bits());
            assert_eq!(sin(x).to_bits(), x.sin().to_bits());
            assert_eq!(cos(x).to_bits(), x.cos().to_bits());
            assert_eq!(tan(x).to_bits(), x.tan().to_bits());
            assert_eq!(sinh(c).to_bits(), c.sinh().to_bits());
            assert_eq!(hypot(x, 1.5).to_bits(), x.hypot(1.5).to_bits());
            assert_eq!(atan2(x, 0.7).to_bits(), x.atan2(0.7).to_bits());
            if x > 0.0 {
                assert_eq!(log(x).to_bits(), x.ln().to_bits());
            }
            if x > -1.0 {
                assert_eq!(log1p(x).to_bits(), x.ln_1p().to_bits());
            }
        }
    }
}
