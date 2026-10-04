//! Real-valued conversions from raw `u32`/`u64` SFMT draws.
//!
//! Port of `extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT-real.c`. These
//! must be bit-identical to the C versions because they appear inside
//! mVMC sampling hot loops (e.g. `do { r = genrand_real2(); s = r<0.5 ? 0 : 1; ... }`
//! in `vmc_sampling.jl:1019`).

/// `to_real1(v) = v * (1.0 / (2^32 - 1))` — maps `u32` -> `[0, 1]`.
#[inline]
pub fn to_real1(v: u32) -> f64 {
    v as f64 * (1.0 / 4_294_967_295.0)
}

/// `to_real2(v) = v * (1.0 / 2^32)` — maps `u32` -> `[0, 1)`.
///
/// This is the function the mVMC sampler actually consumes
/// (`genrand_real2()` is the half-open uniform that powers all
/// acceptance / spin-pick branches).
#[inline]
pub fn to_real2(v: u32) -> f64 {
    v as f64 * (1.0 / 4_294_967_296.0)
}

/// `to_real3(v) = ((v as f64) + 0.5) * (1.0 / 2^32)` — maps `u32` -> `(0, 1)`.
#[inline]
pub fn to_real3(v: u32) -> f64 {
    (v as f64 + 0.5) * (1.0 / 4_294_967_296.0)
}

/// `to_res53(v) = v * (1.0 / 2^64)` for 53-bit resolution from one `u64` draw.
///
/// The C reference uses the `long double` literal `18446744073709551616.0L`
/// to coerce the division to higher precision before truncating to
/// `double`. On x86-64 GCC `long double` is 80-bit extended precision
/// and the expression is folded at compile time to a `double` that is
/// exactly representable: `1.0 / 2^64` rounds to `0x3BF0_0000_0000_0000`
/// (i.e. `5.421010862427522e-20`). We hard-code that constant so the
/// result is platform-independent regardless of how Rust models `f128`.
#[inline]
pub fn to_res53(v: u64) -> f64 {
    const INV_2_64: f64 = 5.421010862427522e-20; // f64::from_bits(0x3BF0_0000_0000_0000)
    v as f64 * INV_2_64
}

/// `to_res53_mix(x, y) = to_res53(x | (y << 32))`.
#[inline]
pub fn to_res53_mix(x: u32, y: u32) -> f64 {
    to_res53((x as u64) | ((y as u64) << 32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_real2_bounds() {
        assert_eq!(to_real2(0), 0.0);
        // Largest u32 maps to 1 - 2^-32 == 0.9999999997671694 exactly.
        // This is the C reference's `(uint32_t)0xFFFFFFFF * (1.0/4294967296.0)`
        // evaluated under IEEE 754 round-to-nearest-even.
        let v = to_real2(u32::MAX);
        assert!(v < 1.0);
        assert_eq!(v, 1.0 - 2.0_f64.powi(-32));
        // Ulp bookkeeping: we are well above 0.5 and far below 1.0.
        assert!(v > 0.999_999_999);
    }

    #[test]
    fn to_real1_inclusive_one() {
        // u32::MAX / (2^32 - 1) == 1.0 exactly under C semantics.
        assert_eq!(to_real1(u32::MAX), 1.0);
    }

    #[test]
    fn to_real3_strictly_open() {
        assert!(to_real3(0) > 0.0);
        assert!(to_real3(u32::MAX) < 1.0);
    }

    #[test]
    fn to_res53_inv_constant_matches_c_long_double() {
        // The C reference uses `1.0 / 18446744073709551616.0L` which folds
        // to the f64 bit pattern below.
        let inv = 5.421010862427522e-20_f64;
        assert_eq!(inv.to_bits(), 0x3BF0_0000_0000_0000);
    }
}
