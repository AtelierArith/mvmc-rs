//! Complex operations used by RBM ratios and derivatives, as C calls them: the glibc
//! `cexp`/`clog`/`ctanh`/`ccosh` ports in `glibc_complex` (issue #457).
use num_complex::Complex64;

use mvmc_expert_parsers::utils::glibc_complex;

/// C `cexp` (glibc port).
pub(crate) fn exp(z: Complex64) -> Complex64 {
    glibc_complex::cexp(z)
}

/// C `clog` (glibc port).
pub(super) fn log(z: Complex64) -> Complex64 {
    glibc_complex::clog(z)
}

/// C `ctanh` (glibc port).
pub(super) fn tanh(z: Complex64) -> Complex64 {
    glibc_complex::ctanh(z)
}

/// `log(cosh(z))` of the RBM weights: C evaluates `clog(ccosh(z))` (`rbm.c:41,58`).
pub(crate) fn log_cosh(z: Complex64) -> Complex64 {
    glibc_complex::clog(glibc_complex::ccosh(z))
}
