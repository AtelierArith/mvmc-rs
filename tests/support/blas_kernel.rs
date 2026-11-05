//! BLAS kernel lineage of the C reference outputs.
//!
//! The C 1.4.0 references are generated on Linux x86_64 with OpenBLAS Haswell kernels (the
//! Zen kernel set reproduces them). On those kernels amplified comparisons (CG recurrences,
//! ill-conditioned solves) reproduce the references at their strict bounds. Other kernels
//! (no FMA, AVX512, ARM, Accelerate) differ by one ulp per GEMV and legitimately exceed
//! them, see docs/NUMERICAL_COMPARISONS.md ("BLAS provider matrix", #455).

/// Name of the linked OpenBLAS kernel (`openblas_get_corename`), or `None` when the
/// BLAS provider does not export it (reference BLAS).
///
/// Resolved with `dlsym` so that test binaries still link against providers that lack
/// the symbol.
pub fn openblas_core_name() -> Option<String> {
    use std::ffi::{c_char, c_void, CStr};
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    #[cfg(target_os = "macos")]
    let rtld_default = -2isize as *mut c_void;
    #[cfg(not(target_os = "macos"))]
    let rtld_default = std::ptr::null_mut::<c_void>();
    // SAFETY: `dlsym` is passed a valid NUL-terminated name and the documented
    // RTLD_DEFAULT handle; the resolved symbol has the C signature below.
    let symbol = unsafe { dlsym(rtld_default, c"openblas_get_corename".as_ptr()) };
    if symbol.is_null() {
        return None;
    }
    let function: extern "C" fn() -> *const c_char = unsafe { std::mem::transmute(symbol) };
    let name = unsafe { CStr::from_ptr(function()) }
        .to_str()
        .ok()?
        .to_owned();
    name.bytes()
        .all(|b| b.is_ascii_alphanumeric())
        .then_some(name)
}

/// Whether the linked BLAS reproduces the arithmetic of the C reference lineage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelClass {
    Reference,
    Unverified,
}

/// `MVMC_BLAS_KERNEL_CLASS=reference|unverified` overrides the detection.
pub fn kernel_class() -> KernelClass {
    match std::env::var("MVMC_BLAS_KERNEL_CLASS").as_deref() {
        Ok("reference") => return KernelClass::Reference,
        Ok("unverified") => return KernelClass::Unverified,
        _ => {}
    }
    let haswell_linux = cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "x86_64"
    )) && openblas_core_name().is_some_and(|core| {
        ["haswell", "zen"]
            .iter()
            .any(|k| core.eq_ignore_ascii_case(k))
    });
    if haswell_linux {
        KernelClass::Reference
    } else {
        KernelClass::Unverified
    }
}
