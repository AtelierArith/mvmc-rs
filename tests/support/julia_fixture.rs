//! Platform-specific, independently generated Julia reference overlays.
//!
//! Identical archived files are reused; Linux overrides are listed with their
//! independent oracle hashes in linux_gnu_julia/SHA256.json.

use std::path::{Path, PathBuf};

fn compressed_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".gz");
    PathBuf::from(name)
}

pub fn read_text(path: impl AsRef<Path>) -> std::io::Result<String> {
    use std::io::Read;
    let path = path.as_ref();
    if path.is_file() {
        return std::fs::read_to_string(path);
    }
    let file = std::fs::File::open(compressed_path(path))?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(file).read_to_string(&mut text)?;
    Ok(text)
}

pub fn exists(path: &Path) -> bool {
    path.is_file() || compressed_path(path).is_file()
}

/// Name of the linked OpenBLAS kernel (`openblas_get_corename`), or `None` when the
/// BLAS provider does not export it (reference BLAS).
///
/// Resolved with `dlsym` so that test binaries still link against providers that lack
/// the symbol.
#[allow(dead_code)]
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

/// Whether the linked BLAS reproduces the arithmetic of the checked-in reference lineage.
///
/// The archived Julia/C fixtures were generated with FMA-based OpenBLAS Haswell kernels
/// (Linux x86_64; the Zen kernel set reproduces them, checked with OPENBLAS_CORETYPE=Zen) or with the per-core macOS ARM overlays under `macos_arm_julia/`. On
/// those kernels the amplified comparisons (CG recurrences, ill-conditioned solves,
/// cancelling Green-function sums) reproduce the references at the strict bounds. Other
/// kernels (Sandybridge/Nehalem without FMA, AVX512, other ARM cores such as the generic armv8) differ
/// by one ulp per GEMV and legitimately exceed those bounds, see
/// docs/NUMERICAL_COMPARISONS.md ("BLAS provider matrix", #455).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum KernelClass {
    Reference,
    Unverified,
}

#[allow(dead_code)]
pub fn kernel_class(fixtures: &Path) -> KernelClass {
    match std::env::var("MVMC_BLAS_KERNEL_CLASS").as_deref() {
        Ok("reference") => return KernelClass::Reference,
        Ok("unverified") => return KernelClass::Unverified,
        _ => {}
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if arm_directory(fixtures).is_some_and(|dir| dir.is_dir()) {
            KernelClass::Reference
        } else {
            KernelClass::Unverified
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = fixtures;
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
}

pub fn arm_directory(fixtures: &Path) -> Option<PathBuf> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let core = openblas_core_name()?;
        Some(fixtures.join("macos_arm_julia").join(core))
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = fixtures;
        None
    }
}

pub fn fixture_path(fixtures: &Path, relative: impl AsRef<Path>) -> PathBuf {
    let relative = relative.as_ref();
    if let Some(root) = arm_directory(fixtures) {
        let arm = root.join(relative);
        if exists(&arm) {
            return arm;
        }
    }
    if cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "x86_64"
    )) {
        let linux = fixtures.join("linux_gnu_julia").join(relative);
        if linux.is_file() {
            return linux;
        }
    }
    fixtures.join(relative)
}
