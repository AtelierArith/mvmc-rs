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

pub fn arm_directory(fixtures: &Path) -> Option<PathBuf> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        extern "C" {
            fn openblas_get_corename() -> *const std::ffi::c_char;
        }
        let core = unsafe { std::ffi::CStr::from_ptr(openblas_get_corename()) }
            .to_str()
            .expect("OpenBLAS core name");
        assert!(core.bytes().all(|b| b.is_ascii_alphanumeric()));
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
