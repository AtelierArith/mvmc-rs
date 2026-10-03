//! Platform-specific, independently generated Julia reference overlays.
//!
//! Identical archived files are reused; Linux overrides are listed with their
//! independent oracle hashes in linux_gnu_julia/SHA256.json.

use std::path::{Path, PathBuf};

pub fn fixture_path(fixtures: &Path, relative: impl AsRef<Path>) -> PathBuf {
    let relative = relative.as_ref();
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
