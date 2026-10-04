//! Explicit inheritance of independently verified, byte-identical fixtures.
use std::path::{Path, PathBuf};

pub fn directory(fixtures: &Path) -> Option<PathBuf> {
    if let Some(arm) = super::julia_fixture::arm_directory(fixtures) {
        let root = arm.join("c_kernel_order/native_fsz");
        if root.is_dir() {
            return Some(root);
        }
    }
    let root = fixtures.join("c_kernel_order/native_fsz");
    if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        Some(root.join("linux_gnu"))
    } else if cfg!(target_os = "macos") {
        Some(root)
    } else {
        None
    }
}

pub fn resolve(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if super::julia_fixture::exists(path) {
        return path.to_path_buf();
    }
    let root = path
        .ancestors()
        .find(|p| p.file_name().is_some_and(|name| name == "native_fsz"))
        .unwrap_or_else(|| panic!("missing reference fixture {}", path.display()));
    if let Some(fixtures) = root
        .ancestors()
        .find(|p| p.file_name().is_some_and(|n| n == "fixtures"))
    {
        let canonical = fixtures.join("c_kernel_order/native_fsz");
        if root != canonical && root.starts_with(fixtures.join("macos_arm_julia")) {
            return resolve(canonical.join(path.strip_prefix(root).unwrap()));
        }
    }
    let key = path.strip_prefix(root).unwrap().to_str().unwrap();
    let manifest = std::fs::read_to_string(root.join("inheritance.tsv")).unwrap();
    let mut entries = manifest
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 3, "invalid native fixture inheritance row");
            (fields[0] == key).then_some(fields)
        });
    let entry = entries
        .next()
        .unwrap_or_else(|| panic!("missing independent fixture {}", path.display()));
    assert!(
        entries.next().is_none(),
        "duplicate inherited fixture {key}"
    );
    assert!(
        entry[2].len() == 64 && entry[2].bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid inherited fixture SHA-256 {key}"
    );
    let relative = Path::new(entry[1]);
    assert!(
        relative
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_))),
        "invalid historical fixture path {key}"
    );
    let historical = root.parent().unwrap().parent().unwrap().join(relative);
    assert!(historical.is_file(), "missing inherited fixture {key}");
    historical
}
