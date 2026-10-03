//! Selected-gate diagnostics; classification never substitutes for verification.
use std::{fmt, fs, io, path::Path};

#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    MissingFixture,
    Failure,
    Unsupported,
}

#[derive(Debug)]
pub struct GateError {
    pub status: Status,
    pub detail: String,
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.status, self.detail)
    }
}

/// Missing paths are MissingFixture; nonregular/inaccessible artifacts are Failure.
fn regular_required(path: &Path) -> Result<(), GateError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| GateError {
        status: if error.kind() == io::ErrorKind::NotFound {
            Status::MissingFixture
        } else {
            Status::Failure
        },
        detail: format!("{}: {error}", path.display()),
    })?;
    if !metadata.is_file() {
        return Err(GateError {
            status: Status::Failure,
            detail: format!("nonregular fixture: {}", path.display()),
        });
    }
    // Probe readability without loading large numerical artifacts into memory.
    fs::File::open(path).map_err(|error| read_failure(path, error))?;
    Ok(())
}

pub fn read_required(path: &Path) -> Result<Vec<u8>, GateError> {
    regular_required(path)?;
    // Once observed, disappearance/read failure is Failure, not an absent bundle.
    fs::read(path).map_err(|error| read_failure(path, error))
}

fn bundle_file(root: &Path, name: &str) -> Result<std::path::PathBuf, GateError> {
    // Root's parent may be a system alias; root itself and every descendant must
    // be real directories/files. Canonicalizing an arbitrary member is forbidden.
    let mut path = root.to_path_buf();
    let components = std::iter::once(None).chain(Path::new(name).components().map(Some));
    for component in components {
        if let Some(std::path::Component::Normal(value)) = component {
            path.push(value);
        } else if component.is_some() {
            return Err(GateError {
                status: Status::Failure,
                detail: "unsafe bundle path".into(),
            });
        }
        let metadata = fs::symlink_metadata(&path).map_err(|error| GateError {
            status: if error.kind() == io::ErrorKind::NotFound {
                Status::MissingFixture
            } else {
                Status::Failure
            },
            detail: format!("{}: {error}", path.display()),
        })?;
        if metadata.file_type().is_symlink() {
            return Err(GateError {
                status: Status::Failure,
                detail: format!("symlink bundle component: {}", path.display()),
            });
        }
    }
    regular_required(&path)?;
    Ok(path)
}

#[cfg(test)]
fn require_supported<'a>(selected: &'a str, supported: &[&str]) -> Result<&'a str, GateError> {
    if supported.contains(&selected) {
        Ok(selected)
    } else {
        Err(GateError {
            status: Status::Unsupported,
            detail: format!("unknown/empty selected model: {selected:?}"),
        })
    }
}

fn read_failure(path: &Path, error: io::Error) -> GateError {
    GateError {
        status: Status::Failure,
        detail: format!("{}: {error}", path.display()),
    }
}

/// Preserve the existing independent verifier unchanged, after missing-file preflight.
pub fn require_bundle(root: &Path, manifest: &str) -> Result<(), GateError> {
    let bytes = read_required(&bundle_file(root, manifest)?)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| GateError {
        status: Status::Failure,
        detail: format!("invalid manifest UTF8: {error}"),
    })?;
    if text.trim().is_empty() {
        return Err(GateError {
            status: Status::Failure,
            detail: "empty manifest".into(),
        });
    }
    let mut names = std::collections::BTreeSet::new();
    for row in text.lines() {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields.len() != 2
            || fields[0].len() != 64
            || !fields[0].bytes().all(|b| b.is_ascii_hexdigit())
            || fields[1].contains('\\')
            || !Path::new(fields[1])
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
            || !names.insert(fields[1])
        {
            return Err(GateError {
                status: Status::Failure,
                detail: format!("invalid manifest row: {row:?}"),
            });
        }
    }
    // Validate ALL syntax before absence can mask a malformed later row.
    for name in names {
        bundle_file(root, name)?;
    }
    // This preflight does NOT validate hashes or grant Pass. The original pinned
    // archive/provenance/settings verifier MUST run next, including corruption checks.
    Ok(())
}

pub fn verify_selected(
    root: &Path,
    manifest: &str,
    verify: impl FnOnce(),
) -> Result<(), GateError> {
    // These are consumed before/alongside the archive verifier. Diagnose absence
    // before its read_to_string/unwrap, without trusting their content here.
    for file in [
        "provenance.txt",
        "reviewed-source.sha256",
        "reference-source/Manifest-v1.13.toml",
    ] {
        bundle_file(root, file)?;
    }
    require_bundle(root, manifest)?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(verify)).map_err(|_| GateError {
        status: Status::Failure,
        detail: "independent fixture identity/settings/hash verification failed".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_fail_closed() {
        for kind in [
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::NotFound,
            io::ErrorKind::InvalidData,
        ] {
            assert_eq!(
                read_failure(Path::new("already-observed-member"), io::Error::from(kind)).status,
                Status::Failure
            );
        }
        for value in ["", "skip", "unknown"] {
            assert_eq!(
                require_supported(value, &["general_rbm_cmp"])
                    .unwrap_err()
                    .status,
                Status::Unsupported
            );
        }
        assert_eq!(
            require_supported("general_rbm_cmp", &["general_rbm_cmp"]).unwrap(),
            "general_rbm_cmp"
        );
    }

    #[test]
    fn absence_and_nonregular_are_distinct() {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fixture-status-{}-{id}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        struct Owned(std::path::PathBuf);
        impl Drop for Owned {
            fn drop(&mut self) {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let owned = Owned(dir);
        assert_eq!(
            read_required(&owned.0.join("absent")).unwrap_err().status,
            Status::MissingFixture
        );
        assert_eq!(read_required(&owned.0).unwrap_err().status, Status::Failure);
        fs::write(owned.0.join("archive.sha256"), "broken row\n").unwrap();
        assert_eq!(
            require_bundle(&owned.0, "archive.sha256")
                .unwrap_err()
                .status,
            Status::Failure
        );
        fs::write(
            owned.0.join("archive.sha256"),
            format!("{} missing.txt\n", "0".repeat(64)),
        )
        .unwrap();
        assert_eq!(
            require_bundle(&owned.0, "archive.sha256")
                .unwrap_err()
                .status,
            Status::MissingFixture
        );
    }

    #[test]
    fn missing_metadata_precedes_verifier_and_corruption_is_failure() {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fixture-verifier-{}-{id}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        struct Owned(std::path::PathBuf);
        impl Drop for Owned {
            fn drop(&mut self) {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let owned = Owned(dir);
        let called = std::cell::Cell::new(false);
        let error = verify_selected(&owned.0, "archive.sha256", || called.set(true)).unwrap_err();
        assert_eq!(error.status, Status::MissingFixture);
        assert!(
            !called.get(),
            "missing metadata must not reach provenance unwrap"
        );
        fs::create_dir(owned.0.join("reference-source")).unwrap();
        for file in [
            "provenance.txt",
            "reviewed-source.sha256",
            "reference-source/Manifest-v1.13.toml",
            "member.txt",
        ] {
            fs::write(owned.0.join(file), b"synthetic control only").unwrap();
        }
        fs::write(
            owned.0.join("archive.sha256"),
            format!("{} member.txt\n", "0".repeat(64)),
        )
        .unwrap();
        let verifier = || {
            assert_eq!(
                fs::read(owned.0.join("member.txt")).unwrap(),
                b"independent fixed expected bytes"
            )
        };
        let error = verify_selected(&owned.0, "archive.sha256", verifier).unwrap_err();
        assert_eq!(error.status, Status::Failure);
        fs::write(
            owned.0.join("member.txt"),
            b"independent fixed expected bytes",
        )
        .unwrap();
        verify_selected(&owned.0, "archive.sha256", || {
            verifier();
            called.set(true);
        })
        .unwrap();
        assert!(
            called.get(),
            "successful preflight must still invoke verifier"
        );
        // A successful synthetic closure is NOT a real archive validation.
        fs::remove_file(owned.0.join("member.txt")).unwrap();
        called.set(false);
        assert_eq!(
            verify_selected(&owned.0, "archive.sha256", || called.set(true))
                .unwrap_err()
                .status,
            Status::MissingFixture
        );
        assert!(!called.get());
    }

    #[test]
    fn malformed_duplicate_and_escaping_manifests_are_failure() {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fixture-manifest-{}-{id}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        struct Owned(std::path::PathBuf);
        impl Drop for Owned {
            fn drop(&mut self) {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let owned = Owned(dir);
        fs::write(owned.0.join("member.txt"), b"control").unwrap();
        let row = format!("{} member.txt\n", "0".repeat(64));
        for invalid in [
            String::new(),
            "bad member.txt\n".into(),
            format!("{row}{row}"),
            format!("{} ../escape\n", "0".repeat(64)),
            format!("{} /absolute\n", "0".repeat(64)),
        ] {
            fs::write(owned.0.join("archive.sha256"), invalid).unwrap();
            assert_eq!(
                require_bundle(&owned.0, "archive.sha256")
                    .unwrap_err()
                    .status,
                Status::Failure
            );
        }
        fs::write(owned.0.join("archive.sha256"), [0xff]).unwrap();
        assert_eq!(
            require_bundle(&owned.0, "archive.sha256")
                .unwrap_err()
                .status,
            Status::Failure
        );
        // A missing first member must not mask a malformed later row.
        fs::write(
            owned.0.join("archive.sha256"),
            format!("{} absent.txt\nbad later-row\n", "0".repeat(64)),
        )
        .unwrap();
        assert_eq!(
            require_bundle(&owned.0, "archive.sha256")
                .unwrap_err()
                .status,
            Status::Failure
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_is_failure_not_missing_or_verified() {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("fixture-symlink-{}-{id}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        struct Owned(std::path::PathBuf);
        impl Drop for Owned {
            fn drop(&mut self) {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let owned = Owned(dir);
        std::os::unix::fs::symlink(owned.0.join("absent"), owned.0.join("link")).unwrap();
        assert_eq!(
            read_required(&owned.0.join("link")).unwrap_err().status,
            Status::Failure
        );
        fs::create_dir(owned.0.join("real-child")).unwrap();
        fs::write(owned.0.join("real-child/member.txt"), b"control").unwrap();
        std::os::unix::fs::symlink(owned.0.join("real-child"), owned.0.join("child")).unwrap();
        fs::write(
            owned.0.join("archive.sha256"),
            format!("{} child/member.txt\n", "0".repeat(64)),
        )
        .unwrap();
        assert_eq!(
            require_bundle(&owned.0, "archive.sha256")
                .unwrap_err()
                .status,
            Status::Failure
        );
        assert_eq!(
            bundle_file(&owned.0.join("child"), "member.txt")
                .unwrap_err()
                .status,
            Status::Failure
        );
    }
}
