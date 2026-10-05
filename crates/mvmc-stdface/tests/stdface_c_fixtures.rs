//! Byte-for-byte comparison against the C StdFace reference outputs (issue #353).
//!
//! Every `tests/fixtures/stdface/<case>/` directory holds the input `StdFace.def` and the C
//! results under `expected/`: each generated Expert file, the complete C `stdout` and the exit
//! status. The Rust port must reproduce all of them exactly: text formatting is deterministic
//! (`%f`/`%e`/`%d`), signed zeros included, so no numerical tolerance is needed or applied.
//! Fixture provenance: `tests/fixtures/stdface/PROVENANCE.md`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use mvmc_stdface::{stdface_main_bytes_in, StdFaceError};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/stdface")
}

struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("mvmc-stdface-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cases() -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(fixture_root())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Run one case; returns a list of mismatch descriptions.
fn check_case(case: &str) -> Vec<String> {
    let dir = fixture_root().join(case);
    // `expected/` is the historical C output; `expected_fixed/` (where present) is the corrected
    // behaviour of issue #404 (patched C build, tests/fixtures/stdface/README.md).
    let expected = if dir.join("expected_fixed").is_dir() {
        dir.join("expected_fixed")
    } else {
        dir.join("expected")
    };
    let work = TempDir::new(case);
    let input = fs::read(dir.join("StdFace.def")).ok();
    // Wannier90 cases ship their data files in the case directory (C reads the cwd).
    let result = stdface_main_bytes_in("StdFace.def", input.as_deref(), &dir, &work.0);
    let (status, log, stderr) = match &result {
        Ok(report) => (0, report.log.clone(), report.stderr.clone()),
        Err(failure) => match failure.error {
            StdFaceError::Exit(code) => (code & 0xff, failure.log.clone(), failure.stderr.clone()),
            StdFaceError::Io(ref message) => panic!("{case}: I/O failure: {message}"),
        },
    };
    let mut problems = Vec::new();
    let expected_status: i32 = fs::read_to_string(expected.join("exit_status"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    if status != expected_status {
        problems.push(format!("exit status {status} != C {expected_status}"));
    }
    let expected_stderr = fs::read(expected.join("stderr.txt")).unwrap_or_default();
    if stderr.as_bytes() != expected_stderr.as_slice() {
        problems.push(first_difference(
            "stderr",
            stderr.as_bytes(),
            &expected_stderr,
        ));
    }
    let expected_log = fs::read(expected.join("stdout.txt")).unwrap();
    if log.as_bytes() != expected_log.as_slice() {
        problems.push(first_difference("stdout", log.as_bytes(), &expected_log));
    }
    let mut expected_files = BTreeSet::new();
    for entry in fs::read_dir(&expected).unwrap().filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if matches!(name.as_str(), "stdout.txt" | "exit_status" | "stderr.txt") {
            continue;
        }
        let want = fs::read(entry.path()).unwrap();
        match fs::read(work.0.join(&name)) {
            Ok(got) if got == want => {}
            Ok(got) => problems.push(first_difference(&name, &got, &want)),
            Err(_) => problems.push(format!("{name}: not generated")),
        }
        expected_files.insert(name);
    }
    for entry in fs::read_dir(&work.0).unwrap().filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !expected_files.contains(&name) {
            problems.push(format!("{name}: generated but C wrote no such file"));
        }
    }
    problems
}

fn first_difference(name: &str, got: &[u8], want: &[u8]) -> String {
    let (got, want) = (String::from_utf8_lossy(got), String::from_utf8_lossy(want));
    for (index, (g, w)) in got.lines().zip(want.lines()).enumerate() {
        if g != w {
            return format!("{name}: line {}: rust {g:?} != C {w:?}", index + 1);
        }
    }
    format!(
        "{name}: {} lines (rust) vs {} lines (C)",
        got.lines().count(),
        want.lines().count()
    )
}

#[test]
fn fixtures_cover_the_required_cases() {
    let names = cases();
    for required in [
        "HubbardChain",
        "HeisenbergChain",
        "KondoChain",
        "err_unknown_keyword",
        "err_missing_L",
        "err_missing_ncond",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "missing fixture {required}"
        );
    }
    assert!(names.len() >= 60, "{} cases", names.len());
}

#[test]
fn rust_output_is_byte_identical_to_c() {
    let mut failures = Vec::new();
    for case in cases() {
        for problem in check_case(&case) {
            failures.push(format!("{case}: {problem}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
