//! Parity of the Rust `greenr2k` with the Fortran `tool/greenr2k.F90` (gfortran 13.3.0,
//! `-O2`, system LAPACK/BLAS) on the fixtures in `tests/fixtures/greenr2k`.
//!
//! Provenance and the tolerance justification are in
//! `tests/fixtures/greenr2k/PROVENANCE.md` and `docs/NUMERICAL_COMPARISONS.md`.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use numerical_comparison::within;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/greenr2k")
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn work_dir(case: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("greenr2k-{case}"));
    let _ = fs::remove_dir_all(&dir);
    copy_dir(&fixtures().join(case).join("inputs"), &dir);
    dir
}

fn list_outputs(dir: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    if dir.join("kpath.gp").exists() {
        names.insert("kpath.gp".to_string());
    }
    for entry in fs::read_dir(dir.join("output")).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        if name.starts_with("zvo_corr") {
            names.insert(format!("output/{name}"));
        }
    }
    names
}

/// Numerical bound for a token that parses as a number on both sides.
///
/// Output numbers are printed with a fixed number of digits. A difference below the
/// rounding boundary of the printed digits may appear as one unit of the last
/// printed digit, so the bound is one unit of the expected token's last digit
/// (`E15.5`: five mantissa digits; `F15.10`: ten decimals; list-directed REAL(4):
/// nine significant digits; REAL(8): seventeen) plus an absolute floor for values
/// that are zero up to roundoff (sums of at most ~100 terms of magnitude <= 1,
/// i.e. about 100 * 2.2e-16 = 2.2e-14).
struct Bounds {
    floor: f64,
}

fn decimals(mantissa: &str) -> i32 {
    mantissa.split('.').nth(1).map_or(0, str::len) as i32
}

impl Bounds {
    fn allowed(&self, token: &str) -> f64 {
        let t = token.trim_end_matches(',');
        let unit = if let Some(p) = t.find(['E', 'e']) {
            // 0.12345E+01 (Ew.d) or 1.2345E+01 (list-directed): one unit of the last digit.
            let exp: i32 = t[p + 1..].parse().unwrap_or(0);
            let leading = if t[..p].trim_start_matches('-').starts_with("0.") {
                0
            } else {
                1
            };
            10f64.powi(exp + leading - decimals(&t[..p]))
        } else if let Some(p) = t.rfind(['+', '-']).filter(|&p| p > 0) {
            // Ew.d with a three-digit exponent drops the "E": 0.10000+101
            let exp: i32 = t[p..].parse().unwrap_or(0);
            10f64.powi(exp - decimals(&t[..p]))
        } else {
            10f64.powi(-decimals(t))
        };
        self.floor.max(unit)
    }
}

/// Line-by-line comparison. Text, line lengths and token layout must be identical;
/// tokens that parse as numbers on both sides compare with the given bounds
/// (signed zeros are equivalent).
fn assert_text_close(actual: &str, expected: &str, bounds: &Bounds, context: &str) {
    let a: Vec<&str> = actual.lines().collect();
    let e: Vec<&str> = expected.lines().collect();
    assert_eq!(a.len(), e.len(), "{context}: line count");
    for (n, (la, le)) in a.iter().zip(&e).enumerate() {
        if la == le {
            continue;
        }
        assert_eq!(
            la.len(),
            le.len(),
            "{context}:{}: line length\n{la}\n{le}",
            n + 1
        );
        let ta: Vec<&str> = la.split_whitespace().collect();
        let te: Vec<&str> = le.split_whitespace().collect();
        assert_eq!(ta.len(), te.len(), "{context}:{}: token count", n + 1);
        for (x, y) in ta.iter().zip(&te) {
            if x == y {
                continue;
            }
            let vx = x.trim_end_matches(',').parse::<f64>();
            let vy = y.trim_end_matches(',').parse::<f64>();
            match (vx, vy) {
                (Ok(vx), Ok(vy)) => assert!(
                    within(vx, vy, bounds.allowed(y), 0.0),
                    "{context}:{}: {x} vs {y}",
                    n + 1
                ),
                _ => panic!("{context}:{}: text differs: {x:?} vs {y:?}", n + 1),
            }
        }
    }
}

fn run_case(case: &str) {
    let dir = work_dir(case);
    let mut stdout = Vec::new();
    mvmc_greenr2k::run("namelist.def", "geometry.dat", &dir, &mut stdout).unwrap();
    let expected_dir = fixtures().join(case).join("expected");
    // Console output: only the rounded geometry numbers may differ, by one unit of
    // the last printed digit (the reciprocal vectors come from a 3x3 inverse that
    // is not LAPACK's).
    assert_text_close(
        &String::from_utf8(stdout).unwrap(),
        &fs::read_to_string(expected_dir.join("stdout.txt")).unwrap(),
        &Bounds { floor: 0.0 },
        &format!("{case} stdout"),
    );
    let mut expected_files = BTreeSet::new();
    expected_files.insert("kpath.gp".to_string());
    for entry in fs::read_dir(expected_dir.join("output")).unwrap() {
        expected_files.insert(format!(
            "output/{}",
            entry.unwrap().file_name().to_string_lossy()
        ));
    }
    assert_eq!(
        list_outputs(&dir),
        expected_files,
        "{case}: set of output files"
    );
    for name in &expected_files {
        let actual = fs::read_to_string(dir.join(name)).unwrap();
        let expected = fs::read_to_string(expected_dir.join(name)).unwrap();
        assert_text_close(
            &actual,
            &expected,
            &Bounds { floor: 1e-13 },
            &format!("{case} {name}"),
        );
    }
}

#[test]
fn chain6_mvmc_matches_fortran() {
    run_case("chain6_mvmc");
}

#[test]
fn honeycomb_mvmc_matches_fortran() {
    run_case("honeycomb_mvmc");
}

#[test]
fn honeycomb_lanczos_matches_fortran() {
    run_case("honeycomb_lanczos");
}

#[test]
fn chain6_tpq_matches_fortran() {
    run_case("chain6_tpq");
}

#[test]
fn chain6_lobcg_matches_fortran() {
    run_case("chain6_lobcg");
}
