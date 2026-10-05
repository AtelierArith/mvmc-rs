//! #350: the Rust ComplexUHF port against the unmodified C `UHF` executable.
//!
//! Expectations in `tests/fixtures/complex_uhf/` were produced by
//! `scripts/check_complex_uhf_c_parity.py` (provenance in `PROVENANCE.txt`).
//! Rust never runs C here. Compared quantities and bounds:
//!
//! * iteration counts, indices, headers, row counts: exact (deterministic
//!   control path: the SFMT start and the mixing/convergence rule are ported
//!   operation for operation);
//! * `_check.dat`: residual `abs 2e-12 + rel 1e-8` (12 printed decimals of a
//!   difference of O(1) numbers), energy `abs 1e-10` (12 decimals), particle
//!   number within one printed `%lf` quantum (1.1e-6);
//! * `_result/_eigen/_gap/_UHF_cisajs`: `abs 2e-10 + rel 1e-10`. The files
//!   print 10 decimals (quantum 1e-10); a different BLAS/LAPACK provider
//!   changes eigenvectors at the 1e-15 level and the linear-mixing fixed point
//!   iteration (contraction factor `rho <= 0.9` in these cases, so roundoff is
//!   amplified by at most `1/(1-rho)`) keeps the final values within ~1e-13,
//!   so only the print quantum matters;
//! * `f_ij` files: eigenvector phases/rotations are gauge dependent, so only
//!   `F F^dagger` (the occupied-space projector, invariant under any unitary
//!   rotation inside the occupied space) is compared, with `abs 3e-5` = the
//!   propagated 6-decimal print quantum of up to 12 products of two entries;
//!   the SFMT noise added to every orbital parameter is compared separately
//!   (`param - class mean of the printed F`) where it is >= 1e-4 wide
//!   (`EpsSlater 3` cases), with `abs 2e-6`.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use mvmc_uhf::{read_definition, run, OrbitalOutputMode, UhfOptions};
use num_complex::Complex64;
use numerical_comparison::assert_close;

struct TestDir(PathBuf);

impl TestDir {
    fn new(case: &str) -> Self {
        let path = std::env::temp_dir().join(format!("mvmc-uhf-{}-{case}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/complex_uhf")
        .join(case)
}

fn copy_inputs(case: &str, dir: &Path) {
    for entry in fs::read_dir(fixture(case).join("input")).unwrap().flatten() {
        fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
    }
}

fn expected(case: &str, name: &str) -> String {
    fs::read_to_string(fixture(case).join("expected").join(name)).unwrap()
}

fn actual(dir: &Path, name: &str) -> String {
    fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Rows of numbers with the first `exact` columns compared as text.
fn compare_rows(
    actual: &str,
    expected: &str,
    exact: usize,
    tolerances: &[(f64, f64)],
    context: &str,
) {
    let a: Vec<_> = actual.lines().filter(|l| !l.starts_with('#')).collect();
    let e: Vec<_> = expected.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(a.len(), e.len(), "{context}: row count");
    for (row, (a, e)) in a.iter().zip(&e).enumerate() {
        let a: Vec<_> = a.split_whitespace().collect();
        let e: Vec<_> = e.split_whitespace().collect();
        assert_eq!(a.len(), e.len(), "{context}: row {row} columns");
        assert_eq!(
            a[..exact],
            e[..exact],
            "{context}: row {row} discrete columns"
        );
        for (k, (a, e)) in a[exact..].iter().zip(&e[exact..]).enumerate() {
            let (abs, rel) = tolerances[k.min(tolerances.len() - 1)];
            assert_close(
                a.parse().unwrap(),
                e.parse().unwrap(),
                abs,
                rel,
                format!("{context}: row {row} value {k}"),
            );
        }
    }
}

/// `F F^dagger` of the matrix assembled from the printed Fij files.
fn projector(
    def: &mvmc_uhf::UhfDefinition,
    dir: &Path,
    case: &str,
    from_expected: bool,
) -> Vec<Complex64> {
    let read = |name: &str| {
        if from_expected {
            expected(case, name)
        } else {
            actual(dir, name)
        }
    };
    let ns = def.nsite;
    let para = &def.para_file_head;
    let entries = |name: &str| -> Vec<(usize, usize, Complex64)> {
        read(name)
            .lines()
            .map(|line| {
                let w: Vec<_> = line.split_whitespace().collect();
                (
                    w[0].parse().unwrap(),
                    w[1].parse().unwrap(),
                    Complex64::new(w[2].parse().unwrap(), w[3].parse().unwrap()),
                )
            })
            .collect()
    };
    let (n, f) = match def.orbital_mode.unwrap() {
        OrbitalOutputMode::AntiParallel => {
            let mut f = vec![Complex64::new(0.0, 0.0); ns * ns];
            for (i, j, v) in entries(&format!("{para}_AP_Fij.dat")) {
                f[i * ns + j] = v;
            }
            (ns, f)
        }
        OrbitalOutputMode::General => {
            let mut f = vec![Complex64::new(0.0, 0.0); 4 * ns * ns];
            for (i, j, v) in entries(&format!("{para}_General_Fij.dat")) {
                f[i * 2 * ns + j] = v;
            }
            (2 * ns, f)
        }
        OrbitalOutputMode::AntiParallelAndParallel => {
            let mut f = vec![Complex64::new(0.0, 0.0); 4 * ns * ns];
            for (i, j, v) in entries(&format!("{para}_AP_Fij.dat")) {
                f[i * 2 * ns + (j + ns)] = v;
                f[(j + ns) * 2 * ns + i] = -v;
            }
            for (i, j, v) in entries(&format!("{para}_P_Fij.dat")) {
                f[i * 2 * ns + j] = v;
                f[j * 2 * ns + i] = -v;
            }
            (2 * ns, f)
        }
    };
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for a in 0..n {
        for b in 0..n {
            for k in 0..n {
                m[a * n + b] += f[a * n + k] * f[b * n + k].conj();
            }
        }
    }
    m
}

/// Orbital parameter file as `index -> value`.
fn opt_params(text: &str) -> (Vec<String>, BTreeMap<usize, Complex64>) {
    let lines: Vec<_> = text.lines().collect();
    let header = lines[..5].iter().map(|s| s.to_string()).collect();
    let params = lines[5..]
        .iter()
        .map(|line| {
            let w: Vec<_> = line.split_whitespace().collect();
            (
                w[0].parse().unwrap(),
                Complex64::new(w[1].parse().unwrap(), w[2].parse().unwrap()),
            )
        })
        .collect();
    (header, params)
}

/// Per-parameter noise `param - class mean(printed F)` for the files of `dir`.
fn noise(
    def: &mvmc_uhf::UhfDefinition,
    dir: &Path,
    case: &str,
    from_expected: bool,
) -> BTreeMap<usize, f64> {
    let read = |name: &str| {
        if from_expected {
            expected(case, name)
        } else {
            actual(dir, name)
        }
    };
    let ns = def.nsite;
    let n2 = 2 * ns;
    let para = &def.para_file_head;
    let mut sums = vec![Complex64::new(0.0, 0.0); def.n_orbital_idx];
    let mut counts = vec![0.0_f64; def.n_orbital_idx];
    let mut add = |name: &str, shift: usize| {
        for line in read(name).lines() {
            let w: Vec<_> = line.split_whitespace().collect();
            let (i, j): (usize, usize) = (w[0].parse().unwrap(), w[1].parse().unwrap());
            let value = Complex64::new(w[2].parse().unwrap(), w[3].parse().unwrap());
            let (row, col) = (i, j + shift);
            let idx = def.orbital_idx[row * n2 + col];
            if idx != -1 {
                let sign = f64::from(def.orbital_sgn[row * n2 + col]);
                sums[idx as usize] += value * sign;
                counts[idx as usize] += 1.0;
            }
        }
    };
    let mut files = Vec::new();
    match def.orbital_mode.unwrap() {
        OrbitalOutputMode::AntiParallel => {
            add(&format!("{para}_AP_Fij.dat"), ns);
            files.push(("APOrbital", 0));
        }
        OrbitalOutputMode::General => {
            add(&format!("{para}_General_Fij.dat"), 0);
            files.push(("GeneralOrbital", 0));
        }
        OrbitalOutputMode::AntiParallelAndParallel => {
            add(&format!("{para}_AP_Fij.dat"), ns);
            add(&format!("{para}_P_Fij.dat"), 0);
            files.push(("APOrbital", 0));
            files.push(("POrbital", def.n_orbital_ap));
        }
    }
    let mut out = BTreeMap::new();
    for (stem, offset) in files {
        let (_, params) = opt_params(&read(&format!("{para}_{stem}_opt.dat")));
        for (k, value) in params {
            let idx = k + offset;
            out.insert(idx, value.re - sums[idx].re / counts[idx]);
        }
    }
    out
}

fn check_case(case: &str) {
    let dir = TestDir::new(case);
    copy_inputs(case, &dir.0);
    let options = UhfOptions {
        base_dir: dir.0.clone(),
    };
    let report = run(Path::new("namelist.def"), &options).unwrap();
    let c_status: i32 = fs::read_to_string(fixture(case).join("status.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // C returns -1 (exit status 255) after writing the files of a non-converged run.
    assert_eq!(
        report.converged,
        c_status == 0,
        "{case}: convergence status"
    );

    let name = |suffix: &str| format!("zvo{suffix}");
    // Iteration count and per-step columns of the convergence log.
    let check_actual = actual(&dir.0, &name("_check.dat"));
    let check_expected = expected(case, &name("_check.dat"));
    assert_eq!(
        check_actual.lines().next(),
        check_expected.lines().next(),
        "{case}: check header"
    );
    compare_rows(
        &check_actual,
        &check_expected,
        1,
        &[(2e-12, 1e-8), (1e-10, 0.0), (1.1e-6, 0.0)],
        &format!("{case} check"),
    );
    let rows = check_expected.lines().count() - 1;
    let logged = if report.converged {
        report.steps + 1
    } else {
        report.steps
    };
    assert_eq!(logged as usize, rows, "{case}: logged iterations");

    let tol = [(2e-10, 1e-10)];
    compare_rows(
        &actual(&dir.0, &name("_result.dat"))
            .replace("energy", "")
            .replace("num", ""),
        &expected(case, &name("_result.dat"))
            .replace("energy", "")
            .replace("num", ""),
        0,
        &tol,
        &format!("{case} result"),
    );
    compare_rows(
        &actual(&dir.0, &name("_eigen.dat")),
        &expected(case, &name("_eigen.dat")),
        1,
        &tol,
        &format!("{case} eigen"),
    );
    compare_rows(
        &actual(&dir.0, &name("_gap.dat")),
        &expected(case, &name("_gap.dat")),
        0,
        &tol,
        &format!("{case} gap"),
    );
    compare_rows(
        &actual(&dir.0, &name("_UHF_cisajs.dat")),
        &expected(case, &name("_UHF_cisajs.dat")),
        4,
        &tol,
        &format!("{case} cisajs"),
    );

    // Orbital files: exactly the C set of files, headers and indices.
    let expected_files: Vec<String> = fs::read_dir(fixture(case).join("expected"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("zqp_"))
        .collect();
    let produced: Vec<String> = fs::read_dir(&dir.0)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("zqp_"))
        .collect();
    let mut sorted_expected = expected_files.clone();
    let mut sorted_produced = produced;
    sorted_expected.sort();
    sorted_produced.sort();
    assert_eq!(sorted_produced, sorted_expected, "{case}: orbital file set");
    if expected_files.is_empty() {
        return;
    }
    let def = read_definition(Path::new("namelist.def"), &dir.0).unwrap();
    for file in expected_files.iter().filter(|n| n.ends_with("_opt.dat")) {
        let (h_actual, p_actual) = opt_params(&actual(&dir.0, file));
        let (h_expected, p_expected) = opt_params(&expected(case, file));
        assert_eq!(h_actual, h_expected, "{case} {file}: header");
        assert_eq!(
            p_actual.keys().collect::<Vec<_>>(),
            p_expected.keys().collect::<Vec<_>>(),
            "{case} {file}: indices"
        );
    }
    let m_actual = projector(&def, &dir.0, case, false);
    let m_expected = projector(&def, &dir.0, case, true);
    for (k, (a, e)) in m_actual.iter().zip(&m_expected).enumerate() {
        assert_close(a.re, e.re, 3e-5, 0.0, format!("{case}: F F^+ re {k}"));
        assert_close(a.im, e.im, 3e-5, 0.0, format!("{case}: F F^+ im {k}"));
    }
    if def.eps_int_slater <= 4 {
        let n_actual = noise(&def, &dir.0, case, false);
        let n_expected = noise(&def, &dir.0, case, true);
        assert_eq!(n_actual.len(), n_expected.len());
        let mut visible = 0;
        for (idx, a) in &n_actual {
            let e = n_expected[idx];
            assert_close(
                *a,
                e,
                2e-6,
                0.0,
                format!("{case}: SFMT orbital noise {idx}"),
            );
            assert!(
                (-2e-6..10f64.powi(-def.eps_int_slater) + 2e-6).contains(a),
                "{case}: noise {idx} = {a} outside [0, 10^-EpsSlater)"
            );
            if *a > 1e-4 {
                visible += 1;
            }
        }
        assert!(visible > 0, "{case}: noise should be resolvable");
    }
}

macro_rules! case_tests {
    ($($test:ident => $case:literal),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                check_case($case);
            }
        )*
    };
}

case_tests! {
    hubbard_chain_random_start_matches_c => "hubbard_chain_real",
    square_initial_file_matches_c => "uhf_hubbard_square",
    triangular_initial_file_matches_c => "uhf_hubbard_triangular",
    interall_random_start_matches_c => "uhf_interall_n2",
    unconverged_run_writes_files_like_c => "uhf_hubbard_square_unconverged",
    interall_exchange_without_orbital_matches_c => "uhf_interall_exchange",
    antiparallel_neel_seed_with_mixing_matches_c => "chain_ap_neel",
    general_orbital_with_all_two_body_families_matches_c => "ring_general_mixed",
    antiparallel_plus_parallel_orbitals_match_c => "ring_ap_parallel",
    two_body_families_without_orbital_match_c => "ring_no_orbital",
}
