//! Cross-language golden parity for the pfapack crate.
//!
//! These tests pin the Rust port (`crates/pfapack/src/{ltl,pfaffian,utu2}.rs`)
//! against PfaPack.jl by loading the deterministic text fixtures dumped
//! by `extern/Julia-mVMC/tools/dump_pfapack_reference.jl`. Each
//! fixture file is a `(kind, n, seed, kind_value)` triple followed by
//! the payload in column-major order.
//!
//! Regenerate after touching either side:
//!
//! ```sh
//! cd extern/Julia-mVMC
//! julia --startup-file=no --project=. tools/dump_pfapack_reference.jl
//! ```
//!
//! Tolerance policy:
//! - LTL form: relative 1e-14 for real and 1e-11 for complex (one
//!   DSKR2 / ZSKR2 rank-2 update per step; the larger benchmark-size
//!   complex fixtures accumulate a little more order-of-operations
//!   drift).
//! - Pfaffian: relative 1e-13.
//! - Inverse: relative 1e-11 -- looser because `utu2inv!` chains
//!   trtri + skew-tridiagonal solve + trmm + two permutations, and the
//!   Julia path runs `BLAS.trmm!` while the Rust port hand-rolls the
//!   `M^T * A` loop. Order-of-summation differences fall well inside
//!   the 1e-11 envelope at the sizes used here (n <= 16).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use pfapack::{
    dsktf2, pfaffian_ltl_complex, pfaffian_ltl_real, utu2inv_complex, utu2inv_real,
    utu2pfa_complex, utu2pfa_real, zsktf2, PivotIndex1Based, SqMat,
};

const FIXTURE_DIR: &str = "../../tests/fixtures/pfapack";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name)
}

struct Fixture {
    n: usize,
    kind: String,         // "real" | "complex"
    kind_value: String,   // "orig" | "ltl" | "pfaffian" | "inverse"
    payload: Vec<String>, // raw payload lines, comments / blanks dropped
}

fn load_fixture(path: &Path) -> Fixture {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!(
            "failed to open fixture {} (did you run dump_pfapack_reference.jl?): {}",
            path.display(),
            e
        )
    });
    let mut n: Option<usize> = None;
    let mut kind: Option<String> = None;
    let mut kind_value: Option<String> = None;
    let mut payload = Vec::new();

    for line in BufReader::new(file).lines() {
        let line = line.expect("fixture line read failed");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            // Header comment, drop. We use a strict prefix here so an
            // accidental '#' inside the payload would fail loudly
            // rather than silently swallowing data.
            let _ = rest;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("kind ") {
            kind = Some(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("n ") {
            n = Some(rest.parse().expect("fixture n must be an int"));
            continue;
        }
        if let Some(_rest) = trimmed.strip_prefix("seed ") {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("kind_value ") {
            kind_value = Some(rest.to_string());
            continue;
        }
        payload.push(trimmed.to_string());
    }

    Fixture {
        n: n.expect("fixture missing `n`"),
        kind: kind.expect("fixture missing `kind`"),
        kind_value: kind_value.expect("fixture missing `kind_value`"),
        payload,
    }
}

fn parse_real_vector(lines: &[String]) -> Vec<f64> {
    lines
        .iter()
        .map(|s| {
            s.parse::<f64>()
                .unwrap_or_else(|_| panic!("fixture line not f64: {s:?}"))
        })
        .collect()
}

fn parse_complex_vector(lines: &[String]) -> Vec<Complex64> {
    lines
        .iter()
        .map(|s| {
            let mut parts = s.split_whitespace();
            let re = parts
                .next()
                .expect("complex line missing real")
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("fixture line bad real: {s:?}"));
            let im = parts
                .next()
                .expect("complex line missing imag")
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("fixture line bad imag: {s:?}"));
            assert!(
                parts.next().is_none(),
                "complex line has trailing junk: {s:?}"
            );
            Complex64::new(re, im)
        })
        .collect()
}

fn rel_close_f64(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    let diff = (a - b).abs();
    diff <= abs || diff <= rel * a.abs().max(b.abs())
}

fn rel_close_complex(a: Complex64, b: Complex64, rel: f64, abs: f64) -> bool {
    let diff = (a - b).norm();
    diff <= abs || diff <= rel * a.norm().max(b.norm())
}

/// Strip the LTL-fixture payload into `(matrix_entries, pivots_1based)`.
fn split_ltl_payload(payload: &[String], n: usize, complex: bool) -> (Vec<String>, Vec<u32>) {
    let payload_lines_per_entry = if complex { n * n } else { n * n };
    let _ = complex;
    let mat_lines: Vec<String> = payload
        .iter()
        .take(payload_lines_per_entry)
        .cloned()
        .collect();
    let pivot_lines: Vec<u32> = payload
        .iter()
        .skip(payload_lines_per_entry)
        .take(n)
        .map(|s| s.parse::<u32>().expect("pivot must be u32"))
        .collect();
    assert_eq!(
        pivot_lines.len(),
        n,
        "LTL fixture missing trailing pivot block"
    );
    (mat_lines, pivot_lines)
}

// ---------------------------------------------------------------------------
// Real cases
// ---------------------------------------------------------------------------

fn check_real_case(n: usize, seed: u64) {
    let stem = format!("real_n{n}_seed{seed}");
    let orig_fx = load_fixture(&fixture_path(&format!("{stem}.orig.txt")));
    let ltl_fx = load_fixture(&fixture_path(&format!("{stem}.ltl.txt")));
    let pf_fx = load_fixture(&fixture_path(&format!("{stem}.pfaffian.txt")));
    let inv_fx = load_fixture(&fixture_path(&format!("{stem}.inverse.txt")));

    assert_eq!(orig_fx.kind, "real");
    assert_eq!(orig_fx.kind_value, "orig");
    assert_eq!(orig_fx.n, n);
    assert_eq!(ltl_fx.kind_value, "ltl");
    assert_eq!(pf_fx.kind_value, "pfaffian");
    assert_eq!(inv_fx.kind_value, "inverse");

    let orig = parse_real_vector(&orig_fx.payload);
    let (ltl_mat_lines, ltl_pivots_julia) = split_ltl_payload(&ltl_fx.payload, n, false);
    let ltl_expected = parse_real_vector(&ltl_mat_lines);
    let pf_expected = parse_real_vector(&pf_fx.payload);
    let inv_expected = parse_real_vector(&inv_fx.payload);

    assert_eq!(orig.len(), n * n);
    assert_eq!(ltl_expected.len(), n * n);
    assert_eq!(pf_expected.len(), 1);
    assert_eq!(inv_expected.len(), n * n);

    // 1) Rust LTL must equal Julia LTL byte-for-bit-ish.
    let mut a_for_ltl = orig.clone();
    let mut pivots = vec![PivotIndex1Based(0); n];
    {
        let mut sm = SqMat::new(&mut a_for_ltl, n);
        dsktf2(&mut sm, &mut pivots).expect("dsktf2 failed on golden case");
    }
    for (idx, (got, want)) in a_for_ltl.iter().zip(ltl_expected.iter()).enumerate() {
        assert!(
            rel_close_f64(*got, *want, 1e-14, 1e-15),
            "real LTL mismatch at idx {idx} (n={n}, seed={seed}): rust={got:.17e} julia={want:.17e}"
        );
    }
    for (idx, (got, want)) in pivots.iter().zip(ltl_pivots_julia.iter()).enumerate() {
        assert_eq!(
            got.0, *want,
            "real pivot mismatch at idx {idx} (n={n}, seed={seed})"
        );
    }

    // 2) Pfaffian via direct Parlett-Reid on a fresh copy.
    let mut a_for_pr = orig.clone();
    let pf_pr = {
        let mut sm = SqMat::new(&mut a_for_pr, n);
        pfaffian_ltl_real(&mut sm)
    };
    assert!(
        rel_close_f64(pf_pr, pf_expected[0], 1e-13, 1e-15),
        "real Pfaffian (Parlett-Reid) mismatch (n={n}, seed={seed}): rust={pf_pr:.17e} julia={julia:.17e}",
        julia = pf_expected[0]
    );

    // 3) Pfaffian via utu2pfa on the LTL form (must agree).
    let pf_utu2 = {
        let sm = SqMat::new(&mut a_for_ltl, n);
        utu2pfa_real(&sm, &pivots)
    };
    assert!(
        rel_close_f64(pf_utu2, pf_expected[0], 1e-13, 1e-15),
        "real Pfaffian (utu2pfa) mismatch (n={n}, seed={seed}): rust={pf_utu2:.17e} julia={julia:.17e}",
        julia = pf_expected[0]
    );

    // 4) Inverse via utu2inv on the LTL form (consumes a_for_ltl).
    let mut a_inv = a_for_ltl.clone();
    let mut vt = vec![0.0; n - 1];
    let mut m_buf = vec![0.0; n * n];
    {
        let mut sm = SqMat::new(&mut a_inv, n);
        let mut mm = SqMat::new(&mut m_buf, n);
        utu2inv_real(&mut sm, &pivots, &mut vt, &mut mm);
    }
    for (idx, (got, want)) in a_inv.iter().zip(inv_expected.iter()).enumerate() {
        assert!(
            rel_close_f64(*got, *want, 1e-11, 1e-12),
            "real inverse mismatch at idx {idx} (n={n}, seed={seed}): rust={got:.17e} julia={want:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Complex cases
// ---------------------------------------------------------------------------

fn check_complex_case(n: usize, seed: u64) {
    let stem = format!("complex_n{n}_seed{seed}");
    let orig_fx = load_fixture(&fixture_path(&format!("{stem}.orig.txt")));
    let ltl_fx = load_fixture(&fixture_path(&format!("{stem}.ltl.txt")));
    let pf_fx = load_fixture(&fixture_path(&format!("{stem}.pfaffian.txt")));
    let inv_fx = load_fixture(&fixture_path(&format!("{stem}.inverse.txt")));

    assert_eq!(orig_fx.kind, "complex");
    assert_eq!(orig_fx.kind_value, "orig");
    assert_eq!(orig_fx.n, n);
    assert_eq!(ltl_fx.kind_value, "ltl");
    assert_eq!(pf_fx.kind_value, "pfaffian");
    assert_eq!(inv_fx.kind_value, "inverse");

    let orig = parse_complex_vector(&orig_fx.payload);
    let (ltl_mat_lines, ltl_pivots_julia) = split_ltl_payload(&ltl_fx.payload, n, true);
    let ltl_expected = parse_complex_vector(&ltl_mat_lines);
    let pf_expected = parse_complex_vector(&pf_fx.payload);
    let inv_expected = parse_complex_vector(&inv_fx.payload);

    assert_eq!(orig.len(), n * n);
    assert_eq!(ltl_expected.len(), n * n);
    assert_eq!(pf_expected.len(), 1);
    assert_eq!(inv_expected.len(), n * n);

    // 1) Rust LTL must match Julia LTL.
    let mut a_for_ltl = orig.clone();
    let mut pivots = vec![PivotIndex1Based(0); n];
    {
        let mut sm = SqMat::new(&mut a_for_ltl, n);
        zsktf2(&mut sm, &mut pivots).expect("zsktf2 failed on golden case");
    }
    for (idx, (got, want)) in a_for_ltl.iter().zip(ltl_expected.iter()).enumerate() {
        assert!(
            rel_close_complex(*got, *want, 1e-11, 1e-15),
            "complex LTL mismatch at idx {idx} (n={n}, seed={seed}): rust={got:?} julia={want:?}"
        );
    }
    for (idx, (got, want)) in pivots.iter().zip(ltl_pivots_julia.iter()).enumerate() {
        assert_eq!(
            got.0, *want,
            "complex pivot mismatch at idx {idx} (n={n}, seed={seed})"
        );
    }

    // 2) Pfaffian via direct Parlett-Reid.
    let mut a_for_pr = orig.clone();
    let pf_pr = {
        let mut sm = SqMat::new(&mut a_for_pr, n);
        pfaffian_ltl_complex(&mut sm)
    };
    assert!(
        rel_close_complex(pf_pr, pf_expected[0], 1e-12, 1e-14),
        "complex Pfaffian (Parlett-Reid) mismatch (n={n}, seed={seed}): rust={pf_pr:?} julia={julia:?}",
        julia = pf_expected[0]
    );

    // 3) Pfaffian via utu2pfa on the LTL form.
    let pf_utu2 = {
        let sm = SqMat::new(&mut a_for_ltl, n);
        utu2pfa_complex(&sm, &pivots)
    };
    assert!(
        rel_close_complex(pf_utu2, pf_expected[0], 1e-12, 1e-14),
        "complex Pfaffian (utu2pfa) mismatch (n={n}, seed={seed}): rust={pf_utu2:?} julia={julia:?}",
        julia = pf_expected[0]
    );

    // 4) Inverse via utu2inv on the LTL form (consumes a_for_ltl).
    let mut a_inv = a_for_ltl.clone();
    let mut vt = vec![Complex64::new(0.0, 0.0); n - 1];
    let mut m_buf = vec![Complex64::new(0.0, 0.0); n * n];
    {
        let mut sm = SqMat::new(&mut a_inv, n);
        let mut mm = SqMat::new(&mut m_buf, n);
        utu2inv_complex(&mut sm, &pivots, &mut vt, &mut mm);
    }
    for (idx, (got, want)) in a_inv.iter().zip(inv_expected.iter()).enumerate() {
        assert!(
            rel_close_complex(*got, *want, 1e-11, 1e-12),
            "complex inverse mismatch at idx {idx} (n={n}, seed={seed}): rust={got:?} julia={want:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

#[test]
fn real_n4_seed42() {
    check_real_case(4, 42);
}

#[test]
fn real_n6_seed1234() {
    check_real_case(6, 1234);
}

#[test]
fn real_n16_seed20240301() {
    check_real_case(16, 20240301);
}

#[test]
fn complex_n4_seed42() {
    check_complex_case(4, 42);
}

#[test]
fn complex_n6_seed1234() {
    check_complex_case(6, 1234);
}

#[test]
fn complex_n16_seed20240301() {
    check_complex_case(16, 20240301);
}

#[cfg(feature = "blas-backend")]
#[test]
fn complex_n32_seed42_blas_utu2inv_regression() {
    check_complex_case(32, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn real_n32_seed42_large() {
    check_real_case(32, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn real_n64_seed42_large() {
    check_real_case(64, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn real_n128_seed42_large() {
    check_real_case(128, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn real_n256_seed42_large() {
    check_real_case(256, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn complex_n32_seed42_large() {
    check_complex_case(32, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn complex_n64_seed42_large() {
    check_complex_case(64, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn complex_n128_seed42_large() {
    check_complex_case(128, 42);
}

#[test]
#[ignore = "large benchmark-size Julia parity fixture"]
fn complex_n256_seed42_large() {
    check_complex_case(256, 42);
}
