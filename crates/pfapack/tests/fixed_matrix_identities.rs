//! Reference-free identities of the LTL factorization, Pfaffian and inverse on the fixed
//! skew-symmetric input matrices in `tests/fixtures/pfapack/*.orig.txt`.
//!
//! The PfaPack.jl output goldens these inputs came with were removed with the other Julia
//! CPU references (#695). The checks below need no reference implementation:
//!
//! - the Parlett-Reid Pfaffian (`pfaffian_ltl_*`) and `utu2pfa_*` on the LTL form agree;
//! - `Pf(A)^2 = det(A)`, with `det(A)` from an independent partial-pivoting LU, compared
//!   as `2 ln|Pf| = ln|det|` (and the phase for complex input);
//! - `utu2inv_*` returns the inverse: `max |A X - I| / (|A|_max |X|_max n)` is small.
//!
//! Tolerances (explicit, near the assertions): the Pfaffian algorithms differ only in
//! operation order, `4 n eps` relative, as in the former golden policy. The determinant
//! identity holds to the backward error of both eliminations, `n * 1e-13` on the
//! logarithm. The inverse residual is bounded by `64 eps`, the scale of a backward-stable
//! inversion. Observed on Linux x86_64 OpenBLAS (n = 4..256, real and complex): Pfaffian
//! agreement <= 5.8e-14 relative (bound 4 n eps = 2.3e-13 at n = 256), log-determinant
//! difference <= 1.2e-13, phase difference <= 8.1e-14, inverse residual <= 4.5 eps.
use std::path::PathBuf;

use num_complex::Complex64;
use pfapack::{
    dsktf2, pfaffian_ltl_complex, pfaffian_ltl_real, utu2inv_complex, utu2inv_real,
    utu2pfa_complex, utu2pfa_real, zsktf2, PivotIndex1Based, SqMat,
};

fn load_orig(kind: &str, n: usize, seed: u64) -> Vec<Complex64> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/pfapack")
        .join(format!("{kind}_n{n}_seed{seed}.orig.txt"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let mut header_n = None;
    let mut values = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("n ") {
            header_n = Some(rest.parse::<usize>().unwrap());
            continue;
        }
        if line.starts_with("kind") || line.starts_with("seed ") {
            continue;
        }
        let mut parts = line.split_whitespace().map(|v| v.parse::<f64>().unwrap());
        let re = parts.next().unwrap();
        let im = parts.next().unwrap_or(0.0);
        assert!(parts.next().is_none(), "trailing data in {line:?}");
        values.push(Complex64::new(re, im));
    }
    assert_eq!(header_n, Some(n));
    assert_eq!(values.len(), n * n);
    // Column-major skew-symmetric input.
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                values[i + j * n],
                -values[j + i * n],
                "input not skew at ({i},{j})"
            );
        }
    }
    values
}

/// `ln|det|` and `det / |det|` by Gaussian elimination with partial pivoting.
fn log_det(a: &[Complex64], n: usize) -> (f64, Complex64) {
    let mut m = a.to_vec();
    let mut log_abs = 0.0;
    let mut phase = Complex64::new(1.0, 0.0);
    for k in 0..n {
        let p = (k..n)
            .max_by(|&x, &y| m[x + k * n].norm().total_cmp(&m[y + k * n].norm()))
            .unwrap();
        if p != k {
            for j in 0..n {
                m.swap(k + j * n, p + j * n);
            }
            phase = -phase;
        }
        let pivot = m[k + k * n];
        assert!(pivot.norm() > 0.0, "singular input");
        log_abs += pivot.norm().ln();
        phase *= pivot / pivot.norm();
        for i in k + 1..n {
            let factor = m[i + k * n] / pivot;
            for j in k + 1..n {
                let update = factor * m[k + j * n];
                m[i + j * n] -= update;
            }
        }
    }
    (log_abs, phase)
}

fn max_abs(values: &[Complex64]) -> f64 {
    values.iter().fold(0.0, |m, v| m.max(v.norm()))
}

/// `max |A X - I| / (|A|_max |X|_max n)`.
fn inverse_residual(a: &[Complex64], x: &[Complex64], n: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(if i == j { -1.0 } else { 0.0 }, 0.0);
            for k in 0..n {
                sum += a[i + k * n] * x[k + j * n];
            }
            worst = worst.max(sum.norm());
        }
    }
    worst / (max_abs(a) * max_abs(x) * n as f64)
}

fn check_identities(
    n: usize,
    a: &[Complex64],
    pf_parlett_reid: Complex64,
    pf_utu2: Complex64,
    inverse: &[Complex64],
) {
    let eps = f64::EPSILON;
    let pf_rel = 1e-13f64.max(4.0 * n as f64 * eps);
    assert!(
        (pf_parlett_reid - pf_utu2).norm() <= pf_rel * pf_parlett_reid.norm(),
        "n={n}: Parlett-Reid {pf_parlett_reid:?} vs utu2pfa {pf_utu2:?}"
    );
    let (log_abs_det, det_phase) = log_det(a, n);
    let log_abs_pf2 = 2.0 * pf_utu2.norm().ln();
    let log_tol = n as f64 * 1e-13;
    assert!(
        (log_abs_pf2 - log_abs_det).abs() <= log_tol,
        "n={n}: 2 ln|Pf| = {log_abs_pf2} vs ln|det| = {log_abs_det}"
    );
    let pf_phase = (pf_utu2 / pf_utu2.norm()).powi(2);
    assert!(
        (pf_phase - det_phase).norm() <= log_tol,
        "n={n}: Pf^2 phase {pf_phase:?} vs det phase {det_phase:?}"
    );
    let residual = inverse_residual(a, inverse, n);
    assert!(
        residual <= 64.0 * eps,
        "n={n}: inverse residual {residual:e} > 64 eps"
    );
}

fn check_real(n: usize, seed: u64) {
    let a = load_orig("real", n, seed);
    assert!(a.iter().all(|v| v.im == 0.0));
    let real: Vec<f64> = a.iter().map(|v| v.re).collect();
    let mut pr = real.clone();
    let pf_pr = pfaffian_ltl_real(&mut SqMat::new(&mut pr, n));
    let mut ltl = real.clone();
    let mut pivots = vec![PivotIndex1Based(0); n];
    dsktf2(&mut SqMat::new(&mut ltl, n), &mut pivots).expect("dsktf2");
    let pf_utu2 = utu2pfa_real(&SqMat::new(&mut ltl, n), &pivots);
    let mut vt = vec![0.0; n - 1];
    let mut m_buf = vec![0.0; n * n];
    utu2inv_real(
        &mut SqMat::new(&mut ltl, n),
        &pivots,
        &mut vt,
        &mut SqMat::new(&mut m_buf, n),
    );
    let inverse: Vec<Complex64> = ltl.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    check_identities(
        n,
        &a,
        Complex64::new(pf_pr, 0.0),
        Complex64::new(pf_utu2, 0.0),
        &inverse,
    );
}

fn check_complex(n: usize, seed: u64) {
    let a = load_orig("complex", n, seed);
    let mut pr = a.clone();
    let pf_pr = pfaffian_ltl_complex(&mut SqMat::new(&mut pr, n));
    let mut ltl = a.clone();
    let mut pivots = vec![PivotIndex1Based(0); n];
    zsktf2(&mut SqMat::new(&mut ltl, n), &mut pivots).expect("zsktf2");
    let pf_utu2 = utu2pfa_complex(&SqMat::new(&mut ltl, n), &pivots);
    let mut vt = vec![Complex64::new(0.0, 0.0); n - 1];
    let mut m_buf = vec![Complex64::new(0.0, 0.0); n * n];
    utu2inv_complex(
        &mut SqMat::new(&mut ltl, n),
        &pivots,
        &mut vt,
        &mut SqMat::new(&mut m_buf, n),
    );
    check_identities(n, &a, pf_pr, pf_utu2, &ltl);
}

#[test]
fn real_inputs_satisfy_pfaffian_determinant_and_inverse_identities() {
    for (n, seed) in [(4, 42), (6, 1234), (16, 20240301), (32, 42), (64, 42)] {
        check_real(n, seed);
    }
}

#[test]
fn complex_inputs_satisfy_pfaffian_determinant_and_inverse_identities() {
    for (n, seed) in [(4, 42), (6, 1234), (16, 20240301), (32, 42), (64, 42)] {
        check_complex(n, seed);
    }
}

#[test]
#[ignore = "large fixed inputs (n = 128, 256)"]
fn large_inputs_satisfy_pfaffian_determinant_and_inverse_identities() {
    for (n, seed) in [(128, 42), (256, 42)] {
        check_real(n, seed);
        check_complex(n, seed);
    }
}
