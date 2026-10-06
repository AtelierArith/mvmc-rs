//! Tensor-native tenferro backend against the pfapack reference.
//!
//! The LTL factorization keeps pfapack's elementwise operation order, so the real Pfaffian
//! is expected to be bit-identical; the inverse uses a different operation order
//! (repeated-squaring triangular inverse, `dot_general` permutations) and is compared with
//! an explicit tolerance derived from `n * eps * cond`.

use mvmc_gpu::testkit::*;
use mvmc_gpu::{pfaffian_inverse_batched, Backend, PlaneStatus};
use num_complex::Complex64;

fn compare<T: TestScalar>(n: usize, nqp: usize, batch: usize, seed: u64) -> (f64, f64, bool) {
    let planes = random_planes::<T>(n, nqp * batch, seed);
    let want = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nqp, batch).unwrap();
    let got = pfaffian_inverse_batched(&Backend::TenferroNative, &planes, n, nqp, batch).unwrap();
    assert_eq!(got.status, want.status);
    let mut pf_rel = 0.0f64;
    let mut identical = true;
    for p in 0..nqp * batch {
        let d = (got.pf[p] - want.pf[p]).modulus() / want.pf[p].modulus();
        pf_rel = pf_rel.max(d);
        identical &= T::as_f64_lanes(&got.pf[p..p + 1]) == T::as_f64_lanes(&want.pf[p..p + 1]);
    }
    let inv_rel = {
        let diff: Vec<f64> = T::as_f64_lanes(&got.inv)
            .iter()
            .zip(T::as_f64_lanes(&want.inv))
            .map(|(a, b)| a - b)
            .collect();
        let num = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        let den = T::as_f64_lanes(&want.inv)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        num / den
    };
    let inv = invariants(&planes, &got);
    assert!(
        inv.max_identity_residual < 1e-8,
        "residual {}",
        inv.max_identity_residual
    );
    assert!(inv.max_skew_defect < 1e-10, "skew {}", inv.max_skew_defect);
    assert!(
        inv.max_pf2_det_rel < 1e-8,
        "pf2/det {}",
        inv.max_pf2_det_rel
    );
    (pf_rel, inv_rel, identical)
}

#[test]
fn native_real_matches_pfapack() {
    for n in [2usize, 4, 6, 16, 24] {
        let (pf_rel, inv_rel, identical) = compare::<f64>(n, 2, 3, 31 + n as u64);
        eprintln!(
            "real n={n}: pf rel {pf_rel:.2e}, inv rel {inv_rel:.2e}, pf bit-identical {identical}"
        );
        // Same pivots and update order: the Pfaffian agrees to rounding of the final product.
        assert!(pf_rel < 1e-13, "n={n} pf rel {pf_rel}");
        assert!(inv_rel < 1e-10, "n={n} inv rel {inv_rel}");
    }
}

#[test]
fn native_complex_matches_pfapack() {
    for n in [2usize, 4, 6, 16, 24] {
        let (pf_rel, inv_rel, identical) = compare::<Complex64>(n, 2, 3, 77 + n as u64);
        eprintln!("complex n={n}: pf rel {pf_rel:.2e}, inv rel {inv_rel:.2e}, pf bit-identical {identical}");
        assert!(pf_rel < 1e-12, "n={n} pf rel {pf_rel}");
        assert!(inv_rel < 1e-9, "n={n} inv rel {inv_rel}");
    }
}

#[test]
fn native_reports_zero_pivot_and_isolates_planes() {
    let n = 6;
    let mut planes = random_planes::<f64>(n, 4, 5);
    planes[n * n..2 * n * n].fill(0.0);
    let out = pfaffian_inverse_batched(&Backend::TenferroNative, &planes, n, 2, 2).unwrap();
    let want = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, 2, 2).unwrap();
    assert_eq!(out.status, want.status);
    assert_eq!(out.status[1], PlaneStatus::ZeroPivot { row: 5 });
    assert_eq!(out.pf[1], 0.0);
    assert!(out.inv[n * n..2 * n * n].iter().all(|&x| x == 0.0));
    for p in [0usize, 2, 3] {
        assert_eq!(out.status[p], PlaneStatus::Ok);
        assert!((out.pf[p] - want.pf[p]).abs() <= 1e-13 * want.pf[p].abs());
    }
}

#[test]
fn native_reports_non_finite_plane() {
    let n = 4;
    let mut planes = random_planes::<f64>(n, 2, 8);
    planes[n * n + 2 + 3 * n] = f64::NAN;
    let out = pfaffian_inverse_batched(&Backend::TenferroNative, &planes, n, 2, 1).unwrap();
    assert_eq!(out.status[0], PlaneStatus::Ok);
    assert_ne!(out.status[1], PlaneStatus::Ok);
}
