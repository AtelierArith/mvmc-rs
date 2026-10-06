//! CPU backends: pfapack loop (serial/rayon) and the tenferro `ExtensionOp` must be
//! bit-identical to calling pfapack per plane; the invariants hold; statuses are correct.

use mvmc_gpu::testkit::*;
use mvmc_gpu::{pfaffian_inverse_batched, Backend, BatchedPfaffian, Error, PfScalar, PlaneStatus};
use num_complex::Complex64;
use pfapack::{PivotIndex1Based, SqMat};

/// Direct pfapack call for one plane, independent of the crate's own wrapper.
trait Direct: TestScalar {
    fn direct(a: &mut [Self], n: usize) -> (Self, PlaneStatus);
}

impl Direct for f64 {
    fn direct(a: &mut [f64], n: usize) -> (f64, PlaneStatus) {
        let mut piv = vec![PivotIndex1Based(0); n];
        let mut sm = SqMat::new(&mut *a, n);
        if let Err(row) = pfapack::dsktf2(&mut sm, &mut piv) {
            return (0.0, PlaneStatus::ZeroPivot { row });
        }
        let pf = pfapack::utu2pfa_real(&sm, &piv);
        let mut vt = vec![0.0; n - 1];
        let mut m = vec![0.0; n * n];
        let mut mm = SqMat::new(&mut m, n);
        pfapack::utu2inv_real(&mut sm, &piv, &mut vt, &mut mm);
        (pf, PlaneStatus::Ok)
    }
}

impl Direct for Complex64 {
    fn direct(a: &mut [Complex64], n: usize) -> (Complex64, PlaneStatus) {
        let mut piv = vec![PivotIndex1Based(0); n];
        let mut sm = SqMat::new(&mut *a, n);
        if let Err(row) = pfapack::zsktf2(&mut sm, &mut piv) {
            return (Complex64::new(0.0, 0.0), PlaneStatus::ZeroPivot { row });
        }
        let pf = pfapack::utu2pfa_complex(&sm, &piv);
        let mut vt = vec![Complex64::new(0.0, 0.0); n - 1];
        let mut m = vec![Complex64::new(0.0, 0.0); n * n];
        let mut mm = SqMat::new(&mut m, n);
        pfapack::utu2inv_complex(&mut sm, &piv, &mut vt, &mut mm);
        (pf, PlaneStatus::Ok)
    }
}

fn bits<T: PfScalar>(v: &[T]) -> Vec<u64> {
    T::as_f64_lanes(v).iter().map(|x| x.to_bits()).collect()
}

fn assert_bit_identical<T: Direct>(backend: Backend, n: usize, nqp: usize, batch: usize) {
    let planes = random_planes::<T>(n, nqp * batch, 17 + n as u64);
    let out = pfaffian_inverse_batched(&backend, &planes, n, nqp, batch).unwrap();
    let nn = n * n;
    let mut want_pf = Vec::new();
    let mut want_inv = Vec::new();
    for p in 0..nqp * batch {
        let mut a = planes[p * nn..(p + 1) * nn].to_vec();
        let (pf, st) = T::direct(&mut a, n);
        assert_eq!(st, PlaneStatus::Ok);
        assert_eq!(out.status[p], PlaneStatus::Ok);
        want_pf.push(pf);
        want_inv.extend(a);
    }
    assert_eq!(
        bits(&out.pf),
        bits(&want_pf),
        "pf must be bit-identical ({backend:?}, n={n})"
    );
    assert_eq!(
        bits(&out.inv),
        bits(&want_inv),
        "inv must be bit-identical ({backend:?}, n={n})"
    );
}

#[test]
fn cpu_batched_matches_pfapack_bitwise_real() {
    for n in [2usize, 4, 16, 32] {
        assert_bit_identical::<f64>(Backend::CpuPfapack, n, 3, 2);
        assert_bit_identical::<f64>(Backend::CpuPfapackRayon, n, 3, 2);
    }
}

#[test]
fn cpu_batched_matches_pfapack_bitwise_complex() {
    for n in [2usize, 4, 16, 32] {
        assert_bit_identical::<Complex64>(Backend::CpuPfapack, n, 3, 2);
        assert_bit_identical::<Complex64>(Backend::CpuPfapackRayon, n, 3, 2);
    }
}

#[test]
fn tenferro_extension_matches_pfapack_bitwise() {
    for n in [2usize, 4, 16, 32] {
        assert_bit_identical::<f64>(Backend::TenferroExtension, n, 3, 2);
        assert_bit_identical::<Complex64>(Backend::TenferroExtension, n, 3, 2);
    }
}

fn check_invariants<T: TestScalar>(backend: Backend, n: usize) {
    let planes = random_planes::<T>(n, 6, 5 + n as u64);
    let out = pfaffian_inverse_batched(&backend, &planes, n, 3, 2).unwrap();
    let inv = invariants(&planes, &out);
    // Gaussian-like skew matrices of side <= 32 have cond ~ 1e1..1e3; n * eps * cond << 1e-9.
    assert!(
        inv.max_identity_residual < 1e-9,
        "{backend:?} n={n}: {}",
        inv.max_identity_residual
    );
    assert!(
        inv.max_skew_defect < 1e-11,
        "{backend:?} n={n}: {}",
        inv.max_skew_defect
    );
    assert!(
        inv.max_pf2_det_rel < 1e-9,
        "{backend:?} n={n}: {}",
        inv.max_pf2_det_rel
    );
}

#[test]
fn cpu_backends_satisfy_pf_squared_det_and_identity_residual() {
    for backend in [
        Backend::CpuPfapack,
        Backend::CpuPfapackRayon,
        Backend::TenferroExtension,
    ] {
        for n in [2usize, 8, 32] {
            check_invariants::<f64>(backend, n);
            check_invariants::<Complex64>(backend, n);
        }
    }
}

#[test]
fn pfaffian_of_2x2_has_the_sign_of_the_upper_entry() {
    let planes = vec![0.0, -3.0, 3.0, 0.0]; // A[0,1] = 3 (column-major: A[0,1] = data[0 + 1*2])
    let planes = {
        let mut p = planes;
        p[2] = 3.0;
        p[1] = -3.0;
        p
    };
    for backend in [Backend::CpuPfapack, Backend::TenferroExtension] {
        let out = pfaffian_inverse_batched(&backend, &planes, 2, 1, 1).unwrap();
        assert_eq!(out.pf, vec![3.0]);
        assert_eq!(out.status, vec![PlaneStatus::Ok]);
    }
}

fn statuses<T: TestScalar>(backend: Backend) -> BatchedPfaffian<T> {
    let n = 6;
    let mut planes = random_planes::<T>(n, 4, 99);
    // plane 1: all zero -> zero pivot at the first scanned column.
    planes[n * n..2 * n * n].fill(T::zero());
    // plane 2: NaN entry.
    planes[2 * n * n + 3 * n + 1] = T::from_pair(f64::NAN, f64::NAN);
    pfaffian_inverse_batched(&backend, &planes, n, 2, 2).unwrap()
}

#[test]
fn zero_and_nan_planes_report_status_and_isolate_failures() {
    for backend in [
        Backend::CpuPfapack,
        Backend::CpuPfapackRayon,
        Backend::TenferroExtension,
    ] {
        let out = statuses::<f64>(backend);
        assert_eq!(out.status[0], PlaneStatus::Ok);
        assert_eq!(
            out.status[1],
            PlaneStatus::ZeroPivot { row: 5 },
            "{backend:?}"
        );
        assert!(matches!(
            out.status[2],
            PlaneStatus::NonFinite | PlaneStatus::ZeroPivot { .. }
        ));
        assert_eq!(out.status[3], PlaneStatus::Ok);
        assert_eq!(out.pf[1], 0.0);
        assert!(out.inv[36..72].iter().all(|&x| x == 0.0));
        assert!(out.pf[0].is_finite() && out.pf[3].is_finite());
        let c = statuses::<Complex64>(backend);
        assert_eq!(c.status[1], PlaneStatus::ZeroPivot { row: 5 });
        assert_eq!(c.status[0], PlaneStatus::Ok);
    }
}

#[test]
fn invalid_shapes_are_typed_errors() {
    let data = vec![0.0f64; 16];
    assert!(matches!(
        pfaffian_inverse_batched(&Backend::CpuPfapack, &data, 3, 1, 1),
        Err(Error::InvalidShape(_))
    ));
    assert!(matches!(
        pfaffian_inverse_batched(&Backend::CpuPfapack, &data, 4, 2, 1),
        Err(Error::InvalidShape(_))
    ));
    assert!(matches!(
        pfaffian_inverse_batched(&Backend::CpuPfapack, &data[..0], 0, 1, 1),
        Err(Error::InvalidShape(_))
    ));
}
