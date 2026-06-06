//! Thin BLAS-2 / LAPACK backend layer used by the LTL and Pfaffian
//! kernels.
//!
//! The pfapack crate ships two interchangeable implementations:
//!
//! * **default (no feature)** — pure Rust scalar loops that mirror the
//!   upstream Julia loop ordering and produce bit-deterministic
//!   output across platforms. Suitable for golden tests and
//!   environments without a system BLAS.
//! * **`blas-backend` feature** — dispatches the hot routines to the
//!   `blas` / `lapack` crates (= whatever BLAS / LAPACK the leaf
//!   binary links against). This is what Julia uses by default via
//!   `LinearAlgebra`. Bit-level results may differ in the last 1-2
//!   ULPs depending on the backend's reduction order.
//!
//! Only the four kernels actually invoked by `pfaffian.rs`, `ltl.rs`,
//! and `utu2.rs` are exposed:
//!
//! | kernel | upstream call | semantics |
//! |---|---|---|
//! | [`skr2_neg`] | two `BLAS.ger!` calls | `A += α (x yᵀ − y xᵀ)` |
//! | [`scal_strided`] | `BLAS.scal!` | `x ← α · x` |
//! | [`trtri_uu_inplace`] | `LAPACK.trtri!('U','U', view)` | unit-upper triangular inverse |
//! | [`trmm_lutu`] | `BLAS.trmm!('L','U','T','U', 1.0, M, A)` | `A ← Mᵀ · A` |
//!
//! The `skew-tridiagonal` solve from `sktdsmx!` has no BLAS analogue
//! and stays scalar in both backends.

#![cfg_attr(feature = "blas-backend", allow(unsafe_code))]

use num_complex::Complex64;

/// Generic scalar field trait covered by the BLAS backend.
///
/// `f64` uses `dger / dscal / dtrmm / dtrtri`.
/// `Complex64` uses `zgeru / zscal / ztrmm / ztrtri`.
pub trait BlasScalar:
    Copy
    + core::ops::Neg<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::AddAssign
    + core::ops::MulAssign
    + core::ops::Div<Output = Self>
    + crate::pfaffian::PfafOne
    + BlasKernels
{
}

impl BlasScalar for f64 {}
impl BlasScalar for Complex64 {}

// ---------------------------------------------------------------------------
// skr2_neg: A += α (x yᵀ − y xᵀ)  (skew-symmetric rank-2 update)
// ---------------------------------------------------------------------------
//
// Mirrors the Julia idiom inside both `pfaffian_ltl!` and `julia_dsktf2!`:
//
//     BLAS.ger!( α, x, y, Asub)
//     BLAS.ger!(-α, y, x, Asub)
//
// Asub is column-major, n_rows × n_cols, and lives inside a larger column-
// major buffer with leading dimension `lda`. `x` and `y` are length-n_rows
// vectors stored contiguously in their own column of that buffer (stride 1).
//
// The default scalar implementation walks the dense (n_rows × n_cols)
// block; the BLAS implementation issues two `dger` / `zgeru` calls so
// the heavy lifting goes through the platform BLAS.

// `skr2_neg` (offset-based variant) was the original entry point but
// is currently unused: every caller has to gather `x` / `y` from
// non-unit strides, so `skr2_neg_vec` is the only call site. Drop the
// dead in-place variant to keep the surface area minimal; bring back
// from git history if a true stride-1 caller appears.

/// Variant of [`skr2_neg`] that takes `x` / `y` as separately-owned
/// contiguous slices instead of offsets into the matrix buffer.
///
/// Used by callers that have to gather `x` / `y` from non-unit strides
/// (e.g. `pfaffian_ltl_generic`, where `tau` is a row of A).
#[allow(clippy::too_many_arguments)]
pub fn skr2_neg_vec<T: BlasScalar>(
    a: &mut [T],
    lda: usize,
    a_row0: usize,
    a_col0: usize,
    n: usize,
    alpha: T,
    x: &mut [T],
    y: &mut [T],
) {
    if n == 0 {
        return;
    }
    assert!(x.len() >= n && y.len() >= n, "skr2_neg_vec: x/y too short");
    T::skr2_neg_vec(a, lda, a_row0, a_col0, n, alpha, x, y);
}

/// `x ← α · x` for a contiguous strided slice (stride 1, length `n`).
pub fn scal_strided<T: BlasScalar>(x: &mut [T], n: usize, alpha: T) {
    if n == 0 {
        return;
    }
    T::scal_strided(x, n, alpha);
}

/// In-place inverse of an `n × n` unit-upper-triangular block, mirroring
/// `LAPACK.trtri!('U','U', view)`.
///
/// The block is anchored at `a[row0 + col0 * lda]` inside the column-major
/// buffer `a` (leading dimension `lda`). Only the strict upper triangle is
/// read/written; the diagonal is implicitly 1.
pub fn trtri_uu_inplace<T: BlasScalar>(
    a: &mut [T],
    lda: usize,
    row0: usize,
    col0: usize,
    n: usize,
) {
    if n < 2 {
        return;
    }
    T::trtri_uu(a, lda, row0, col0, n);
}

/// `A ← Mᵀ · A` where M is `n × n` unit upper triangular.
///
/// Both matrices are column-major with leading dimension `n`. Mirrors
/// `BLAS.trmm!('L','U','T','U', 1.0, M, A)`.
pub fn trmm_lutu<T: BlasScalar>(m: &[T], a: &mut [T], n: usize) {
    if n == 0 {
        return;
    }
    T::trmm_lutu(m, a, n);
}

// ---------------------------------------------------------------------------
// Per-scalar trait that the kernel implementations dispatch to.
// ---------------------------------------------------------------------------

#[doc(hidden)]
pub trait BlasKernels: Sized + Copy {
    #[allow(clippy::too_many_arguments)]
    fn skr2_neg_vec(
        a: &mut [Self],
        lda: usize,
        a_row0: usize,
        a_col0: usize,
        n: usize,
        alpha: Self,
        x: &[Self],
        y: &[Self],
    );
    fn scal_strided(x: &mut [Self], n: usize, alpha: Self);
    fn trtri_uu(a: &mut [Self], lda: usize, row0: usize, col0: usize, n: usize);
    fn trmm_lutu(m: &[Self], a: &mut [Self], n: usize);
}

#[inline]
fn idx(lda: usize, row: usize, col: usize) -> usize {
    col * lda + row
}

#[inline]
fn scalar_trtri_uu_impl<T>(a: &mut [T], lda: usize, row0: usize, col0: usize, n: usize)
where
    T: Copy + core::ops::Neg<Output = T> + core::ops::Mul<Output = T> + core::ops::AddAssign,
{
    if n < 2 {
        return;
    }
    for j in 1..n {
        for i in 0..j {
            let mut acc = a[idx(lda, row0 + i, col0 + j)];
            for k in (i + 1)..j {
                let x_ik = a[idx(lda, row0 + i, col0 + k)];
                let x_kj = a[idx(lda, row0 + k, col0 + j)];
                acc += x_ik * x_kj;
            }
            let pos = idx(lda, row0 + i, col0 + j);
            a[pos] = -acc;
        }
    }
}

#[inline]
fn scalar_trmm_lutu_impl<T>(m: &[T], a: &mut [T], n: usize)
where
    T: Copy + core::ops::Mul<Output = T> + core::ops::AddAssign,
{
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        for j in 0..n {
            let mut acc = a[idx(n, i, j)];
            for k in 0..i {
                acc += m[idx(n, k, i)] * a[idx(n, k, j)];
            }
            a[idx(n, i, j)] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust default implementations (also serve as the reference oracle).
// ---------------------------------------------------------------------------

#[cfg(not(feature = "blas-backend"))]
mod scalar_impl {
    use super::*;

    // Concrete scalar implementations for the two supported types.
    // (We can't use a blanket impl because `BlasScalar` is now a
    //  supertrait of `BlasKernels`.)
    macro_rules! scalar_impls {
        ($t:ty) => {
            impl BlasKernels for $t {
                fn skr2_neg_vec(
                    a: &mut [$t],
                    lda: usize,
                    a_row0: usize,
                    a_col0: usize,
                    n: usize,
                    alpha: $t,
                    x: &[$t],
                    y: &[$t],
                ) {
                    for j in 0..n {
                        let y_j = y[j];
                        let x_j = x[j];
                        for i in 0..n {
                            let x_i = x[i];
                            let y_i = y[i];
                            let upd = alpha * (x_i * y_j - y_i * x_j);
                            let k = idx(lda, a_row0 + i, a_col0 + j);
                            a[k] += upd;
                        }
                    }
                }

                fn scal_strided(x: &mut [$t], n: usize, alpha: $t) {
                    for v in &mut x[..n] {
                        *v *= alpha;
                    }
                }

                fn trtri_uu(a: &mut [$t], lda: usize, row0: usize, col0: usize, n: usize) {
                    // Column-by-column STRTI2 (DIAG='U') in place. Mirrors the
                    // upstream `LAPACK.trtri!('U','U', view)` behaviour.
                    scalar_trtri_uu_impl(a, lda, row0, col0, n);
                }

                fn trmm_lutu(m: &[$t], a: &mut [$t], n: usize) {
                    // A_new[i, j] = A[i, j] + sum_{k=0..i-1} M[k, i] * A[k, j].
                    // Process rows i from n-1 down to 0 so updated values don't
                    // contaminate the dependencies for smaller i (which read A[k, j]
                    // for k < i, untouched at that point).
                    scalar_trmm_lutu_impl(m, a, n);
                }
            }
        };
    }

    scalar_impls!(f64);
    scalar_impls!(Complex64);
}

// ---------------------------------------------------------------------------
// BLAS / LAPACK implementations (feature = "blas-backend").
// ---------------------------------------------------------------------------

#[cfg(feature = "blas-backend")]
mod blas_impl {
    use super::*;

    /// Translate a usize `lda` to BLAS `i32`. Matrices we feed in are
    /// small (`n <= a few hundred`); we still guard against the rare
    /// case of `n` exceeding `i32::MAX`.
    #[inline]
    fn as_i32(v: usize, what: &str) -> i32 {
        i32::try_from(v)
            .unwrap_or_else(|_| panic!("pfapack BLAS backend: {what} = {v} does not fit in i32"))
    }

    impl BlasKernels for f64 {
        fn skr2_neg_vec(
            a: &mut [f64],
            lda: usize,
            a_row0: usize,
            a_col0: usize,
            n: usize,
            alpha: f64,
            x: &[f64],
            y: &[f64],
        ) {
            let n_i = as_i32(n, "n");
            let lda_i = as_i32(lda, "lda");
            let a_off = a_col0 * lda + a_row0;
            let a_view = &mut a[a_off..a_off + (n - 1) * lda + n];
            unsafe {
                blas::dger(n_i, n_i, alpha, x, 1, y, 1, a_view, lda_i);
                blas::dger(n_i, n_i, -alpha, y, 1, x, 1, a_view, lda_i);
            }
        }

        fn scal_strided(x: &mut [f64], n: usize, alpha: f64) {
            let n_i = as_i32(n, "n");
            // SAFETY: x has at least `n` contiguous f64 entries by the caller's contract.
            unsafe { blas::dscal(n_i, alpha, x, 1) };
        }

        fn trtri_uu(a: &mut [f64], lda: usize, row0: usize, col0: usize, n: usize) {
            if n <= 64 {
                scalar_trtri_uu_impl(a, lda, row0, col0, n);
                return;
            }
            let n_i = as_i32(n, "n");
            let lda_i = as_i32(lda, "lda");
            let mut info: i32 = 0;
            let a_off = col0 * lda + row0;
            let view = &mut a[a_off..a_off + (n - 1) * lda + n];
            // SAFETY: `view` covers the (n × n) block with leading dim `lda`.
            unsafe { lapack::dtrtri(b'U', b'U', n_i, view, lda_i, &mut info) };
            if info != 0 {
                panic!("pfapack BLAS backend: dtrtri failed with INFO={info}");
            }
        }

        fn trmm_lutu(m: &[f64], a: &mut [f64], n: usize) {
            if n <= 64 {
                scalar_trmm_lutu_impl(m, a, n);
                return;
            }
            let n_i = as_i32(n, "n");
            unsafe {
                blas::dtrmm(b'L', b'U', b'T', b'U', n_i, n_i, 1.0, m, n_i, a, n_i);
            }
        }
    }

    impl BlasKernels for Complex64 {
        fn skr2_neg_vec(
            a: &mut [Complex64],
            lda: usize,
            a_row0: usize,
            a_col0: usize,
            n: usize,
            alpha: Complex64,
            x: &[Complex64],
            y: &[Complex64],
        ) {
            let n_i = as_i32(n, "n");
            let lda_i = as_i32(lda, "lda");
            let a_off = a_col0 * lda + a_row0;
            let a_view = &mut a[a_off..a_off + (n - 1) * lda + n];
            unsafe {
                blas::zgeru(n_i, n_i, alpha, x, 1, y, 1, a_view, lda_i);
                blas::zgeru(n_i, n_i, -alpha, y, 1, x, 1, a_view, lda_i);
            }
        }

        fn scal_strided(x: &mut [Complex64], n: usize, alpha: Complex64) {
            let n_i = as_i32(n, "n");
            unsafe { blas::zscal(n_i, alpha, x, 1) };
        }

        fn trtri_uu(a: &mut [Complex64], lda: usize, row0: usize, col0: usize, n: usize) {
            if n <= 32 {
                scalar_trtri_uu_impl(a, lda, row0, col0, n);
                return;
            }
            let n_i = as_i32(n, "n");
            let lda_i = as_i32(lda, "lda");
            let mut info: i32 = 0;
            let a_off = col0 * lda + row0;
            let view = &mut a[a_off..a_off + (n - 1) * lda + n];
            unsafe { lapack::ztrtri(b'U', b'U', n_i, view, lda_i, &mut info) };
            if info != 0 {
                panic!("pfapack BLAS backend: ztrtri failed with INFO={info}");
            }
        }

        fn trmm_lutu(m: &[Complex64], a: &mut [Complex64], n: usize) {
            if n <= 32 {
                scalar_trmm_lutu_impl(m, a, n);
                return;
            }
            let n_i = as_i32(n, "n");
            // For complex 'T' = transpose (no conjugate). Julia's call
            // `BLAS.trmm!('L','U','T','U', 1.0, M, A)` likewise uses 'T'.
            let one = Complex64::new(1.0, 0.0);
            unsafe {
                blas::ztrmm(b'L', b'U', b'T', b'U', n_i, n_i, one, m, n_i, a, n_i);
            }
        }
    }
}

#[cfg(all(test, feature = "blas-backend"))]
mod tests {
    use super::*;

    #[inline]
    fn idx(lda: usize, row: usize, col: usize) -> usize {
        col * lda + row
    }

    #[inline]
    fn z(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn assert_complex_slices_close(got: &[Complex64], want: &[Complex64]) {
        assert_eq!(got.len(), want.len());
        for (i, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (*got - *want).norm();
            let scale = got.norm().max(want.norm()).max(1.0);
            assert!(
                diff <= 1e-12 * scale,
                "complex slice mismatch at idx {i}: got={got:?} want={want:?} diff={diff:.3e}"
            );
        }
    }

    fn reference_trtri_uu(a: &mut [Complex64], lda: usize, row0: usize, col0: usize, n: usize) {
        if n < 2 {
            return;
        }

        let mut tmp = vec![Complex64::new(0.0, 0.0); n - 1];
        for j in 1..n {
            for i in 0..j {
                tmp[i] = a[idx(lda, row0 + i, col0 + j)];
            }
            for i in 0..j {
                let mut acc = tmp[i];
                for k in (i + 1)..j {
                    acc += a[idx(lda, row0 + i, col0 + k)] * tmp[k];
                }
                a[idx(lda, row0 + i, col0 + j)] = -acc;
            }
        }
    }

    fn reference_trmm_lutu(m: &[Complex64], a: &mut [Complex64], n: usize) {
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            for j in 0..n {
                let mut acc = a[idx(n, i, j)];
                for k in 0..i {
                    acc += m[idx(n, k, i)] * a[idx(n, k, j)];
                }
                a[idx(n, i, j)] = acc;
            }
        }
    }

    #[test]
    fn complex_trtri_uu_offset_submatrix_matches_scalar_reference() {
        let lda = 7;
        let row0 = 1;
        let col0 = 2;
        let n = 4;
        let mut got: Vec<_> = (0..lda * lda)
            .map(|k| z(0.125 + 0.031 * k as f64, -0.25 + 0.017 * k as f64))
            .collect();

        for j in 0..n {
            for i in 0..n {
                let value = if i < j {
                    z(
                        0.2 + 0.13 * i as f64 + 0.19 * j as f64,
                        -0.3 + 0.07 * i as f64 - 0.11 * j as f64,
                    )
                } else {
                    z(
                        7.0 + 0.3 * i as f64 + 0.2 * j as f64,
                        -5.0 + 0.4 * i as f64 - 0.1 * j as f64,
                    )
                };
                got[idx(lda, row0 + i, col0 + j)] = value;
            }
        }

        let mut want = got.clone();
        reference_trtri_uu(&mut want, lda, row0, col0, n);

        trtri_uu_inplace::<Complex64>(&mut got, lda, row0, col0, n);

        assert_complex_slices_close(&got, &want);
    }

    #[test]
    fn complex_trmm_lutu_uses_plain_transpose_not_conjugate_transpose() {
        let n = 5;
        let mut m = vec![Complex64::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                m[idx(n, i, j)] = z(
                    3.0 + 0.2 * i as f64 + 0.1 * j as f64,
                    -2.0 + 0.05 * i as f64 - 0.07 * j as f64,
                );
            }
        }
        for j in 0..n {
            for i in 0..j {
                m[idx(n, i, j)] = z(
                    0.35 + 0.09 * i as f64 + 0.14 * j as f64,
                    -0.42 + 0.08 * i as f64 - 0.06 * j as f64,
                );
            }
            m[idx(n, j, j)] = z(19.0 + j as f64, -23.0 - j as f64);
        }

        let mut got: Vec<_> = (0..n * n)
            .map(|k| z(-0.7 + 0.03 * k as f64, 0.45 - 0.04 * k as f64))
            .collect();
        let mut want = got.clone();
        reference_trmm_lutu(&m, &mut want, n);

        trmm_lutu::<Complex64>(&m, &mut got, n);

        assert_complex_slices_close(&got, &want);
    }
}
