//! Pfaffian via the Parlett–Reid algorithm.
//!
//! Port target: `extern/Julia-mVMC/PfaPack.jl/src/pfaffian.jl::pfaffian_ltl!`.
//!
//! Algorithm reference: M. Wimmer, *Algorithm 923*, ACM TOMS 38 (2012),
//! 30:1–30:17. The upstream Julia is itself a line-by-line port of the
//! Python reference in PfaPack 2014-09 (`python/pfaffian.py:247-308`),
//! and the Rust port below is a line-by-line port of the Julia one.
//! Per-line `// Julia: …` comments mark the correspondence.
//!
//! License: BSD-3-Clause.
//!
//! ## Numerical contract (mirrors Julia)
//!
//! * Returns `T::zero()` when `n` is odd — do NOT use `Option`; downstream
//!   `vmc_sampling` divide-by-zero checks compare against zero directly.
//! * Overwrites `a` (the matrix is in LTL form on exit). The caller can
//!   pass a `clone()` if it needs to keep the original.

use num_complex::Complex64;

use crate::backend::{self, BlasScalar};
use crate::mat::SqMat;

/// Real Pfaffian via the Parlett–Reid LTL algorithm (in-place).
///
/// Mirrors `pfaffian_ltl!(A; overwrite_a=true)` from `pfaffian.jl`.
pub fn pfaffian_ltl_real(a: &mut SqMat<'_, f64>) -> f64 {
    pfaffian_ltl_generic::<f64, _, _>(a, |z| z.abs() * z.abs(), |z| z != 0.0)
}

/// Complex Pfaffian via the Parlett–Reid LTL algorithm (in-place).
pub fn pfaffian_ltl_complex(a: &mut SqMat<'_, Complex64>) -> Complex64 {
    pfaffian_ltl_generic::<Complex64, _, _>(
        a,
        // |z|^2 = re^2 + im^2 — Julia uses `abs2` which is the same.
        |z| z.norm_sqr(),
        |z| z != Complex64::new(0.0, 0.0),
    )
}

// Generic core. Parameterised over the absolute-square pivot metric (so
// the real version can avoid the complex `norm_sqr` cost) and the
// non-zero predicate (so the complex version compares against
// `Complex64::new(0, 0)`, mirroring Julia's `iszero`).
fn pfaffian_ltl_generic<T, Abs2, IsNz>(a: &mut SqMat<'_, T>, abs2: Abs2, is_nz: IsNz) -> T
where
    T: BlasScalar,
    Abs2: Fn(T) -> f64,
    IsNz: Fn(T) -> bool,
{
    let n = a.n();

    // Julia: if n % 2 == 1 -> return zero(T)
    if n % 2 == 1 {
        return T::pfaf_zero();
    }

    let mut pf = T::pfaf_one();
    let mut tau_buf = Vec::<T>::with_capacity(n);
    let mut y_buf = Vec::<T>::with_capacity(n);

    // Julia: for k in 0:2:(n-2)
    let mut k = 0usize;
    while k + 1 < n {
        // Julia uses k1 = k + 1 and k2 = k + 2 as the 1-based indices
        // of Python's k and k+1. In Rust 0-based that becomes:
        let i_piv_col = k; // Julia's k1-1
        let i_piv_row_start = k + 1; // Julia's k2-1

        // Find pivot: argmax_{i in (k+1)..n} |A[i, k]|
        let mut max_abs2 = abs2(a.get(i_piv_row_start, i_piv_col));
        let mut max_idx = i_piv_row_start;
        for i in (i_piv_row_start + 1)..n {
            let v = abs2(a.get(i, i_piv_col));
            if v > max_abs2 {
                max_abs2 = v;
                max_idx = i;
            }
        }

        // Swap row k+1 with max_idx and column k+1 with max_idx if needed.
        if max_idx != i_piv_row_start {
            // Julia: swap rows k2 and max_idx for j in 1:n
            for j in 0..n {
                let v = a.get(i_piv_row_start, j);
                a.set(i_piv_row_start, j, a.get(max_idx, j));
                a.set(max_idx, j, v);
            }
            // Julia: swap cols k2 and max_idx for i in 1:n
            for i in 0..n {
                let v = a.get(i, i_piv_row_start);
                a.set(i, i_piv_row_start, a.get(i, max_idx));
                a.set(i, max_idx, v);
            }
            pf = -pf;
        }

        // Julia: if iszero(A[k1, k2]) -> return zero(T)
        let pivot = a.get(i_piv_col, i_piv_row_start);
        if !is_nz(pivot) {
            return T::pfaf_zero();
        }

        // Julia: if k2 + 1 <= n -> form tau and do the rank-2 update.
        //
        // In 0-based:
        //   - tau = A[k, k+2 ..= n-1] / A[k, k+1]
        //   - y   = A[k+2 ..= n-1, k+1]
        //   - Asub = A[k+2 ..= n-1, k+2 ..= n-1]
        //   - Asub += outer(tau, y) - outer(y, tau)
        if i_piv_row_start + 1 < n {
            let start = i_piv_row_start + 1; // first row/col of the trailing submatrix
            pf *= pivot;
            let inv = T::pfaf_one() / pivot;

            // Skew-symmetric rank-2 update of the trailing submatrix:
            //   Asub += inv * (tau yᵀ − y tauᵀ)
            // where tau lives in row `i_piv_col` of A and y in column
            // `i_piv_row_start`. Both vectors are length `len`.
            //
            // tau[r] == A[i_piv_col, start + r]
            //        → column-major offset (start + r) * lda + i_piv_col,
            //          i.e. stride = lda (not 1).
            // y[r]   == A[start + r, i_piv_row_start]
            //        → column-major offset i_piv_row_start * lda + (start + r),
            //          i.e. stride = 1.
            let len = n - start;
            let lda = a.lda();
            tau_buf.clear();
            y_buf.clear();
            for r in 0..len {
                tau_buf.push(a.get(i_piv_col, start + r));
                y_buf.push(a.get(start + r, i_piv_row_start));
            }
            backend::skr2_neg_vec::<T>(
                a.as_mut_slice(),
                lda,
                start,
                start,
                len,
                inv,
                &mut tau_buf,
                &mut y_buf,
            );
        } else {
            // Julia: last iteration -- just multiply by the off-diagonal.
            pf *= pivot;
        }

        k += 2;
    }

    pf
}

/// Tiny trait providing `0` and `1` constants for the scalar types we
/// support. Avoids dragging in `num-traits` for a one-off use.
pub trait PfafOne: Sized {
    /// Multiplicative identity (`1`).
    fn pfaf_one() -> Self;
    /// Additive identity (`0`).
    fn pfaf_zero() -> Self;
}

impl PfafOne for f64 {
    #[inline]
    fn pfaf_zero() -> Self {
        0.0
    }
    #[inline]
    fn pfaf_one() -> Self {
        1.0
    }
}

impl PfafOne for Complex64 {
    #[inline]
    fn pfaf_zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    #[inline]
    fn pfaf_one() -> Self {
        Complex64::new(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn odd_returns_zero() {
        let mut buf = vec![0.0; 9];
        let mut m = SqMat::new(&mut buf, 3);
        assert_eq!(pfaffian_ltl_real(&mut m), 0.0);
    }

    #[test]
    fn two_by_two_real() {
        // A = [[0, a], [-a, 0]] -> Pf(A) = a
        let mut buf = vec![0.0, -3.5, 3.5, 0.0]; // column-major: A[0,0]=0, A[1,0]=-3.5, A[0,1]=3.5, A[1,1]=0
        let mut m = SqMat::new(&mut buf, 2);
        let pf = pfaffian_ltl_real(&mut m);
        assert_relative_eq!(pf, 3.5, max_relative = 1e-15);
    }

    #[test]
    fn four_by_four_real_against_known() {
        // Block-diagonal: A = diag([[0, 2], [-2, 0]], [[0, 5], [-5, 0]])
        // Pf(A) = 2 * 5 = 10.
        let n = 4;
        let mut buf = vec![0.0; n * n];
        let mut m = SqMat::new(&mut buf, n);
        m.set(0, 1, 2.0);
        m.set(1, 0, -2.0);
        m.set(2, 3, 5.0);
        m.set(3, 2, -5.0);
        let pf = pfaffian_ltl_real(&mut m);
        assert_relative_eq!(pf, 10.0, max_relative = 1e-15);
    }

    #[test]
    fn pf_squared_equals_det_for_random_skew() {
        // Pf(A)^2 = det(A) for any skew-symmetric A. The 4x4 case is
        // small enough to compute det by hand via Leibniz expansion.
        let n = 4;
        let raw = [
            [0.0, 1.5, -0.7, 2.1],
            [-1.5, 0.0, 0.3, -0.4],
            [0.7, -0.3, 0.0, 1.1],
            [-2.1, 0.4, -1.1, 0.0],
        ];
        // Column-major flatten.
        let mut buf = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                buf[j * n + i] = raw[i][j];
            }
        }
        // Snapshot for det computation.
        let work = buf.clone();
        let det = det_4x4_dense(&work, n);
        let mut m = SqMat::new(&mut buf, n);
        let pf = pfaffian_ltl_real(&mut m);
        // Allow loose tolerance because det is computed with a slightly
        // different summation order than Pf^2.
        assert_relative_eq!(pf * pf, det, max_relative = 1e-12);
    }

    /// Tiny Leibniz-expansion det for 4x4 dense column-major. Used only
    /// as a cross-check oracle for the Pfaffian test above.
    fn det_4x4_dense(buf: &[f64], n: usize) -> f64 {
        assert_eq!(n, 4);
        let at = |i: usize, j: usize| buf[j * n + i];
        let mut det = 0.0;
        let perms: &[([usize; 4], i32)] = &[
            ([0, 1, 2, 3], 1),
            ([0, 1, 3, 2], -1),
            ([0, 2, 1, 3], -1),
            ([0, 2, 3, 1], 1),
            ([0, 3, 1, 2], 1),
            ([0, 3, 2, 1], -1),
            ([1, 0, 2, 3], -1),
            ([1, 0, 3, 2], 1),
            ([1, 2, 0, 3], 1),
            ([1, 2, 3, 0], -1),
            ([1, 3, 0, 2], -1),
            ([1, 3, 2, 0], 1),
            ([2, 0, 1, 3], 1),
            ([2, 0, 3, 1], -1),
            ([2, 1, 0, 3], -1),
            ([2, 1, 3, 0], 1),
            ([2, 3, 0, 1], 1),
            ([2, 3, 1, 0], -1),
            ([3, 0, 1, 2], -1),
            ([3, 0, 2, 1], 1),
            ([3, 1, 0, 2], 1),
            ([3, 1, 2, 0], -1),
            ([3, 2, 0, 1], -1),
            ([3, 2, 1, 0], 1),
        ];
        for (p, s) in perms {
            let term = at(p[0], 0) * at(p[1], 1) * at(p[2], 2) * at(p[3], 3);
            det += (*s as f64) * term;
        }
        det
    }

    #[test]
    fn two_by_two_complex() {
        let a = Complex64::new(1.0, 2.0);
        let mut buf = vec![Complex64::new(0.0, 0.0), -a, a, Complex64::new(0.0, 0.0)];
        let mut m = SqMat::new(&mut buf, 2);
        let pf = pfaffian_ltl_complex(&mut m);
        assert_relative_eq!(pf.re, a.re, max_relative = 1e-15);
        assert_relative_eq!(pf.im, a.im, max_relative = 1e-15);
    }
}
