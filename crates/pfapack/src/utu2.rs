// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Port of `extern/Julia-mVMC/PfaPack.jl/src/utu2.jl`, which is itself a
// Julia port of the C++ `ltl2inv` implementations originally written
// by RuQing Xu (@xrq-phys) as part of xrq-phys/Pfaffine
// (https://github.com/xrq-phys/Pfaffine) under the Mozilla Public
// License 2.0. The Rust port preserves the upstream MPL-2.0 license
// per MPL §1.10 (Modifications). See ../../THIRD_PARTY_LICENSES.md.

//! LTL-form post-decomposition helpers.
//!
//! * [`utu2pfa_real`] / [`utu2pfa_complex`] — extract the Pfaffian from
//!   a matrix that has already been put into upper-triangular LTL form
//!   by [`crate::ltl::dsktf2`] / [`crate::ltl::zsktf2`].
//! * [`utu2inv_real`] / [`utu2inv_complex`] — compute the inverse of
//!   the same matrix.
//!
//! The eight-step inverse pipeline mirrors the upstream
//! `deps/invert.tcc:76-118` block in xrq-phys/Pfaffine:
//!
//! ```text
//! 1. M ← I
//! 2. trtri  (invert the unit upper-triangular submatrix A[0..n-1, 1..n])
//! 3. lacpy  (copy upper-triangular block A[0..n-2, 2..n] → M[0..n-2, 1..n-1])
//! 4. extract negated tridiagonal vT[i] = -A[i, i+1]
//! 5. solve the skew-tridiagonal system (sktdsmx)
//! 6. column permutation by iPiv
//! 7. A ← M^T * A  (trmm, M unit upper-triangular)
//! 8. row permutation by iPiv
//! ```
//!
//! For Phase 2 we provide BLAS-free hand-rolled `trtri` and `trmm`
//! versions limited to the unit-upper-triangular case actually used
//! here. They are O(n^3) scalar loops; at the matrix sizes mVMC feeds
//! Pfaffian (a few hundred rows at most), the cost is well below the
//! sampler's own cost.

use num_complex::Complex64;

use crate::backend::{self, BlasScalar};
use crate::mat::SqMat;
use crate::pfaffian::PfafOne;
use crate::PivotIndex1Based;

// ---------------------------------------------------------------------------
// utu2pfa
// ---------------------------------------------------------------------------

/// Extract the Pfaffian from a real upper-triangular LTL form.
pub fn utu2pfa_real(a: &SqMat<'_, f64>, pivots: &[PivotIndex1Based]) -> f64 {
    utu2pfa_generic::<f64>(a, pivots)
}

/// Extract the Pfaffian from a complex upper-triangular LTL form.
pub fn utu2pfa_complex(a: &SqMat<'_, Complex64>, pivots: &[PivotIndex1Based]) -> Complex64 {
    utu2pfa_generic::<Complex64>(a, pivots)
}

fn utu2pfa_generic<T>(a: &SqMat<'_, T>, pivots: &[PivotIndex1Based]) -> T
where
    T: Copy + core::ops::MulAssign + core::ops::Neg<Output = T> + PfafOne + BlasScalar,
{
    let n = a.n();
    if n == 0 {
        return T::pfaf_one();
    }

    let mut pf = T::pfaf_one();
    // Julia: for i in 1:2:(n-1); pfaff *= A[i, i+1]
    let mut i = 0usize;
    while i + 1 < n {
        pf *= a.get(i, i + 1);
        i += 2;
    }

    // Sign from the LTL pivot permutation: each swap flips the sign.
    let mut sign_negative = false;
    for (k, p) in pivots.iter().enumerate() {
        if p.0 != (k as u32) + 1 {
            sign_negative = !sign_negative;
        }
    }
    // Julia returns T(sign) * pfaff, including complex multiplication's
    // signed-zero arithmetic. Unary negation differs on a real-axis result.
    let sign = if sign_negative {
        T::pfaf_zero() - T::pfaf_one()
    } else {
        T::pfaf_one()
    };
    sign * pf
}

// ---------------------------------------------------------------------------
// sktdsmx (internal helper for utu2inv)
// ---------------------------------------------------------------------------

/// Solve T * C = B where T is skew-tridiagonal with sub/super-diagonal
/// entries in `vt` (length n-1). `b` is the input, `c` is the output.
///
/// Matches `sktdsmx!` from `utu2.jl` line-by-line; the Julia version
/// pre-computes the inverses `inv(-vT[i+1])` etc., which we mirror so
/// the floating-point arithmetic order matches.
///
/// Index translation note: Julia uses 1-based indices throughout. The
/// forward/backward helpers below walk Julia indices verbatim and only
/// subtract 1 at the array boundary.
#[allow(clippy::needless_range_loop)]
fn forward_julia_style<T>(vt: &[T], b: &SqMat<'_, T>, c: &mut SqMat<'_, T>)
where
    T: Copy
        + core::ops::Neg<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>
        + core::ops::Div<Output = T>
        + PfafOne
        + BlasScalar,
{
    let n = b.n();
    if n == 0 {
        return;
    }
    let b_data = b.as_slice();
    let c_data = c.as_mut_slice();

    // Julia 1-based: vT[1..n-1], B[1..n, 1..n], C[1..n, 1..n].
    //   inv_minus_vT_1 = inv(-vT[1])
    //   C[2, j] = B[1, j] * inv_minus_vT_1  for j in 1..n
    let inv_minus_vt_1 = (-vt[0]).julia_inv();
    let mut base = 0usize;
    let unrolled = n / 4 * 4;
    for _ in (0..unrolled).step_by(4) {
        c_data[base + 1] = b_data[base] * inv_minus_vt_1;
        base += n;
        c_data[base + 1] = b_data[base] * inv_minus_vt_1;
        base += n;
        c_data[base + 1] = b_data[base] * inv_minus_vt_1;
        base += n;
        c_data[base + 1] = b_data[base] * inv_minus_vt_1;
        base += n;
    }
    for _ in unrolled..n {
        c_data[base + 1] = b_data[base] * inv_minus_vt_1;
        base += n;
    }
    // Julia: i_cpp = 2; while i_cpp < n;
    //   inv_minus_vT_i_cpp_1 = inv(-vT[i_cpp+1])
    //   vT_i_cpp = vT[i_cpp]
    //   for j in 1:n;
    //     C[i_cpp + 2, j] = (B[i_cpp + 1, j] - C[i_cpp, j] * vT_i_cpp) * inv_minus_vT_i_cpp_1
    //   i_cpp += 2
    let mut i_julia = 2usize;
    while i_julia < n {
        let inv_minus_vt_ipp1 = (-vt[i_julia]).julia_inv(); // vT[i_cpp+1] (1-based) -> vt[i_cpp+1 - 1] = vt[i_julia]
        let vt_i_julia = vt[i_julia - 1]; // vT[i_cpp] (1-based) -> vt[i_cpp - 1]
        let read_row = i_julia - 1; // C[i_cpp, j] (1-based) -> c[i_julia-1, j]
        let b_row = i_julia; // B[i_cpp + 1, j] (1-based) -> b[i_julia, j]
        let write_row = i_julia + 1; // C[i_cpp + 2, j] (1-based) -> c[i_julia + 1, j]
        let mut base = 0usize;
        let unrolled = n / 4 * 4;
        for _ in (0..unrolled).step_by(4) {
            let cij = c_data[base + read_row];
            let bipij = b_data[base + b_row];
            c_data[base + write_row] = (bipij - cij * vt_i_julia) * inv_minus_vt_ipp1;
            base += n;

            let cij = c_data[base + read_row];
            let bipij = b_data[base + b_row];
            c_data[base + write_row] = (bipij - cij * vt_i_julia) * inv_minus_vt_ipp1;
            base += n;

            let cij = c_data[base + read_row];
            let bipij = b_data[base + b_row];
            c_data[base + write_row] = (bipij - cij * vt_i_julia) * inv_minus_vt_ipp1;
            base += n;

            let cij = c_data[base + read_row];
            let bipij = b_data[base + b_row];
            c_data[base + write_row] = (bipij - cij * vt_i_julia) * inv_minus_vt_ipp1;
            base += n;
        }
        for _ in unrolled..n {
            let cij = c_data[base + read_row];
            let bipij = b_data[base + b_row];
            c_data[base + write_row] = (bipij - cij * vt_i_julia) * inv_minus_vt_ipp1;
            base += n;
        }
        i_julia += 2;
    }
}

#[allow(clippy::needless_range_loop)]
fn backward_julia_style<T>(vt: &[T], b: &SqMat<'_, T>, c: &mut SqMat<'_, T>)
where
    T: Copy
        + core::ops::Mul<Output = T>
        + core::ops::Add<Output = T>
        + core::ops::Div<Output = T>
        + PfafOne
        + BlasScalar,
{
    let n = b.n();
    if n == 0 {
        return;
    }
    let b_data = b.as_slice();
    let c_data = c.as_mut_slice();

    // Julia: vT_n_1 = inv(vT[n - 1])
    //        C[n - 1, j] = B[n, j] * vT_n_1   for j in 1..n
    let inv_vt_nm1 = vt[n - 2].julia_inv(); // vT[n-1] (1-based) -> vt[n-2]
    let mut base = 0usize;
    let unrolled = n / 4 * 4;
    for _ in (0..unrolled).step_by(4) {
        c_data[base + n - 2] = b_data[base + n - 1] * inv_vt_nm1;
        base += n;
        c_data[base + n - 2] = b_data[base + n - 1] * inv_vt_nm1;
        base += n;
        c_data[base + n - 2] = b_data[base + n - 1] * inv_vt_nm1;
        base += n;
        c_data[base + n - 2] = b_data[base + n - 1] * inv_vt_nm1;
        base += n;
    }
    for _ in unrolled..n {
        c_data[base + n - 2] = b_data[base + n - 1] * inv_vt_nm1;
        base += n;
    }
    // Julia: i_cpp = n - 3; while i_cpp >= 1;
    //   inv_vT_i_cpp   = inv(vT[i_cpp])
    //   vT_i_cpp_1     = vT[i_cpp + 1]
    //   for j in 1:n;
    //     C[i_cpp, j] = (B[i_cpp + 1, j] + C[i_cpp + 2, j] * vT_i_cpp_1) * inv_vT_i_cpp
    //   i_cpp -= 2
    if n < 4 {
        return;
    }
    let mut i_julia: isize = (n as isize) - 3;
    while i_julia >= 1 {
        let i = i_julia as usize;
        let inv_vt_i = vt[i - 1].julia_inv(); // vT[i_cpp] (1-based) -> vt[i_cpp - 1]
        let vt_ip1 = vt[i]; // vT[i_cpp + 1] (1-based) -> vt[i_cpp]
        let b_row = i; // B[i_cpp + 1, j] (1-based) -> b[i_cpp, j]
        let read_row = i + 1; // C[i_cpp + 2, j] (1-based) -> c[i_cpp + 1, j]
        let write_row = i - 1; // C[i_cpp, j] (1-based) -> c[i_cpp - 1, j]
        let mut base = 0usize;
        let unrolled = n / 4 * 4;
        for _ in (0..unrolled).step_by(4) {
            let bip = b_data[base + b_row];
            let cip2 = c_data[base + read_row];
            c_data[base + write_row] = (bip + cip2 * vt_ip1) * inv_vt_i;
            base += n;

            let bip = b_data[base + b_row];
            let cip2 = c_data[base + read_row];
            c_data[base + write_row] = (bip + cip2 * vt_ip1) * inv_vt_i;
            base += n;

            let bip = b_data[base + b_row];
            let cip2 = c_data[base + read_row];
            c_data[base + write_row] = (bip + cip2 * vt_ip1) * inv_vt_i;
            base += n;

            let bip = b_data[base + b_row];
            let cip2 = c_data[base + read_row];
            c_data[base + write_row] = (bip + cip2 * vt_ip1) * inv_vt_i;
            base += n;
        }
        for _ in unrolled..n {
            let bip = b_data[base + b_row];
            let cip2 = c_data[base + read_row];
            c_data[base + write_row] = (bip + cip2 * vt_ip1) * inv_vt_i;
            base += n;
        }
        i_julia -= 2;
    }
}

// Public, cleaner entry point to the skew-tridiagonal solver. The
// `sktdsmx` function above ended up with the failed-derivation
// dead-code that triggers an unreachable!() and panics; we route the
// real callers through `forward_julia_style` + `backward_julia_style`.
pub(crate) fn solve_sktd<T>(vt: &[T], b: &SqMat<'_, T>, c: &mut SqMat<'_, T>)
where
    T: Copy
        + core::ops::Neg<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>
        + core::ops::Div<Output = T>
        + PfafOne
        + BlasScalar,
{
    forward_julia_style(vt, b, c);
    backward_julia_style(vt, b, c);
}

#[inline]
fn should_use_panel_trmmt(n: usize) -> bool {
    const PANEL: usize = 64;
    cfg!(not(feature = "blas-backend")) && PANEL > 1 && PANEL < n
}

fn fill_lower_from_upper_skew<T>(a: &mut [T], n: usize)
where
    T: Copy + core::ops::Neg<Output = T> + PfafOne + BlasScalar,
{
    for j in 0..n {
        a[j * n + j] = T::pfaf_zero();
        for i in (j + 1)..n {
            a[j * n + i] = -a[i * n + j];
        }
    }
}

// ---------------------------------------------------------------------------
// utu2inv (eight-step pipeline)
// ---------------------------------------------------------------------------

/// Compute the inverse of a real skew-symmetric matrix from its
/// upper-triangular LTL form (in-place on `a`).
pub fn utu2inv_real(
    a: &mut SqMat<'_, f64>,
    pivots: &[PivotIndex1Based],
    vt: &mut [f64],
    m_work: &mut SqMat<'_, f64>,
) {
    utu2inv_generic::<f64>(a, pivots, vt, m_work, None);
}

/// Compute the inverse of a complex skew-symmetric matrix from its
/// upper-triangular LTL form (in-place on `a`).
pub fn utu2inv_complex(
    a: &mut SqMat<'_, Complex64>,
    pivots: &[PivotIndex1Based],
    vt: &mut [Complex64],
    m_work: &mut SqMat<'_, Complex64>,
) {
    utu2inv_generic::<Complex64>(a, pivots, vt, m_work, None);
}

/// Inverse arithmetic used by Julia's FSZ runtime: direct tridiagonal
/// divisions and its native BLAS provider, without a PfaPack wrapper call.
/// `divide` specifies the caller's platform complex division arithmetic;
/// the ordinary PfaPack inverse keeps its separate scalar reference path.
pub fn utu2inv_complex_fsz(
    a: &mut SqMat<'_, Complex64>,
    pivots: &[PivotIndex1Based],
    vt: &mut [Complex64],
    m_work: &mut SqMat<'_, Complex64>,
    divide: fn(Complex64, Complex64) -> Complex64,
) {
    utu2inv_generic::<Complex64>(a, pivots, vt, m_work, Some(divide));
}

fn solve_sktd_direct<T: BlasScalar>(
    vt: &[T],
    b: &SqMat<'_, T>,
    c: &mut SqMat<'_, T>,
    divide: fn(T, T) -> T,
) {
    let n = b.n();
    for j in 0..n {
        c.set(1, j, divide(b.get(0, j), -vt[0]));
        for i in (2..n).step_by(2) {
            c.set(
                i + 1,
                j,
                divide(b.get(i, j) - c.get(i - 1, j) * vt[i - 1], -vt[i]),
            );
        }
        c.set(n - 2, j, divide(b.get(n - 1, j), vt[n - 2]));
        for i in (1..n - 2).rev().step_by(2) {
            c.set(
                i - 1,
                j,
                divide(b.get(i, j) + c.get(i + 1, j) * vt[i], vt[i - 1]),
            );
        }
    }
}

fn utu2inv_generic<T>(
    a: &mut SqMat<'_, T>,
    pivots: &[PivotIndex1Based],
    vt: &mut [T],
    m: &mut SqMat<'_, T>,
    fsz: Option<fn(T, T) -> T>,
) where
    T: BlasScalar,
{
    let n = a.n();
    assert_eq!(m.n(), n);
    assert_eq!(vt.len(), n - 1);

    // Step 1: M <- I.
    //
    // Match the authoritative C workspace contract: every entry is reset.
    // Step 3 only copies into columns 1..n-2; the final strict-upper column
    // must also be zero before the skew-tridiagonal solve reads M.
    let m_data = m.as_mut_slice();
    m_data.fill(T::pfaf_zero());
    for i in 0..n {
        m_data[i * n + i] = T::pfaf_one();
    }

    // Step 2: trtri on the unit upper-triangular submatrix A[0..n-1, 1..n].
    // (n-1) x (n-1) submatrix in zero-based; in the upstream this is the
    // sub-block that lives in the strict upper triangle of A together
    // with implicit unit diagonal entries.
    if n > 1 {
        // The submatrix is unit upper triangular if we treat A[i, i+1]
        // (the diagonal of the submatrix) as 1. But in the upstream
        // (PfaPack.jl line 130) the call is
        //   trtri!('U', 'U', @view(A[1:n-1, 2:n]))
        // which is `LAPACK.trtri!('U' upper, 'U' unit diagonal,
        // submatrix)`. So the submatrix is *interpreted* as unit upper
        // triangular (diagonal is forced to 1); we replicate that with
        // our own implementation.
        let lda = a.lda();
        if fsz.is_some() {
            T::fsz_trtri(a.as_mut_slice(), lda, n - 1);
        } else {
            backend::trtri_uu_inplace::<T>(a.as_mut_slice(), lda, 0, 1, n - 1);
        }
    }

    // Step 3: lacpy -- copy upper-triangular block A[0..n-2, 2..n]
    //                  -> M[0..n-2, 1..n-1].
    // Julia: for j_rel in 1:n-2; for i_rel in 1:j_rel; M[i_rel, j_rel+1] = A[i_rel, j_rel+2]
    // 0-based: for jr in 0..n-2; for ir in 0..=jr; m[ir, jr+1] = a[ir, jr+2]
    if n > 2 {
        let a_data = a.as_slice();
        let m_data = m.as_mut_slice();
        for jr in 0..(n - 2) {
            let src_col = (jr + 2) * n;
            let dst_col = (jr + 1) * n;
            m_data[dst_col..=(dst_col + jr)].copy_from_slice(&a_data[src_col..=(src_col + jr)]);
        }
    }

    // Step 4: vT[i] = -A[i, i+1]  (0-based: vt[i] = -a[i, i+1] for i in 0..n-1)
    let a_data = a.as_slice();
    for i in 0..(n - 1) {
        vt[i] = -a_data[(i + 1) * n + i];
    }

    // Step 5: skew-tridiagonal solve (input M, output A).
    // Julia: sktdsmx!(n, vT, M, A)  -- B=M, C=A.
    //
    // We pass `m` by shared reference and `a` by mutable reference;
    // the helper only writes into the output (`a`).
    if let Some(divide) = fsz {
        solve_sktd_direct(vt, m, a, divide);
    } else {
        solve_sktd::<T>(vt, m, a);
    }

    let panel_trmmt = should_use_panel_trmmt(n);
    if panel_trmmt {
        // Mirrors deps/invert.tcc::utu2inv: for large matrices, compute only
        // the upper-triangular part of M^T * A by 64-column panels, then
        // restore skew-symmetry before applying the pivot permutations.
        backend::trmmt_upper_lutu::<T>(m.as_slice(), a.as_mut_slice(), n);
        fill_lower_from_upper_skew::<T>(a.as_mut_slice(), n);
    }

    // Step 6: column permutation by iPiv (forward direction).
    // Julia: for j in 1:n; target = iPiv[j]; if target != j; swap cols.
    let a_data = a.as_mut_slice();
    for (j, pivot) in pivots.iter().enumerate().take(n) {
        let target = (pivot.0 as usize) - 1;
        if target != j {
            let col_j = j * n;
            let col_target = target * n;
            for i in 0..n {
                a_data.swap(col_j + i, col_target + i);
            }
        }
    }

    // Step 7: A <- M^T * A   (trmm, M unit upper-triangular)
    if !panel_trmmt {
        let n_local = a.n();
        if fsz.is_some() {
            T::fsz_trmm(m.as_slice(), a.as_mut_slice(), n_local);
        } else {
            backend::trmm_lutu::<T>(m.as_slice(), a.as_mut_slice(), n_local);
        }
    }

    // Step 8: row permutation by iPiv (forward direction, sequential)
    let a_data = a.as_mut_slice();
    for (i, pivot) in pivots.iter().enumerate().take(n) {
        let target = (pivot.0 as usize) - 1;
        if target != i {
            for j in 0..n {
                let col = j * n;
                a_data.swap(col + i, col + target);
            }
        }
    }
}

// trtri('U', 'U', submatrix) -- invert in place the (sz x sz) unit
// upper-triangular submatrix anchored at A[row0..row0+sz, col0..col0+sz].
//
// Local trtri_uu / trmm_left_upper_unit_transpose helpers were
// replaced by `backend::trtri_uu_inplace` and `backend::trmm_lutu`.
// See `crates/pfapack/src/backend.rs` for the two implementations
// (pure-Rust default, BLAS / LAPACK when `--features blas-backend`).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ltl::{dsktf2, zsktf2};
    use approx::assert_relative_eq;

    fn random_skew_real(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic small LCG to avoid pulling in `rand`.
        let mut s = seed.wrapping_add(0xdeadbeef);
        let mut next = || -> f64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map upper 32 bits to (-1, 1).
            let bits = (s >> 32) as u32;
            (bits as f64 / u32::MAX as f64) * 2.0 - 1.0
        };
        let mut buf = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..j {
                let v = next();
                buf[j * n + i] = v;
                buf[i * n + j] = -v;
            }
        }
        buf
    }

    #[test]
    fn utu2inv_then_multiply_equals_identity_real() {
        for n in [4usize, 6, 8] {
            let orig = random_skew_real(n, 12345 + n as u64);
            let mut a = orig.clone();
            let mut m_buf = vec![0.0; n * n];
            let mut vt = vec![0.0; n - 1];
            let mut piv = vec![PivotIndex1Based(0); n];
            {
                let mut sm = SqMat::new(&mut a, n);
                dsktf2(&mut sm, &mut piv).unwrap();
            }
            // Compute the inverse into `a`.
            {
                let mut sm = SqMat::new(&mut a, n);
                let mut mm = SqMat::new(&mut m_buf, n);
                utu2inv_real(&mut sm, &piv, &mut vt, &mut mm);
            }
            // Verify orig * a == I to within tolerance.
            let mut prod = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0;
                    for k in 0..n {
                        // Column-major access:
                        //   M[i, j] = data[j * n + i]
                        // (orig * a)[i, j] = sum_k orig[i, k] * a[k, j]
                        let orig_ik = orig[k * n + i];
                        let a_kj = a[j * n + k];
                        acc += orig_ik * a_kj;
                    }
                    prod[j * n + i] = acc;
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert_relative_eq!(prod[j * n + i], want, epsilon = 1e-9);
                }
            }
        }
    }

    #[test]
    fn utu2inv_complex_round_trip() {
        let n = 6;
        // Random complex skew matrix via the real builder + i * (independent).
        let re = random_skew_real(n, 1234);
        let im = random_skew_real(n, 5678);
        let orig: Vec<Complex64> = (0..n * n).map(|k| Complex64::new(re[k], im[k])).collect();
        let mut a = orig.clone();
        let mut m_buf = vec![Complex64::new(0.0, 0.0); n * n];
        let mut vt = vec![Complex64::new(0.0, 0.0); n - 1];
        let mut piv = vec![PivotIndex1Based(0); n];
        {
            let mut sm = SqMat::new(&mut a, n);
            zsktf2(&mut sm, &mut piv).unwrap();
        }
        {
            let mut sm = SqMat::new(&mut a, n);
            let mut mm = SqMat::new(&mut m_buf, n);
            utu2inv_complex(&mut sm, &piv, &mut vt, &mut mm);
        }
        // Check orig * a == I.
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    // Column-major: a[k, j] -> a[j * n + k]
                    acc += orig[k * n + i] * a[j * n + k];
                }
                let want = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_relative_eq!(acc.re, want.re, epsilon = 1e-9);
                assert_relative_eq!(acc.im, want.im, epsilon = 1e-9);
            }
        }
    }
}
