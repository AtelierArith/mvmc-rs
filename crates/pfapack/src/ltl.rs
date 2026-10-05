//! LTL decomposition of skew-symmetric matrices (upper-triangular form).
//!
//! Port targets: `extern/Julia-mVMC/PfaPack.jl/src/ltl_decomposition.jl`
//! (`julia_dsktf2!` for Real / `julia_zsktf2!` for Complex), which in
//! turn are line-by-line ports of Wimmer PfaPack 2014-09's Fortran
//! `dsktf2.f` / `zsktf2.f` (`UPLO='U'`, `MODE='N'`).
//!
//! License: BSD-3-Clause.
//!
//! On exit the upper-triangular portion of `a` holds the LTL form. The
//! pivot vector `pivots` (length `n`, **1-based** to mirror the Fortran
//! `IPIV` convention) tells the post-decomposition routines
//! (`utu2pfa` / `utu2inv`) which row/column swaps were applied.
//!
//! Returns `Ok(())` on success or `Err(zero_pivot_row)` mirroring the
//! Fortran `INFO > 0` semantics.

#[cfg(test)]
#[path = "rank2_real_regression.rs"]
mod rank2_real_regression;

use num_complex::Complex64;

use crate::backend::{self, BlasScalar};
use crate::mat::SqMat;
use crate::PivotIndex1Based;

/// LTL decomposition for real skew-symmetric matrices (`dsktf2`).
///
/// Mirrors `julia_dsktf2!(A, iPiv)` from `ltl_decomposition.jl`.
pub fn dsktf2(a: &mut SqMat<'_, f64>, pivots: &mut [PivotIndex1Based]) -> Result<(), usize> {
    sktf2_generic::<f64, _>(a, pivots, |x| x.abs(), false, false)
}

/// Real LTL decomposition with C's `DSKR2` update order.
///
/// The upper rank-2 update evaluates `(A + X*t1) - Y*t2` as in
/// `extern/mVMC-1.3.0/src/pfapack/fortran/dskr2.f`. [`dsktf2`] keeps Julia's
/// `A += (X*t1 - Y*t2)` grouping, which can differ in the last bit.
pub fn dsktf2_c_compat(
    a: &mut SqMat<'_, f64>,
    pivots: &mut [PivotIndex1Based],
) -> Result<(), usize> {
    sktf2_generic::<f64, _>(a, pivots, |x| x.abs(), false, true)
}

/// LTL decomposition for complex skew-symmetric matrices (`zsktf2`).
pub fn zsktf2(a: &mut SqMat<'_, Complex64>, pivots: &mut [PivotIndex1Based]) -> Result<(), usize> {
    // IZAMAX uses |Re| + |Im|, the BLAS 1-norm. We mirror it exactly.
    sktf2_generic::<Complex64, _>(a, pivots, |z| z.re.abs() + z.im.abs(), false, false)
}

/// Complex LTL decomposition matching Julia's StructArray/@turbo production path.
pub fn zsktf2_turbo(
    a: &mut SqMat<'_, Complex64>,
    pivots: &mut [PivotIndex1Based],
) -> Result<(), usize> {
    sktf2_generic::<Complex64, _>(a, pivots, |z| z.re.abs() + z.im.abs(), true, false)
}

/// Complex LTL decomposition with C's standard complex division and ZSKR2
/// column update order.
pub fn zsktf2_c_compat(
    a: &mut SqMat<'_, Complex64>,
    pivots: &mut [PivotIndex1Based],
) -> Result<(), usize> {
    sktf2_generic::<Complex64, _>(a, pivots, |z| z.re.abs() + z.im.abs(), true, true)
}

fn sktf2_generic<T, Mag>(
    a: &mut SqMat<'_, T>,
    pivots: &mut [PivotIndex1Based],
    mag: Mag,
    turbo: bool,
    c_order: bool,
) -> Result<(), usize>
where
    T: BlasScalar + UpperRank2Kernel,
    Mag: Fn(T) -> f64,
{
    let n = a.n();
    assert_eq!(pivots.len(), n, "pivots length must equal matrix side");

    let mut info: Option<usize> = None;

    // Julia: for i in 1:n; iPiv[i] = i
    for (i, pivot) in pivots.iter_mut().enumerate().take(n) {
        *pivot = PivotIndex1Based((i as u32) + 1);
    }

    if n < 2 {
        return Ok(());
    }

    // Iterate K from N down to 2 (Fortran: DO K=N, 2, -1). The Julia
    // version uses 1-based `k`; in 0-based Rust the loop runs over
    // `k0 = n-1 ..= 1` so `kk0 = k0 - 1` is the row/col to swap with kp.
    let mut k0 = n; // sentinel; we decrement at the top of the loop
    while k0 > 1 {
        k0 -= 1;
        let kk0 = k0 - 1; // Julia's kk = k - 1, 0-based

        // Pivot search: argmax_{j in 0..kk0+1} |A[j, k0]| via the
        // appropriate 1-norm. Julia uses IDAMAX / IZAMAX which return
        // the FIRST index achieving the max; we mirror that with a
        // strict `>` comparison.
        let mut kp = 0usize;
        let mut colmax = mag(a.get(0, k0));
        for j in 1..=kk0 {
            let v = mag(a.get(j, k0));
            if v > colmax {
                colmax = v;
                kp = j;
            }
        }

        if colmax == 0.0 {
            // Column is zero -- record INFO if first time, set pivot to
            // kk0 (no swap), continue the outer loop.
            if info.is_none() {
                // Fortran INFO is 1-based row index; Julia stores k-1
                // which is the 1-based kk. Translate to 1-based here.
                info = Some(kk0 + 1);
            }
            pivots[kk0] = PivotIndex1Based((kk0 as u32) + 1);
            continue;
        }

        // Swap rows/columns kk0 and kp if needed.
        if kp != kk0 {
            // Julia: for j in 1:(kp-1); A[j, kk], A[j, kp] = A[j, kp], A[j, kk]
            for j in 0..kp {
                let t = a.get(j, kk0);
                a.set(j, kk0, a.get(j, kp));
                a.set(j, kp, t);
            }
            // Julia: for j in (kp+1):(kk-1); A[j, kk], A[kp, j] = A[kp, j], A[j, kk]
            for j in (kp + 1)..kk0 {
                let t = a.get(j, kk0);
                a.set(j, kk0, a.get(kp, j));
                a.set(kp, j, t);
            }
            // Julia: for j in k:n; A[kk, j], A[kp, j] = A[kp, j], A[kk, j]
            for j in k0..n {
                let t = a.get(kk0, j);
                a.set(kk0, j, a.get(kp, j));
                a.set(kp, j, t);
            }
            // Julia: for j in kp:(kk-1); A[j, kk] = -A[j, kk]
            for j in kp..kk0 {
                let v = -a.get(j, kk0);
                a.set(j, kk0, v);
            }
            // Julia: for j in (kp+1):(kk-1); A[kp, j] = -A[kp, j]
            for j in (kp + 1)..kk0 {
                let v = -a.get(kp, j);
                a.set(kp, j, v);
            }
        }

        pivots[kk0] = PivotIndex1Based((kp as u32) + 1);

        // Skew-symmetric rank-2 update of A[0..kk0, 0..kk0].
        // Julia: if k >= 3 ... (kk = k-1 >= 2, i.e. kk0 >= 1 in 0-based).
        if kk0 >= 1 {
            let pivot = a.get(kk0, k0);
            // Note: pivot is `A[kk, k]` in Julia (a single off-diagonal entry).
            let alpha = T::ltl_alpha(pivot, turbo);

            // Skew-symmetric rank-2 update of the **upper-triangular** part
            // of A[0..kk0, 0..kk0], following the operation order of Julia
            // `julia_dsktf2!` (ltl_decomposition.jl:124-150). Julia
            // deliberately uses a hand-rolled upper-triangle-only DSKR2
            // here ("Optimized skew-symmetric rank-2 update") instead of
            // `BLAS.ger!`, because the LTL output contract requires the
            // strict lower triangle to be preserved as the original
            // matrix entries — the golden fixtures under
            // `tests/fixtures/dump_pfapack_reference/` compare the full
            // n×n buffer.
            //
            // We mirror that here: a single `dger` would touch the lower
            // triangle, which `utu2pfa` doesn't care about but the golden
            // diff does, so the scalar form preserves this storage contract
            // backend even when `--features blas-backend` is on.
            let lda = a.lda();
            T::update_rank2_mode(a.as_mut_slice(), lda, kk0, k0, alpha, turbo, c_order);

            // Julia: BLAS.scal!(k-2, alpha, A[1:k-2, k], 1) — backend
            // route: dscal / zscal when BLAS is on, plain loop otherwise.
            let n_sub = kk0;
            let data = a.as_mut_slice();
            let col_k0 = k0 * lda;
            T::scale_column_mode(&mut data[col_k0..col_k0 + n_sub], n_sub, alpha, turbo);
        }
    }

    match info {
        None => Ok(()),
        Some(i) => Err(i),
    }
}

#[inline]
fn update_upper_rank2<T: UpperRank2Kernel>(
    data: &mut [T],
    lda: usize,
    kk0: usize,
    k0: usize,
    alpha: T,
) {
    T::update_upper_rank2(data, lda, kk0, k0, alpha);
}

trait UpperRank2Kernel: BlasScalar {
    fn ltl_alpha(pivot: Self, _turbo: bool) -> Self {
        Self::pfaf_one().julia_div(pivot)
    }
    fn update_rank2_mode(
        data: &mut [Self],
        lda: usize,
        kk0: usize,
        k0: usize,
        alpha: Self,
        _turbo: bool,
        _c_order: bool,
    ) {
        update_upper_rank2(data, lda, kk0, k0, alpha);
    }
    fn scale_column_mode(data: &mut [Self], n: usize, alpha: Self, _turbo: bool) {
        backend::scal_strided::<Self>(data, n, alpha);
    }
    fn update_upper_rank2(data: &mut [Self], lda: usize, kk0: usize, k0: usize, alpha: Self);
}

impl UpperRank2Kernel for f64 {
    fn update_rank2_mode(
        data: &mut [Self],
        lda: usize,
        kk0: usize,
        k0: usize,
        alpha: Self,
        _turbo: bool,
        c_order: bool,
    ) {
        if c_order {
            update_upper_rank2_f64_c_order(data, lda, kk0, k0, alpha);
        } else {
            update_upper_rank2_f64(data, lda, kk0, k0, alpha);
        }
    }
    #[inline]
    fn update_upper_rank2(data: &mut [Self], lda: usize, kk0: usize, k0: usize, alpha: Self) {
        update_upper_rank2_f64(data, lda, kk0, k0, alpha);
    }
}

impl UpperRank2Kernel for Complex64 {
    fn ltl_alpha(pivot: Self, turbo: bool) -> Self {
        if turbo {
            let denom = pivot.re * pivot.re + pivot.im * pivot.im;
            Self::new(pivot.re / denom, -pivot.im / denom)
        } else {
            Self::new(1.0, 0.0).julia_div(pivot)
        }
    }
    fn update_rank2_mode(
        data: &mut [Self],
        lda: usize,
        kk0: usize,
        k0: usize,
        alpha: Self,
        turbo: bool,
        c_order: bool,
    ) {
        if c_order {
            update_upper_rank2_c64_c_order(data, lda, kk0, k0, alpha);
            return;
        }
        if !turbo {
            Self::update_upper_rank2(data, lda, kk0, k0, alpha);
            return;
        }
        for j in 0..kk0 {
            let t1 = alpha * data[j + kk0 * lda];
            let t2 = alpha * data[j + k0 * lda];
            for i in 0..j {
                let x = data[i + k0 * lda];
                let y = data[i + kk0 * lda];
                let old = data[i + j * lda];
                // LoopVectorization fuses from the right-hand products through
                // the old entry; product-wise complex FMAs use a different tree.
                let re = y.im.mul_add(t2.im, old.re);
                let re = y.re.mul_add(t2.re, -re);
                let re = x.im.mul_add(t1.im, re);
                let re = x.re.mul_add(t1.re, -re);
                let im = y.im.mul_add(t2.re, -old.im);
                let im = y.re.mul_add(t2.im, im);
                let im = (-x.im).mul_add(t1.re, im);
                let im = x.re.mul_add(t1.im, -im);
                data[i + j * lda] = Self::new(re, im);
            }
            data[j + j * lda] = Self::new(0.0, 0.0);
        }
    }
    fn scale_column_mode(data: &mut [Self], n: usize, alpha: Self, turbo: bool) {
        if !turbo {
            backend::scal_strided::<Self>(data, n, alpha);
            return;
        }
        for value in &mut data[..n] {
            let v = *value;
            *value = Self::new(
                alpha.re.mul_add(v.re, -alpha.im * v.im),
                alpha.re.mul_add(v.im, alpha.im * v.re),
            );
        }
    }
    #[inline]
    fn update_upper_rank2(data: &mut [Self], lda: usize, kk0: usize, k0: usize, alpha: Self) {
        #[cfg(feature = "simd-backend")]
        {
            update_upper_rank2_c64_simd(data, lda, kk0, k0, alpha);
        }
        #[cfg(not(feature = "simd-backend"))]
        {
            update_upper_rank2_c64_scalar(data, lda, kk0, k0, alpha);
        }
    }
}

#[cfg(feature = "simd-backend")]
mod simd_rank2 {
    use num_complex::Complex64;
    use pulp::{Arch, Simd, WithSimd};
    use std::sync::OnceLock;

    static ARCH: OnceLock<Arch> = OnceLock::new();

    pub(super) fn update_upper_rank2_c64(
        write_cols: &mut [Complex64],
        col_kk0_data: &[Complex64],
        col_k0_data: &[Complex64],
        lda: usize,
        kk0: usize,
        alpha: Complex64,
    ) {
        (*ARCH.get_or_init(Arch::new)).dispatch(UpdateUpperRank2C64 {
            write_cols,
            col_kk0_data,
            col_k0_data,
            lda,
            kk0,
            alpha,
        });
    }

    struct UpdateUpperRank2C64<'a> {
        write_cols: &'a mut [Complex64],
        col_kk0_data: &'a [Complex64],
        col_k0_data: &'a [Complex64],
        lda: usize,
        kk0: usize,
        alpha: Complex64,
    }

    impl WithSimd for UpdateUpperRank2C64<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let Self {
                write_cols,
                col_kk0_data,
                col_k0_data,
                lda,
                kk0,
                alpha,
            } = self;

            let mut cols = write_cols.chunks_exact_mut(lda);
            let mut j = 0usize;
            while j + 1 < kk0 {
                let col_j0 = cols.next().expect("column j must exist");
                let col_j1 = cols.next().expect("column j+1 must exist");

                let temp1_j0 = alpha * col_kk0_data[j];
                let temp2_j0 = alpha * col_k0_data[j];
                let temp1_j1 = alpha * col_kk0_data[j + 1];
                let temp2_j1 = alpha * col_k0_data[j + 1];

                update_col2(
                    simd,
                    &mut col_j0[..j],
                    &mut col_j1[..j],
                    &col_k0_data[..j],
                    &col_kk0_data[..j],
                    temp1_j0,
                    temp2_j0,
                    temp1_j1,
                    temp2_j1,
                );

                col_j0[j] = Complex64::new(0.0, 0.0);
                col_j1[j] += col_k0_data[j] * temp1_j1 - col_kk0_data[j] * temp2_j1;
                col_j1[j + 1] = Complex64::new(0.0, 0.0);
                j += 2;
            }

            if j < kk0 {
                let col_j = cols.next().expect("last column must exist");
                let temp1 = alpha * col_kk0_data[j];
                let temp2 = alpha * col_k0_data[j];
                update_col(
                    simd,
                    &mut col_j[..j],
                    &col_k0_data[..j],
                    &col_kk0_data[..j],
                    temp1,
                    temp2,
                );
                col_j[j] = Complex64::new(0.0, 0.0);
            }
        }
    }

    #[inline(always)]
    fn update_col<S: Simd>(
        simd: S,
        dst: &mut [Complex64],
        x: &[Complex64],
        y: &[Complex64],
        temp1: Complex64,
        temp2: Complex64,
    ) {
        let temp1_scalar = temp1;
        let temp2_scalar = temp2;
        let temp1 = simd.splat_c64s(temp1_scalar);
        let neg_temp2 = simd.splat_c64s(-temp2_scalar);

        let (dst_head, dst_tail) = S::as_mut_simd_c64s(dst);
        let (x_head, x_tail) = S::as_simd_c64s(x);
        let (y_head, y_tail) = S::as_simd_c64s(y);

        for ((dst, x), y) in dst_head.iter_mut().zip(x_head).zip(y_head) {
            let acc = simd.mul_add_c64s(*x, temp1, *dst);
            *dst = simd.mul_add_c64s(*y, neg_temp2, acc);
        }

        for ((dst, x), y) in dst_tail.iter_mut().zip(x_tail).zip(y_tail) {
            *dst += *x * temp1_scalar - *y * temp2_scalar;
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn update_col2<S: Simd>(
        simd: S,
        dst0: &mut [Complex64],
        dst1: &mut [Complex64],
        x: &[Complex64],
        y: &[Complex64],
        temp1_0: Complex64,
        temp2_0: Complex64,
        temp1_1: Complex64,
        temp2_1: Complex64,
    ) {
        let temp1_0_scalar = temp1_0;
        let temp2_0_scalar = temp2_0;
        let temp1_1_scalar = temp1_1;
        let temp2_1_scalar = temp2_1;
        let temp1_0 = simd.splat_c64s(temp1_0_scalar);
        let neg_temp2_0 = simd.splat_c64s(-temp2_0_scalar);
        let temp1_1 = simd.splat_c64s(temp1_1_scalar);
        let neg_temp2_1 = simd.splat_c64s(-temp2_1_scalar);

        let (dst0_head, dst0_tail) = S::as_mut_simd_c64s(dst0);
        let (dst1_head, dst1_tail) = S::as_mut_simd_c64s(dst1);
        let (x_head, x_tail) = S::as_simd_c64s(x);
        let (y_head, y_tail) = S::as_simd_c64s(y);

        for (((dst0, dst1), x), y) in dst0_head
            .iter_mut()
            .zip(dst1_head.iter_mut())
            .zip(x_head)
            .zip(y_head)
        {
            let acc0 = simd.mul_add_c64s(*x, temp1_0, *dst0);
            *dst0 = simd.mul_add_c64s(*y, neg_temp2_0, acc0);

            let acc1 = simd.mul_add_c64s(*x, temp1_1, *dst1);
            *dst1 = simd.mul_add_c64s(*y, neg_temp2_1, acc1);
        }

        for (((dst0, dst1), x), y) in dst0_tail
            .iter_mut()
            .zip(dst1_tail.iter_mut())
            .zip(x_tail)
            .zip(y_tail)
        {
            *dst0 += *x * temp1_0_scalar - *y * temp2_0_scalar;
            *dst1 += *x * temp1_1_scalar - *y * temp2_1_scalar;
        }
    }
}

#[cfg(feature = "simd-backend")]
#[inline]
fn update_upper_rank2_c64_simd(
    data: &mut [Complex64],
    lda: usize,
    kk0: usize,
    k0: usize,
    alpha: Complex64,
) {
    // The current AoS Complex64 pulp kernel loses badly at the matrix sizes
    // exercised by mVMC and the comparison benchmark. Julia's fast path uses
    // a StructArray SoA layout before vectorizing; until Rust has the same
    // layout-specialized kernel, keep the scalar hand-expanded complex loop
    // for these sizes so enabling `simd-backend` is not a performance trap.
    if kk0 < 512 {
        update_upper_rank2_c64_scalar(data, lda, kk0, k0, alpha);
        return;
    }

    let (write_cols, col_kk0_data, col_k0_data) = split_update_upper_rank2_cols(data, lda, kk0, k0);
    simd_rank2::update_upper_rank2_c64(write_cols, col_kk0_data, col_k0_data, lda, kk0, alpha);
}

fn split_update_upper_rank2_cols<T>(
    data: &mut [T],
    lda: usize,
    kk0: usize,
    k0: usize,
) -> (&mut [T], &mut [T], &mut [T]) {
    let col_kk0 = kk0 * lda;
    let col_k0 = k0 * lda;

    let (write_cols, tail) = data.split_at_mut(col_kk0);
    let (col_kk0_data, tail) = tail.split_at_mut(lda);
    let skip_to_k0 = col_k0 - col_kk0 - lda;
    let (_, tail) = tail.split_at_mut(skip_to_k0);
    let (col_k0_data, _) = tail.split_at_mut(lda);
    (write_cols, col_kk0_data, col_k0_data)
}

#[inline]
fn check_update_upper_rank2_args<T>(data: &[T], lda: usize, kk0: usize, k0: usize) {
    let col_k0 = k0 * lda;

    debug_assert!(kk0 < k0);
    debug_assert!(k0 < lda);
    debug_assert!(col_k0 + lda <= data.len());
}

#[inline]
fn update_upper_rank2_f64(data: &mut [f64], lda: usize, kk0: usize, k0: usize, alpha: f64) {
    check_update_upper_rank2_args(data, lda, kk0, k0);
    let (write_cols, col_kk0_data, col_k0_data) = split_update_upper_rank2_cols(data, lda, kk0, k0);

    for (j, col_j) in write_cols.chunks_exact_mut(lda).take(kk0).enumerate() {
        let temp1 = alpha * col_kk0_data[j];
        let temp2 = alpha * col_k0_data[j];

        for i in 0..j {
            col_j[i] += col_k0_data[i] * temp1 - col_kk0_data[i] * temp2;
        }
        col_j[j] = 0.0;
    }
}

/// C `DSKR2` column update: first addition, then subtraction.
#[inline]
fn update_upper_rank2_f64_c_order(data: &mut [f64], lda: usize, kk0: usize, k0: usize, alpha: f64) {
    check_update_upper_rank2_args(data, lda, kk0, k0);
    let (write_cols, col_kk0_data, col_k0_data) = split_update_upper_rank2_cols(data, lda, kk0, k0);

    for (j, col_j) in write_cols.chunks_exact_mut(lda).take(kk0).enumerate() {
        let temp1 = alpha * col_kk0_data[j];
        let temp2 = alpha * col_k0_data[j];

        for i in 0..j {
            col_j[i] = (col_j[i] + col_k0_data[i] * temp1) - col_kk0_data[i] * temp2;
        }
        col_j[j] = 0.0;
    }
}

#[inline]
fn update_upper_rank2_c64_scalar(
    data: &mut [Complex64],
    lda: usize,
    kk0: usize,
    k0: usize,
    alpha: Complex64,
) {
    check_update_upper_rank2_args(data, lda, kk0, k0);
    let ar = alpha.re;
    let ai = alpha.im;

    let (write_cols, col_kk0_data, col_k0_data) = split_update_upper_rank2_cols(data, lda, kk0, k0);

    let mut cols = write_cols.chunks_exact_mut(lda);
    let mut j = 0usize;
    while j + 1 < kk0 {
        let col_j0 = cols.next().expect("column j must exist");
        let col_j1 = cols.next().expect("column j+1 must exist");

        let kk_j0 = col_kk0_data[j];
        let k_j0 = col_k0_data[j];
        let temp1_0_re = ar * kk_j0.re - ai * kk_j0.im;
        let temp1_0_im = ar * kk_j0.im + ai * kk_j0.re;
        let temp2_0_re = ar * k_j0.re - ai * k_j0.im;
        let temp2_0_im = ar * k_j0.im + ai * k_j0.re;

        let kk_j1 = col_kk0_data[j + 1];
        let k_j1 = col_k0_data[j + 1];
        let temp1_1_re = ar * kk_j1.re - ai * kk_j1.im;
        let temp1_1_im = ar * kk_j1.im + ai * kk_j1.re;
        let temp2_1_re = ar * k_j1.re - ai * k_j1.im;
        let temp2_1_im = ar * k_j1.im + ai * k_j1.re;

        for i in 0..j {
            let x = col_k0_data[i];
            let y = col_kk0_data[i];

            let x_temp1_re = x.re * temp1_0_re - x.im * temp1_0_im;
            let x_temp1_im = x.re * temp1_0_im + x.im * temp1_0_re;
            let y_temp2_re = y.re * temp2_0_re - y.im * temp2_0_im;
            let y_temp2_im = y.re * temp2_0_im + y.im * temp2_0_re;
            col_j0[i].re += x_temp1_re - y_temp2_re;
            col_j0[i].im += x_temp1_im - y_temp2_im;

            let x_temp1_re = x.re * temp1_1_re - x.im * temp1_1_im;
            let x_temp1_im = x.re * temp1_1_im + x.im * temp1_1_re;
            let y_temp2_re = y.re * temp2_1_re - y.im * temp2_1_im;
            let y_temp2_im = y.re * temp2_1_im + y.im * temp2_1_re;
            col_j1[i].re += x_temp1_re - y_temp2_re;
            col_j1[i].im += x_temp1_im - y_temp2_im;
        }

        col_j0[j] = Complex64::new(0.0, 0.0);

        let x = col_k0_data[j];
        let y = col_kk0_data[j];
        let x_temp1_re = x.re * temp1_1_re - x.im * temp1_1_im;
        let x_temp1_im = x.re * temp1_1_im + x.im * temp1_1_re;
        let y_temp2_re = y.re * temp2_1_re - y.im * temp2_1_im;
        let y_temp2_im = y.re * temp2_1_im + y.im * temp2_1_re;
        col_j1[j].re += x_temp1_re - y_temp2_re;
        col_j1[j].im += x_temp1_im - y_temp2_im;

        col_j1[j + 1] = Complex64::new(0.0, 0.0);
        j += 2;
    }

    if j < kk0 {
        let col_j = cols.next().expect("last column must exist");
        let kk_j = col_kk0_data[j];
        let k_j = col_k0_data[j];
        let temp1_re = ar * kk_j.re - ai * kk_j.im;
        let temp1_im = ar * kk_j.im + ai * kk_j.re;
        let temp2_re = ar * k_j.re - ai * k_j.im;
        let temp2_im = ar * k_j.im + ai * k_j.re;

        for i in 0..j {
            let x = col_k0_data[i];
            let y = col_kk0_data[i];

            let x_temp1_re = x.re * temp1_re - x.im * temp1_im;
            let x_temp1_im = x.re * temp1_im + x.im * temp1_re;
            let y_temp2_re = y.re * temp2_re - y.im * temp2_im;
            let y_temp2_im = y.re * temp2_im + y.im * temp2_re;

            col_j[i].re += x_temp1_re - y_temp2_re;
            col_j[i].im += x_temp1_im - y_temp2_im;
        }
        col_j[j] = Complex64::new(0.0, 0.0);
    }
}

#[inline]
fn update_upper_rank2_c64_c_order(
    data: &mut [Complex64],
    lda: usize,
    kk0: usize,
    k0: usize,
    alpha: Complex64,
) {
    check_update_upper_rank2_args(data, lda, kk0, k0);
    let col_k = k0 * lda;
    let col_kk = kk0 * lda;
    for j in 0..kk0 {
        let temp1 = alpha * data[j + col_kk];
        let temp2 = alpha * data[j + col_k];
        for i in 0..j {
            let idx = i + j * lda;
            data[idx] += data[i + col_k] * temp1 - data[i + col_kk] * temp2;
        }
        data[j + j * lda] = Complex64::new(0.0, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pfaffian::pfaffian_ltl_real;
    use crate::utu2::utu2pfa_real;
    use approx::assert_relative_eq;

    // After dsktf2, utu2pfa should reproduce the same Pfaffian as the
    // direct Parlett-Reid pass on a fresh copy. This is the
    // "end-to-end" property test the upstream package documents.
    #[test]
    fn dsktf2_then_utu2pfa_matches_parlett_reid_real() {
        let n = 6;
        let raw = [
            [0.0, 1.5, -0.7, 2.1, 0.4, -0.2],
            [-1.5, 0.0, 0.3, -0.4, 1.2, 0.9],
            [0.7, -0.3, 0.0, 1.1, -0.5, 0.1],
            [-2.1, 0.4, -1.1, 0.0, 0.8, -0.6],
            [-0.4, -1.2, 0.5, -0.8, 0.0, 0.7],
            [0.2, -0.9, -0.1, 0.6, -0.7, 0.0],
        ];
        let mut a_copy_for_pr = vec![0.0; n * n];
        let mut a_copy_for_ltl = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                a_copy_for_pr[j * n + i] = raw[i][j];
                a_copy_for_ltl[j * n + i] = raw[i][j];
            }
        }
        let pf_pr = {
            let mut m = SqMat::new(&mut a_copy_for_pr, n);
            pfaffian_ltl_real(&mut m)
        };
        let pf_utu2 = {
            let mut m = SqMat::new(&mut a_copy_for_ltl, n);
            let mut piv = vec![PivotIndex1Based(0); n];
            dsktf2(&mut m, &mut piv).expect("dsktf2 should succeed on non-singular skew matrix");
            utu2pfa_real(&m, &piv)
        };
        assert_relative_eq!(pf_pr, pf_utu2, max_relative = 1e-12);
    }

    #[test]
    fn zero_matrix_returns_info() {
        // A 4x4 all-zero matrix: every pivot column is zero, so `info`
        // should fire on the very first iteration (k = n = 4, kk = 3,
        // 1-based info = 3).
        let n = 4;
        let mut buf = vec![0.0; n * n];
        let mut m = SqMat::new(&mut buf, n);
        let mut piv = vec![PivotIndex1Based(0); n];
        let err = dsktf2(&mut m, &mut piv).unwrap_err();
        // Fortran INFO is the 1-based first zero-pivot row, which is
        // n-1 here (k = n, kk = n-1, info = kk = 3).
        assert_eq!(err, n - 1);
    }
}
