//! Issue #641 kernel microbenchmark: scalar / production hand AVX2 / pulp variants of the
//! hot kernels that the L128 Hubbard Opt and PhysCal profiles execute on `main` b1c0870d.
//!
//! Kernels (production sources copied verbatim, only the slice offset removed):
//! - K1 `update_one_real` (UpdateMAll one-hop Sherman-Morrison),
//!   `crates/mvmc-core/src/sampling/updates.rs`;
//! - K2 `update_upper_rank2_f64_c_order` (dsktf2 skew rank-2 update), the full sequence of
//!   calls one `dsktf2` makes on an n x n matrix, `crates/pfapack/src/ltl.rs`;
//! - K3 `two_hop_bilinear_c_order` (CalculateNewPfMTwo `b^T M a`), `updates.rs`;
//! - K4 `solve_sktd_c_real_f64` (utu2inv skew-tridiagonal solve), `crates/pfapack/src/utu2.rs`.
//!
//! Variants: `scalar` (production scalar reference path), `avx2` (production hand AVX2),
//! `pulp-scalar` (pulp generic code on `pulp::Scalar`, which is what `Arch::new()` returns
//! under the workspace's pulp features), `pulp-v3` (pulp generic, separate mul/add in the
//! production order; AVX2 lanes) and `pulp-v3-fma` (same with fused multiply-add).
//!
//! usage: pulp_lv_641 [n=128] [rounds=15]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use pulp::{Simd, WithSimd};
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.unif()).collect()
    }
}

#[inline(always)]
unsafe fn ld<S: Simd>(p: *const f64) -> S::f64s {
    (p as *const S::f64s).read_unaligned()
}
#[inline(always)]
unsafe fn st<S: Simd>(p: *mut f64, v: S::f64s) {
    (p as *mut S::f64s).write_unaligned(v)
}

fn bits_differ(a: &[f64], b: &[f64]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

// ---------------------------------------------------------------------------------------
// K1 UpdateMAll one-hop (production copies)
// ---------------------------------------------------------------------------------------

fn k1_scalar(
    msa: usize,
    inv: &mut [f64],
    slt_vec: &[f64],
    vec1: &mut [f64],
    vec2: &mut [f64],
    n_size: usize,
    pf: &mut f64,
) {
    let inv = &mut inv[..n_size * n_size];
    let (vec1, vec2) = (&mut vec1[..n_size], &mut vec2[..n_size]);
    vec1.fill(0.0);
    for (column, &slt) in inv.chunks_exact(n_size).zip(slt_vec) {
        for (v, &x) in vec1.iter_mut().zip(column) {
            *v += -x * slt;
        }
    }
    let tmp = vec1[msa];
    *pf *= -tmp;
    let inv_vec1_a = -1.0 / tmp;
    for (v2, &x) in vec2.iter_mut().zip(&inv[msa * n_size..(msa + 1) * n_size]) {
        *v2 = x * inv_vec1_a;
    }
    for (msi, row) in inv.chunks_exact_mut(n_size).enumerate() {
        let vec1_i = vec1[msi];
        let vec2_i = vec2[msi];
        for ((x, &vec1_j), &vec2_j) in row.iter_mut().zip(vec1.iter()).zip(vec2.iter()) {
            *x += vec1_i * vec2_j - vec1_j * vec2_i;
        }
        row[msa] -= vec2_i;
    }
    for (x, &v2) in inv[msa * n_size..(msa + 1) * n_size]
        .iter_mut()
        .zip(vec2.iter())
    {
        *x += v2;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn k1_avx2(
    msa: usize,
    inv: &mut [f64],
    slt_vec: &[f64],
    vec1: &mut [f64],
    vec2: &mut [f64],
    n: usize,
    pf: &mut f64,
) {
    use std::arch::x86_64::*;
    assert!(inv.len() >= n * n && slt_vec.len() >= n && vec1.len() >= n);
    let column0 = inv.as_ptr();
    let sign = _mm256_set1_pd(-0.0);
    let mut i = 0;
    while i + 16 <= n {
        let (mut a0, mut a1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        let (mut a2, mut a3) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        for (j, &slt) in slt_vec[..n].iter().enumerate() {
            let s = _mm256_set1_pd(slt);
            let c = column0.add(j * n + i);
            let x0 = _mm256_xor_pd(_mm256_loadu_pd(c), sign);
            let x1 = _mm256_xor_pd(_mm256_loadu_pd(c.add(4)), sign);
            let x2 = _mm256_xor_pd(_mm256_loadu_pd(c.add(8)), sign);
            let x3 = _mm256_xor_pd(_mm256_loadu_pd(c.add(12)), sign);
            a0 = _mm256_add_pd(a0, _mm256_mul_pd(x0, s));
            a1 = _mm256_add_pd(a1, _mm256_mul_pd(x1, s));
            a2 = _mm256_add_pd(a2, _mm256_mul_pd(x2, s));
            a3 = _mm256_add_pd(a3, _mm256_mul_pd(x3, s));
        }
        let out = vec1.as_mut_ptr().add(i);
        _mm256_storeu_pd(out, a0);
        _mm256_storeu_pd(out.add(4), a1);
        _mm256_storeu_pd(out.add(8), a2);
        _mm256_storeu_pd(out.add(12), a3);
        i += 16;
    }
    while i + 4 <= n {
        let mut a = _mm256_setzero_pd();
        for (j, &slt) in slt_vec[..n].iter().enumerate() {
            let x = _mm256_xor_pd(_mm256_loadu_pd(column0.add(j * n + i)), sign);
            a = _mm256_add_pd(a, _mm256_mul_pd(x, _mm256_set1_pd(slt)));
        }
        _mm256_storeu_pd(vec1.as_mut_ptr().add(i), a);
        i += 4;
    }
    for k in i..n {
        let mut acc = 0.0;
        for (column, &slt) in inv.chunks_exact(n).zip(&slt_vec[..n]) {
            acc += -column[k] * slt;
        }
        vec1[k] = acc;
    }
    let tmp = vec1[msa];
    *pf *= -tmp;
    let inv_vec1_a = -1.0 / tmp;
    for (v2, &x) in vec2.iter_mut().zip(&inv[msa * n..(msa + 1) * n]) {
        *v2 = x * inv_vec1_a;
    }
    for (msi, row) in inv.chunks_exact_mut(n).enumerate() {
        let v1i = _mm256_set1_pd(vec1[msi]);
        let v2i = _mm256_set1_pd(vec2[msi]);
        let mut j = 0;
        while j + 4 <= n {
            let x = _mm256_loadu_pd(row.as_ptr().add(j));
            let v1 = _mm256_loadu_pd(vec1.as_ptr().add(j));
            let v2 = _mm256_loadu_pd(vec2.as_ptr().add(j));
            let delta = _mm256_sub_pd(_mm256_mul_pd(v1i, v2), _mm256_mul_pd(v1, v2i));
            _mm256_storeu_pd(row.as_mut_ptr().add(j), _mm256_add_pd(x, delta));
            j += 4;
        }
        for k in j..n {
            row[k] += vec1[msi] * vec2[k] - vec1[k] * vec2[msi];
        }
        row[msa] -= vec2[msi];
    }
    for (x, &v2) in inv[msa * n..(msa + 1) * n].iter_mut().zip(vec2.iter()) {
        *x += v2;
    }
}

/// pulp port of `k1_avx2`: same register blocking (4 vectors of vec1 held across all
/// columns), lane width from the selected ISA. `FMA=false` keeps every separately rounded
/// multiply/add of the C order; `FMA=true` fuses them (what LoopVectorization emits).
struct K1Pulp<'a, const FMA: bool> {
    msa: usize,
    inv: &'a mut [f64],
    slt: &'a [f64],
    vec1: &'a mut [f64],
    vec2: &'a mut [f64],
    n: usize,
    pf: &'a mut f64,
}

impl<const FMA: bool> WithSimd for K1Pulp<'_, FMA> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            msa,
            inv,
            slt,
            vec1,
            vec2,
            n,
            pf,
        } = self;
        assert!(inv.len() >= n * n && slt.len() >= n && vec1.len() >= n && vec2.len() >= n);
        let l = S::F64_LANES;
        let p = inv.as_mut_ptr();
        let zero = simd.splat_f64s(0.0);
        #[inline(always)]
        fn acc<S: Simd, const FMA: bool>(simd: S, a: S::f64s, x: S::f64s, s: S::f64s) -> S::f64s {
            if FMA {
                simd.mul_add_e_f64s(x, s, a)
            } else {
                simd.add_f64s(a, simd.mul_f64s(x, s))
            }
        }
        unsafe {
            let mut i = 0;
            while i + 4 * l <= n {
                let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
                for j in 0..n {
                    let s = simd.splat_f64s(*slt.get_unchecked(j));
                    let c = p.add(j * n + i);
                    a0 = acc::<S, FMA>(simd, a0, simd.neg_f64s(ld::<S>(c)), s);
                    a1 = acc::<S, FMA>(simd, a1, simd.neg_f64s(ld::<S>(c.add(l))), s);
                    a2 = acc::<S, FMA>(simd, a2, simd.neg_f64s(ld::<S>(c.add(2 * l))), s);
                    a3 = acc::<S, FMA>(simd, a3, simd.neg_f64s(ld::<S>(c.add(3 * l))), s);
                }
                let o = vec1.as_mut_ptr().add(i);
                st::<S>(o, a0);
                st::<S>(o.add(l), a1);
                st::<S>(o.add(2 * l), a2);
                st::<S>(o.add(3 * l), a3);
                i += 4 * l;
            }
            while i + l <= n {
                let mut a = zero;
                for j in 0..n {
                    let s = simd.splat_f64s(*slt.get_unchecked(j));
                    a = acc::<S, FMA>(simd, a, simd.neg_f64s(ld::<S>(p.add(j * n + i))), s);
                }
                st::<S>(vec1.as_mut_ptr().add(i), a);
                i += l;
            }
            for k in i..n {
                let mut a = 0.0;
                for j in 0..n {
                    let x = -*p.add(j * n + k);
                    a = if FMA {
                        x.mul_add(slt[j], a)
                    } else {
                        a + x * slt[j]
                    };
                }
                vec1[k] = a;
            }
        }
        let tmp = vec1[msa];
        *pf *= -tmp;
        let inv_vec1_a = -1.0 / tmp;
        for k in 0..n {
            vec2[k] = inv[msa * n + k] * inv_vec1_a;
        }
        unsafe {
            let (v1p, v2p) = (vec1.as_ptr(), vec2.as_ptr());
            for msi in 0..n {
                let row = p.add(msi * n);
                let (v1s, v2s) = (*v1p.add(msi), *v2p.add(msi));
                let (v1i, v2i) = (simd.splat_f64s(v1s), simd.splat_f64s(v2s));
                let mut j = 0;
                while j + l <= n {
                    let x = ld::<S>(row.add(j));
                    let v1 = ld::<S>(v1p.add(j));
                    let v2 = ld::<S>(v2p.add(j));
                    let r = if FMA {
                        let t = simd.mul_add_e_f64s(simd.neg_f64s(v1), v2i, x);
                        simd.mul_add_e_f64s(v1i, v2, t)
                    } else {
                        simd.add_f64s(
                            x,
                            simd.sub_f64s(simd.mul_f64s(v1i, v2), simd.mul_f64s(v1, v2i)),
                        )
                    };
                    st::<S>(row.add(j), r);
                    j += l;
                }
                for k in j..n {
                    *row.add(k) += v1s * *v2p.add(k) - *v1p.add(k) * v2s;
                }
                *row.add(msa) -= v2s;
            }
        }
        for k in 0..n {
            inv[msa * n + k] += vec2[k];
        }
    }
}

// ---------------------------------------------------------------------------------------
// K2 dsktf2 skew rank-2 update (production copies)
// ---------------------------------------------------------------------------------------

fn k2_split(data: &mut [f64], lda: usize, kk0: usize, k0: usize) -> (&mut [f64], &[f64], &[f64]) {
    let (write_cols, rest) = data.split_at_mut(kk0 * lda);
    let (col_kk0, rest) = rest.split_at_mut(lda);
    let col_k0 = &rest[(k0 - kk0 - 1) * lda..(k0 - kk0) * lda];
    (write_cols, col_kk0, col_k0)
}

fn k2_scalar(data: &mut [f64], lda: usize, kk0: usize, k0: usize, alpha: f64) {
    let (write_cols, col_kk0_data, col_k0_data) = k2_split(data, lda, kk0, k0);
    for (j, col_j) in write_cols.chunks_exact_mut(lda).take(kk0).enumerate() {
        let temp1 = alpha * col_kk0_data[j];
        let temp2 = alpha * col_k0_data[j];
        let (upper, rest) = col_j.split_at_mut(j);
        let (x, y) = (&col_k0_data[..j], &col_kk0_data[..j]);
        for ((c, &xk), &yk) in upper.iter_mut().zip(x).zip(y) {
            *c = (*c + xk * temp1) - yk * temp2;
        }
        rest[0] = 0.0;
    }
}

/// `MASKED=true` is the production kernel (masked AVX2 tail); `MASKED=false` replaces only
/// the <4-element column tail by the same scalar `(c + x*t1) - y*t2` (bitwise identical).
#[target_feature(enable = "avx2")]
unsafe fn k2_avx2_inner<const MASKED: bool>(
    write_cols: &mut [f64],
    col_kk0_data: &[f64],
    col_k0_data: &[f64],
    lda: usize,
    kk0: usize,
    alpha: f64,
) {
    use core::arch::x86_64::*;
    const MASKS: [[i64; 4]; 4] = [[0; 4], [-1, 0, 0, 0], [-1, -1, 0, 0], [-1, -1, -1, 0]];
    assert!(write_cols.len() >= kk0 * lda && col_kk0_data.len() >= kk0 && col_k0_data.len() >= kk0);
    let x = col_k0_data.as_ptr();
    let y = col_kk0_data.as_ptr();
    let finish = |col: *mut f64, mut i: usize, len: usize, t1: __m256d, t2: __m256d| {
        while i + 4 <= len {
            let c = _mm256_loadu_pd(col.add(i));
            let xv = _mm256_loadu_pd(x.add(i));
            let yv = _mm256_loadu_pd(y.add(i));
            let r = _mm256_sub_pd(
                _mm256_add_pd(c, _mm256_mul_pd(xv, t1)),
                _mm256_mul_pd(yv, t2),
            );
            _mm256_storeu_pd(col.add(i), r);
            i += 4;
        }
        let rem = len - i;
        if !MASKED {
            let (s1, s2) = (_mm256_cvtsd_f64(t1), _mm256_cvtsd_f64(t2));
            for k in i..len {
                *col.add(k) = (*col.add(k) + *x.add(k) * s1) - *y.add(k) * s2;
            }
        } else if rem > 0 {
            let mask = _mm256_loadu_si256(MASKS[rem].as_ptr() as *const __m256i);
            let c = _mm256_maskload_pd(col.add(i), mask);
            let xv = _mm256_maskload_pd(x.add(i), mask);
            let yv = _mm256_maskload_pd(y.add(i), mask);
            let r = _mm256_sub_pd(
                _mm256_add_pd(c, _mm256_mul_pd(xv, t1)),
                _mm256_mul_pd(yv, t2),
            );
            _mm256_maskstore_pd(col.add(i), mask, r);
        }
    };
    let base = write_cols.as_mut_ptr();
    let mut j = 0;
    while j + 1 < kk0 {
        let (a1, a2) = (alpha * *y.add(j), alpha * *x.add(j));
        let (b1, b2) = (alpha * *y.add(j + 1), alpha * *x.add(j + 1));
        let (ta1, ta2) = (_mm256_set1_pd(a1), _mm256_set1_pd(a2));
        let (tb1, tb2) = (_mm256_set1_pd(b1), _mm256_set1_pd(b2));
        let col_a = base.add(j * lda);
        let col_b = base.add((j + 1) * lda);
        let mut i = 0;
        while i + 4 <= j {
            let xv = _mm256_loadu_pd(x.add(i));
            let yv = _mm256_loadu_pd(y.add(i));
            let ca = _mm256_loadu_pd(col_a.add(i));
            let cb = _mm256_loadu_pd(col_b.add(i));
            let ra = _mm256_sub_pd(
                _mm256_add_pd(ca, _mm256_mul_pd(xv, ta1)),
                _mm256_mul_pd(yv, ta2),
            );
            let rb = _mm256_sub_pd(
                _mm256_add_pd(cb, _mm256_mul_pd(xv, tb1)),
                _mm256_mul_pd(yv, tb2),
            );
            _mm256_storeu_pd(col_a.add(i), ra);
            _mm256_storeu_pd(col_b.add(i), rb);
            i += 4;
        }
        finish(col_a, i, j, ta1, ta2);
        *col_a.add(j) = 0.0;
        finish(col_b, i, j + 1, tb1, tb2);
        *col_b.add(j + 1) = 0.0;
        j += 2;
    }
    if j < kk0 {
        let t1 = _mm256_set1_pd(alpha * *y.add(j));
        let t2 = _mm256_set1_pd(alpha * *x.add(j));
        let col = base.add(j * lda);
        finish(col, 0, j, t1, t2);
        *col.add(j) = 0.0;
    }
}

fn k2_avx2(data: &mut [f64], lda: usize, kk0: usize, k0: usize, alpha: f64) {
    let (w, y, x) = k2_split(data, lda, kk0, k0);
    unsafe { k2_avx2_inner::<true>(w, y, x, lda, kk0, alpha) }
}

fn k2_avx2_scalar_tail(data: &mut [f64], lda: usize, kk0: usize, k0: usize, alpha: f64) {
    let (w, y, x) = k2_split(data, lda, kk0, k0);
    unsafe { k2_avx2_inner::<false>(w, y, x, lda, kk0, alpha) }
}

/// pulp port of `k2_avx2` (paired columns, pulp masked partial load/store for the tail).
struct K2Pulp<'a, const FMA: bool> {
    w: &'a mut [f64],
    y: &'a [f64],
    x: &'a [f64],
    lda: usize,
    kk0: usize,
    alpha: f64,
}

impl<const FMA: bool> WithSimd for K2Pulp<'_, FMA> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            w,
            y,
            x,
            lda,
            kk0,
            alpha,
        } = self;
        assert!(w.len() >= kk0 * lda && y.len() >= kk0 && x.len() >= kk0);
        let l = S::F64_LANES;
        #[inline(always)]
        fn upd<S: Simd, const FMA: bool>(
            simd: S,
            c: S::f64s,
            xv: S::f64s,
            yv: S::f64s,
            t1: S::f64s,
            t2: S::f64s,
        ) -> S::f64s {
            if FMA {
                simd.mul_add_e_f64s(simd.neg_f64s(yv), t2, simd.mul_add_e_f64s(xv, t1, c))
            } else {
                simd.sub_f64s(
                    simd.add_f64s(c, simd.mul_f64s(xv, t1)),
                    simd.mul_f64s(yv, t2),
                )
            }
        }
        #[inline(always)]
        fn finish<S: Simd, const FMA: bool>(
            simd: S,
            col: &mut [f64],
            x: &[f64],
            y: &[f64],
            mut i: usize,
            len: usize,
            t1: S::f64s,
            t2: S::f64s,
        ) {
            let l = S::F64_LANES;
            unsafe {
                while i + l <= len {
                    let r = upd::<S, FMA>(
                        simd,
                        ld::<S>(col.as_ptr().add(i)),
                        ld::<S>(x.as_ptr().add(i)),
                        ld::<S>(y.as_ptr().add(i)),
                        t1,
                        t2,
                    );
                    st::<S>(col.as_mut_ptr().add(i), r);
                    i += l;
                }
            }
            if i < len {
                let c = simd.partial_load_f64s(&col[i..len]);
                let xv = simd.partial_load_f64s(&x[i..len]);
                let yv = simd.partial_load_f64s(&y[i..len]);
                simd.partial_store_f64s(&mut col[i..len], upd::<S, FMA>(simd, c, xv, yv, t1, t2));
            }
        }
        let mut j = 0;
        while j + 1 < kk0 {
            let (ta1, ta2) = (simd.splat_f64s(alpha * y[j]), simd.splat_f64s(alpha * x[j]));
            let (tb1, tb2) = (
                simd.splat_f64s(alpha * y[j + 1]),
                simd.splat_f64s(alpha * x[j + 1]),
            );
            let (ca, cb) = w[j * lda..(j + 2) * lda].split_at_mut(lda);
            let mut i = 0;
            unsafe {
                while i + l <= j {
                    let xv = ld::<S>(x.as_ptr().add(i));
                    let yv = ld::<S>(y.as_ptr().add(i));
                    let ra = upd::<S, FMA>(simd, ld::<S>(ca.as_ptr().add(i)), xv, yv, ta1, ta2);
                    let rb = upd::<S, FMA>(simd, ld::<S>(cb.as_ptr().add(i)), xv, yv, tb1, tb2);
                    st::<S>(ca.as_mut_ptr().add(i), ra);
                    st::<S>(cb.as_mut_ptr().add(i), rb);
                    i += l;
                }
            }
            finish::<S, FMA>(simd, ca, x, y, i, j, ta1, ta2);
            ca[j] = 0.0;
            finish::<S, FMA>(simd, cb, x, y, i, j + 1, tb1, tb2);
            cb[j + 1] = 0.0;
            j += 2;
        }
        if j < kk0 {
            let (t1, t2) = (simd.splat_f64s(alpha * y[j]), simd.splat_f64s(alpha * x[j]));
            let col = &mut w[j * lda..(j + 1) * lda];
            finish::<S, FMA>(simd, col, x, y, 0, j, t1, t2);
            col[j] = 0.0;
        }
    }
}

fn k2_pulp<const FMA: bool>(
    arch: Arch,
    data: &mut [f64],
    lda: usize,
    kk0: usize,
    k0: usize,
    alpha: f64,
) {
    let (w, y, x) = k2_split(data, lda, kk0, k0);
    let op = K2Pulp::<FMA> {
        w,
        y,
        x,
        lda,
        kk0,
        alpha,
    };
    match arch {
        Arch::Scalar => pulp::Scalar::new().vectorize(op),
        Arch::Best(a) => a.dispatch(op),
    }
}

/// The rank-2 calls of one `dsktf2` on an n x n matrix (k0 = n-1, n-3, ..., kk0 = k0-1).
fn k2_sweep(data: &mut [f64], n: usize, f: &mut dyn FnMut(&mut [f64], usize, usize, usize, f64)) {
    let alpha = 1.0 / n as f64;
    let mut k0 = n - 1;
    while k0 > 1 {
        f(data, n, k0 - 1, k0, alpha);
        k0 -= 2;
    }
}

// ---------------------------------------------------------------------------------------
// K3 two-hop bilinear b^T M a (production copy + SIMD candidates)
// ---------------------------------------------------------------------------------------

fn k3_scalar(inv: &[f64], n: usize, a: &[f64], b: &[f64]) -> f64 {
    let base = 0;
    let mut b_ma = 0.0;
    let mut i = 0;
    while i + 4 <= n {
        let r0 = &inv[base + i * n..base + i * n + n];
        let r1 = &inv[base + (i + 1) * n..base + (i + 1) * n + n];
        let r2 = &inv[base + (i + 2) * n..base + (i + 2) * n + n];
        let r3 = &inv[base + (i + 3) * n..base + (i + 3) * n + n];
        let (mut t0, mut t1, mut t2, mut t3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for (j, &aj) in a[..n].iter().enumerate() {
            t0 += r0[j] * aj;
            t1 += r1[j] * aj;
            t2 += r2[j] * aj;
            t3 += r3[j] * aj;
        }
        b_ma += b[i] * t0;
        b_ma += b[i + 1] * t1;
        b_ma += b[i + 2] * t2;
        b_ma += b[i + 3] * t3;
        i += 4;
    }
    while i < n {
        let row = &inv[base + i * n..base + i * n + n];
        let mut tmp = 0.0;
        for (j, &aj) in a[..n].iter().enumerate() {
            tmp += row[j] * aj;
        }
        b_ma += b[i] * tmp;
        i += 1;
    }
    b_ma
}

/// Exact AVX2 candidate: lanes are rows. A 4x4 block of four rows is transposed in
/// registers so every row keeps its serial left-to-right sum; 16 rows (four independent
/// accumulators) hide the add latency; `b_ma` is accumulated over rows in order.
#[target_feature(enable = "avx2")]
unsafe fn k3_avx2_rows(inv: &[f64], n: usize, a: &[f64], b: &[f64]) -> f64 {
    use core::arch::x86_64::*;
    assert!(inv.len() >= n * n && a.len() >= n && b.len() >= n);
    let p = inv.as_ptr();
    let mut b_ma = 0.0;
    let mut i = 0;
    #[inline(always)]
    unsafe fn tr(r0: *const f64, r1: *const f64, r2: *const f64, r3: *const f64) -> [__m256d; 4] {
        let (x0, x1, x2, x3) = (
            _mm256_loadu_pd(r0),
            _mm256_loadu_pd(r1),
            _mm256_loadu_pd(r2),
            _mm256_loadu_pd(r3),
        );
        let t0 = _mm256_unpacklo_pd(x0, x1); // r0[0] r1[0] r0[2] r1[2]
        let t1 = _mm256_unpackhi_pd(x0, x1); // r0[1] r1[1] r0[3] r1[3]
        let t2 = _mm256_unpacklo_pd(x2, x3);
        let t3 = _mm256_unpackhi_pd(x2, x3);
        [
            _mm256_permute2f128_pd(t0, t2, 0x20),
            _mm256_permute2f128_pd(t1, t3, 0x20),
            _mm256_permute2f128_pd(t0, t2, 0x31),
            _mm256_permute2f128_pd(t1, t3, 0x31),
        ]
    }
    if n.is_multiple_of(4) {
        while i + 16 <= n {
            let mut acc = [_mm256_setzero_pd(); 4];
            let mut j = 0;
            while j + 4 <= n {
                let s = [
                    _mm256_set1_pd(a[j]),
                    _mm256_set1_pd(a[j + 1]),
                    _mm256_set1_pd(a[j + 2]),
                    _mm256_set1_pd(a[j + 3]),
                ];
                for g in 0..4 {
                    let r = p.add((i + 4 * g) * n + j);
                    let c = tr(r, r.add(n), r.add(2 * n), r.add(3 * n));
                    for q in 0..4 {
                        acc[g] = _mm256_add_pd(acc[g], _mm256_mul_pd(c[q], s[q]));
                    }
                }
                j += 4;
            }
            let mut t = [0.0f64; 16];
            for g in 0..4 {
                _mm256_storeu_pd(t.as_mut_ptr().add(4 * g), acc[g]);
            }
            for r in 0..16 {
                b_ma += b[i + r] * t[r];
            }
            i += 16;
        }
    }
    while i < n {
        let row = &inv[i * n..i * n + n];
        let mut tmp = 0.0;
        for (j, &aj) in a[..n].iter().enumerate() {
            tmp += row[j] * aj;
        }
        b_ma += b[i] * tmp;
        i += 1;
    }
    b_ma
}

/// pulp candidate with lanes over `j` (the natural portable form, as LoopVectorization):
/// every row's dot product is lane-split and FMA-contracted, so the sum is reassociated.
struct K3PulpFma<'a> {
    inv: &'a [f64],
    n: usize,
    a: &'a [f64],
    b: &'a [f64],
}

impl WithSimd for K3PulpFma<'_> {
    type Output = f64;
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let Self { inv, n, a, b } = self;
        let l = S::F64_LANES;
        let mut b_ma = 0.0;
        for i in 0..n {
            let row = &inv[i * n..i * n + n];
            let (mut s0, mut s1) = (simd.splat_f64s(0.0), simd.splat_f64s(0.0));
            let mut j = 0;
            unsafe {
                while j + 2 * l <= n {
                    s0 = simd.mul_add_e_f64s(
                        ld::<S>(row.as_ptr().add(j)),
                        ld::<S>(a.as_ptr().add(j)),
                        s0,
                    );
                    s1 = simd.mul_add_e_f64s(
                        ld::<S>(row.as_ptr().add(j + l)),
                        ld::<S>(a.as_ptr().add(j + l)),
                        s1,
                    );
                    j += 2 * l;
                }
            }
            let mut tmp = simd.reduce_sum_f64s(simd.add_f64s(s0, s1));
            for k in j..n {
                tmp = row[k].mul_add(a[k], tmp);
            }
            b_ma = b[i].mul_add(tmp, b_ma);
        }
        b_ma
    }
}

// ---------------------------------------------------------------------------------------
// K4 sktd solve (production copies + pulp)
// ---------------------------------------------------------------------------------------

fn k4_scalar(vt: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    let mut b_groups = b.chunks_exact(4 * n);
    let mut c_groups = c.chunks_exact_mut(4 * n);
    for (bg, cg) in (&mut b_groups).zip(&mut c_groups) {
        let (b0, br) = bg.split_at(n);
        let (b1, br) = br.split_at(n);
        let (b2, b3) = br.split_at(n);
        let (c0, cr) = cg.split_at_mut(n);
        let (c1, cr) = cr.split_at_mut(n);
        let (c2, c3) = cr.split_at_mut(n);
        let bs = [b0, b1, b2, b3];
        let cs = [c0, c1, c2, c3];
        let d0 = -vt[0];
        for w in 0..4 {
            cs[w][1] = bs[w][0] / d0;
        }
        for i in (2..n).step_by(2) {
            let (v, d) = (vt[i - 1], -vt[i]);
            for w in 0..4 {
                cs[w][i + 1] = (bs[w][i] - cs[w][i - 1] * v) / d;
            }
        }
        let dn = vt[n - 2];
        for w in 0..4 {
            cs[w][n - 2] = bs[w][n - 1] / dn;
        }
        for i in (1..n - 2).rev().step_by(2) {
            let (v, d) = (vt[i], vt[i - 1]);
            for w in 0..4 {
                cs[w][i - 1] = (bs[w][i] + cs[w][i + 1] * v) / d;
            }
        }
    }
    for (bc, cc) in b_groups
        .remainder()
        .chunks_exact(n)
        .zip(c_groups.into_remainder().chunks_exact_mut(n))
    {
        cc[1] = bc[0] / -vt[0];
        for i in (2..n).step_by(2) {
            cc[i + 1] = (bc[i] - cc[i - 1] * vt[i - 1]) / -vt[i];
        }
        cc[n - 2] = bc[n - 1] / vt[n - 2];
        for i in (1..n - 2).rev().step_by(2) {
            cc[i - 1] = (bc[i] + cc[i + 1] * vt[i]) / vt[i - 1];
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn k4_avx2_inner(vt: &[f64], b: &[f64], c: &mut [f64], n: usize, cols: usize) {
    use core::arch::x86_64::*;
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let load = |j: usize, i: usize| {
        _mm256_set_pd(
            *bp.add((j + 3) * n + i),
            *bp.add((j + 2) * n + i),
            *bp.add((j + 1) * n + i),
            *bp.add(j * n + i),
        )
    };
    let store = |j: usize, i: usize, v: __m256d| {
        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), v);
        for (w, lane) in lanes.into_iter().enumerate() {
            *cp.add((j + w) * n + i) = lane;
        }
    };
    for j in (0..cols).step_by(8) {
        let d0 = _mm256_set1_pd(-vt[0]);
        let mut pa = _mm256_div_pd(load(j, 0), d0);
        let mut pb = _mm256_div_pd(load(j + 4, 0), d0);
        store(j, 1, pa);
        store(j + 4, 1, pb);
        for i in (2..n).step_by(2) {
            let v = _mm256_set1_pd(vt[i - 1]);
            let d = _mm256_set1_pd(-vt[i]);
            pa = _mm256_div_pd(_mm256_sub_pd(load(j, i), _mm256_mul_pd(pa, v)), d);
            pb = _mm256_div_pd(_mm256_sub_pd(load(j + 4, i), _mm256_mul_pd(pb, v)), d);
            store(j, i + 1, pa);
            store(j + 4, i + 1, pb);
        }
        let dn = _mm256_set1_pd(vt[n - 2]);
        pa = _mm256_div_pd(load(j, n - 1), dn);
        pb = _mm256_div_pd(load(j + 4, n - 1), dn);
        store(j, n - 2, pa);
        store(j + 4, n - 2, pb);
        for i in (1..n - 2).rev().step_by(2) {
            let v = _mm256_set1_pd(vt[i]);
            let d = _mm256_set1_pd(vt[i - 1]);
            pa = _mm256_div_pd(_mm256_add_pd(load(j, i), _mm256_mul_pd(pa, v)), d);
            pb = _mm256_div_pd(_mm256_add_pd(load(j + 4, i), _mm256_mul_pd(pb, v)), d);
            store(j, i - 1, pa);
            store(j + 4, i - 1, pb);
        }
    }
}

fn k4_avx2(vt: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    let done = n / 8 * 8;
    unsafe { k4_avx2_inner(vt, b, c, n, done) };
    k4_scalar(vt, &b[done * n..n * n], &mut c[done * n..n * n], n);
}

/// pulp port of `k4_avx2`: lanes are columns (strided gather/scatter through a stack
/// buffer, as the hand kernel does), two vectors in flight, no FMA, IEEE division.
struct K4Pulp<'a> {
    vt: &'a [f64],
    b: &'a [f64],
    c: &'a mut [f64],
    n: usize,
}

impl WithSimd for K4Pulp<'_> {
    type Output = usize;
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> usize {
        let Self { vt, b, c, n } = self;
        let l = S::F64_LANES;
        let w = 2 * l;
        let cols = n / w * w;
        let load = |j: usize, i: usize| {
            let mut t = [0.0f64; 16];
            for q in 0..l {
                t[q] = b[(j + q) * n + i];
            }
            simd.partial_load_f64s(&t[..l])
        };
        let mut store = |j: usize, i: usize, v: S::f64s| {
            let mut t = [0.0f64; 16];
            simd.partial_store_f64s(&mut t[..l], v);
            for q in 0..l {
                c[(j + q) * n + i] = t[q];
            }
        };
        for j in (0..cols).step_by(w) {
            let d0 = simd.splat_f64s(-vt[0]);
            let mut pa = simd.div_f64s(load(j, 0), d0);
            let mut pb = simd.div_f64s(load(j + l, 0), d0);
            store(j, 1, pa);
            store(j + l, 1, pb);
            for i in (2..n).step_by(2) {
                let v = simd.splat_f64s(vt[i - 1]);
                let d = simd.splat_f64s(-vt[i]);
                pa = simd.div_f64s(simd.sub_f64s(load(j, i), simd.mul_f64s(pa, v)), d);
                pb = simd.div_f64s(simd.sub_f64s(load(j + l, i), simd.mul_f64s(pb, v)), d);
                store(j, i + 1, pa);
                store(j + l, i + 1, pb);
            }
            let dn = simd.splat_f64s(vt[n - 2]);
            pa = simd.div_f64s(load(j, n - 1), dn);
            pb = simd.div_f64s(load(j + l, n - 1), dn);
            store(j, n - 2, pa);
            store(j + l, n - 2, pb);
            for i in (1..n - 2).rev().step_by(2) {
                let v = simd.splat_f64s(vt[i]);
                let d = simd.splat_f64s(vt[i - 1]);
                pa = simd.div_f64s(simd.add_f64s(load(j, i), simd.mul_f64s(pa, v)), d);
                pb = simd.div_f64s(simd.add_f64s(load(j + l, i), simd.mul_f64s(pb, v)), d);
                store(j, i - 1, pa);
                store(j + l, i - 1, pb);
            }
        }
        cols
    }
}

fn k4_pulp(arch: Arch, vt: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    let op = K4Pulp { vt, b, c, n };
    let done = match arch {
        Arch::Scalar => pulp::Scalar::new().vectorize(op),
        Arch::Best(a) => a.dispatch(op),
    };
    k4_scalar(vt, &b[done * n..n * n], &mut c[done * n..n * n], n);
}

// ---------------------------------------------------------------------------------------
// dispatch helper and bench harness
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Arch {
    Scalar,
    Best(pulp::Arch),
}

fn k1_pulp<const FMA: bool>(
    arch: Arch,
    msa: usize,
    inv: &mut [f64],
    slt: &[f64],
    vec1: &mut [f64],
    vec2: &mut [f64],
    n: usize,
    pf: &mut f64,
) {
    let op = K1Pulp::<FMA> {
        msa,
        inv,
        slt,
        vec1,
        vec2,
        n,
        pf,
    };
    match arch {
        Arch::Scalar => pulp::Scalar::new().vectorize(op),
        Arch::Best(a) => a.dispatch(op),
    }
}

type Variant<'a> = (&'static str, Box<dyn FnMut() + 'a>);

/// Round-robin timing: each round times `iters` calls of every variant (rotating the
/// order); returns per-variant (median, min) ns/call over the rounds.
fn run(title: &str, rounds: usize, iters: usize, vs: &mut [Variant<'_>]) {
    for (_, f) in vs.iter_mut() {
        for _ in 0..iters {
            f();
        }
    }
    let mut samples = vec![Vec::new(); vs.len()];
    for r in 0..rounds {
        for k in 0..vs.len() {
            let idx = (k + r) % vs.len();
            let f = &mut vs[idx].1;
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            samples[idx].push(t.elapsed().as_nanos() as f64 / iters as f64);
        }
    }
    println!("## {title}");
    for (k, (name, _)) in vs.iter().enumerate() {
        let s = &mut samples[k];
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{title}\t{name}\tmedian_ns={:.1}\tmin_ns={:.1}\trounds={}",
            s[s.len() / 2],
            s[0],
            s.len()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map_or(128, |s| s.parse().unwrap());
    let rounds: usize = args.get(2).map_or(15, |s| s.parse().unwrap());
    let best = pulp::Arch::new();
    println!("n={n} rounds={rounds}");
    println!("pulp::Arch::new() = {best:?}  (x86-v3 feature enabled in this crate)");
    println!(
        "cpu: avx2={} fma={} avx512f={}",
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx512f")
    );
    let best = Arch::Best(best);
    let mut rng = Rng(0x9E3779B97F4A7C15);

    // --- setup cost of pulp dispatch ---
    {
        let reps = 10_000_000;
        let t = Instant::now();
        for _ in 0..reps {
            black_box(pulp::Arch::new());
        }
        println!(
            "setup\tpulp::Arch::new\tns_per_call={:.2}",
            t.elapsed().as_nanos() as f64 / reps as f64
        );
        let t = Instant::now();
        for _ in 0..reps {
            black_box(is_x86_feature_detected!("avx2"));
        }
        println!(
            "setup\tis_x86_feature_detected(avx2)\tns_per_call={:.2}",
            t.elapsed().as_nanos() as f64 / reps as f64
        );
    }

    // --- genuine skew matrix and inverse for K1 residual checks ---
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let v = rng.unif();
            m[i * n + j] = v;
            m[j * n + i] = -v;
        }
    }
    let inverse = |m: &[f64]| -> Vec<f64> {
        let mut buf: Vec<f64> = m.iter().map(|x| -x).collect();
        let mut piv = vec![pfapack::PivotIndex1Based(0); n];
        let mut vt = vec![0.0; n - 1];
        let mut work = vec![0.0; n * n];
        let mut a = pfapack::SqMat::new(&mut buf, n);
        pfapack::dsktf2(&mut a, &mut piv).expect("nonsingular");
        let mut w = pfapack::SqMat::new(&mut work, n);
        pfapack::utu2inv_real(&mut a, &piv, &mut vt, &mut w);
        buf.iter().map(|x| -x).collect()
    };
    let residual = |m: &[f64], x: &[f64]| -> f64 {
        let mut r = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += m[i * n + k] * x[k * n + j];
                }
                r = r.max((s - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        r
    };
    let inv0 = inverse(&m);
    println!(
        "check\tK1 initial inverse residual max|M X - I| = {:.3e}",
        residual(&m, &inv0)
    );

    // --- K1 correctness: chain of accepted hops on a genuine inverse ---
    {
        type K1Fn = fn(Arch, usize, &mut [f64], &[f64], &mut [f64], &mut [f64], usize, &mut f64);
        let variants: [(&str, Option<K1Fn>); 5] = [
            ("scalar", None),
            ("avx2", None),
            ("pulp-scalar", Some(k1_pulp::<false>)),
            ("pulp-v3", Some(k1_pulp::<false>)),
            ("pulp-v3-fma", Some(k1_pulp::<true>)),
        ];
        let steps = 64;
        let mut results: Vec<(Vec<f64>, f64, Vec<f64>)> = Vec::new();
        for (name, f) in variants.iter() {
            let mut rng = Rng(12345);
            let mut mm = m.clone();
            let mut x = inv0.clone();
            let (mut v1, mut v2) = (vec![0.0; n], vec![0.0; n]);
            let mut pf = 1.0;
            for _ in 0..steps {
                let msa = (rng.next() % n as u64) as usize;
                let mut slt = rng.vec(n);
                slt[msa] = 0.0;
                match (*name, f) {
                    ("scalar", _) => k1_scalar(msa, &mut x, &slt, &mut v1, &mut v2, n, &mut pf),
                    ("avx2", _) => unsafe {
                        k1_avx2(msa, &mut x, &slt, &mut v1, &mut v2, n, &mut pf)
                    },
                    ("pulp-scalar", Some(f)) => f(
                        Arch::Scalar,
                        msa,
                        &mut x,
                        &slt,
                        &mut v1,
                        &mut v2,
                        n,
                        &mut pf,
                    ),
                    (_, Some(f)) => f(best, msa, &mut x, &slt, &mut v1, &mut v2, n, &mut pf),
                    _ => unreachable!(),
                }
                for j in 0..n {
                    mm[msa * n + j] = slt[j];
                    mm[j * n + msa] = -slt[j];
                }
            }
            let res = residual(&mm, &x);
            results.push((x, pf, mm));
            let (x0, pf0, _) = &results[0];
            let (x, pf, _) = results.last().unwrap();
            println!(
                "check\tK1 {name}\tsteps={steps}\tresidual={res:.3e}\tbits_differ_vs_scalar={}/{}\tmax_rel_vs_scalar={:.3e}\tpf_rel_vs_scalar={:.3e}",
                bits_differ(x, x0),
                n * n,
                max_rel(x, x0),
                ((pf - pf0) / pf0).abs()
            );
        }
        // Fresh inverse of the final matrix as an independent reference.
        let fresh = inverse(&results[0].2);
        for ((x, _, _), (name, _)) in results.iter().zip(variants.iter()) {
            println!(
                "check\tK1 {name}\tmax_rel_vs_fresh_inverse={:.3e}",
                max_rel(x, &fresh)
            );
        }
    }

    // --- K1 timing ---
    {
        let pristine = inv0.clone();
        let mut msa_rng = Rng(777);
        let msa = (msa_rng.next() % n as u64) as usize;
        let mut slt = rng.vec(n);
        slt[msa] = 0.0;
        let mk = |pristine: &[f64]| (pristine.to_vec(), vec![0.0; n], vec![0.0; n], 1.0f64);
        let (mut w0, mut w1, mut w2, mut w3, mut w4, mut w5, mut w6) = (
            mk(&pristine),
            mk(&pristine),
            mk(&pristine),
            mk(&pristine),
            mk(&pristine),
            mk(&pristine),
            mk(&pristine),
        );
        let (p, s) = (&pristine, &slt);
        let mut vs: Vec<Variant> = vec![
            (
                "copy-only",
                Box::new(|| {
                    w0.0.copy_from_slice(p);
                    black_box(&mut w0.0);
                }),
            ),
            (
                "scalar",
                Box::new(|| {
                    w1.0.copy_from_slice(p);
                    k1_scalar(msa, &mut w1.0, s, &mut w1.1, &mut w1.2, n, &mut w1.3);
                    black_box(&mut w1.0);
                }),
            ),
            (
                "avx2",
                Box::new(|| {
                    w2.0.copy_from_slice(p);
                    unsafe { k1_avx2(msa, &mut w2.0, s, &mut w2.1, &mut w2.2, n, &mut w2.3) };
                    black_box(&mut w2.0);
                }),
            ),
            (
                "pulp-scalar",
                Box::new(|| {
                    w3.0.copy_from_slice(p);
                    k1_pulp::<false>(
                        Arch::Scalar,
                        msa,
                        &mut w3.0,
                        s,
                        &mut w3.1,
                        &mut w3.2,
                        n,
                        &mut w3.3,
                    );
                    black_box(&mut w3.0);
                }),
            ),
            (
                "pulp-v3",
                Box::new(|| {
                    w4.0.copy_from_slice(p);
                    k1_pulp::<false>(best, msa, &mut w4.0, s, &mut w4.1, &mut w4.2, n, &mut w4.3);
                    black_box(&mut w4.0);
                }),
            ),
            (
                "pulp-v3-fma",
                Box::new(|| {
                    w5.0.copy_from_slice(p);
                    k1_pulp::<true>(best, msa, &mut w5.0, s, &mut w5.1, &mut w5.2, n, &mut w5.3);
                    black_box(&mut w5.0);
                }),
            ),
            (
                "pulp-v3+Arch::new/call",
                Box::new(|| {
                    w6.0.copy_from_slice(p);
                    let a = Arch::Best(pulp::Arch::new());
                    k1_pulp::<false>(a, msa, &mut w6.0, s, &mut w6.1, &mut w6.2, n, &mut w6.3);
                    black_box(&mut w6.0);
                }),
            ),
        ];
        run("K1 update_one_real (copy+kernel)", rounds, 2000, &mut vs);
    }

    // --- K2 correctness + timing ---
    {
        let mut a0 = vec![0.0; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let v = rng.unif();
                a0[j * n + i] = v; // column-major upper
                a0[i * n + j] = -v;
            }
        }
        let sweep = |f: &mut dyn FnMut(&mut [f64], usize, usize, usize, f64)| {
            let mut d = a0.clone();
            k2_sweep(&mut d, n, f);
            d
        };
        let r_s = sweep(&mut |d, l, kk, k, al| k2_scalar(d, l, kk, k, al));
        let r_a = sweep(&mut |d, l, kk, k, al| k2_avx2(d, l, kk, k, al));
        let r_at = sweep(&mut |d, l, kk, k, al| k2_avx2_scalar_tail(d, l, kk, k, al));
        let r_ps = sweep(&mut |d, l, kk, k, al| k2_pulp::<false>(Arch::Scalar, d, l, kk, k, al));
        let r_p = sweep(&mut |d, l, kk, k, al| k2_pulp::<false>(best, d, l, kk, k, al));
        let r_pf = sweep(&mut |d, l, kk, k, al| k2_pulp::<true>(best, d, l, kk, k, al));
        for (name, r) in [
            ("avx2", &r_a),
            ("avx2-scalar-tail", &r_at),
            ("pulp-scalar", &r_ps),
            ("pulp-v3", &r_p),
            ("pulp-v3-fma", &r_pf),
        ] {
            println!(
                "check\tK2 {name}\tbits_differ_vs_scalar={}/{}\tmax_rel_vs_scalar={:.3e}",
                bits_differ(r, &r_s),
                n * n,
                max_rel(r, &r_s)
            );
        }
        let p = &a0;
        let mut ws: Vec<Vec<f64>> = (0..7).map(|_| a0.clone()).collect();
        let [w0, w1, w2, w3, w4, w5, w6] = &mut ws[..] else {
            unreachable!()
        };
        let mut vs: Vec<Variant> = vec![
            (
                "copy-only",
                Box::new(|| {
                    w0.copy_from_slice(p);
                    black_box(&mut *w0);
                }),
            ),
            (
                "scalar",
                Box::new(|| {
                    w1.copy_from_slice(p);
                    k2_sweep(w1, n, &mut |d, l, kk, k, al| k2_scalar(d, l, kk, k, al));
                    black_box(&mut *w1);
                }),
            ),
            (
                "avx2",
                Box::new(|| {
                    w2.copy_from_slice(p);
                    k2_sweep(w2, n, &mut |d, l, kk, k, al| k2_avx2(d, l, kk, k, al));
                    black_box(&mut *w2);
                }),
            ),
            (
                "avx2-scalar-tail",
                Box::new(|| {
                    w6.copy_from_slice(p);
                    k2_sweep(w6, n, &mut |d, l, kk, k, al| {
                        k2_avx2_scalar_tail(d, l, kk, k, al)
                    });
                    black_box(&mut *w6);
                }),
            ),
            (
                "pulp-scalar",
                Box::new(|| {
                    w3.copy_from_slice(p);
                    k2_sweep(w3, n, &mut |d, l, kk, k, al| {
                        k2_pulp::<false>(Arch::Scalar, d, l, kk, k, al)
                    });
                    black_box(&mut *w3);
                }),
            ),
            (
                "pulp-v3",
                Box::new(|| {
                    w4.copy_from_slice(p);
                    k2_sweep(w4, n, &mut |d, l, kk, k, al| {
                        k2_pulp::<false>(best, d, l, kk, k, al)
                    });
                    black_box(&mut *w4);
                }),
            ),
            (
                "pulp-v3-fma",
                Box::new(|| {
                    w5.copy_from_slice(p);
                    k2_sweep(w5, n, &mut |d, l, kk, k, al| {
                        k2_pulp::<true>(best, d, l, kk, k, al)
                    });
                    black_box(&mut *w5);
                }),
            ),
        ];
        run(
            "K2 dsktf2 rank-2 sweep (copy+kernel)",
            rounds,
            1000,
            &mut vs,
        );
    }

    // --- K3 correctness + timing ---
    {
        let a = rng.vec(n);
        let b = rng.vec(n);
        let x = &inv0;
        let r_s = k3_scalar(x, n, &a, &b);
        let r_a = unsafe { k3_avx2_rows(x, n, &a, &b) };
        let r_pf = best_dispatch_k3(best, x, n, &a, &b);
        println!(
            "check\tK3 avx2-rows\tbitwise_equal={}\trel={:.3e}",
            r_a.to_bits() == r_s.to_bits(),
            ((r_a - r_s) / r_s).abs()
        );
        println!(
            "check\tK3 pulp-v3-fma\tbitwise_equal={}\trel={:.3e}",
            r_pf.to_bits() == r_s.to_bits(),
            ((r_pf - r_s) / r_s).abs()
        );
        let (a, b) = (&a, &b);
        let mut vs: Vec<Variant> = vec![
            (
                "scalar",
                Box::new(|| {
                    black_box(k3_scalar(black_box(x), n, a, b));
                }),
            ),
            (
                "avx2-rows(exact,new)",
                Box::new(|| {
                    black_box(unsafe { k3_avx2_rows(black_box(x), n, a, b) });
                }),
            ),
            (
                "pulp-scalar-fma",
                Box::new(|| {
                    black_box(pulp::Scalar::new().vectorize(K3PulpFma {
                        inv: black_box(x),
                        n,
                        a,
                        b,
                    }));
                }),
            ),
            (
                "pulp-v3-fma",
                Box::new(|| {
                    black_box(best_dispatch_k3(best, black_box(x), n, a, b));
                }),
            ),
        ];
        run("K3 two_hop_bilinear", rounds, 5000, &mut vs);
    }

    // --- K4 correctness + timing ---
    {
        let vt: Vec<f64> = (0..n - 1).map(|_| 0.5 + rng.unif().abs()).collect();
        let bm = rng.vec(n * n);
        let mut c_s = vec![0.0; n * n];
        let mut c_a = vec![0.0; n * n];
        let mut c_ps = vec![0.0; n * n];
        let mut c_p = vec![0.0; n * n];
        k4_scalar(&vt, &bm, &mut c_s, n);
        k4_avx2(&vt, &bm, &mut c_a, n);
        k4_pulp(Arch::Scalar, &vt, &bm, &mut c_ps, n);
        k4_pulp(best, &vt, &bm, &mut c_p, n);
        for (name, r) in [("avx2", &c_a), ("pulp-scalar", &c_ps), ("pulp-v3", &c_p)] {
            println!(
                "check\tK4 {name}\tbits_differ_vs_scalar={}/{}",
                bits_differ(r, &c_s),
                n * n
            );
        }
        let (vt, bm) = (&vt, &bm);
        let mut vs: Vec<Variant> = vec![
            (
                "scalar",
                Box::new(|| {
                    k4_scalar(vt, bm, &mut c_s, n);
                    black_box(&mut c_s);
                }),
            ),
            (
                "avx2",
                Box::new(|| {
                    k4_avx2(vt, bm, &mut c_a, n);
                    black_box(&mut c_a);
                }),
            ),
            (
                "pulp-scalar",
                Box::new(|| {
                    k4_pulp(Arch::Scalar, vt, bm, &mut c_ps, n);
                    black_box(&mut c_ps);
                }),
            ),
            (
                "pulp-v3",
                Box::new(|| {
                    k4_pulp(best, vt, bm, &mut c_p, n);
                    black_box(&mut c_p);
                }),
            ),
        ];
        run("K4 solve_sktd", rounds, 2000, &mut vs);
    }
}

fn best_dispatch_k3(arch: Arch, inv: &[f64], n: usize, a: &[f64], b: &[f64]) -> f64 {
    let op = K3PulpFma { inv, n, a, b };
    match arch {
        Arch::Scalar => pulp::Scalar::new().vectorize(op),
        Arch::Best(x) => x.dispatch(op),
    }
}
