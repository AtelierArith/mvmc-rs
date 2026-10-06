//! Tensor-native batched Pfaffian and inverse on tenferro `CpuBackend` (faer provider).
//!
//! This is a reference for how a device-portable formulation looks, not a fast CPU path. All
//! state lives in tenferro tensors with the plane axis `P = NQP * B` trailing (`[n, n, P]`
//! matrices, `[n, P]` vectors, `[P]` scalars); every step is expressed with whole-batch
//! elementwise/structural/reduction/`dot_general` ops, with no per-plane host loop.
//!
//! Algorithm (mirrors `pfapack::dsktf2` + `utu2pfa` + `utu2inv`):
//!
//! 1. LTL^T with partial pivoting. For `k0 = n-1 .. 1` the pivot row is the first argmax of
//!    `|A[0..kk, k0]|` (`kk = k0 - 1`), found with `reduce_max` + `compare` + `select` +
//!    `reduce_min` (there is no argmax op: tensor4all/tenferro-rs#1976). The row/column swap
//!    is a full congruence `A <- P A P^T` with batch-varying `kp`, expressed with one-hot
//!    masks, `select` and `reduce_sum` (there is no batched row/column gather/swap with
//!    batch-varying indices: tensor4all/tenferro-rs#2008). The skew rank-2 update uses the same
//!    elementwise operation order as pfapack (`(c + x t1) - y t2`), so the factor and hence the
//!    real Pfaffian are bit-identical to pfapack for real planes (complex differs only through
//!    the division convention).
//! 2. Pfaffian: sequential product of `A[i, i+1]` times the pivot sign.
//! 3. Inverse: the unit upper-triangular inverse `(I+N)^-1` by batched back substitution over
//!    rows (elementwise multiply + `reduce_sum`; the former repeated-squaring expansion was
//!    unstable for n >= 64, issue #466), the skew-tridiagonal
//!    solve as row recurrences over `[n, P]` slices, and the pivot permutations as batched
//!    one-hot permutation matrices applied with `dot_general`. The operation order differs
//!    from pfapack, so the inverse agrees only to rounding.
//!
//! Zero-pivot planes keep their state frozen (masked `select`) like pfapack's `continue`.

use std::marker::PhantomData;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, CompareDir, DType, DotGeneralConfig, PadConfig,
    SliceConfig, Tensor, TypedTensor,
};

use crate::scalar::private::Impl;
use crate::{BatchOutput, Error, PfScalar, PlaneStatus};

type R<X> = tenferro_tensor::Result<X>;

struct Ops<'a, T: PfScalar> {
    sess: &'a mut dyn BackendSession,
    /// Number of tenferro session operations issued (reported with `MVMC_GPU_NATIVE_OPCOUNT=1`).
    ops: usize,
    n: usize,
    p: usize,
    _marker: PhantomData<T>,
}

impl<'a, T: PfScalar> Ops<'a, T> {
    fn s(&mut self) -> &mut dyn BackendSession {
        self.ops += 1;
        &mut *self.sess
    }

    // ----- constants and shape plumbing -------------------------------------------------

    fn scalar(&mut self, v: f64) -> R<Tensor> {
        T::make_tensor(vec![], vec![T::from_parts(v, 0.0)])
    }

    /// `T`-typed tensor filled with `v`.
    fn full(&mut self, shape: &[usize], v: f64) -> R<Tensor> {
        let c = self.scalar(v)?;
        self.s().broadcast_in_dim(&c, shape, &[])
    }

    /// `f64` tensor filled with `v`.
    fn fullf(&mut self, shape: &[usize], v: f64) -> R<Tensor> {
        let c = <f64 as Impl>::make_tensor(vec![], vec![v])?;
        self.s().broadcast_in_dim(&c, shape, &[])
    }

    fn bc(&mut self, t: &Tensor, shape: &[usize], dims: &[usize]) -> R<Tensor> {
        self.s().broadcast_in_dim(t, shape, dims)
    }

    fn rs(&mut self, t: &Tensor, shape: &[usize]) -> R<Tensor> {
        self.s().reshape(t, shape)
    }

    fn sl(&mut self, t: &Tensor, starts: &[usize], limits: &[usize]) -> R<Tensor> {
        let config = SliceConfig {
            starts: starts.to_vec(),
            limits: limits.to_vec(),
            strides: vec![1; starts.len()],
        };
        self.s().slice(t, &config)
    }

    fn pad(&mut self, t: &Tensor, low: &[i64], high: &[i64]) -> R<Tensor> {
        let config = PadConfig {
            edge_padding_low: low.to_vec(),
            edge_padding_high: high.to_vec(),
            interior_padding: vec![0; low.len()],
        };
        self.s().pad(t, &config)
    }

    /// Static boolean mask `[rows, cols]` broadcast to `[rows, cols, P]`.
    fn mask3(&mut self, rows: usize, cols: usize, f: impl Fn(usize, usize) -> bool) -> R<Tensor> {
        let mut data = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                data.push(f(i, j));
            }
        }
        let m =
            Tensor::from_typed::<bool>(TypedTensor::from_vec_col_major(vec![rows, cols], data)?);
        let p = self.p;
        self.s().broadcast_in_dim(&m, &[rows, cols, p], &[0, 1])
    }

    /// `f64` index vector `0..len` broadcast along a trailing plane axis: `[len, P]`.
    fn iota_p(&mut self, len: usize) -> R<Tensor> {
        let v = <f64 as Impl>::make_tensor(vec![len], (0..len).map(|i| i as f64).collect())?;
        let p = self.p;
        self.s().broadcast_in_dim(&v, &[len, p], &[0])
    }

    // ----- small helpers over the session -----------------------------------------------

    fn add(&mut self, a: &Tensor, b: &Tensor) -> R<Tensor> {
        self.s().add(a, b)
    }
    fn sub(&mut self, a: &Tensor, b: &Tensor) -> R<Tensor> {
        self.s().sub(a, b)
    }
    fn mul(&mut self, a: &Tensor, b: &Tensor) -> R<Tensor> {
        self.s().mul(a, b)
    }
    fn div(&mut self, a: &Tensor, b: &Tensor) -> R<Tensor> {
        self.s().div(a, b)
    }
    fn neg(&mut self, a: &Tensor) -> R<Tensor> {
        self.s().neg(a)
    }
    fn select(&mut self, pred: &Tensor, a: &Tensor, b: &Tensor) -> R<Tensor> {
        self.s().select(pred, a, b)
    }
    fn cmp(&mut self, a: &Tensor, b: &Tensor, dir: CompareDir) -> R<Tensor> {
        self.s().compare(a, b, &dir)
    }

    /// Batched matrix product `lhs[:, :, p] * rhs[:, :, p]` (contract `lhs` dim 1 with `rhs`
    /// dim 0, batch on the trailing axis).
    fn bmm(&mut self, lhs: &Tensor, rhs: &Tensor) -> R<Tensor> {
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        };
        self.s().dot_general(lhs, rhs, &config)
    }

    /// `lhs[:, :, p]^T * rhs[:, :, p]`.
    fn bmm_tn(&mut self, lhs: &Tensor, rhs: &Tensor) -> R<Tensor> {
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        };
        self.s().dot_general(lhs, rhs, &config)
    }

    /// `|re| + |im|` (BLAS `izamax` magnitude; `|x|` for reals) as an `f64` tensor.
    fn magnitude(&mut self, t: &Tensor) -> R<Tensor> {
        if !T::IS_COMPLEX {
            return self.s().abs(t);
        }
        let shape = t.shape().to_vec();
        let re = self.s().cast(t, DType::F64)?;
        let i_unit = {
            let c = self.scalar_complex(0.0, 1.0)?;
            self.s().broadcast_in_dim(&c, &shape, &[])?
        };
        let iz = self.mul(t, &i_unit)?;
        let im = self.s().cast(&iz, DType::F64)?;
        let are = self.s().abs(&re)?;
        let aim = self.s().abs(&im)?;
        self.s().add(&are, &aim)
    }

    fn scalar_complex(&mut self, re: f64, im: f64) -> R<Tensor> {
        T::make_tensor(vec![], vec![T::from_parts(re, im)])
    }

    /// Finite-value predicate of a `[P]` `T` tensor, as bool `[P]`.
    fn is_finite(&mut self, t: &Tensor) -> R<Tensor> {
        let mag = self.s().abs(t)?;
        let max = self.fullf(&[self.p], f64::MAX)?;
        self.cmp(&mag, &max, CompareDir::Le)
    }

    // ----- stage 1: LTL^T ----------------------------------------------------------------

    /// Returns `(A_factored, pivots (n x [P] f64, 0-based), info [P] f64 (1-based row, 0 = ok))`.
    fn ltl(&mut self, mut a: Tensor) -> R<(Tensor, Vec<Tensor>, Tensor)> {
        let (n, p) = (self.n, self.p);
        let zeros3 = self.full(&[n, n, p], 0.0)?;
        let iota_n = self.iota_p(n)?;
        let mut info = self.fullf(&[p], 0.0)?;
        let mut pivs: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
        pivs[n - 1] = Some(self.fullf(&[p], (n - 1) as f64)?);

        for k0 in (1..n).rev() {
            let kk = k0 - 1;
            // Pivot search: first argmax of |A[0..=kk, k0]| (tenferro-rs#1976: no argmax op).
            let col = self.sl(&a, &[0, k0, 0], &[kk + 1, k0 + 1, p])?;
            let col = self.rs(&col, &[kk + 1, p])?;
            let mag = self.magnitude(&col)?;
            let colmax = self.s().reduce_max(&mag, &[0])?;
            let colmax_b = self.bc(&colmax, &[kk + 1, p], &[1])?;
            let eq = self.cmp(&mag, &colmax_b, CompareDir::Eq)?;
            let iota_k = self.iota_p(kk + 1)?;
            let big = self.fullf(&[kk + 1, p], 1.0e30)?;
            let cand = self.select(&eq, &iota_k, &big)?;
            let kp = self.s().reduce_min(&cand, &[0])?;
            let cap = self.fullf(&[p], kk as f64)?;
            let kp = self.s().minimum(&kp, &cap)?; // NaN columns would leave `big` behind
            let zero_f = self.fullf(&[p], 0.0)?;
            let is_zero = self.cmp(&colmax, &zero_f, CompareDir::Eq)?;
            let kp = self.select(&is_zero, &cap, &kp)?; // no swap on a zero column
            let info_is_unset = self.cmp(&info, &zero_f, CompareDir::Eq)?;
            let row_f = self.fullf(&[p], (kk + 1) as f64)?;
            let first = self.select(&is_zero, &row_f, &zero_f)?;
            info = self.select(&info_is_unset, &first, &info)?;

            // Row then column swap kk <-> kp with batch-varying kp (tenferro-rs#2008).
            let kp_b = self.bc(&kp, &[n, p], &[1])?;
            let oh = self.cmp(&iota_n, &kp_b, CompareDir::Eq)?; // [n, P] bool
            let oh_rows = self.bc(&oh, &[n, n, p], &[0, 2])?;
            let sel = self.select(&oh_rows, &a, &zeros3)?;
            let row_kp = self.s().reduce_sum(&sel, &[0])?; // [n(j), P]
            let row_kk = self.sl(&a, &[kk, 0, 0], &[kk + 1, n, p])?;
            let row_kk = self.rs(&row_kk, &[n, p])?;
            let m_kk = self.mask3(n, n, |i, _| i == kk)?;
            let row_kp3 = self.bc(&row_kp, &[n, n, p], &[1, 2])?;
            let row_kk3 = self.bc(&row_kk, &[n, n, p], &[1, 2])?;
            let inner = self.select(&oh_rows, &row_kk3, &a)?;
            let a1 = self.select(&m_kk, &row_kp3, &inner)?;

            let oh_cols = self.bc(&oh, &[n, n, p], &[1, 2])?;
            let sel = self.select(&oh_cols, &a1, &zeros3)?;
            let col_kp = self.s().reduce_sum(&sel, &[1])?; // [n(i), P]
            let col_kk = self.sl(&a1, &[0, kk, 0], &[n, kk + 1, p])?;
            let col_kk = self.rs(&col_kk, &[n, p])?;
            let m_kkc = self.mask3(n, n, |_, j| j == kk)?;
            let col_kp3 = self.bc(&col_kp, &[n, n, p], &[0, 2])?;
            let col_kk3 = self.bc(&col_kk, &[n, n, p], &[0, 2])?;
            let inner = self.select(&oh_cols, &col_kk3, &a1)?;
            let a2 = self.select(&m_kkc, &col_kp3, &inner)?;

            let a4 = if kk >= 1 {
                let tau = self.sl(&a2, &[kk, k0, 0], &[kk + 1, k0 + 1, p])?;
                let tau = self.rs(&tau, &[p])?;
                let one = self.full(&[p], 1.0)?;
                let alpha = self.div(&one, &tau)?;
                let alpha_k = self.bc(&alpha, &[kk, p], &[1])?;
                let x = self.sl(&a2, &[0, k0, 0], &[kk, k0 + 1, p])?;
                let x = self.rs(&x, &[kk, p])?;
                let y = self.sl(&a2, &[0, kk, 0], &[kk, kk + 1, p])?;
                let y = self.rs(&y, &[kk, p])?;
                let t1 = self.mul(&alpha_k, &y)?;
                let t2 = self.mul(&alpha_k, &x)?;
                let blk = self.sl(&a2, &[0, 0, 0], &[kk, kk, p])?;
                let xi = self.bc(&x, &[kk, kk, p], &[0, 2])?;
                let yi = self.bc(&y, &[kk, kk, p], &[0, 2])?;
                let t1j = self.bc(&t1, &[kk, kk, p], &[1, 2])?;
                let t2j = self.bc(&t2, &[kk, kk, p], &[1, 2])?;
                let xt = self.mul(&xi, &t1j)?;
                let yt = self.mul(&yi, &t2j)?;
                let acc = self.add(&blk, &xt)?;
                let u = self.sub(&acc, &yt)?; // (c + x t1) - y t2, as pfapack
                                              // Keep the block exactly skew: upper strict from `u`, lower = -u^T, diag 0.
                let ut = self.s().transpose(&u, &[1, 0, 2])?;
                let nut = self.neg(&ut)?;
                let zk = self.full(&[kk, kk, p], 0.0)?;
                let upper = self.mask3(kk, kk, |i, j| i < j)?;
                let lower = self.mask3(kk, kk, |i, j| i > j)?;
                let low_part = self.select(&lower, &nut, &zk)?;
                let unew = self.select(&upper, &u, &low_part)?;
                let padded = self.pad(&unew, &[0, 0, 0], &[(n - kk) as i64, (n - kk) as i64, 0])?;
                let in_block = self.mask3(n, n, |i, j| i < kk && j < kk)?;
                let a3 = self.select(&in_block, &padded, &a2)?;
                // Multipliers: column k0, rows 0..kk scaled by alpha.
                let xs = self.mul(&x, &alpha_k)?;
                let xs3 = self.rs(&xs, &[kk, 1, p])?;
                let padded = self.pad(
                    &xs3,
                    &[0, k0 as i64, 0],
                    &[(n - kk) as i64, (n - k0 - 1) as i64, 0],
                )?;
                let col_mask = self.mask3(n, n, |i, j| i < kk && j == k0)?;
                self.select(&col_mask, &padded, &a3)?
            } else {
                a2
            };
            // Zero-pivot planes are frozen (pfapack `continue`s before swapping/updating).
            let frozen = self.bc(&is_zero, &[n, n, p], &[2])?;
            a = self.select(&frozen, &a, &a4)?;
            pivs[kk] = Some(kp);
        }
        let pivs = pivs
            .into_iter()
            .map(|x| x.expect("all pivots set"))
            .collect();
        Ok((a, pivs, info))
    }

    // ----- stage 2: Pfaffian ---------------------------------------------------------------

    fn pfaffian(&mut self, a: &Tensor, pivs: &[Tensor]) -> R<Tensor> {
        let (n, p) = (self.n, self.p);
        let mut pf = self.full(&[p], 1.0)?;
        for i in (0..n - 1).step_by(2) {
            let e = self.sl(a, &[i, i + 1, 0], &[i + 1, i + 2, p])?;
            let e = self.rs(&e, &[p])?;
            pf = self.mul(&pf, &e)?;
        }
        let mut sign = self.full(&[p], 1.0)?;
        for (k, piv) in pivs.iter().enumerate() {
            let kf = self.fullf(&[p], k as f64)?;
            let same = self.cmp(piv, &kf, CompareDir::Eq)?;
            let flipped = self.neg(&sign)?;
            sign = self.select(&same, &sign, &flipped)?;
        }
        self.mul(&sign, &pf)
    }

    // ----- stage 3: inverse ----------------------------------------------------------------

    fn inverse(&mut self, a: &Tensor, pivs: &[Tensor]) -> R<Tensor> {
        let (n, p) = (self.n, self.p);
        let m = n - 1;
        // X = (I + N)^-1 for the unit upper-triangular S = A[0..m, 1..n].
        let s = self.sl(a, &[0, 1, 0], &[m, n, p])?;
        let zm = self.full(&[m, m, p], 0.0)?;
        let strict = self.mask3(m, m, |i, j| i < j)?;
        let nn = self.select(&strict, &s, &zm)?;
        // Back substitution (I + N) X = I, from the last row up: X_i = e_i - sum_{j>i} N_ij X_j.
        // (The former explicit expansion (I+N)^-1 = prod_j (I + (-N)^(2^j)) is mathematically
        // exact but numerically unstable: the entries of the powers of the multiplier matrix
        // grow exponentially with n and the factors cancel, so the residual grew from 3e-14 at
        // n = 32 to 1e-8 at n = 128 (issue #466). Substitution has the stability of
        // `dtrtri`/pfapack.)
        let mut x_rows: Option<Tensor> = None; // rows i+1..m as [k, m, P]
        let ident_row = |ops: &mut Self, i: usize| -> R<Tensor> {
            let mut data = vec![T::from_parts(0.0, 0.0); m];
            data[i] = T::from_parts(1.0, 0.0);
            let e = T::make_tensor(vec![m], data)?;
            ops.bc(&e, &[m, p], &[0])
        };
        for i in (0..m).rev() {
            let e_i = ident_row(self, i)?;
            let row = match &x_rows {
                None => e_i,
                Some(sub) => {
                    let k = m - 1 - i;
                    let coef = self.sl(&nn, &[i, i + 1, 0], &[i + 1, m, p])?;
                    let coef = self.rs(&coef, &[k, p])?;
                    let coef3 = self.bc(&coef, &[k, m, p], &[0, 2])?;
                    let prod = self.mul(&coef3, sub)?;
                    let sum = self.s().reduce_sum(&prod, &[0])?; // [m, P]
                    self.sub(&e_i, &sum)?
                }
            };
            let row3 = self.rs(&row, &[1, m, p])?;
            x_rows = Some(match x_rows {
                None => row3,
                Some(sub) => {
                    let parts = [&row3, &sub];
                    self.s().concatenate(&parts, 0)?
                }
            });
        }
        let x = x_rows.expect("m >= 1");
        // M = [X 0; 0 1] with the last column e_{n-1}.
        let mpad = self.pad(&x, &[0, 0, 0], &[1, 1, 0])?;
        let last = self.mask3(n, n, |i, j| i == n - 1 && j == n - 1)?;
        let one_n = self.full(&[n, n, p], 1.0)?;
        let mm = self.select(&last, &one_n, &mpad)?;

        // vt[i] = -A[i, i+1].
        let mut vt = Vec::with_capacity(m);
        for i in 0..m {
            let e = self.sl(a, &[i, i + 1, 0], &[i + 1, i + 2, p])?;
            let e = self.rs(&e, &[p])?;
            vt.push(self.neg(&e)?);
        }
        // Skew-tridiagonal solve C = T^-1 M, rows as [n(cols), P] vectors.
        let row = |ops: &mut Self, t: &Tensor, i: usize| -> R<Tensor> {
            let r = ops.sl(t, &[i, 0, 0], &[i + 1, n, p])?;
            ops.rs(&r, &[n, p])
        };
        let bvec = |ops: &mut Self, v: &Tensor| ops.bc(v, &[n, p], &[1]);
        let mut rows: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
        let b0 = row(self, &mm, 0)?;
        let d0 = bvec(self, &vt[0])?; // vt[0] = -T0; divide by -vt[0] = T0 handled below
        let nd0 = self.neg(&d0)?;
        rows[1] = Some(self.div(&b0, &nd0)?);
        for i in (2..n).step_by(2) {
            let bi = row(self, &mm, i)?;
            let v = bvec(self, &vt[i - 1])?;
            let pv = self.mul(rows[i - 1].as_ref().expect("odd row computed"), &v)?;
            let num = self.sub(&bi, &pv)?;
            let d = bvec(self, &vt[i])?;
            let nd = self.neg(&d)?;
            rows[i + 1] = Some(self.div(&num, &nd)?);
        }
        let bl = row(self, &mm, n - 1)?;
        let dl = bvec(self, &vt[n - 2])?;
        rows[n - 2] = Some(self.div(&bl, &dl)?);
        let mut i = n as isize - 3;
        while i >= 1 {
            let iu = i as usize;
            let bi = row(self, &mm, iu)?;
            let v = bvec(self, &vt[iu])?;
            let pv = self.mul(rows[iu + 1].as_ref().expect("even row computed"), &v)?;
            let num = self.add(&bi, &pv)?;
            let d = bvec(self, &vt[iu - 1])?;
            rows[iu - 1] = Some(self.div(&num, &d)?);
            i -= 2;
        }
        let rows3: Vec<Tensor> = rows
            .into_iter()
            .map(|r| {
                let r = r.expect("all rows computed");
                self.rs(&r, &[1, n, p])
            })
            .collect::<R<_>>()?;
        let refs: Vec<&Tensor> = rows3.iter().collect();
        let c = self.s().concatenate(&refs, 0)?; // [n, n, P]

        // W = M^T C, then the symmetric pivot permutation Out = Q^T W Q.
        let w = self.bmm_tn(&mm, &c)?;
        let iota_n = self.iota_p(n)?;
        let mut idx = self.iota_p(n)?;
        let zeros_np = self.fullf(&[n, p], 0.0)?;
        for (j, piv) in pivs.iter().enumerate() {
            // swap idx[j] <-> idx[piv[j]]
            let piv_b = self.bc(piv, &[n, p], &[1])?;
            let oh = self.cmp(&iota_n, &piv_b, CompareDir::Eq)?;
            let sel = self.select(&oh, &idx, &zeros_np)?;
            let idx_t = self.s().reduce_sum(&sel, &[0])?; // [P]
            let idx_j = self.sl(&idx, &[j, 0], &[j + 1, p])?;
            let idx_j = self.rs(&idx_j, &[p])?;
            let idx_t_b = self.bc(&idx_t, &[n, p], &[1])?;
            let idx_j_b = self.bc(&idx_j, &[n, p], &[1])?;
            let jmask = {
                let mut data = vec![false; n];
                data[j] = true;
                let mk =
                    Tensor::from_typed::<bool>(TypedTensor::from_vec_col_major(vec![n], data)?);
                self.s().broadcast_in_dim(&mk, &[n, p], &[0])?
            };
            let inner = self.select(&oh, &idx_j_b, &idx)?;
            idx = self.select(&jmask, &idx_t_b, &inner)?;
        }
        // Q[i, c, P] = (i == idx[c, P]).
        let i3 = {
            let v = <f64 as Impl>::make_tensor(vec![n], (0..n).map(|i| i as f64).collect())?;
            self.s().broadcast_in_dim(&v, &[n, n, p], &[0])?
        };
        let idx3 = self.bc(&idx, &[n, n, p], &[1, 2])?;
        let hit = self.cmp(&i3, &idx3, CompareDir::Eq)?;
        let one_n = self.full(&[n, n, p], 1.0)?;
        let zero_n = self.full(&[n, n, p], 0.0)?;
        let q = self.select(&hit, &one_n, &zero_n)?;
        let wq = self.bmm(&w, &q)?;
        self.bmm_tn(&q, &wq)
    }
}

fn backend_err(err: impl std::fmt::Display) -> Error {
    Error::Backend(err.to_string())
}

fn solve<T: PfScalar>(
    s: &mut dyn BackendSession,
    n: usize,
    p: usize,
    planes: Tensor,
) -> R<(Tensor, Tensor, Tensor)> {
    let mut ops = Ops::<T> {
        sess: s,
        ops: 0,
        n,
        p,
        _marker: PhantomData,
    };
    let (a, pivs, info) = ops.ltl(planes)?;
    let pf_raw = ops.pfaffian(&a, &pivs)?;
    let inv_raw = ops.inverse(&a, &pivs)?;

    // Status and masking: zero pivot -> info, else non-finite pf -> -1, else 0.
    let zero_f = ops.fullf(&[p], 0.0)?;
    let failed_pivot = ops.cmp(&info, &zero_f, CompareDir::Gt)?;
    let finite = ops.is_finite(&pf_raw)?;
    let minus_one = ops.fullf(&[p], -1.0)?;
    let nonfinite_code = ops.select(&finite, &zero_f, &minus_one)?;
    let status = ops.select(&failed_pivot, &info, &nonfinite_code)?;
    let ok = ops.cmp(&status, &zero_f, CompareDir::Eq)?;
    let zero_t = ops.full(&[p], 0.0)?;
    // pf = 0 on a zero pivot, the computed (non-finite) value otherwise.
    let pf = ops.select(&failed_pivot, &zero_t, &pf_raw)?;
    let ok3 = ops.bc(&ok, &[n, n, p], &[2])?;
    let zeros3 = ops.full(&[n, n, p], 0.0)?;
    let inv = ops.select(&ok3, &inv_raw, &zeros3)?;
    if std::env::var_os("MVMC_GPU_NATIVE_OPCOUNT").is_some() {
        eprintln!("tenferro-native: n={n} planes={p} session ops={}", ops.ops);
    }
    Ok((pf, inv, status))
}

pub(crate) fn run<T: PfScalar>(
    planes: &[T],
    n: usize,
    count: usize,
) -> crate::Result<BatchOutput<T>> {
    let input = T::make_tensor(vec![n, n, count], planes.to_vec()).map_err(backend_err)?;
    let mut backend = CpuBackend::new();
    let (pf, inv, status) = backend
        .with_backend_session(|session| solve::<T>(session, n, count, input))
        .map_err(backend_err)?;
    let pf = T::read_vec(&pf).map_err(backend_err)?;
    let inv = T::read_vec(&inv).map_err(backend_err)?;
    let status = <f64 as Impl>::read_vec(&status)
        .map_err(backend_err)?
        .into_iter()
        .map(|c| PlaneStatus::from_code(c as i32))
        .collect();
    Ok((pf, inv, status))
}
