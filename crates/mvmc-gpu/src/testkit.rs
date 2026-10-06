//! Test helpers shared by the root tests and the CUDA gate: deterministic plane generators,
//! a dense LU determinant and the invariants (`Pf^2 = det`, `||A A^-1 - I||`, skew
//! structure) used by all backends. Not part of the stable API.
#![allow(missing_docs)]

use crate::{BatchedPfaffian, PfScalar, PlaneStatus};
use num_complex::Complex64;

/// Scalar helpers needed by the generic test code.
pub trait TestScalar:
    PfScalar
    + std::ops::Mul<Output = Self>
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const COMPLEX: bool;
    fn from_pair(re: f64, im: f64) -> Self;
    fn modulus(self) -> f64;
    fn zero() -> Self;
    fn one() -> Self;
}

impl TestScalar for f64 {
    const COMPLEX: bool = false;
    fn from_pair(re: f64, _im: f64) -> Self {
        re
    }
    fn modulus(self) -> f64 {
        self.abs()
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl TestScalar for Complex64 {
    const COMPLEX: bool = true;
    fn from_pair(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn modulus(self) -> f64 {
        self.norm()
    }
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
}

/// Small LCG, uniform in (-1, 1).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// `planes` random skew-symmetric `n x n` planes (column-major, both triangles filled).
pub fn random_planes<T: TestScalar>(n: usize, planes: usize, seed: u64) -> Vec<T> {
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut out = vec![T::zero(); n * n * planes];
    for p in 0..planes {
        let a = &mut out[p * n * n..(p + 1) * n * n];
        for j in 0..n {
            for i in 0..j {
                let v = T::from_pair(rng.uniform(), if T::COMPLEX { rng.uniform() } else { 0.0 });
                a[i + j * n] = v;
                a[j + i * n] = -v;
            }
        }
    }
    out
}

/// Dense LU determinant with partial pivoting (column-major `n x n`, consumed).
pub fn det<T: TestScalar>(mut a: Vec<T>, n: usize) -> T {
    let mut det = T::one();
    for k in 0..n {
        let mut piv = k;
        let mut best = a[k + k * n].modulus();
        for i in k + 1..n {
            let v = a[i + k * n].modulus();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best == 0.0 {
            return T::zero();
        }
        if piv != k {
            for j in 0..n {
                a.swap(k + j * n, piv + j * n);
            }
            det = -det;
        }
        let d = a[k + k * n];
        det = det * d;
        for i in k + 1..n {
            let f = a[i + k * n] / d;
            for j in k + 1..n {
                let t = f * a[k + j * n];
                a[i + j * n] = a[i + j * n] - t;
            }
        }
    }
    det
}

/// Frobenius norm.
pub fn fro<T: TestScalar>(a: &[T]) -> f64 {
    a.iter().map(|x| x.modulus().powi(2)).sum::<f64>().sqrt()
}

/// `max_ij |(A B)_ij - delta_ij|` (column-major).
pub fn identity_residual<T: TestScalar>(a: &[T], b: &[T], n: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = T::zero();
            for k in 0..n {
                acc = acc + a[i + k * n] * b[k + j * n];
            }
            let want = if i == j { T::one() } else { T::zero() };
            worst = worst.max((acc - want).modulus());
        }
    }
    worst
}

/// Largest `|A[i,j] + A[j,i]|` relative to the matrix scale.
pub fn skew_defect<T: TestScalar>(a: &[T], n: usize) -> f64 {
    let scale = a
        .iter()
        .map(|x| x.modulus())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((a[i + j * n] + a[j + i * n]).modulus());
        }
    }
    worst / scale
}

/// Invariant report for one backend result against its input.
pub struct Invariants {
    pub max_identity_residual: f64,
    pub max_skew_defect: f64,
    pub max_pf2_det_rel: f64,
}

/// Check `Pf^2 = det`, `A * inv = I`, skew structure of `inv` for every `Ok` plane.
pub fn invariants<T: TestScalar>(planes: &[T], out: &BatchedPfaffian<T>) -> Invariants {
    let n = out.n;
    let nn = n * n;
    let mut inv = Invariants {
        max_identity_residual: 0.0,
        max_skew_defect: 0.0,
        max_pf2_det_rel: 0.0,
    };
    for p in 0..out.planes() {
        if out.status[p] != PlaneStatus::Ok {
            continue;
        }
        let a = &planes[p * nn..(p + 1) * nn];
        let ainv = &out.inv[p * nn..(p + 1) * nn];
        inv.max_identity_residual = inv.max_identity_residual.max(identity_residual(a, ainv, n));
        inv.max_skew_defect = inv.max_skew_defect.max(skew_defect(ainv, n));
        let d = det(a.to_vec(), n);
        let pf2 = out.pf[p] * out.pf[p];
        let rel = (pf2 - d).modulus() / d.modulus().max(f64::MIN_POSITIVE);
        inv.max_pf2_det_rel = inv.max_pf2_det_rel.max(rel);
    }
    inv
}
