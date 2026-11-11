//! Non-destructive panelized Pfaffians with reusable caller-owned scratch.
//!
//! The strict lower triangle is authoritative. Complex matrices must satisfy
//! `A[j,i] = -A[i,j]`, without conjugation. Unlike the scalar reference API,
//! odd dimensions, zero pivots and numerical failures are explicit errors.
//! Sequential execution avoids creating a competing team inside QP workers.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{factor, inverse};
use faer::linalg::pfaffian::{pfaffian, pfaffian_scratch};
use faer::{MatMut, MatRef, Par, ParShape};
use num_complex::Complex64;

pub use faer::linalg::pfaffian::PfaffianError;

/// Reusable workspace for real matrices of one fixed dimension.
pub struct RealPfaffianWorkspace {
    n: usize,
    memory: MemBuffer,
}

impl RealPfaffianWorkspace {
    /// Allocate scratch once, outside the evaluation loop.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            memory: MemBuffer::new(pfaffian_scratch::<f64>(n)),
        }
    }

    /// Evaluate a compact column-major matrix, leaving its input untouched.
    ///
    /// Panics if `a.len()` is not the square of the workspace dimension.
    pub fn compute(&mut self, a: &[f64]) -> Result<f64, PfaffianError> {
        assert_eq!(a.len(), self.n * self.n);
        pfaffian(
            MatRef::from_column_major_slice(a, self.n, self.n),
            Par::Seq,
            MemStack::new(&mut self.memory),
        )
    }
}

/// Reusable workspace for complex skew-symmetric matrices of one fixed dimension.
pub struct ComplexPfaffianWorkspace {
    n: usize,
    memory: MemBuffer,
}

impl ComplexPfaffianWorkspace {
    /// Allocate scratch once, outside the evaluation loop.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            memory: MemBuffer::new(pfaffian_scratch::<Complex64>(n)),
        }
    }

    /// Evaluate a compact column-major matrix, leaving its input untouched.
    ///
    /// Panics if `a.len()` is not the square of the workspace dimension.
    pub fn compute(&mut self, a: &[Complex64]) -> Result<Complex64, PfaffianError> {
        assert_eq!(a.len(), self.n * self.n);
        pfaffian(
            MatRef::from_column_major_slice(a, self.n, self.n),
            Par::Seq,
            MemStack::new(&mut self.memory),
        )
    }
}

// The Pfaffian provider does not expose its elimination factors. Use faer's
// blocked LU/inverse for the inverse instead of factoring a second time with
// the scalar LTL kernel. All three operations share one scratch allocation;
// no team or allocation is created by compute().
macro_rules! inverse_workspace {
    ($name:ident, $scalar:ty, $finite:expr) => {
        /// Reusable sequential faer workspace for a Pfaffian and true inverse.
        ///
        /// Pfaffians use panelized skew elimination; inverses use blocked LU.
        /// The scalar LTL routines remain the C-order reference implementation.
        pub struct $name {
            n: usize,
            memory: MemBuffer,
            lu: Vec<$scalar>,
            perm: Vec<usize>,
            perm_inv: Vec<usize>,
        }

        impl $name {
            /// Allocate all scratch for matrices of dimension `n`.
            pub fn new(n: usize) -> Self {
                let req = faer::dyn_stack::StackReq::any_of(&[
                    pfaffian_scratch::<$scalar>(n),
                    factor::lu_in_place_scratch::<usize, $scalar>(
                        n,
                        n,
                        ParShape::Seq,
                        Default::default(),
                    ),
                    inverse::inverse_scratch::<usize, $scalar>(n, ParShape::Seq),
                ]);
                Self {
                    n,
                    memory: MemBuffer::new(req),
                    lu: vec![<$scalar>::default(); n * n],
                    perm: vec![0; n],
                    perm_inv: vec![0; n],
                }
            }

            /// Matrix dimension accepted by this workspace.
            pub fn n(&self) -> usize {
                self.n
            }

            /// Compute Pfaffian and true inverse of compact column-major `a`.
            ///
            /// Strict lower-triangle input is authoritative for both outputs.
            /// Input is preserved; `out` is unspecified on error. Nonfinite
            /// inverses are errors even when the Pfaffian itself is finite.
            /// Panics unless both slices have exactly `n*n` entries.
            pub fn compute(
                &mut self,
                a: &[$scalar],
                out: &mut [$scalar],
            ) -> Result<$scalar, PfaffianError> {
                let n = self.n;
                assert_eq!(a.len(), n * n);
                assert_eq!(out.len(), n * n);
                let value = pfaffian(
                    MatRef::from_column_major_slice(a, n, n),
                    Par::Seq,
                    MemStack::new(&mut self.memory),
                )?;
                if n == 0 {
                    return Ok(value);
                }
                for j in 0..n {
                    self.lu[j + j * n] = <$scalar>::default();
                    for i in (j + 1)..n {
                        self.lu[i + j * n] = a[i + j * n];
                        self.lu[j + i * n] = -a[i + j * n];
                    }
                }
                let (_, perm) = factor::lu_in_place(
                    MatMut::from_column_major_slice_mut(&mut self.lu, n, n),
                    &mut self.perm,
                    &mut self.perm_inv,
                    Par::Seq,
                    MemStack::new(&mut self.memory),
                    Default::default(),
                );
                for i in 0..n {
                    let diagonal = self.lu[i + i * n];
                    if !($finite)(diagonal) {
                        return Err(PfaffianError::NumericalFailure);
                    }
                    if diagonal == <$scalar>::default() {
                        return Err(PfaffianError::ZeroPivot { index: i });
                    }
                }
                let lu = MatRef::from_column_major_slice(&self.lu, n, n);
                inverse::inverse(
                    MatMut::from_column_major_slice_mut(out, n, n),
                    lu,
                    lu,
                    perm,
                    Par::Seq,
                    MemStack::new(&mut self.memory),
                );
                if out.iter().any(|&x| !($finite)(x)) {
                    return Err(PfaffianError::NumericalFailure);
                }
                // The exact inverse of a skew matrix is skew. Project LU
                // roundoff onto that structure, without conjugating complex
                // values. Halving before subtracting avoids finite overflow.
                for j in 0..n {
                    out[j + j * n] = <$scalar>::default();
                    for i in (j + 1)..n {
                        let value = out[i + j * n] * 0.5 - out[j + i * n] * 0.5;
                        out[i + j * n] = value;
                        out[j + i * n] = -value;
                    }
                }
                Ok(value)
            }
        }

        // Scratch is not mathematical state. Cloning allocates fresh scratch;
        // comparison and debugging describe only its configured dimension.
        impl Clone for $name {
            fn clone(&self) -> Self {
                Self::new(self.n)
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.n == other.n
            }
        }
        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("n", &self.n)
                    .finish_non_exhaustive()
            }
        }
    };
}

inverse_workspace!(RealPfaffianInverseWorkspace, f64, |x: f64| x.is_finite());
inverse_workspace!(
    ComplexPfaffianInverseWorkspace,
    Complex64,
    |x: Complex64| x.re.is_finite() && x.im.is_finite()
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_and_complex_inverse_residuals_across_panel_boundaries() {
        for n in [0, 2, 4, 24, 26, 64] {
            let mut a = vec![Complex64::default(); n * n];
            for j in 0..n {
                for i in 0..j {
                    let re = if i % 2 == 0 && j == i + 1 {
                        8.0
                    } else {
                        ((i * 17 + j * 31) % 23) as f64 / 23.0 - 0.5
                    };
                    let value = Complex64::new(re, ((i * 11 + j * 7) % 19) as f64 / 38.0);
                    a[i + j * n] = value;
                    a[j + i * n] = -value;
                }
            }
            let real: Vec<_> = a.iter().map(|z| z.re).collect();
            let mut inv = vec![Complex64::default(); n * n];
            let mut inv_real = vec![0.0; n * n];
            let mut complex_ws = ComplexPfaffianInverseWorkspace::new(n);
            let mut real_ws = RealPfaffianInverseWorkspace::new(n);
            let mut scalar = a.clone();
            let mut scalar_real = real.clone();
            let expected = crate::pfaffian_ltl_complex(&mut crate::SqMat::new(&mut scalar, n));
            let expected_real =
                crate::pfaffian_ltl_real(&mut crate::SqMat::new(&mut scalar_real, n));
            for _ in 0..2 {
                let pf = complex_ws.compute(&a, &mut inv).unwrap();
                let pf_real = real_ws.compute(&real, &mut inv_real).unwrap();
                assert!((pf - expected).norm() <= 1e-12 * expected.norm().max(1.0));
                assert!((pf_real - expected_real).abs() <= 1e-12 * expected_real.abs().max(1.0));
                for j in 0..n {
                    for i in 0..n {
                        let mut residual = -Complex64::new(f64::from(i == j), 0.0);
                        let mut residual_real = -f64::from(i == j);
                        for k in 0..n {
                            residual += a[i + k * n] * inv[k + j * n];
                            residual_real += real[i + k * n] * inv_real[k + j * n];
                        }
                        // Bounded, block-diagonally dominant inputs; 64-term dots.
                        assert!(residual.norm() <= 1e-12, "n={n} ({i},{j}) {residual}");
                        assert!(
                            residual_real.abs() <= 1e-12,
                            "n={n} ({i},{j}) {residual_real}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_errors_do_not_conflate_underflow_nonfinite_and_singular() {
        let mut out = [0.0; 16];
        let mut ws = RealPfaffianInverseWorkspace::new(4);
        assert!(matches!(
            ws.compute(&[0.0; 16], &mut out),
            Err(PfaffianError::ZeroPivot { .. })
        ));
        let mut a = [0.0; 16];
        a[1] = -1e-200;
        a[4] = 1e-200;
        a[11] = -1e-200;
        a[14] = 1e-200;
        assert_eq!(
            ws.compute(&a, &mut out),
            Err(PfaffianError::NumericalFailure)
        );
        let mut tiny = RealPfaffianInverseWorkspace::new(2);
        assert_eq!(
            tiny.compute(&[0.0, -1e-320, 1e-320, 0.0], &mut [0.0; 4]),
            Err(PfaffianError::NumericalFailure)
        );
        a[1] = f64::NAN;
        assert_eq!(
            ws.compute(&a, &mut out),
            Err(PfaffianError::NumericalFailure)
        );
    }

    #[test]
    fn boundary_and_failure_contracts() {
        assert_eq!(RealPfaffianWorkspace::new(0).compute(&[]), Ok(1.0));
        assert_eq!(
            RealPfaffianWorkspace::new(3).compute(&[0.0; 9]),
            Err(PfaffianError::OddDimension)
        );
        assert!(matches!(
            RealPfaffianWorkspace::new(4).compute(&[0.0; 16]),
            Err(PfaffianError::ZeroPivot { .. })
        ));
        assert_eq!(
            RealPfaffianWorkspace::new(2).compute(&[0.0, f64::NAN, 1.0, 0.0]),
            Err(PfaffianError::NumericalFailure)
        );
        // The upper triangle is not read by the provider.
        assert_eq!(
            RealPfaffianWorkspace::new(2).compute(&[0.0, -2.0, f64::NAN, 0.0]),
            Ok(2.0)
        );
    }

    #[test]
    fn complex_plain_transpose_sign_and_workspace_reuse() {
        let z = Complex64::new(0.0, 0.0);
        let [a, b, c, d, e, f] = [
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.25, 3.0),
            Complex64::new(4.0, 2.0),
            Complex64::new(-1.0, 1.0),
        ];
        let input = [z, -a, -b, -c, a, z, -d, -e, b, d, z, -f, c, e, f, z];
        // Independent 4x4 combinatorial formula.
        let expected = a * f - b * e + c * d;
        let mut ws = ComplexPfaffianWorkspace::new(4);
        for _ in 0..2 {
            let value = ws.compute(&input).unwrap();
            assert!((value - expected).norm() <= 1e-13 * expected.norm().max(1.0));
        }
    }

    #[test]
    fn panel_updates_match_scalar_reference() {
        // n=64 exercises more than one panel. Bounded integer generation is
        // independent of RNG libraries; diagonal 2x2 blocks improve conditioning.
        let n = 64;
        let mut input = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..j {
                let value = if i % 2 == 0 && j == i + 1 {
                    8.0
                } else {
                    ((i * 17 + j * 31) % 23) as f64 / 23.0 - 0.5
                };
                input[i + j * n] = value;
                input[j + i * n] = -value;
            }
        }
        let mut reference = input.clone();
        let expected = crate::pfaffian_ltl_real(&mut crate::SqMat::new(&mut reference, n));
        let value = RealPfaffianWorkspace::new(n).compute(&input).unwrap();
        // Different panel summation order over 32 elimination steps.
        assert!((value - expected).abs() <= 1e-12 * expected.abs());
    }
}
