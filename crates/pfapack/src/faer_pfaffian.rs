//! Non-destructive panelized Pfaffians with reusable caller-owned scratch.
//!
//! The strict lower triangle is authoritative. Complex matrices must satisfy
//! `A[j,i] = -A[i,j]`, without conjugation. Unlike the scalar reference API,
//! odd dimensions, zero pivots and numerical failures are explicit errors.
//! Sequential execution avoids creating a competing team inside QP workers.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::pfaffian::{pfaffian, pfaffian_scratch};
use faer::{MatRef, Par};
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

#[cfg(test)]
mod tests {
    use super::*;

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
