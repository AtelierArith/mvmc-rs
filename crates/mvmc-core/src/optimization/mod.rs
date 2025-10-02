//! Optimization algorithms for variational Monte Carlo calculations
//!
//! This module provides optimization algorithms for VMC calculations:
//! - Stochastic Reconfiguration (SR) method
//! - Conjugate Gradient (CG) method
//! - Lanczos method for eigenvalue problems

pub mod conjugate_gradient;
pub mod lanczos;
pub mod stochastic_reconfiguration;

pub use conjugate_gradient::{ConjugateGradientSolver, CGSolver};
pub use lanczos::{LanczosSolver, LanczosEigenvalue};
pub use stochastic_reconfiguration::{SROptimizer, SRMatrix, SROptimizationResult};
