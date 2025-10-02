//! VMC calculation engine
//!
//! This module provides the main VMC (Variational Monte Carlo) calculation engine
//! that integrates wavefunction, Monte Carlo sampling, and optimization algorithms.

pub mod engine;

pub use engine::{VmcEngine, VmcResult, OptimizationResult, ExpectationResult};

