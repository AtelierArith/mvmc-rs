//! VMC calculation engine
//!
//! This module provides the main VMC (Variational Monte Carlo) calculation engine
//! that integrates wavefunction, Monte Carlo sampling, and optimization algorithms.

pub mod engine;
pub mod simple_heisenberg;
pub mod improved_heisenberg;
pub mod adaptive_heisenberg;
pub mod integration_test;
pub mod sr_optimization;

pub use engine::{VmcEngine, VmcResult, OptimizationResult, ExpectationResult};
pub use simple_heisenberg::SimpleHeisenbergVMC;
pub use improved_heisenberg::{ImprovedHeisenbergVMC, VmcStatistics, MultipleVmcStatistics};
pub use adaptive_heisenberg::{AdaptiveHeisenbergVMC, AdaptiveVmcStatistics};
pub use integration_test::{VmcIntegrationTest, VmcBenchmark, IntegrationTestResults, BenchmarkResults};

