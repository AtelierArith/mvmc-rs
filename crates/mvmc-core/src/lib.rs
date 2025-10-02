//! Core library for mVMC (many-variable Variational Monte Carlo method)
//!
//! This crate provides the fundamental building blocks for VMC calculations:
//! - Type-safe wrappers for physical quantities
//! - Error handling
//! - Configuration and parameter management
//! - Wavefunction representations
//! - Core data structures and algorithms
//!
//! # Example
//!
//! ```
//! use mvmc_core::types::{SiteIndex, SiteCount, ElectronCount};
//! use mvmc_core::config::VmcParameters;
//! use mvmc_core::wavefunction::SlaterDeterminant;
//!
//! let nsite = SiteCount::new(4);
//! let ne = ElectronCount::new(2);
//!
//! let site = SiteIndex::new(0);
//! assert_eq!(site.get(), 0);
//!
//! let slater = SlaterDeterminant::new(nsite, ne);
//! ```

pub mod config;
pub mod error;
pub mod monte_carlo;
pub mod optimization;
pub mod types;
pub mod vmc;
pub mod wavefunction;

// Re-export commonly used items
pub use config::{MonteCarloParameters, SRParameters, VmcParameters, ParameterValidator};
pub use error::{Result, VmcError};
pub use types::{
    CalcMode, ElectronCount, LanczosMode, RandomSeed, SiteCount, SiteIndex, TwoSz,
};
pub use monte_carlo::{
    ElectronConfiguration, MetropolisSampler, MetropolisStep, ObservableCalculator,
    PhysicalObservables, SamplingStatistics, Sampler,
};
pub use monte_carlo::sampler::SamplingAnalysis;
pub use optimization::{
    ConjugateGradientSolver, CGSolver, LanczosEigenvalue, LanczosSolver,
    SROptimizationResult, SROptimizer, SRMatrix,
};
pub use vmc::{VmcEngine, VmcResult, OptimizationResult, ExpectationResult};
pub use wavefunction::{
    AmplitudeResult, CombinedWavefunction, PfaffianMatrix, PfaffianWavefunction,
    ProjectionCount, ProjectionOperator, RBMCounter, RBMParameters, RBMWavefunction,
    SlaterDeterminant, SlaterMatrix, log_projection_ratio, log_projection_value,
    make_projection_count, projection_ratio,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);

        assert_eq!(nsite.get(), 4);
        assert_eq!(ne.get(), 2);
    }

    #[test]
    fn test_error_usage() {
        let result: Result<i32> = Err(VmcError::invalid_param("test"));
        assert!(result.is_err());
    }

    #[test]
    fn test_type_safety() {
        // Demonstrate that we cannot accidentally mix types
        let site_idx = SiteIndex::new(0);
        let electron_count = ElectronCount::new(2);

        // These have different types and cannot be compared directly
        // This would not compile:
        // assert_eq!(site_idx, electron_count);

        assert_eq!(site_idx.get(), 0);
        assert_eq!(electron_count.get(), 2);
    }
}
