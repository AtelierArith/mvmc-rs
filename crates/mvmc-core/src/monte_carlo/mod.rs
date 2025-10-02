//! Monte Carlo sampling for variational Monte Carlo calculations
//!
//! This module provides Monte Carlo sampling algorithms for VMC calculations:
//! - Metropolis sampling for electron configurations
//! - Physical observables calculation
//! - Statistical analysis and error estimation

pub mod metropolis;
pub mod observables;
pub mod sampler;

pub use metropolis::{ElectronConfiguration, MetropolisSampler, MetropolisStep, SamplingResult};
pub use observables::{ObservableCalculator, PhysicalObservables};
pub use sampler::{Sampler, SamplingStatistics};
