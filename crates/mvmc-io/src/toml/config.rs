//! TOML configuration data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TOML configuration structure.
///
/// This structure represents a TOML configuration file
/// for mVMC calculations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TomlConfig {
    /// Lattice configuration
    pub lattice: LatticeConfig,

    /// Model configuration
    pub model: ModelConfig,

    /// Calculation configuration
    pub calculation: CalculationConfig,

    /// Optimization configuration
    pub optimization: OptimizationConfig,

    /// Monte Carlo configuration
    pub monte_carlo: MonteCarloConfig,

    /// Additional parameters
    #[serde(flatten)]
    pub additional: HashMap<String, toml::Value>,
}

/// Lattice configuration for TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeConfig {
    /// Lattice type
    pub lattice_type: String,

    /// Lattice dimensions
    pub dimensions: Vec<usize>,

    /// Sublattice dimensions
    pub sub_dimensions: Vec<usize>,
}

/// Model configuration for TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type
    pub model_type: String,

    /// Model parameters
    pub parameters: HashMap<String, f64>,
}

/// Calculation configuration for TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalculationConfig {
    /// Number of particles
    pub n_particles: Option<usize>,

    /// Total spin Sz
    pub total_sz: Option<i32>,

    /// Random seed
    pub random_seed: Option<u64>,
}

/// Optimization configuration for TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// SR optimization steps
    pub sr_steps: Option<usize>,

    /// SR reduction cutoff
    pub sr_reduction_cutoff: Option<f64>,

    /// SR stabilization delta
    pub sr_stabilization_delta: Option<f64>,

    /// SR step delta
    pub sr_step_delta: Option<f64>,
}

/// Monte Carlo configuration for TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// VMC samples
    pub vmc_samples: Option<usize>,

    /// VMC calculation mode
    pub vmc_calculation_mode: Option<usize>,
}

impl Default for TomlConfig {
    fn default() -> Self {
        Self {
            lattice: LatticeConfig {
                lattice_type: "chain".to_string(),
                dimensions: vec![6],
                sub_dimensions: vec![2],
            },
            model: ModelConfig {
                model_type: "Hubbard".to_string(),
                parameters: HashMap::new(),
            },
            calculation: CalculationConfig {
                n_particles: Some(6),
                total_sz: Some(0),
                random_seed: Some(1),
            },
            optimization: OptimizationConfig {
                sr_steps: Some(500),
                sr_reduction_cutoff: Some(1e-8),
                sr_stabilization_delta: Some(1e-2),
                sr_step_delta: Some(3e-3),
            },
            monte_carlo: MonteCarloConfig {
                vmc_samples: Some(100),
                vmc_calculation_mode: Some(0),
            },
            additional: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TomlConfig::default();
        assert_eq!(config.lattice.lattice_type, "chain");
        assert_eq!(config.lattice.dimensions, vec![6]);
        assert_eq!(config.model.model_type, "Hubbard");
    }
}
