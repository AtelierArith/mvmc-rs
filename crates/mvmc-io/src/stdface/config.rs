//! StdFace configuration data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::error::{StdFaceError, Result};

/// StdFace configuration structure.
///
/// This structure represents a parsed StdFace configuration file,
/// containing all the parameters needed for mVMC calculations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StdFaceConfig {
    /// Lattice parameters
    pub lattice: LatticeConfig,

    /// Model parameters
    pub model: ModelConfig,

    /// Calculation parameters
    pub calculation: CalculationConfig,

    /// Optimization parameters
    pub optimization: OptimizationConfig,

    /// Monte Carlo parameters
    pub monte_carlo: MonteCarloConfig,

    /// Additional parameters
    pub additional: HashMap<String, String>,
}

/// Lattice configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeConfig {
    /// Lattice type (e.g., "chain", "Tetragonal")
    pub lattice_type: String,

    /// Lattice dimensions
    pub dimensions: Vec<usize>,

    /// Sublattice dimensions
    pub sub_dimensions: Vec<usize>,
}

/// Model configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type (e.g., "Hubbard", "Spin", "FermionHubbard")
    pub model_type: String,

    /// Model-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Calculation configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalculationConfig {
    /// Number of electrons/particles
    pub n_particles: Option<usize>,

    /// Total spin Sz
    pub total_sz: Option<i32>,

    /// Random seed
    pub random_seed: Option<u64>,
}

/// Optimization configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Number of SR optimization steps
    pub sr_steps: Option<usize>,

    /// SR reduction cutoff
    pub sr_reduction_cutoff: Option<f64>,

    /// SR stabilization delta
    pub sr_stabilization_delta: Option<f64>,

    /// SR step delta
    pub sr_step_delta: Option<f64>,
}

/// Monte Carlo configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of VMC samples
    pub vmc_samples: Option<usize>,

    /// VMC calculation mode
    pub vmc_calculation_mode: Option<usize>,
}

impl Default for StdFaceConfig {
    fn default() -> Self {
        Self {
            lattice: LatticeConfig {
                lattice_type: "chain".to_string(),
                dimensions: vec![],
                sub_dimensions: vec![],
            },
            model: ModelConfig {
                model_type: "Hubbard".to_string(),
                parameters: HashMap::new(),
            },
            calculation: CalculationConfig {
                n_particles: None, // Will be determined by model type
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

impl StdFaceConfig {
    /// Creates a new StdFace configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<()> {
        // Validate lattice dimensions
        if self.lattice.dimensions.is_empty() {
            return Err(StdFaceError::ValidationError {
                message: "Lattice dimensions cannot be empty".to_string(),
            });
        }

        for &dim in &self.lattice.dimensions {
            if dim == 0 {
                return Err(StdFaceError::ValidationError {
                    message: "Lattice dimensions must be positive".to_string(),
                });
            }
        }

        // Validate sublattice dimensions
        if self.lattice.sub_dimensions.len() != self.lattice.dimensions.len() {
            return Err(StdFaceError::ValidationError {
                message: "Sublattice dimensions must match lattice dimensions".to_string(),
            });
        }

        for (i, &sub_dim) in self.lattice.sub_dimensions.iter().enumerate() {
            if sub_dim == 0 || sub_dim > self.lattice.dimensions[i] {
                return Err(StdFaceError::ValidationError {
                    message: format!(
                        "Sublattice dimension {} must be positive and <= lattice dimension {}",
                        sub_dim, self.lattice.dimensions[i]
                    ),
                });
            }
        }

        // Validate model parameters
        match self.model.model_type.as_str() {
            "Hubbard" | "FermionHubbard" => {
                if !self.model.parameters.contains_key("t") {
                    return Err(StdFaceError::MissingParameter {
                        parameter: "t (hopping parameter)".to_string(),
                    });
                }
                if !self.model.parameters.contains_key("U") {
                    return Err(StdFaceError::MissingParameter {
                        parameter: "U (interaction parameter)".to_string(),
                    });
                }
            }
            "Spin" => {
                if !self.model.parameters.contains_key("J") {
                    return Err(StdFaceError::MissingParameter {
                        parameter: "J (exchange parameter)".to_string(),
                    });
                }
            }
            _ => {
                return Err(StdFaceError::UnsupportedModel {
                    model: self.model.model_type.clone(),
                });
            }
        }

        // Validate calculation parameters
        if let Some(n_particles) = self.calculation.n_particles {
            let total_sites: usize = self.lattice.dimensions.iter().product();

            // For spin models, particles should be 0
            if self.model.model_type == "Spin" && n_particles != 0 {
                return Err(StdFaceError::ValidationError {
                    message: format!(
                        "Spin models should have 0 particles, but got {}",
                        n_particles
                    ),
                });
            }

            // For other models, check capacity
            if self.model.model_type != "Spin" && n_particles > total_sites * 2 {
                return Err(StdFaceError::ValidationError {
                    message: format!(
                        "Number of particles ({}) exceeds maximum capacity ({} sites * 2)",
                        n_particles, total_sites
                    ),
                });
            }
        }

        Ok(())
    }

    /// Returns the total number of lattice sites.
    pub fn total_sites(&self) -> usize {
        self.lattice.dimensions.iter().product()
    }

    /// Returns the lattice dimension (1D, 2D, etc.).
    pub fn lattice_dimension(&self) -> usize {
        self.lattice.dimensions.len()
    }

    /// Returns true if the lattice is periodic.
    pub fn is_periodic(&self) -> bool {
        // For now, assume all lattices are periodic
        // This could be made configurable in the future
        true
    }

    /// Sets default particle count based on model type.
    pub fn set_default_particle_count(&mut self) {
        if self.calculation.n_particles.is_none() {
            self.calculation.n_particles = Some(match self.model.model_type.as_str() {
                "Spin" => 0, // No electrons for spin models
                "Hubbard" | "FermionHubbard" => self.total_sites(), // Half-filling
                _ => self.total_sites(), // Default to half-filling
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StdFaceConfig::new();
        assert_eq!(config.lattice.lattice_type, "chain");
        assert_eq!(config.lattice.dimensions, vec![] as Vec<usize>);
        assert_eq!(config.model.model_type, "Hubbard");
        assert_eq!(config.calculation.n_particles, Some(6));
    }

    #[test]
    fn test_validate_hubbard_config() {
        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![6];
        config.lattice.sub_dimensions = vec![2];
        config.model.parameters.insert("t".to_string(), 1.0);
        config.model.parameters.insert("U".to_string(), 4.0);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_parameters() {
        let config = StdFaceConfig::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_dimensions() {
        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![0];
        config.model.parameters.insert("t".to_string(), 1.0);
        config.model.parameters.insert("U".to_string(), 4.0);

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_total_sites() {
        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![6];
        assert_eq!(config.total_sites(), 6);

        let mut config2d = StdFaceConfig::new();
        config2d.lattice.dimensions = vec![4, 3];
        config2d.lattice.sub_dimensions = vec![2, 2];
        assert_eq!(config2d.total_sites(), 12);
    }

    #[test]
    fn test_lattice_dimension() {
        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![6];
        assert_eq!(config.lattice_dimension(), 1);

        let mut config2d = StdFaceConfig::new();
        config2d.lattice.dimensions = vec![4, 3];
        config2d.lattice.sub_dimensions = vec![2, 2];
        assert_eq!(config2d.lattice_dimension(), 2);
    }

    #[test]
    fn test_set_default_particle_count() {
        // Test Hubbard model
        let mut config_hubbard = StdFaceConfig::new();
        config_hubbard.lattice.dimensions = vec![6];
        config_hubbard.model.model_type = "Hubbard".to_string();
        config_hubbard.set_default_particle_count();
        assert_eq!(config_hubbard.calculation.n_particles, Some(6));

        // Test Spin model
        let mut config_spin = StdFaceConfig::new();
        config_spin.lattice.dimensions = vec![6];
        config_spin.model.model_type = "Spin".to_string();
        config_spin.set_default_particle_count();
        assert_eq!(config_spin.calculation.n_particles, Some(0));

        // Test that existing value is not overwritten
        let mut config_existing = StdFaceConfig::new();
        config_existing.lattice.dimensions = vec![6];
        config_existing.model.model_type = "Hubbard".to_string();
        config_existing.calculation.n_particles = Some(4);
        config_existing.set_default_particle_count();
        assert_eq!(config_existing.calculation.n_particles, Some(4));
    }

    #[test]
    fn test_validate_spin_model_particles() {
        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![6];
        config.model.model_type = "Spin".to_string();
        config.model.parameters.insert("J".to_string(), 1.0);
        config.calculation.n_particles = Some(0); // Correct for spin model

        assert!(config.validate().is_ok());

        // Test invalid particle count for spin model
        config.calculation.n_particles = Some(2);
        assert!(config.validate().is_err());
    }
}
