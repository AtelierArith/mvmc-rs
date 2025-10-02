//! Physical observables for quantum lattice models.
//!
//! This module provides various physical observables that can be calculated
//! for quantum lattice models, such as energy, magnetization, and correlation functions.

pub mod energy;
pub mod magnetization;
pub mod correlation;

pub use energy::EnergyCalculator;
pub use magnetization::MagnetizationCalculator;
pub use correlation::CorrelationCalculator;

use crate::hamiltonian::{HubbardHamiltonian, Spin};
use crate::lattice::Lattice;

/// Trait for physical observables.
///
/// This trait defines the common interface for all observable calculators,
/// allowing for generic algorithms that work with any observable.
pub trait Observable {
    /// Returns the name of the observable.
    fn name(&self) -> &str;

    /// Calculates the observable value for a given configuration.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The observable value
    fn calculate(&self, config: &[Spin]) -> f64;

    /// Calculates the observable value with error estimation.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// (value, error) tuple
    fn calculate_with_error(&self, config: &[Spin]) -> (f64, f64) {
        let value = self.calculate(config);
        (value, 0.0) // Default: no error estimation
    }
}

/// Error types for observable calculations.
#[derive(Debug, thiserror::Error)]
pub enum ObservableError {
    #[error("Configuration length {config_len} does not match lattice size {lattice_size}")]
    ConfigurationLengthMismatch { config_len: usize, lattice_size: usize },

    #[error("Invalid observable parameter: {message}")]
    InvalidParameter { message: String },

    #[error("Calculation failed: {message}")]
    CalculationFailed { message: String },

    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),
}

/// Result type for observable operations.
pub type Result<T> = std::result::Result<T, ObservableError>;

/// Validates a configuration for observable calculations.
///
/// # Arguments
/// * `config` - Spin configuration
/// * `lattice` - Lattice structure
///
/// # Returns
/// * `Ok(())` - Configuration is valid
/// * `Err(ObservableError)` - Configuration is invalid
pub fn validate_configuration(config: &[Spin], lattice: &dyn Lattice) -> Result<()> {
    if config.len() != lattice.n_sites() {
        return Err(ObservableError::ConfigurationLengthMismatch {
            config_len: config.len(),
            lattice_size: lattice.n_sites(),
        });
    }

    Ok(())
}

/// Calculates the average of an observable over multiple configurations.
///
/// # Arguments
/// * `observable` - Observable calculator
/// * `configurations` - List of configurations
///
/// # Returns
/// (average, standard_deviation) tuple
pub fn calculate_average<O: Observable>(
    observable: &O,
    configurations: &[Vec<Spin>],
) -> (f64, f64) {
    if configurations.is_empty() {
        return (0.0, 0.0);
    }

    let values: Vec<f64> = configurations
        .iter()
        .map(|config| observable.calculate(config))
        .collect();

    let mean = values.iter().sum::<f64>() / values.len() as f64;

    let variance = values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;

    let std_dev = variance.sqrt();

    (mean, std_dev)
}

/// Calculates the correlation function between two sites.
///
/// # Arguments
/// * `config` - Spin configuration
/// * `site1` - First site index
/// * `site2` - Second site index
///
/// # Returns
/// The correlation value
pub fn calculate_correlation(config: &[Spin], site1: usize, site2: usize) -> f64 {
    if site1 >= config.len() || site2 >= config.len() {
        return 0.0;
    }

    let spin1 = config[site1].value_f64();
    let spin2 = config[site2].value_f64();

    spin1 * spin2
}

/// Calculates the structure factor for a given momentum.
///
/// # Arguments
/// * `config` - Spin configuration
/// * `lattice` - Lattice structure
/// * `momentum` - Momentum vector (in units of 2π/L)
///
/// # Returns
/// The structure factor value
pub fn calculate_structure_factor(
    config: &[Spin],
    lattice: &dyn Lattice,
    momentum: &[f64],
) -> f64 {
    if momentum.len() != lattice.dimension() {
        return 0.0;
    }

    let mut structure_factor = 0.0;

    for i in 0..lattice.n_sites() {
        for j in 0..lattice.n_sites() {
            let spin_i = config[i].value_f64();
            let spin_j = config[j].value_f64();

            // Calculate the phase factor
            let mut phase = 0.0;
            for (dim, &k) in momentum.iter().enumerate() {
                let size = lattice.size()[dim];
                let pos_i = (i % size) as f64;
                let pos_j = (j % size) as f64;
                phase += k * (pos_i - pos_j);
            }

            structure_factor += spin_i * spin_j * (2.0 * std::f64::consts::PI * phase).cos();
        }
    }

    structure_factor / lattice.n_sites() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ChainLattice;
    use crate::hamiltonian::HubbardHamiltonian;

    #[test]
    fn test_validate_configuration() {
        let lattice = ChainLattice::new(4, true).unwrap();

        // Valid configuration
        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        assert!(validate_configuration(&config, &lattice).is_ok());

        // Invalid configuration (wrong length)
        let config = vec![Spin::Up, Spin::Down];
        assert!(validate_configuration(&config, &lattice).is_err());
    }

    #[test]
    fn test_calculate_correlation() {
        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // Same site correlation
        assert_eq!(calculate_correlation(&config, 0, 0), 1.0);

        // Different sites
        assert_eq!(calculate_correlation(&config, 0, 1), -1.0);
        assert_eq!(calculate_correlation(&config, 0, 2), 1.0);

        // Out of bounds
        assert_eq!(calculate_correlation(&config, 0, 10), 0.0);
    }

    #[test]
    fn test_calculate_structure_factor() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // Zero momentum
        let momentum = vec![0.0];
        let sf = calculate_structure_factor(&config, &lattice, &momentum);
        assert!(sf.is_finite());

        // Non-zero momentum
        let momentum = vec![1.0];
        let sf = calculate_structure_factor(&config, &lattice, &momentum);
        assert!(sf.is_finite());
    }

    #[test]
    fn test_calculate_average() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let configurations = vec![
            vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down],
            vec![Spin::Down, Spin::Up, Spin::Down, Spin::Up],
            vec![Spin::Up, Spin::Up, Spin::Down, Spin::Down],
        ];

        let (mean, std_dev) = calculate_average(&energy_calc, &configurations);

        assert!(mean.is_finite());
        assert!(std_dev >= 0.0);
    }
}
