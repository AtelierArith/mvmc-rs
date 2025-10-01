//! Magnetization observable implementation.
//!
//! This module provides the `MagnetizationCalculator` for calculating various
//! magnetization-related observables for quantum lattice models.

use super::{Observable, ObservableError, Result, validate_configuration};
use crate::hamiltonian::Spin;
use crate::lattice::Lattice;

/// Magnetization calculator for quantum lattice models.
///
/// This calculator computes various magnetization-related observables
/// for a given configuration.
///
/// # Examples
///
/// ```
/// use mvmc_physics::observables::MagnetizationCalculator;
/// use mvmc_physics::lattice::ChainLattice;
/// use mvmc_physics::hamiltonian::Spin;
///
/// let lattice = ChainLattice::new(6, true).unwrap();
/// let mag_calc = MagnetizationCalculator::new(Box::new(lattice));
///
/// let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down, Spin::Up, Spin::Down];
/// let magnetization = mag_calc.calculate(&config);
/// ```
#[derive(Debug)]
pub struct MagnetizationCalculator {
    /// Lattice structure
    lattice: Box<dyn Lattice>,
}

impl MagnetizationCalculator {
    /// Creates a new magnetization calculator.
    ///
    /// # Arguments
    /// * `lattice` - Lattice structure
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::observables::MagnetizationCalculator;
    /// use mvmc_physics::lattice::ChainLattice;
    ///
    /// let lattice = ChainLattice::new(6, true).unwrap();
    /// let mag_calc = MagnetizationCalculator::new(Box::new(lattice));
    /// ```
    pub fn new(lattice: Box<dyn Lattice>) -> Self {
        Self { lattice }
    }

    /// Returns the lattice used for magnetization calculation.
    pub fn lattice(&self) -> &dyn Lattice {
        self.lattice.as_ref()
    }

    /// Calculates the total magnetization (sum of all spins).
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The total magnetization
    pub fn total_magnetization(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let magnetization = config.iter().map(|spin| spin.value_f64()).sum();
        Ok(magnetization)
    }

    /// Calculates the magnetization per site.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The magnetization per site
    pub fn magnetization_per_site(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let total_mag = self.total_magnetization(config)?;
        let n_sites = self.lattice.n_sites();

        Ok(total_mag / n_sites as f64)
    }

    /// Calculates the absolute magnetization (magnitude of total magnetization).
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The absolute magnetization
    pub fn absolute_magnetization(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let total_mag = self.total_magnetization(config)?;
        Ok(total_mag.abs())
    }

    /// Calculates the magnetization in a specific direction.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `direction` - Direction index (0=x, 1=y, 2=z)
    ///
    /// # Returns
    /// The magnetization in the specified direction
    pub fn magnetization_direction(&self, config: &[Spin], direction: usize) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        if direction > 2 {
            return Err(ObservableError::InvalidParameter {
                message: "Direction must be 0, 1, or 2".to_string(),
            });
        }

        // For simplicity, we assume all magnetization is in the z-direction
        // In a more general implementation, we would track different spin components
        match direction {
            2 => self.total_magnetization(config), // z-direction
            _ => Ok(0.0), // x and y directions (not implemented)
        }
    }

    /// Calculates the staggered magnetization.
    ///
    /// This is useful for antiferromagnetic systems where the magnetization
    /// alternates between sublattices.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The staggered magnetization
    pub fn staggered_magnetization(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let mut staggered_mag = 0.0;

        for (i, &spin) in config.iter().enumerate() {
            // Determine sublattice based on site index
            let sublattice = if i % 2 == 0 { 1.0 } else { -1.0 };
            staggered_mag += spin.value_f64() * sublattice;
        }

        Ok(staggered_mag.abs())
    }

    /// Calculates the magnetization correlation function.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `distance` - Distance between sites
    ///
    /// # Returns
    /// The magnetization correlation at the given distance
    pub fn magnetization_correlation(&self, config: &[Spin], distance: usize) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let mut correlation = 0.0;
        let mut count = 0;

        for i in 0..self.lattice.n_sites() {
            for j in 0..self.lattice.n_sites() {
                if self.lattice.distance(i, j) == distance as f64 {
                    correlation += config[i].value_f64() * config[j].value_f64();
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok(correlation / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates the magnetic susceptibility.
    ///
    /// This is a simplified calculation that assumes the susceptibility
    /// is proportional to the magnetization variance.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The magnetic susceptibility
    pub fn magnetic_susceptibility(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        // This is a simplified implementation
        // In practice, susceptibility should be calculated from fluctuations
        let magnetization = self.total_magnetization(config)?;
        let n_sites = self.lattice.n_sites();

        // Rough estimate: susceptibility ∝ magnetization²
        Ok(magnetization.powi(2) / n_sites as f64)
    }
}

impl Observable for MagnetizationCalculator {
    fn name(&self) -> &str {
        "Magnetization"
    }

    fn calculate(&self, config: &[Spin]) -> f64 {
        self.total_magnetization(config).unwrap_or(0.0)
    }

    fn calculate_with_error(&self, config: &[Spin]) -> (f64, f64) {
        let value = self.calculate(config);
        (value, 0.0) // No error for single configuration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ChainLattice;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_magnetization_calculator_creation() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        assert_eq!(mag_calc.name(), "Magnetization");
    }

    #[test]
    fn test_total_magnetization() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let magnetization = mag_calc.total_magnetization(&config).unwrap();

        // 1 + (-1) + 1 + (-1) = 0
        assert_abs_diff_eq!(magnetization, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetization_per_site() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Up, Spin::Down, Spin::Down];
        let mag_per_site = mag_calc.magnetization_per_site(&config).unwrap();

        // (1 + 1 + (-1) + (-1)) / 4 = 0
        assert_abs_diff_eq!(mag_per_site, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_absolute_magnetization() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Up, Spin::Up, Spin::Up];
        let abs_mag = mag_calc.absolute_magnetization(&config).unwrap();

        // |1 + 1 + 1 + 1| = 4
        assert_abs_diff_eq!(abs_mag, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetization_direction() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // z-direction
        let mag_z = mag_calc.magnetization_direction(&config, 2).unwrap();
        assert_abs_diff_eq!(mag_z, 0.0, epsilon = 1e-10);

        // x and y directions (not implemented)
        let mag_x = mag_calc.magnetization_direction(&config, 0).unwrap();
        let mag_y = mag_calc.magnetization_direction(&config, 1).unwrap();
        assert_abs_diff_eq!(mag_x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mag_y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_staggered_magnetization() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        // Perfect antiferromagnetic configuration
        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let staggered_mag = mag_calc.staggered_magnetization(&config).unwrap();

        // |1*1 + (-1)*(-1) + 1*1 + (-1)*(-1)| = |1 + 1 + 1 + 1| = 4
        assert_abs_diff_eq!(staggered_mag, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetization_correlation() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // Correlation at distance 0 (same site)
        let corr_0 = mag_calc.magnetization_correlation(&config, 0).unwrap();
        assert_abs_diff_eq!(corr_0, 1.0, epsilon = 1e-10);

        // Correlation at distance 1 (nearest neighbors)
        let corr_1 = mag_calc.magnetization_correlation(&config, 1).unwrap();
        assert_abs_diff_eq!(corr_1, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetic_susceptibility() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Up, Spin::Up, Spin::Up];
        let susceptibility = mag_calc.magnetic_susceptibility(&config).unwrap();

        // Should be positive
        assert!(susceptibility > 0.0);
    }

    #[test]
    fn test_invalid_direction() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let result = mag_calc.magnetization_direction(&config, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_configuration() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

        // Invalid configuration (wrong length)
        let config = vec![Spin::Up, Spin::Down];
        let result = mag_calc.total_magnetization(&config);
        assert!(result.is_err());
    }
}

// Property-based tests using proptest
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::lattice::ChainLattice;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_magnetization_finite(config in prop::collection::vec(any::<Spin>(), 4..8)) {
            let lattice = ChainLattice::new(config.len(), true).unwrap();
            let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

            let magnetization = mag_calc.calculate(&config);

            // Magnetization should be finite
            prop_assert!(magnetization.is_finite());
        }

        #[test]
        fn prop_magnetization_per_site_scale(config in prop::collection::vec(any::<Spin>(), 4..8)) {
            let lattice = ChainLattice::new(config.len(), true).unwrap();
            let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

            let total_mag = mag_calc.total_magnetization(&config).unwrap();
            let mag_per_site = mag_calc.magnetization_per_site(&config).unwrap();

            // Magnetization per site should be total magnetization divided by number of sites
            let expected = total_mag / config.len() as f64;
            prop_assert!((mag_per_site - expected).abs() < 1e-10);
        }

        #[test]
        fn prop_absolute_magnetization_positive(config in prop::collection::vec(any::<Spin>(), 4..8)) {
            let lattice = ChainLattice::new(config.len(), true).unwrap();
            let mag_calc = MagnetizationCalculator::new(Box::new(lattice));

            let abs_mag = mag_calc.absolute_magnetization(&config).unwrap();

            // Absolute magnetization should be non-negative
            prop_assert!(abs_mag >= 0.0);
        }
    }
}
