//! Correlation function observable implementation.
//!
//! This module provides the `CorrelationCalculator` for calculating various
//! correlation functions for quantum lattice models.

use super::{Observable, ObservableError, Result, validate_configuration};
use crate::hamiltonian::Spin;
use crate::lattice::Lattice;

/// Correlation function calculator for quantum lattice models.
///
/// This calculator computes various correlation functions
/// for a given configuration.
///
/// # Examples
///
/// ```
/// use mvmc_physics::observables::{CorrelationCalculator, Observable};
/// use mvmc_physics::lattice::ChainLattice;
/// use mvmc_physics::hamiltonian::Spin;
///
/// let lattice = ChainLattice::new(6, true).unwrap();
/// let corr_calc = CorrelationCalculator::new(Box::new(lattice));
///
/// let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down, Spin::Up, Spin::Down];
/// let correlation = corr_calc.calculate(&config);
/// ```
#[derive(Debug)]
pub struct CorrelationCalculator {
    /// Lattice structure
    lattice: Box<dyn Lattice>,
}

impl CorrelationCalculator {
    /// Creates a new correlation calculator.
    ///
    /// # Arguments
    /// * `lattice` - Lattice structure
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::observables::CorrelationCalculator;
    /// use mvmc_physics::lattice::ChainLattice;
    ///
    /// let lattice = ChainLattice::new(6, true).unwrap();
    /// let corr_calc = CorrelationCalculator::new(Box::new(lattice));
    /// ```
    pub fn new(lattice: Box<dyn Lattice>) -> Self {
        Self { lattice }
    }

    /// Returns the lattice used for correlation calculation.
    pub fn lattice(&self) -> &dyn Lattice {
        self.lattice.as_ref()
    }

    /// Calculates the spin-spin correlation function.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `site1` - First site index
    /// * `site2` - Second site index
    ///
    /// # Returns
    /// The spin-spin correlation
    pub fn spin_spin_correlation(&self, config: &[Spin], site1: usize, site2: usize) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        if site1 >= config.len() || site2 >= config.len() {
            return Err(ObservableError::InvalidParameter {
                message: "Site indices out of bounds".to_string(),
            });
        }

        let spin1 = config[site1].value_f64();
        let spin2 = config[site2].value_f64();

        Ok(spin1 * spin2)
    }

    /// Calculates the average correlation function at a given distance.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `distance` - Distance between sites
    ///
    /// # Returns
    /// The average correlation at the given distance
    pub fn correlation_at_distance(&self, config: &[Spin], distance: f64) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let mut correlation_sum = 0.0;
        let mut count = 0;

        for i in 0..self.lattice.n_sites() {
            for j in 0..self.lattice.n_sites() {
                if (self.lattice.distance(i, j) - distance).abs() < 1e-10 {
                    let corr = self.spin_spin_correlation(config, i, j)?;
                    correlation_sum += corr;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok(correlation_sum / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates the structure factor.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `momentum` - Momentum vector (in units of 2π/L)
    ///
    /// # Returns
    /// The structure factor
    pub fn structure_factor(&self, config: &[Spin], momentum: &[f64]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        if momentum.len() != self.lattice.dimension() {
            return Err(ObservableError::InvalidParameter {
                message: "Momentum dimension must match lattice dimension".to_string(),
            });
        }

        let mut structure_factor = 0.0;

        for i in 0..self.lattice.n_sites() {
            for j in 0..self.lattice.n_sites() {
                let spin_i = config[i].value_f64();
                let spin_j = config[j].value_f64();

                // Calculate the phase factor
                let mut phase = 0.0;
                for (dim, &k) in momentum.iter().enumerate() {
                    let size = self.lattice.size()[dim];
                    let pos_i = (i % size) as f64;
                    let pos_j = (j % size) as f64;
                    phase += k * (pos_i - pos_j);
                }

                structure_factor += spin_i * spin_j * (2.0 * std::f64::consts::PI * phase).cos();
            }
        }

        Ok(structure_factor / self.lattice.n_sites() as f64)
    }

    /// Calculates the pair correlation function.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `max_distance` - Maximum distance to consider
    ///
    /// # Returns
    /// Vector of (distance, correlation) pairs
    pub fn pair_correlation_function(&self, config: &[Spin], max_distance: f64) -> Result<Vec<(f64, f64)>> {
        validate_configuration(config, self.lattice.as_ref())?;

        let mut correlations = Vec::new();

        for i in 0..self.lattice.n_sites() {
            for j in 0..self.lattice.n_sites() {
                let distance = self.lattice.distance(i, j);
                if distance <= max_distance {
                    let corr = self.spin_spin_correlation(config, i, j)?;
                    correlations.push((distance, corr));
                }
            }
        }

        // Group by distance and calculate average
        let mut grouped: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
        for (dist, corr) in correlations {
            let dist_key = (dist * 1000.0) as usize; // Round to avoid floating point issues
            grouped.entry(dist_key).or_insert_with(Vec::new).push(corr);
        }

        let mut result = Vec::new();
        for (dist_key, corrs) in grouped {
            let distance = dist_key as f64 / 1000.0;
            let avg_corr = corrs.iter().sum::<f64>() / corrs.len() as f64;
            result.push((distance, avg_corr));
        }

        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(result)
    }

    /// Calculates the correlation length.
    ///
    /// This is a simplified calculation that assumes exponential decay
    /// of correlations with distance.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The correlation length
    pub fn correlation_length(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let pair_corr = self.pair_correlation_function(config, 10.0)?;

        if pair_corr.len() < 2 {
            return Ok(0.0);
        }

        // Find the maximum correlation (excluding distance 0)
        let max_corr = pair_corr.iter()
            .filter(|(dist, _)| *dist > 0.0)
            .map(|(_, corr)| corr.abs())
            .fold(0.0, f64::max);

        if max_corr <= 0.0 {
            return Ok(0.0);
        }

        // Estimate correlation length as the distance where correlation
        // drops to 1/e of its maximum value
        let threshold = max_corr / std::f64::consts::E;

        for (distance, corr) in pair_corr {
            if corr.abs() <= threshold {
                return Ok(distance);
            }
        }

        Ok(0.0)
    }

    /// Calculates the staggered correlation function.
    ///
    /// This is useful for antiferromagnetic systems.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `distance` - Distance between sites
    ///
    /// # Returns
    /// The staggered correlation at the given distance
    pub fn staggered_correlation(&self, config: &[Spin], distance: f64) -> Result<f64> {
        validate_configuration(config, self.lattice.as_ref())?;

        let mut correlation_sum = 0.0;
        let mut count = 0;

        for i in 0..self.lattice.n_sites() {
            for j in 0..self.lattice.n_sites() {
                if (self.lattice.distance(i, j) - distance).abs() < 1e-10 {
                    let spin_i = config[i].value_f64();
                    let spin_j = config[j].value_f64();

                    // Determine sublattice based on site index
                    let sublattice_i = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let sublattice_j = if j % 2 == 0 { 1.0 } else { -1.0 };

                    correlation_sum += spin_i * spin_j * sublattice_i * sublattice_j;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Ok(correlation_sum / count as f64)
        } else {
            Ok(0.0)
        }
    }
}

impl Observable for CorrelationCalculator {
    fn name(&self) -> &str {
        "Correlation"
    }

    fn calculate(&self, config: &[Spin]) -> f64 {
        // Calculate the average nearest-neighbor correlation
        self.correlation_at_distance(config, 1.0).unwrap_or(0.0)
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
    fn test_correlation_calculator_creation() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        assert_eq!(corr_calc.name(), "Correlation");
    }

    #[test]
    fn test_spin_spin_correlation() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // Same site correlation
        let corr_00 = corr_calc.spin_spin_correlation(&config, 0, 0).unwrap();
        assert_abs_diff_eq!(corr_00, 1.0, epsilon = 1e-10);

        // Different sites
        let corr_01 = corr_calc.spin_spin_correlation(&config, 0, 1).unwrap();
        assert_abs_diff_eq!(corr_01, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_at_distance() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // Correlation at distance 0
        let corr_0 = corr_calc.correlation_at_distance(&config, 0.0).unwrap();
        assert_abs_diff_eq!(corr_0, 1.0, epsilon = 1e-10);

        // Correlation at distance 1
        let corr_1 = corr_calc.correlation_at_distance(&config, 1.0).unwrap();
        assert_abs_diff_eq!(corr_1, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_structure_factor() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];

        // Zero momentum
        let momentum = vec![0.0];
        let sf = corr_calc.structure_factor(&config, &momentum).unwrap();
        assert!(sf.is_finite());

        // Non-zero momentum
        let momentum = vec![1.0];
        let sf = corr_calc.structure_factor(&config, &momentum).unwrap();
        assert!(sf.is_finite());
    }

    #[test]
    fn test_pair_correlation_function() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let pair_corr = corr_calc.pair_correlation_function(&config, 2.0).unwrap();

        // Should have correlations at various distances
        assert!(!pair_corr.is_empty());

        // Check that distances are sorted
        for i in 1..pair_corr.len() {
            assert!(pair_corr[i].0 >= pair_corr[i-1].0);
        }
    }

    #[test]
    fn test_correlation_length() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let corr_length = corr_calc.correlation_length(&config).unwrap();

        // Should be non-negative
        assert!(corr_length >= 0.0);
    }

    #[test]
    fn test_staggered_correlation() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        // Perfect antiferromagnetic configuration
        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let staggered_corr = corr_calc.staggered_correlation(&config, 1.0).unwrap();

        // Should be positive for antiferromagnetic configuration
        assert!(staggered_corr > 0.0);
    }

    #[test]
    fn test_invalid_site_indices() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let result = corr_calc.spin_spin_correlation(&config, 0, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_momentum_dimension() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let momentum = vec![0.0, 0.0]; // Wrong dimension for 1D lattice
        let result = corr_calc.structure_factor(&config, &momentum);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_configuration() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let corr_calc = CorrelationCalculator::new(Box::new(lattice));

        // Invalid configuration (wrong length)
        let config = vec![Spin::Up, Spin::Down];
        let result = corr_calc.spin_spin_correlation(&config, 0, 1);
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
        fn prop_correlation_finite(config in prop::collection::vec(any::<Spin>(), 4..8)) {
            let lattice = ChainLattice::new(config.len(), true).unwrap();
            let corr_calc = CorrelationCalculator::new(Box::new(lattice));

            let correlation = corr_calc.calculate(&config);

            // Correlation should be finite
            prop_assert!(correlation.is_finite());
        }

        #[test]
        fn prop_spin_spin_correlation_symmetric(config in prop::collection::vec(any::<Spin>(), 4..8), i in 0usize..8, j in 0usize..8) {
            let lattice = ChainLattice::new(config.len(), true).unwrap();
            let corr_calc = CorrelationCalculator::new(Box::new(lattice));

            if i < config.len() && j < config.len() {
                let corr_ij = corr_calc.spin_spin_correlation(&config, i, j).unwrap();
                let corr_ji = corr_calc.spin_spin_correlation(&config, j, i).unwrap();

                // Spin-spin correlation should be symmetric
                prop_assert!((corr_ij - corr_ji).abs() < 1e-10);
            }
        }

        #[test]
        fn prop_structure_factor_finite(config in prop::collection::vec(any::<Spin>(), 4..8)) {
            let lattice = ChainLattice::new(config.len(), true).unwrap();
            let corr_calc = CorrelationCalculator::new(Box::new(lattice));

            let momentum = vec![0.0]; // 1D lattice
            let sf = corr_calc.structure_factor(&config, &momentum).unwrap();

            // Structure factor should be finite
            prop_assert!(sf.is_finite());
        }
    }
}
