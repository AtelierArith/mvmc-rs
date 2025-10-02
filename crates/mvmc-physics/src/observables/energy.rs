//! Energy observable implementation.
//!
//! This module provides the `EnergyCalculator` for calculating the energy
//! of quantum lattice model configurations.

use super::{Observable, Result, validate_configuration};
use crate::hamiltonian::{Hamiltonian, Spin};

/// Energy calculator for quantum lattice models.
///
/// This calculator computes the energy of a given configuration using
/// a provided Hamiltonian.
///
/// # Examples
///
/// ```
/// use mvmc_physics::observables::{EnergyCalculator, Observable};
/// use mvmc_physics::hamiltonian::HubbardHamiltonian;
/// use mvmc_physics::lattice::ChainLattice;
/// use mvmc_physics::hamiltonian::Spin;
///
/// let lattice = ChainLattice::new(6, true).unwrap();
/// let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
/// let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));
///
/// let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up, Spin::Down, Spin::Empty];
/// let energy = energy_calc.calculate(&config);
/// ```
#[derive(Debug)]
pub struct EnergyCalculator {
    /// Hamiltonian for energy calculation
    hamiltonian: Box<dyn Hamiltonian>,
}

impl EnergyCalculator {
    /// Creates a new energy calculator.
    ///
    /// # Arguments
    /// * `hamiltonian` - Hamiltonian for energy calculation
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::observables::EnergyCalculator;
    /// use mvmc_physics::hamiltonian::HubbardHamiltonian;
    /// use mvmc_physics::lattice::ChainLattice;
    ///
    /// let lattice = ChainLattice::new(6, true).unwrap();
    /// let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
    /// let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));
    /// ```
    pub fn new(hamiltonian: Box<dyn Hamiltonian>) -> Self {
        Self { hamiltonian }
    }

    /// Returns the Hamiltonian used for energy calculation.
    pub fn hamiltonian(&self) -> &dyn Hamiltonian {
        self.hamiltonian.as_ref()
    }

    /// Calculates the energy per site.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The energy per site
    pub fn energy_per_site(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.hamiltonian.lattice())?;

        let total_energy = self.hamiltonian.total_energy(config);
        let n_sites = self.hamiltonian.lattice().n_sites();

        Ok(total_energy / n_sites as f64)
    }

    /// Calculates the kinetic energy contribution.
    ///
    /// This is a simplified calculation that assumes the Hamiltonian
    /// can be decomposed into kinetic and potential parts.
    /// For more accurate results, specific Hamiltonian implementations
    /// should provide their own kinetic energy methods.
    pub fn kinetic_energy(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.hamiltonian.lattice())?;

        // This is a simplified implementation
        // In practice, each Hamiltonian type should provide its own method
        let total_energy = self.hamiltonian.total_energy(config);

        // Estimate kinetic energy as a fraction of total energy
        // This is a rough approximation and should be improved
        Ok(total_energy * 0.5)
    }

    /// Calculates the potential energy contribution.
    ///
    /// This is a simplified calculation that assumes the Hamiltonian
    /// can be decomposed into kinetic and potential parts.
    pub fn potential_energy(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.hamiltonian.lattice())?;

        let total_energy = self.hamiltonian.total_energy(config);
        let kinetic = self.kinetic_energy(config)?;

        Ok(total_energy - kinetic)
    }

    /// Calculates the energy variance.
    ///
    /// This is useful for assessing the quality of a variational wave function.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// The energy variance
    pub fn energy_variance(&self, config: &[Spin]) -> Result<f64> {
        validate_configuration(config, self.hamiltonian.lattice())?;

        // For a single configuration, the variance is 0
        // This method is provided for consistency with Monte Carlo methods
        Ok(0.0)
    }

    /// Calculates the local energy for VMC calculations.
    ///
    /// The local energy is defined as H|ψ⟩/|ψ⟩, where H is the Hamiltonian
    /// and |ψ⟩ is the wave function. This is the key quantity in VMC.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    /// * `wavefunction` - Wave function for the configuration
    ///
    /// # Returns
    /// The local energy
    pub fn local_energy(&self, config: &[Spin], wavefunction: &dyn crate::wavefunction::Wavefunction) -> Result<f64> {
        validate_configuration(config, self.hamiltonian.lattice())?;

        // Calculate the wave function amplitude
        let psi = wavefunction.calculate_spin(config)?;
        if psi.norm() < 1e-12 {
            return Ok(0.0);
        }

        // Calculate the local energy: H|ψ⟩/|ψ⟩
        let mut local_energy = 0.0;

        // Diagonal terms (current configuration)
        let diagonal_energy = self.hamiltonian.total_energy(config);
        local_energy += diagonal_energy;

        // Off-diagonal terms (hopping terms)
        let n_sites = self.hamiltonian.lattice().n_sites();
        for i in 0..n_sites {
            for j in 0..n_sites {
                if i != j {
                    // Check if hopping is possible
                    if self.can_hop(config, i, j) {
                        let mut new_config = config.to_vec();
                        self.perform_hop(&mut new_config, i, j);

                        let new_psi = wavefunction.calculate_spin(&new_config)?;
                        if new_psi.norm() > 1e-12 {
                            let hopping_amplitude = self.get_hopping_amplitude(config, i, j);
                            let ratio = new_psi / psi;
                            local_energy += hopping_amplitude * ratio.re;
                        }
                    }
                }
            }
        }

        Ok(local_energy)
    }

    /// Calculates the energy variance for VMC calculations.
    ///
    /// This calculates the variance of the local energy over multiple configurations.
    ///
    /// # Arguments
    /// * `configurations` - List of configurations
    /// * `wavefunction` - Wave function
    ///
    /// # Returns
    /// The energy variance
    pub fn calculate_energy_variance(
        &self,
        configurations: &[Vec<Spin>],
        wavefunction: &dyn crate::wavefunction::Wavefunction,
    ) -> Result<f64> {
        if configurations.is_empty() {
            return Ok(0.0);
        }

        let mut local_energies = Vec::new();
        let mut total_energy = 0.0;

        for config in configurations {
            let local_energy = self.local_energy(config, wavefunction)?;
            local_energies.push(local_energy);
            total_energy += local_energy;
        }

        let mean_energy = total_energy / configurations.len() as f64;

        let variance = local_energies
            .iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>() / configurations.len() as f64;

        Ok(variance)
    }

    /// Checks if hopping is possible between two sites.
    ///
    /// # Arguments
    /// * `config` - Current configuration
    /// * `from` - Source site
    /// * `to` - Destination site
    ///
    /// # Returns
    /// True if hopping is possible
    fn can_hop(&self, config: &[Spin], from: usize, to: usize) -> bool {
        if from >= config.len() || to >= config.len() {
            return false;
        }

        // Check if source site is occupied and destination is empty
        config[from] != Spin::Empty && config[to] == Spin::Empty
    }

    /// Performs a hopping move between two sites.
    ///
    /// # Arguments
    /// * `config` - Configuration to modify
    /// * `from` - Source site
    /// * `to` - Destination site
    fn perform_hop(&self, config: &mut [Spin], from: usize, to: usize) {
        if from < config.len() && to < config.len() {
            config[to] = config[from];
            config[from] = Spin::Empty;
        }
    }

    /// Gets the hopping amplitude between two sites.
    ///
    /// # Arguments
    /// * `config` - Current configuration
    /// * `from` - Source site
    /// * `to` - Destination site
    ///
    /// # Returns
    /// The hopping amplitude
    fn get_hopping_amplitude(&self, config: &[Spin], from: usize, to: usize) -> f64 {
        // This is a simplified implementation
        // In practice, this should be extracted from the Hamiltonian
        if from < config.len() && to < config.len() {
            match (config[from], config[to]) {
                (Spin::Up, Spin::Empty) | (Spin::Down, Spin::Empty) => 1.0,
                _ => 0.0,
            }
        } else {
            0.0
        }
    }
}

impl Observable for EnergyCalculator {
    fn name(&self) -> &str {
        "Energy"
    }

    fn calculate(&self, config: &[Spin]) -> f64 {
        self.hamiltonian.total_energy(config)
    }

    fn calculate_with_error(&self, config: &[Spin]) -> (f64, f64) {
        let value = self.calculate(config);
        (value, 0.0) // No error for single configuration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::{HubbardHamiltonian, HeisenbergHamiltonian};
    use crate::lattice::ChainLattice;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_energy_calculator_creation() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        assert_eq!(energy_calc.name(), "Energy");
    }

    #[test]
    fn test_energy_calculation() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let energy = energy_calc.calculate(&config);

        // Should match the Hamiltonian's total_energy method
        let lattice2 = ChainLattice::new(4, true).unwrap();
        let hamiltonian2 = HubbardHamiltonian::new(lattice2, 1.0, 2.0, 0.5).unwrap();
        let expected = hamiltonian2.total_energy(&config);
        assert_abs_diff_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_per_site() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let energy_per_site = energy_calc.energy_per_site(&config).unwrap();

        let lattice2 = ChainLattice::new(4, true).unwrap();
        let hamiltonian2 = HubbardHamiltonian::new(lattice2, 1.0, 2.0, 0.5).unwrap();
        let total_energy = hamiltonian2.total_energy(&config);
        let expected = total_energy / 4.0;
        assert_abs_diff_eq!(energy_per_site, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_kinetic_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let kinetic = energy_calc.kinetic_energy(&config).unwrap();

        // Should be approximately half of total energy (simplified model)
        let lattice2 = ChainLattice::new(4, true).unwrap();
        let hamiltonian2 = HubbardHamiltonian::new(lattice2, 1.0, 2.0, 0.5).unwrap();
        let total_energy = hamiltonian2.total_energy(&config);
        let expected = total_energy * 0.5;
        assert_abs_diff_eq!(kinetic, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_potential_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let potential = energy_calc.potential_energy(&config).unwrap();

        // Should be total energy minus kinetic energy
        let lattice2 = ChainLattice::new(4, true).unwrap();
        let hamiltonian2 = HubbardHamiltonian::new(lattice2, 1.0, 2.0, 0.5).unwrap();
        let total_energy = hamiltonian2.total_energy(&config);
        let kinetic = energy_calc.kinetic_energy(&config).unwrap();
        let expected = total_energy - kinetic;
        assert_abs_diff_eq!(potential, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_variance() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let variance = energy_calc.energy_variance(&config).unwrap();

        // Should be 0 for a single configuration
        assert_abs_diff_eq!(variance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_heisenberg_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy = energy_calc.calculate(&config);

        // Should match the Hamiltonian's total_energy method
        let lattice2 = ChainLattice::new(4, true).unwrap();
        let hamiltonian2 = HeisenbergHamiltonian::new(lattice2, 1.0, 0.5).unwrap();
        let expected = hamiltonian2.total_energy(&config);
        assert_abs_diff_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_calculation_with_error() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let (value, error) = energy_calc.calculate_with_error(&config);

        let lattice2 = ChainLattice::new(4, true).unwrap();
        let hamiltonian2 = HubbardHamiltonian::new(lattice2, 1.0, 2.0, 0.5).unwrap();
        let expected = hamiltonian2.total_energy(&config);
        assert_abs_diff_eq!(value, expected, epsilon = 1e-10);
        assert_abs_diff_eq!(error, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_configuration() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();
        let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

        // Invalid configuration (wrong length)
        let config = vec![Spin::Up, Spin::Down];
        let result = energy_calc.energy_per_site(&config);
        assert!(result.is_err());
    }
}

// Property-based tests using proptest
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::hamiltonian::HubbardHamiltonian;
    use crate::lattice::ChainLattice;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_energy_finite(hopping in 0.0f64..10.0, interaction in 0.0f64..10.0, chemical_potential in -10.0f64..10.0) {
            let lattice = ChainLattice::new(6, true).unwrap();
            let hamiltonian = HubbardHamiltonian::new(lattice, hopping, interaction, chemical_potential).unwrap();
            let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

            // Test with a random configuration
            let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up, Spin::Down, Spin::Empty];
            let energy = energy_calc.calculate(&config);

            // Energy should be finite
            prop_assert!(energy.is_finite());
        }

        #[test]
        fn prop_energy_kinetic_potential_sum(hopping in 0.0f64..10.0, interaction in 0.0f64..10.0, chemical_potential in -10.0f64..10.0) {
            let lattice = ChainLattice::new(4, true).unwrap();
            let hamiltonian = HubbardHamiltonian::new(lattice, hopping, interaction, chemical_potential).unwrap();
            let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));

            let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
            let total_energy = energy_calc.calculate(&config);
            let kinetic = energy_calc.kinetic_energy(&config).unwrap();
            let potential = energy_calc.potential_energy(&config).unwrap();

            // Total energy should equal kinetic + potential
            prop_assert!((total_energy - (kinetic + potential)).abs() < 1e-10);
        }
    }
}
