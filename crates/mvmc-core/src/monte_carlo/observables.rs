//! Physical observables calculation for VMC
//!
//! This module provides calculation of physical observables from Monte Carlo samples.
//! It corresponds to the physical quantity calculation in the C implementation.
//! Reference: mVMC/src/mVMC/vmccal.c

use crate::types::{SiteCount, ElectronCount};
use crate::monte_carlo::{ElectronConfiguration, MetropolisStep};

/// Physical observables calculated from Monte Carlo samples
///
/// This structure contains various physical quantities calculated
/// from the Monte Carlo sampling results.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalObservables {
    /// Total energy
    pub energy: f64,

    /// Kinetic energy
    pub kinetic_energy: f64,

    /// Potential energy
    pub potential_energy: f64,

    /// Total magnetization
    pub total_magnetization: f64,

    /// Site magnetization (per site)
    pub site_magnetization: Vec<f64>,

    /// Spin-spin correlation function
    pub spin_correlation: Vec<Vec<f64>>,

    /// Statistical error in energy
    pub energy_error: f64,

    /// Number of samples used
    pub num_samples: usize,
}

/// Observable calculator for VMC calculations
///
/// This structure manages the calculation of physical observables
/// from Monte Carlo sampling results.
///
/// # Examples
///
/// ```
/// use mvmc_core::monte_carlo::ObservableCalculator;
/// use mvmc_core::types::{SiteCount, ElectronCount};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let calculator = ObservableCalculator::new(nsite, ne);
/// ```
#[derive(Debug, Clone)]
pub struct ObservableCalculator {
    /// Number of lattice sites
    nsite: usize,

    /// Number of electrons
    ne: usize,

    /// Energy accumulator
    energy_sum: f64,

    /// Energy squared accumulator
    energy_sq_sum: f64,

    /// Kinetic energy accumulator
    kinetic_sum: f64,

    /// Potential energy accumulator
    potential_sum: f64,

    /// Magnetization accumulator
    magnetization_sum: f64,

    /// Site magnetization accumulator
    site_magnetization_sum: Vec<f64>,

    /// Spin correlation accumulator
    spin_correlation_sum: Vec<Vec<f64>>,

    /// Number of samples
    num_samples: usize,
}

impl PhysicalObservables {
    /// Creates a new PhysicalObservables with default values
    pub fn new(nsite: usize) -> Self {
        Self {
            energy: 0.0,
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            total_magnetization: 0.0,
            site_magnetization: vec![0.0; nsite],
            spin_correlation: vec![vec![0.0; nsite]; nsite],
            energy_error: 0.0,
            num_samples: 0,
        }
    }
}

impl ObservableCalculator {
    /// Creates a new observable calculator
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::ObservableCalculator;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let calculator = ObservableCalculator::new(nsite, ne);
    /// ```
    pub fn new(nsite: SiteCount, ne: ElectronCount) -> Self {
        let nsite_val = nsite.get();
        let ne_val = ne.get();

        Self {
            nsite: nsite_val,
            ne: ne_val,
            energy_sum: 0.0,
            energy_sq_sum: 0.0,
            kinetic_sum: 0.0,
            potential_sum: 0.0,
            magnetization_sum: 0.0,
            site_magnetization_sum: vec![0.0; nsite_val],
            spin_correlation_sum: vec![vec![0.0; nsite_val]; nsite_val],
            num_samples: 0,
        }
    }

    /// Adds a sample to the accumulator
    ///
    /// # Arguments
    ///
    /// * `config` - Electron configuration
    /// * `_step` - Metropolis step result
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::{ObservableCalculator, ElectronConfiguration, MetropolisStep};
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
    /// use num_complex::Complex64;
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let mut calculator = ObservableCalculator::new(nsite, ne);
    ///
    /// let two_sz = TwoSz::new(0);
    /// let config = ElectronConfiguration::new(nsite, ne, two_sz);
    /// let step = MetropolisStep {
    ///     accepted: true,
    ///     proposed_config: config.clone(),
    ///     acceptance_prob: 0.5,
    ///     amplitude_ratio: Complex64::new(1.0, 0.0),
    /// };
    ///
    /// calculator.add_sample(&config, &step);
    /// ```
    pub fn add_sample(&mut self, config: &ElectronConfiguration, _step: &MetropolisStep, local_energy: f64) {
        // Use provided local energy instead of calculating
        let energy = local_energy;
        let kinetic = self.calculate_kinetic_energy(config);
        let potential = self.calculate_potential_energy(config);
        let magnetization = self.calculate_magnetization(config);
        let site_magnetization = self.calculate_site_magnetization(config);
        let spin_correlation = self.calculate_spin_correlation(config);

        // Accumulate values
        self.energy_sum += energy;
        self.energy_sq_sum += energy * energy;
        self.kinetic_sum += kinetic;
        self.potential_sum += potential;
        self.magnetization_sum += magnetization;

        for i in 0..self.nsite {
            self.site_magnetization_sum[i] += site_magnetization[i];
            for j in 0..self.nsite {
                self.spin_correlation_sum[i][j] += spin_correlation[i][j];
            }
        }

        self.num_samples += 1;
    }

    /// Resets the calculator to start fresh
    pub fn reset(&mut self) {
        self.energy_sum = 0.0;
        self.energy_sq_sum = 0.0;
        self.kinetic_sum = 0.0;
        self.potential_sum = 0.0;
        self.magnetization_sum = 0.0;
        self.site_magnetization_sum.fill(0.0);
        for i in 0..self.nsite {
            self.spin_correlation_sum[i].fill(0.0);
        }
        self.num_samples = 0;
    }

    /// Calculates the final observables from accumulated samples
    ///
    /// # Returns
    ///
    /// The calculated physical observables
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::ObservableCalculator;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let mut calculator = ObservableCalculator::new(nsite, ne);
    ///
    /// // Add some samples...
    ///
    /// let observables = calculator.calculate_observables();
    /// println!("Energy: {:.6}", observables.energy);
    /// ```
    pub fn calculate_observables(&self) -> PhysicalObservables {
        if self.num_samples == 0 {
            return PhysicalObservables::new(self.nsite);
        }

        let n = self.num_samples as f64;

        // Calculate means
        let energy = self.energy_sum / n;
        let kinetic_energy = self.kinetic_sum / n;
        let potential_energy = self.potential_sum / n;
        let total_magnetization = self.magnetization_sum / n;

        // Calculate site magnetization
        let site_magnetization = self.site_magnetization_sum
            .iter()
            .map(|&sum| sum / n)
            .collect();

        // Calculate spin correlation
        let spin_correlation = self.spin_correlation_sum
            .iter()
            .map(|row| row.iter().map(|&sum| sum / n).collect())
            .collect();

        // Calculate energy error (standard deviation of the mean)
        let energy_variance = (self.energy_sq_sum / n) - (energy * energy);
        let energy_error = if energy_variance > 0.0 {
            (energy_variance / n).sqrt()
        } else {
            0.0
        };

        PhysicalObservables {
            energy,
            kinetic_energy,
            potential_energy,
            total_magnetization,
            site_magnetization,
            spin_correlation,
            energy_error,
            num_samples: self.num_samples,
        }
    }

    /// Calculates the energy for a given configuration
    ///
    /// This is a simplified energy calculation for testing purposes.
    /// In a real implementation, this would use the Hamiltonian.
    fn calculate_energy(&self, config: &ElectronConfiguration) -> f64 {
        // Simple energy: sum of electron numbers
        let mut energy = 0.0;
        for &n in config.ele_num() {
            energy += n as f64;
        }
        energy
    }

    /// Calculates the kinetic energy for a given configuration
    fn calculate_kinetic_energy(&self, config: &ElectronConfiguration) -> f64 {
        // Simplified kinetic energy
        self.calculate_energy(config) * 0.5
    }

    /// Calculates the potential energy for a given configuration
    fn calculate_potential_energy(&self, config: &ElectronConfiguration) -> f64 {
        // Simplified potential energy
        self.calculate_energy(config) * 0.5
    }

    /// Calculates the total magnetization for a given configuration
    fn calculate_magnetization(&self, config: &ElectronConfiguration) -> f64 {
        let mut magnetization = 0.0;

        for i in 0..self.nsite {
            let n_up = config.ele_num()[i] as f64;
            let n_down = config.ele_num()[i + self.nsite] as f64;
            magnetization += n_up - n_down;
        }

        magnetization
    }

    /// Calculates the site magnetization for a given configuration
    fn calculate_site_magnetization(&self, config: &ElectronConfiguration) -> Vec<f64> {
        let mut site_magnetization = vec![0.0; self.nsite];

        for i in 0..self.nsite {
            let n_up = config.ele_num()[i] as f64;
            let n_down = config.ele_num()[i + self.nsite] as f64;
            site_magnetization[i] = n_up - n_down;
        }

        site_magnetization
    }

    /// Calculates the spin-spin correlation function for a given configuration
    fn calculate_spin_correlation(&self, config: &ElectronConfiguration) -> Vec<Vec<f64>> {
        let mut correlation = vec![vec![0.0; self.nsite]; self.nsite];

        for i in 0..self.nsite {
            for j in 0..self.nsite {
                let s_i = config.ele_num()[i] as f64 - config.ele_num()[i + self.nsite] as f64;
                let s_j = config.ele_num()[j] as f64 - config.ele_num()[j + self.nsite] as f64;
                correlation[i][j] = s_i * s_j;
            }
        }

        correlation
    }

    /// Returns the number of samples accumulated
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount, TwoSz};
    use num_complex::Complex64;

    #[test]
    fn test_observable_calculator_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let calculator = ObservableCalculator::new(nsite, ne);

        assert_eq!(calculator.num_samples(), 0);
    }

    #[test]
    fn test_add_sample() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut calculator = ObservableCalculator::new(nsite, ne);

        let two_sz = TwoSz::new(1);
        let config = ElectronConfiguration::new(nsite, ne, two_sz);
        let step = MetropolisStep {
            accepted: true,
            proposed_config: config.clone(),
            acceptance_prob: 0.5,
            amplitude_ratio: Complex64::new(1.0, 0.0),
        };

        calculator.add_sample(&config, &step, 0.0);
        assert_eq!(calculator.num_samples(), 1);
    }

    #[test]
    fn test_calculate_observables_empty() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let calculator = ObservableCalculator::new(nsite, ne);

        let observables = calculator.calculate_observables();

        assert_eq!(observables.energy, 0.0);
        assert_eq!(observables.num_samples, 0);
        assert_eq!(observables.site_magnetization.len(), 4);
    }

    #[test]
    fn test_calculate_observables_with_samples() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut calculator = ObservableCalculator::new(nsite, ne);

        let two_sz = TwoSz::new(1);
        let config = ElectronConfiguration::new(nsite, ne, two_sz);
        let step = MetropolisStep {
            accepted: true,
            proposed_config: config.clone(),
            acceptance_prob: 0.5,
            amplitude_ratio: Complex64::new(1.0, 0.0),
        };

        // Add multiple samples
        for _ in 0..10 {
            calculator.add_sample(&config, &step, 0.0);
        }

        let observables = calculator.calculate_observables();

        assert_eq!(observables.num_samples, 10);
        assert!(observables.energy >= 0.0);
        assert_eq!(observables.site_magnetization.len(), 2);
    }

    #[test]
    fn test_reset() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let mut calculator = ObservableCalculator::new(nsite, ne);

        let two_sz = TwoSz::new(1);
        let config = ElectronConfiguration::new(nsite, ne, two_sz);
        let step = MetropolisStep {
            accepted: true,
            proposed_config: config.clone(),
            acceptance_prob: 0.5,
            amplitude_ratio: Complex64::new(1.0, 0.0),
        };

        calculator.add_sample(&config, &step, 0.0);
        assert_eq!(calculator.num_samples(), 1);

        calculator.reset();
        assert_eq!(calculator.num_samples(), 0);
    }

    #[test]
    fn test_physical_observables_creation() {
        let observables = PhysicalObservables::new(4);

        assert_eq!(observables.energy, 0.0);
        assert_eq!(observables.site_magnetization.len(), 4);
        assert_eq!(observables.spin_correlation.len(), 4);
        assert_eq!(observables.spin_correlation[0].len(), 4);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount, TwoSz};
    use num_complex::Complex64;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_observable_calculator_accumulation(
            nsite in 1usize..10,
            ne in 0usize..5,
            two_sz in -2i32..3,
            n_samples in 1usize..100
        ) {
            if ne <= nsite * 2 {
                let nsite_val = SiteCount::new(nsite);
                let ne_val = ElectronCount::new(ne);
                let two_sz_val = TwoSz::new(two_sz);
                let mut calculator = ObservableCalculator::new(nsite_val, ne_val);

                let config = ElectronConfiguration::new(nsite_val, ne_val, two_sz_val);
                let step = MetropolisStep {
                    accepted: true,
                    proposed_config: config.clone(),
                    acceptance_prob: 0.5,
                    amplitude_ratio: Complex64::new(1.0, 0.0),
                };

                // Add samples
                for _ in 0..n_samples {
                    calculator.add_sample(&config, &step, 0.0);
                }

                let observables = calculator.calculate_observables();

                prop_assert_eq!(observables.num_samples, n_samples);
                prop_assert!(observables.energy >= 0.0);
                prop_assert_eq!(observables.site_magnetization.len(), nsite);
            }
        }
    }
}
