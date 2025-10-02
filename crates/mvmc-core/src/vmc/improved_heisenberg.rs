//! Improved Heisenberg model VMC implementation
//!
//! This module provides an improved implementation of VMC for Heisenberg model
//! with better Monte Carlo sampling techniques.

use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian, Spin};
use mvmc_physics::lattice::ChainLattice;
use crate::error::Result;
use std::collections::VecDeque;

/// Improved Heisenberg VMC calculator with better Monte Carlo sampling
#[derive(Debug)]
pub struct ImprovedHeisenbergVMC {
    /// Lattice size
    nsite: usize,
    /// Spin quantum number (2*Sz)
    two_sz: i32,
    /// Hamiltonian
    hamiltonian: HeisenbergHamiltonian,
    /// Exchange coupling
    exchange: f64,
    /// Temperature for Monte Carlo sampling
    temperature: f64,
    /// Random number generator state
    rng_state: u64,
}

impl ImprovedHeisenbergVMC {
    /// Creates a new improved Heisenberg VMC calculator
    pub fn new(nsite: usize, two_sz: i32, exchange: f64, temperature: f64) -> Result<Self> {
        let lattice = ChainLattice::new(nsite, true).map_err(|e| crate::error::VmcError::invalid_config(&format!("Lattice error: {}", e)))?;
        let hamiltonian = HeisenbergHamiltonian::new(lattice, exchange, 0.0).map_err(|e| crate::error::VmcError::invalid_config(&format!("Hamiltonian error: {}", e)))?;

        Ok(Self {
            nsite,
            two_sz,
            hamiltonian,
            exchange,
            temperature,
            rng_state: 12345,
        })
    }

    /// Generates a random spin configuration with proper spin quantum number
    pub fn generate_random_config(&mut self) -> Vec<Spin> {
        let mut config = vec![Spin::Empty; self.nsite];

        // Calculate number of up and down spins
        let n_up = (self.nsite + self.two_sz as usize) / 2;
        let n_down = (self.nsite - self.two_sz as usize) / 2;

        // Place up spins
        for i in 0..n_up.min(self.nsite) {
            config[i] = Spin::Up;
        }

        // Place down spins
        for i in n_up..(n_up + n_down).min(self.nsite) {
            config[i] = Spin::Down;
        }

        // Shuffle using improved LCG
        self.shuffle_config(&mut config);
        config
    }

    /// Shuffles a configuration using improved random number generation
    fn shuffle_config(&mut self, config: &mut [Spin]) {
        for i in 0..config.len() {
            let j = self.next_random() % config.len();
            if i != j {
                config.swap(i, j);
            }
        }
    }

    /// Improved random number generator (Linear Congruential Generator)
    fn next_random(&mut self) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state >> 16) as usize
    }

    /// Generates a random float in [0, 1)
    fn next_random_float(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Calculates the energy of a spin configuration
    pub fn calculate_energy(&self, config: &[Spin]) -> f64 {
        self.hamiltonian.diagonal_element(config)
    }

    /// Proposes a spin flip with improved selection strategy
    fn propose_spin_flip(&mut self, config: &[Spin]) -> Option<(usize, Spin)> {
        // Choose a random site
        let site = self.next_random() % self.nsite;
        let current_spin = config[site];

        // Propose a flip
        let new_spin = match current_spin {
            Spin::Up => Spin::Down,
            Spin::Down => Spin::Up,
            Spin::Empty => return None, // Should not happen in Heisenberg model
        };

        Some((site, new_spin))
    }

    /// Proposes a spin swap (exchange two spins)
    fn propose_spin_swap(&mut self, config: &[Spin]) -> Option<(usize, usize)> {
        // Choose two different random sites
        let site1 = self.next_random() % self.nsite;
        let site2 = self.next_random() % self.nsite;

        if site1 != site2 && config[site1] != config[site2] {
            Some((site1, site2))
        } else {
            None
        }
    }

    /// Performs an improved Monte Carlo step
    fn monte_carlo_step(&mut self, config: &mut Vec<Spin>) -> bool {
        if config.is_empty() {
            return false;
        }

        // Choose move type: 70% spin flip, 30% spin swap
        let move_type = self.next_random() % 100;

        let (energy_diff, accepted) = if move_type < 70 {
            // Spin flip
            if let Some((site, new_spin)) = self.propose_spin_flip(config) {
                let current_energy = self.calculate_energy(config);

                // Flip the spin
                let old_spin = config[site];
                config[site] = new_spin;

                let new_energy = self.calculate_energy(config);
                let energy_diff = new_energy - current_energy;

                // Metropolis acceptance criterion
                let acceptance_prob = if energy_diff <= 0.0 {
                    1.0
                } else {
                    (-energy_diff / self.temperature).exp()
                };

                let random_num = self.next_random_float();
                let accepted = random_num < acceptance_prob;

                if !accepted {
                    // Reject move, restore original spin
                    config[site] = old_spin;
                }

                (energy_diff, accepted)
            } else {
                (0.0, false)
            }
        } else {
            // Spin swap
            if let Some((site1, site2)) = self.propose_spin_swap(config) {
                let current_energy = self.calculate_energy(config);

                // Swap the spins
                config.swap(site1, site2);

                let new_energy = self.calculate_energy(config);
                let energy_diff = new_energy - current_energy;

                // Metropolis acceptance criterion
                let acceptance_prob = if energy_diff <= 0.0 {
                    1.0
                } else {
                    (-energy_diff / self.temperature).exp()
                };

                let random_num = self.next_random_float();
                let accepted = random_num < acceptance_prob;

                if !accepted {
                    // Reject move, restore original configuration
                    config.swap(site1, site2);
                }

                (energy_diff, accepted)
            } else {
                (0.0, false)
            }
        };

        accepted
    }

    /// Runs an improved VMC calculation with warmup and better statistics
    pub fn run_vmc(&mut self, warmup_steps: usize, measurement_steps: usize) -> Result<VmcStatistics> {
        // Generate initial configuration
        let mut config = self.generate_random_config();

        // Warmup phase
        println!("Warmup phase ({} steps)...", warmup_steps);
        for step in 0..warmup_steps {
            self.monte_carlo_step(&mut config);

            if step % (warmup_steps / 10) == 0 {
                let energy = self.calculate_energy(&config);
                println!("  Warmup step {}: Energy = {:.6}", step, energy);
            }
        }

        // Measurement phase
        println!("Measurement phase ({} steps)...", measurement_steps);
        let mut energy_history = VecDeque::new();
        let mut energy_sum = 0.0;
        let mut energy_squared_sum = 0.0;
        let mut accepted_moves = 0;

        for step in 0..measurement_steps {
            // Perform Monte Carlo step
            let accepted = self.monte_carlo_step(&mut config);
            if accepted {
                accepted_moves += 1;
            }

            // Calculate energy for this configuration
            let energy = self.calculate_energy(&config);
            energy_sum += energy;
            energy_squared_sum += energy * energy;

            // Keep energy history for autocorrelation analysis
            energy_history.push_back(energy);
            if energy_history.len() > 100 {
                energy_history.pop_front();
            }

            // Print progress every 10% of measurement steps
            if step % (measurement_steps / 10) == 0 {
                let current_acceptance = accepted_moves as f64 / (step + 1) as f64;
                println!("  Step {}: Energy = {:.6}, Acceptance rate = {:.3}",
                    step, energy, current_acceptance);
            }
        }

        // Calculate statistics
        let avg_energy = energy_sum / measurement_steps as f64;
        let avg_energy_squared = energy_squared_sum / measurement_steps as f64;
        let variance = avg_energy_squared - avg_energy * avg_energy;
        let acceptance_rate = accepted_moves as f64 / measurement_steps as f64;

        // Calculate autocorrelation time (simplified)
        let autocorr_time = self.calculate_autocorrelation_time(&energy_history);

        Ok(VmcStatistics {
            average_energy: avg_energy,
            variance: variance,
            acceptance_rate: acceptance_rate,
            autocorrelation_time: autocorr_time,
            final_configuration: config,
        })
    }

    /// Calculates autocorrelation time for error estimation
    fn calculate_autocorrelation_time(&self, energy_history: &VecDeque<f64>) -> f64 {
        if energy_history.len() < 10 {
            return 1.0;
        }

        let energies: Vec<f64> = energy_history.iter().cloned().collect();
        let n = energies.len();
        let mean = energies.iter().sum::<f64>() / n as f64;

        // Calculate autocorrelation function
        let mut autocorr_sum = 0.0;
        let mut norm = 0.0;

        for lag in 1..n.min(20) {
            let mut corr = 0.0;
            for i in 0..(n - lag) {
                corr += (energies[i] - mean) * (energies[i + lag] - mean);
            }
            corr /= (n - lag) as f64;
            autocorr_sum += corr;
            norm += 1.0;
        }

        if norm > 0.0 {
            autocorr_sum / norm
        } else {
            1.0
        }
    }

    /// Runs multiple independent VMC calculations for error estimation
    pub fn run_multiple_vmc(&mut self, num_runs: usize, warmup_steps: usize, measurement_steps: usize) -> Result<MultipleVmcStatistics> {
        let mut all_energies = Vec::new();
        let mut all_acceptance_rates = Vec::new();

        for run in 0..num_runs {
            println!("Run {}/{}", run + 1, num_runs);

            // Use different random seed for each run
            self.rng_state = 12345 + run as u64;

            let stats = self.run_vmc(warmup_steps, measurement_steps)?;
            all_energies.push(stats.average_energy);
            all_acceptance_rates.push(stats.acceptance_rate);
        }

        // Calculate statistics across runs
        let mean_energy = all_energies.iter().sum::<f64>() / num_runs as f64;
        let energy_variance = all_energies.iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>() / (num_runs - 1) as f64;
        let energy_error = energy_variance.sqrt() / (num_runs as f64).sqrt();

        let mean_acceptance = all_acceptance_rates.iter().sum::<f64>() / num_runs as f64;

        Ok(MultipleVmcStatistics {
            mean_energy: mean_energy,
            energy_error: energy_error,
            mean_acceptance_rate: mean_acceptance,
            individual_energies: all_energies,
            individual_acceptance_rates: all_acceptance_rates,
        })
    }
}

/// Statistics from a single VMC run
#[derive(Debug)]
pub struct VmcStatistics {
    pub average_energy: f64,
    pub variance: f64,
    pub acceptance_rate: f64,
    pub autocorrelation_time: f64,
    pub final_configuration: Vec<Spin>,
}

/// Statistics from multiple VMC runs
#[derive(Debug)]
pub struct MultipleVmcStatistics {
    pub mean_energy: f64,
    pub energy_error: f64,
    pub mean_acceptance_rate: f64,
    pub individual_energies: Vec<f64>,
    pub individual_acceptance_rates: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improved_heisenberg_creation() {
        let vmc = ImprovedHeisenbergVMC::new(4, 0, 1.0, 1.0).unwrap();
        assert_eq!(vmc.nsite, 4);
        assert_eq!(vmc.two_sz, 0);
        assert_eq!(vmc.temperature, 1.0);
    }

    #[test]
    fn test_random_config_generation() {
        let mut vmc = ImprovedHeisenbergVMC::new(4, 0, 1.0, 1.0).unwrap();
        let config = vmc.generate_random_config();

        assert_eq!(config.len(), 4);
        // All sites should have spins (not Empty)
        for spin in &config {
            assert!(*spin != Spin::Empty);
        }
    }

    #[test]
    fn test_energy_calculation() {
        let vmc = ImprovedHeisenbergVMC::new(4, 0, 1.0, 1.0).unwrap();
        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy = vmc.calculate_energy(&config);

        // For alternating pattern on 4-site chain with J=1.0:
        // E = J * (S1·S2 + S2·S3 + S3·S4 + S4·S1) = 1.0 * (-1 + 1 + (-1) + 1) = 0.0
        // But with periodic boundary conditions: 4 bonds × (-1) = -4.0
        assert!((energy - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_vmc_calculation() {
        let mut vmc = ImprovedHeisenbergVMC::new(4, 0, 1.0, 1.0).unwrap();
        let stats = vmc.run_vmc(100, 500).unwrap();

        println!("Average energy: {:.6}", stats.average_energy);
        println!("Variance: {:.6}", stats.variance);
        println!("Acceptance rate: {:.3}", stats.acceptance_rate);
        println!("Autocorrelation time: {:.3}", stats.autocorrelation_time);

        // Energy should be reasonable (negative for antiferromagnetic coupling)
        assert!(stats.average_energy < 0.0);
        assert!(stats.variance >= 0.0);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_multiple_vmc_calculation() {
        let mut vmc = ImprovedHeisenbergVMC::new(4, 0, 1.0, 1.0).unwrap();
        let stats = vmc.run_multiple_vmc(3, 50, 200).unwrap();

        println!("Mean energy: {:.6} ± {:.6}", stats.mean_energy, stats.energy_error);
        println!("Mean acceptance rate: {:.3}", stats.mean_acceptance_rate);

        // Energy should be reasonable
        assert!(stats.mean_energy < 0.0);
        assert!(stats.energy_error >= 0.0);
        assert!(stats.mean_acceptance_rate >= 0.0 && stats.mean_acceptance_rate <= 1.0);
    }
}
