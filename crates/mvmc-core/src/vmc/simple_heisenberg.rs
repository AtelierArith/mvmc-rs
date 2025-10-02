//! Simple Heisenberg model VMC implementation
//!
//! This module provides a minimal, working implementation of VMC for Heisenberg model
//! starting from the basics and building up complexity gradually.

use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian, Spin};
use mvmc_physics::lattice::ChainLattice;
use crate::error::Result;

/// Simple Heisenberg VMC calculator
///
/// This is a minimal implementation that focuses on getting the basic
/// energy calculation working correctly.
#[derive(Debug)]
pub struct SimpleHeisenbergVMC {
    /// Lattice size
    nsite: usize,
    /// Spin quantum number (2*Sz)
    two_sz: i32,
    /// Hamiltonian
    hamiltonian: HeisenbergHamiltonian,
    /// Exchange coupling
    exchange: f64,
}

impl SimpleHeisenbergVMC {
    /// Creates a new simple Heisenberg VMC calculator
    pub fn new(nsite: usize, two_sz: i32, exchange: f64) -> Result<Self> {
        let lattice = ChainLattice::new(nsite, true).map_err(|e| crate::error::VmcError::invalid_config(&format!("Lattice error: {}", e)))?;
        let hamiltonian = HeisenbergHamiltonian::new(lattice, exchange, 0.0).map_err(|e| crate::error::VmcError::invalid_config(&format!("Hamiltonian error: {}", e)))?;

        Ok(Self {
            nsite,
            two_sz,
            hamiltonian,
            exchange,
        })
    }

    /// Generates a simple spin configuration
    ///
    /// For testing purposes, this generates a simple alternating pattern
    /// instead of random configurations.
    pub fn generate_simple_config(&self) -> Vec<Spin> {
        let mut config = vec![Spin::Empty; self.nsite];

        // Simple alternating pattern: Up, Down, Up, Down, ...
        for i in 0..self.nsite {
            config[i] = if i % 2 == 0 {
                Spin::Up
            } else {
                Spin::Down
            };
        }

        config
    }

    /// Generates a random spin configuration
    ///
    /// This generates a random configuration with the correct spin quantum number.
    pub fn generate_random_config(&self, seed: u64) -> Vec<Spin> {
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

        // Shuffle using simple LCG
        let mut rng_state = seed;
        for i in 0..self.nsite {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state as usize) % self.nsite;
            config.swap(i, j);
        }

        config
    }

    /// Calculates the energy of a spin configuration
    pub fn calculate_energy(&self, config: &[Spin]) -> f64 {
        self.hamiltonian.diagonal_element(config)
    }

    /// Performs a simple Monte Carlo step
    ///
    /// This proposes a spin exchange (two spins flip) and accepts it based on energy difference.
    pub fn monte_carlo_step(&self, config: &mut Vec<Spin>, rng_state: &mut u64) -> bool {
        if config.len() < 2 {
            return false;
        }

        // Choose two random sites to exchange
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let site1 = (*rng_state as usize) % self.nsite;

        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let site2 = (*rng_state as usize) % self.nsite;

        if site1 == site2 {
            return false; // No exchange possible
        }

        // Calculate current energy
        let current_energy = self.calculate_energy(config);

        // Exchange the spins
        let old_spin1 = config[site1];
        let old_spin2 = config[site2];
        config[site1] = old_spin2;
        config[site2] = old_spin1;

        // Check if the new configuration is physically valid
        let new_two_sz = self.calculate_two_sz(config);
        if new_two_sz != self.two_sz {
            // Reject move, restore original spins
            config[site1] = old_spin1;
            config[site2] = old_spin2;
            return false;
        }

        // Calculate new energy
        let new_energy = self.calculate_energy(config);

        // Metropolis acceptance criterion
        let energy_diff = new_energy - current_energy;
        let acceptance_prob = if energy_diff <= 0.0 {
            1.0 // Always accept if energy decreases
        } else {
            (-energy_diff).exp() // Boltzmann factor
        };

        // Generate random number
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random_num = (*rng_state as f64) / (u64::MAX as f64);

        if random_num < acceptance_prob {
            true // Move accepted
        } else {
            // Reject move, restore original spins
            config[site1] = old_spin1;
            config[site2] = old_spin2;
            false
        }
    }

    /// Calculates the total spin quantum number (2*Sz) for a configuration
    fn calculate_two_sz(&self, config: &[Spin]) -> i32 {
        let mut two_sz = 0;
        for spin in config {
            match spin {
                Spin::Up => two_sz += 1,
                Spin::Down => two_sz -= 1,
                Spin::Empty => {} // Should not happen in Heisenberg model
            }
        }
        two_sz
    }

    /// Runs a simple VMC calculation
    ///
    /// This performs Monte Carlo sampling and calculates the average energy.
    pub fn run_vmc(&self, num_steps: usize, seed: u64) -> Result<(f64, f64, f64)> {
        // Generate initial configuration
        let mut config = self.generate_random_config(seed);

        // Monte Carlo sampling
        let mut rng_state = seed;
        let mut energy_sum = 0.0;
        let mut energy_squared_sum = 0.0;
        let mut accepted_moves = 0;

        for step in 0..num_steps {
            // Perform Monte Carlo step
            let accepted = self.monte_carlo_step(&mut config, &mut rng_state);
            if accepted {
                accepted_moves += 1;
            }

            // Calculate energy for this configuration
            let energy = self.calculate_energy(&config);
            energy_sum += energy;
            energy_squared_sum += energy * energy;

            // Print progress every 100 steps
            if step % 100 == 0 {
                println!("Step {}: Energy = {:.6}, Acceptance rate = {:.3}",
                    step, energy, accepted_moves as f64 / (step + 1) as f64);
            }
        }

        // Calculate statistics
        let avg_energy = energy_sum / num_steps as f64;
        let avg_energy_squared = energy_squared_sum / num_steps as f64;
        let variance = avg_energy_squared - avg_energy * avg_energy;
        let acceptance_rate = accepted_moves as f64 / num_steps as f64;

        Ok((avg_energy, variance, acceptance_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_heisenberg_creation() {
        let vmc = SimpleHeisenbergVMC::new(4, 0, 1.0).unwrap();
        assert_eq!(vmc.nsite, 4);
        assert_eq!(vmc.two_sz, 0);
    }

    #[test]
    fn test_simple_config_generation() {
        let vmc = SimpleHeisenbergVMC::new(4, 0, 1.0).unwrap();
        let config = vmc.generate_simple_config();

        assert_eq!(config.len(), 4);
        assert_eq!(config[0], Spin::Up);
        assert_eq!(config[1], Spin::Down);
        assert_eq!(config[2], Spin::Up);
        assert_eq!(config[3], Spin::Down);
    }

    #[test]
    fn test_energy_calculation() {
        let vmc = SimpleHeisenbergVMC::new(4, 0, 1.0).unwrap();
        let config = vmc.generate_simple_config();
        let energy = vmc.calculate_energy(&config);

        // For alternating pattern on 4-site chain with J=1.0:
        // E = J * (S1·S2 + S2·S3 + S3·S4 + S4·S1) = 1.0 * (-1 + 1 + (-1) + 1) = 0.0
        // But with periodic boundary conditions: 4 bonds × (-1) = -4.0
        assert!((energy - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_monte_carlo_step() {
        let vmc = SimpleHeisenbergVMC::new(4, 0, 1.0).unwrap();
        let mut config = vmc.generate_simple_config();
        let mut rng_state = 12345u64;

        // Perform a few Monte Carlo steps
        for _ in 0..10 {
            vmc.monte_carlo_step(&mut config, &mut rng_state);
        }

        // Configuration should still be valid (all sites should have spins)
        for spin in &config {
            assert!(*spin != Spin::Empty);
        }
    }

    #[test]
    fn test_vmc_calculation() {
        let vmc = SimpleHeisenbergVMC::new(4, 0, 1.0).unwrap();
        let (avg_energy, variance, acceptance_rate) = vmc.run_vmc(1000, 12345).unwrap();

        println!("Average energy: {:.6}", avg_energy);
        println!("Variance: {:.6}", variance);
        println!("Acceptance rate: {:.3}", acceptance_rate);

        // Energy should be reasonable (can be negative for antiferromagnetic coupling)
        // For 4-site chain with alternating pattern, energy can be 0
        assert!(avg_energy <= 0.0);
        assert!(variance >= 0.0);
        assert!(acceptance_rate >= 0.0 && acceptance_rate <= 1.0);
    }
}
