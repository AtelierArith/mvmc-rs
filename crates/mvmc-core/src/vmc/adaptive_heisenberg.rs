//! Adaptive Heisenberg model VMC implementation
//!
//! This module provides an adaptive implementation of VMC for Heisenberg model
//! with self-adjusting Monte Carlo parameters.

use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian, Spin};
use mvmc_physics::lattice::ChainLattice;
use crate::error::Result;
use std::collections::VecDeque;

/// Adaptive Heisenberg VMC calculator with self-adjusting parameters
#[derive(Debug)]
pub struct AdaptiveHeisenbergVMC {
    /// Lattice size
    nsite: usize,
    /// Spin quantum number (2*Sz)
    two_sz: i32,
    /// Hamiltonian
    hamiltonian: HeisenbergHamiltonian,
    /// Exchange coupling
    exchange: f64,
    /// Current temperature (adaptive)
    temperature: f64,
    /// Target acceptance rate
    target_acceptance_rate: f64,
    /// Temperature adjustment factor
    temp_adjustment_factor: f64,
    /// Random number generator state
    rng_state: u64,
    /// Energy history for adaptive adjustments
    energy_history: VecDeque<f64>,
    /// Acceptance rate history
    acceptance_history: VecDeque<f64>,
}

impl AdaptiveHeisenbergVMC {
    /// Creates a new adaptive Heisenberg VMC calculator
    pub fn new(
        nsite: usize,
        two_sz: i32,
        exchange: f64,
        initial_temperature: f64,
        target_acceptance_rate: f64
    ) -> Result<Self> {
        let lattice = ChainLattice::new(nsite, true).map_err(|e| crate::error::VmcError::invalid_config(&format!("Lattice error: {}", e)))?;
        let hamiltonian = HeisenbergHamiltonian::new(lattice, exchange, 0.0).map_err(|e| crate::error::VmcError::invalid_config(&format!("Hamiltonian error: {}", e)))?;

        Ok(Self {
            nsite,
            two_sz,
            hamiltonian,
            exchange,
            temperature: initial_temperature,
            target_acceptance_rate,
            temp_adjustment_factor: 0.1,
            rng_state: 12345,
            energy_history: VecDeque::new(),
            acceptance_history: VecDeque::new(),
        })
    }

    /// Generates a random spin configuration
    pub fn generate_random_config(&mut self) -> Vec<Spin> {
        let mut config = vec![Spin::Empty; self.nsite];

        let n_up = (self.nsite + self.two_sz as usize) / 2;
        let n_down = (self.nsite - self.two_sz as usize) / 2;

        for i in 0..n_up.min(self.nsite) {
            config[i] = Spin::Up;
        }
        for i in n_up..(n_up + n_down).min(self.nsite) {
            config[i] = Spin::Down;
        }

        self.shuffle_config(&mut config);
        config
    }

    /// Shuffles a configuration
    fn shuffle_config(&mut self, config: &mut [Spin]) {
        for i in 0..config.len() {
            let j = self.next_random() % config.len();
            if i != j {
                config.swap(i, j);
            }
        }
    }

    /// Improved random number generator
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

    /// Proposes a spin flip
    fn propose_spin_flip(&mut self, config: &[Spin]) -> Option<(usize, Spin)> {
        let site = self.next_random() % self.nsite;
        let current_spin = config[site];

        let new_spin = match current_spin {
            Spin::Up => Spin::Down,
            Spin::Down => Spin::Up,
            Spin::Empty => return None,
        };

        Some((site, new_spin))
    }

    /// Proposes a spin swap
    fn propose_spin_swap(&mut self, config: &[Spin]) -> Option<(usize, usize)> {
        let site1 = self.next_random() % self.nsite;
        let site2 = self.next_random() % self.nsite;

        if site1 != site2 && config[site1] != config[site2] {
            Some((site1, site2))
        } else {
            None
        }
    }

    /// Adjusts temperature based on acceptance rate
    fn adjust_temperature(&mut self, current_acceptance_rate: f64) {
        let rate_diff = current_acceptance_rate - self.target_acceptance_rate;

        if rate_diff > 0.1 {
            // Acceptance rate too high, increase temperature
            self.temperature *= 1.0 + self.temp_adjustment_factor;
        } else if rate_diff < -0.1 {
            // Acceptance rate too low, decrease temperature
            self.temperature *= 1.0 - self.temp_adjustment_factor;
        }

        // Keep temperature within reasonable bounds
        self.temperature = self.temperature.max(0.01).min(100.0);
    }

    /// Performs an adaptive Monte Carlo step
    fn adaptive_monte_carlo_step(&mut self, config: &mut Vec<Spin>) -> bool {
        if config.is_empty() {
            return false;
        }

        // Choose move type based on current acceptance rate
        let recent_acceptance = if self.acceptance_history.len() > 10 {
            self.acceptance_history.iter().rev().take(10).sum::<f64>() / 10.0
        } else {
            0.5
        };

        let move_type = if recent_acceptance < 0.3 {
            // Low acceptance rate, prefer spin swaps
            self.next_random() % 100 < 50
        } else {
            // Normal acceptance rate, use mixed moves
            self.next_random() % 100 < 70
        };

        let (_energy_diff, accepted) = if move_type {
            // Spin flip
            if let Some((site, new_spin)) = self.propose_spin_flip(config) {
                let current_energy = self.calculate_energy(config);
                let old_spin = config[site];
                config[site] = new_spin;

                let new_energy = self.calculate_energy(config);
                let energy_diff = new_energy - current_energy;

                let acceptance_prob = if energy_diff <= 0.0 {
                    1.0
                } else {
                    (-energy_diff / self.temperature).exp()
                };

                let random_num = self.next_random_float();
                let accepted = random_num < acceptance_prob;

                if !accepted {
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
                config.swap(site1, site2);

                let new_energy = self.calculate_energy(config);
                let energy_diff = new_energy - current_energy;

                let acceptance_prob = if energy_diff <= 0.0 {
                    1.0
                } else {
                    (-energy_diff / self.temperature).exp()
                };

                let random_num = self.next_random_float();
                let accepted = random_num < acceptance_prob;

                if !accepted {
                    config.swap(site1, site2);
                }

                (energy_diff, accepted)
            } else {
                (0.0, false)
            }
        };

        accepted
    }

    /// Runs an adaptive VMC calculation
    pub fn run_adaptive_vmc(&mut self, total_steps: usize) -> Result<AdaptiveVmcStatistics> {
        // Generate initial configuration
        let mut config = self.generate_random_config();

        let mut energy_sum = 0.0;
        let mut energy_squared_sum = 0.0;
        let mut accepted_moves = 0;
        let mut _step_count = 0;

        // Adaptive phase (first 20% of steps)
        let adaptive_steps = total_steps / 5;
        let measurement_steps = total_steps - adaptive_steps;

        println!("Adaptive phase ({} steps)...", adaptive_steps);
        for step in 0..adaptive_steps {
            let accepted = self.adaptive_monte_carlo_step(&mut config);
            if accepted {
                accepted_moves += 1;
            }

            // Update acceptance rate history
            let current_acceptance = accepted_moves as f64 / (step + 1) as f64;
            self.acceptance_history.push_back(current_acceptance);
            if self.acceptance_history.len() > 50 {
                self.acceptance_history.pop_front();
            }

            // Adjust temperature every 10 steps
            if step % 10 == 0 && step > 0 {
                self.adjust_temperature(current_acceptance);
            }

            if step % (adaptive_steps / 10) == 0 {
                let energy = self.calculate_energy(&config);
                println!("  Step {}: Energy = {:.6}, T = {:.3}, Acceptance = {:.3}",
                    step, energy, self.temperature, current_acceptance);
            }
        }

        println!("Measurement phase ({} steps)...", measurement_steps);
        for step in 0..measurement_steps {
            let accepted = self.adaptive_monte_carlo_step(&mut config);
            if accepted {
                accepted_moves += 1;
            }

            let energy = self.calculate_energy(&config);
            energy_sum += energy;
            energy_squared_sum += energy * energy;

            // Update energy history
            self.energy_history.push_back(energy);
            if self.energy_history.len() > 100 {
                self.energy_history.pop_front();
            }

            _step_count += 1;

            if step % (measurement_steps / 10) == 0 {
                let current_acceptance = accepted_moves as f64 / (step + adaptive_steps + 1) as f64;
                println!("  Step {}: Energy = {:.6}, T = {:.3}, Acceptance = {:.3}",
                    step + adaptive_steps, energy, self.temperature, current_acceptance);
            }
        }

        // Calculate final statistics
        let avg_energy = energy_sum / measurement_steps as f64;
        let avg_energy_squared = energy_squared_sum / measurement_steps as f64;
        let variance = avg_energy_squared - avg_energy * avg_energy;
        let final_acceptance_rate = accepted_moves as f64 / total_steps as f64;

        // Calculate autocorrelation time
        let autocorr_time = self.calculate_autocorrelation_time();

        Ok(AdaptiveVmcStatistics {
            average_energy: avg_energy,
            variance: variance,
            final_acceptance_rate: final_acceptance_rate,
            final_temperature: self.temperature,
            autocorrelation_time: autocorr_time,
            final_configuration: config,
            temperature_history: self.acceptance_history.iter().cloned().collect(),
        })
    }

    /// Calculates autocorrelation time
    fn calculate_autocorrelation_time(&self) -> f64 {
        if self.energy_history.len() < 10 {
            return 1.0;
        }

        let energies: Vec<f64> = self.energy_history.iter().cloned().collect();
        let n = energies.len();
        let mean = energies.iter().sum::<f64>() / n as f64;

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
}

/// Statistics from adaptive VMC run
#[derive(Debug)]
pub struct AdaptiveVmcStatistics {
    pub average_energy: f64,
    pub variance: f64,
    pub final_acceptance_rate: f64,
    pub final_temperature: f64,
    pub autocorrelation_time: f64,
    pub final_configuration: Vec<Spin>,
    pub temperature_history: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_heisenberg_creation() {
        let vmc = AdaptiveHeisenbergVMC::new(4, 0, 1.0, 1.0, 0.5).unwrap();
        assert_eq!(vmc.nsite, 4);
        assert_eq!(vmc.two_sz, 0);
        assert_eq!(vmc.temperature, 1.0);
        assert_eq!(vmc.target_acceptance_rate, 0.5);
    }

    #[test]
    fn test_adaptive_vmc_calculation() {
        let mut vmc = AdaptiveHeisenbergVMC::new(4, 0, 1.0, 1.0, 0.5).unwrap();
        let stats = vmc.run_adaptive_vmc(500).unwrap();

        println!("Average energy: {:.6}", stats.average_energy);
        println!("Final temperature: {:.3}", stats.final_temperature);
        println!("Final acceptance rate: {:.3}", stats.final_acceptance_rate);

        // Energy should be reasonable
        assert!(stats.average_energy < 0.0);
        assert!(stats.variance >= 0.0);
        assert!(stats.final_acceptance_rate >= 0.0 && stats.final_acceptance_rate <= 1.0);
    }
}
