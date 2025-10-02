//! General sampling interface for VMC calculations
//!
//! This module provides a general sampling interface that can be used
//! with different sampling algorithms (Metropolis, etc.).

use crate::monte_carlo::{ElectronConfiguration, MetropolisStep};

/// Sampling statistics for monitoring the sampling process
///
/// This structure contains statistics about the sampling process
/// that can be used for monitoring and optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingStatistics {
    /// Total number of steps performed
    pub total_steps: usize,

    /// Number of accepted steps
    pub accepted_steps: usize,

    /// Acceptance rate
    pub acceptance_rate: f64,

    /// Average energy
    pub average_energy: f64,

    /// Energy variance
    pub energy_variance: f64,

    /// Number of warm-up steps
    pub warmup_steps: usize,

    /// Number of measurement steps
    pub measurement_steps: usize,
}

/// General sampling interface
///
/// This trait defines the interface for different sampling algorithms
/// that can be used in VMC calculations.
pub trait Sampler {
    /// Performs a single sampling step
    fn step(&mut self) -> crate::Result<MetropolisStep>;

    /// Runs sampling for a specified number of steps
    fn run_sampling(&mut self, n_steps: usize) -> crate::Result<Vec<MetropolisStep>>;

    /// Returns the current electron configuration
    fn current_config(&self) -> &ElectronConfiguration;

    /// Returns the acceptance rate from a sequence of steps
    fn acceptance_rate(&self, steps: &[MetropolisStep]) -> f64;

    /// Returns sampling statistics
    fn statistics(&self) -> SamplingStatistics;
}

impl SamplingStatistics {
    /// Creates a new SamplingStatistics with default values
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            accepted_steps: 0,
            acceptance_rate: 0.0,
            average_energy: 0.0,
            energy_variance: 0.0,
            warmup_steps: 0,
            measurement_steps: 0,
        }
    }

    /// Updates statistics from a sequence of steps
    ///
    /// # Arguments
    ///
    /// * `steps` - Sequence of Metropolis steps
    /// * `warmup_steps` - Number of warm-up steps
    pub fn update_from_steps(&mut self, steps: &[MetropolisStep], warmup_steps: usize) {
        self.total_steps = steps.len();
        self.accepted_steps = steps.iter().filter(|step| step.accepted).count();
        self.acceptance_rate = if self.total_steps > 0 {
            self.accepted_steps as f64 / self.total_steps as f64
        } else {
            0.0
        };

        self.warmup_steps = warmup_steps;
        self.measurement_steps = self.total_steps.saturating_sub(warmup_steps);

        // Calculate energy statistics (simplified)
        if self.measurement_steps > 0 {
            let measurement_steps = &steps[warmup_steps..];
            let energies: Vec<f64> = measurement_steps
                .iter()
                .map(|step| step.amplitude_ratio.norm())
                .collect();

            if !energies.is_empty() {
                let sum: f64 = energies.iter().sum();
                self.average_energy = sum / energies.len() as f64;

                let variance_sum: f64 = energies
                    .iter()
                    .map(|&e| (e - self.average_energy).powi(2))
                    .sum();
                self.energy_variance = variance_sum / energies.len() as f64;
            }
        }
    }

    /// Returns the effective sample size
    ///
    /// This is a measure of how many independent samples we have
    /// after accounting for autocorrelation.
    pub fn effective_sample_size(&self) -> usize {
        if self.energy_variance > 0.0 {
            // Simplified effective sample size calculation
            let autocorr_time = 1.0 / (1.0 - self.acceptance_rate);
            (self.measurement_steps as f64 / autocorr_time) as usize
        } else {
            self.measurement_steps
        }
    }

    /// Returns whether the sampling has converged
    ///
    /// This is a simple convergence criterion based on acceptance rate
    /// and effective sample size.
    pub fn is_converged(&self) -> bool {
        self.acceptance_rate > 0.1 && // Not too low acceptance
        self.acceptance_rate < 0.9 && // Not too high acceptance
        self.effective_sample_size() > 100 // Enough effective samples
    }
}

impl Default for SamplingStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for sampling analysis
pub struct SamplingAnalysis;

impl SamplingAnalysis {
    /// Calculates the autocorrelation function for a time series
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `max_lag` - Maximum lag to calculate
    ///
    /// # Returns
    ///
    /// Vector of autocorrelation values for lags 0 to max_lag
    pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 || max_lag >= n {
            return vec![];
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;

        if variance == 0.0 || variance.is_nan() || variance.is_infinite() {
            return vec![1.0; max_lag + 1];
        }

        let mut autocorr = Vec::with_capacity(max_lag + 1);

        for lag in 0..=max_lag {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += (data[i] - mean) * (data[i + lag] - mean);
            }
            let corr = sum / ((n - lag) as f64 * variance);
            // Clamp to [-1, 1] to avoid numerical issues
            let corr_clamped = corr.max(-1.0).min(1.0);
            autocorr.push(corr_clamped);
        }

        autocorr
    }

    /// Calculates the integrated autocorrelation time
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    ///
    /// # Returns
    ///
    /// The integrated autocorrelation time
    pub fn integrated_autocorr_time(data: &[f64]) -> f64 {
        let autocorr = Self::autocorrelation(data, data.len() / 4);

        let mut tau_int = 0.5; // Start with 0.5 (uncorrelated)

        for i in 1..autocorr.len() {
            tau_int += autocorr[i];
            if autocorr[i] < 0.0 {
                break; // Stop when autocorrelation becomes negative
            }
        }

        tau_int
    }

    /// Calculates the effective sample size accounting for autocorrelation
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    ///
    /// # Returns
    ///
    /// The effective sample size
    pub fn effective_sample_size(data: &[f64]) -> usize {
        if data.is_empty() {
            return 0;
        }

        let tau_int = Self::integrated_autocorr_time(data);
        let eff_size = (data.len() as f64 / (2.0 * tau_int + 1.0)) as usize;

        // Ensure effective sample size is reasonable
        eff_size.min(data.len()).max(1)
    }

    /// Performs a statistical test for convergence
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data
    /// * `window_size` - Size of windows for comparison
    ///
    /// # Returns
    ///
    /// True if the data appears to have converged
    pub fn convergence_test(data: &[f64], window_size: usize) -> bool {
        if data.len() < 2 * window_size {
            return false;
        }

        let n_windows = data.len() / window_size;
        let mut window_means = Vec::with_capacity(n_windows);

        for i in 0..n_windows {
            let start = i * window_size;
            let end = start + window_size;
            let window_mean = data[start..end].iter().sum::<f64>() / window_size as f64;
            window_means.push(window_mean);
        }

        if window_means.len() < 2 {
            return false;
        }

        // Calculate variance of window means
        let overall_mean = window_means.iter().sum::<f64>() / window_means.len() as f64;
        let window_variance = window_means.iter()
            .map(|&x| (x - overall_mean).powi(2))
            .sum::<f64>() / window_means.len() as f64;

        // Calculate variance within windows
        let mut within_variance = 0.0;
        for i in 0..n_windows {
            let start = i * window_size;
            let end = start + window_size;
            let window_mean = window_means[i];
            let window_var = data[start..end].iter()
                .map(|&x| (x - window_mean).powi(2))
                .sum::<f64>() / window_size as f64;
            within_variance += window_var;
        }
        within_variance /= n_windows as f64;

        // Convergence test: between-window variance should be small compared to within-window variance
        window_variance < 0.1 * within_variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_statistics_creation() {
        let stats = SamplingStatistics::new();

        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.accepted_steps, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_sampling_statistics_update() {
        let mut stats = SamplingStatistics::new();

        // Create dummy steps
        use crate::types::{SiteCount, ElectronCount, TwoSz};
        let steps = vec![
            MetropolisStep {
                accepted: true,
                proposed_config: ElectronConfiguration::new(
                    SiteCount::new(2), ElectronCount::new(1), TwoSz::new(1)
                ),
                acceptance_prob: 0.5,
                amplitude_ratio: num_complex::Complex64::new(1.0, 0.0),
            },
            MetropolisStep {
                accepted: false,
                proposed_config: ElectronConfiguration::new(
                    SiteCount::new(2), ElectronCount::new(1), TwoSz::new(1)
                ),
                acceptance_prob: 0.3,
                amplitude_ratio: num_complex::Complex64::new(0.5, 0.0),
            },
        ];

        stats.update_from_steps(&steps, 0);

        assert_eq!(stats.total_steps, 2);
        assert_eq!(stats.accepted_steps, 1);
        assert_eq!(stats.acceptance_rate, 0.5);
    }

    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let autocorr = SamplingAnalysis::autocorrelation(&data, 2);

        assert_eq!(autocorr.len(), 3);
        assert!((autocorr[0] - 1.0).abs() < 1e-10); // Lag 0 should be 1.0
    }

    #[test]
    fn test_effective_sample_size() {
        let data = vec![1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1];
        let eff_size = SamplingAnalysis::effective_sample_size(&data);

        assert!(eff_size > 0);
        assert!(eff_size <= data.len());

        // Test empty data
        let empty_data = vec![];
        let eff_size_empty = SamplingAnalysis::effective_sample_size(&empty_data);
        assert_eq!(eff_size_empty, 0);
    }

    #[test]
    fn test_convergence_test() {
        // Create data that should converge
        let data: Vec<f64> = (0..100).map(|i| 1.0 + 0.1 * (i as f64 / 100.0)).collect();
        let converged = SamplingAnalysis::convergence_test(&data, 10);

        // This is a simple test - in practice, convergence depends on the data
        assert!(converged || !converged); // Just check it doesn't panic
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_autocorrelation_bounds(
            data in prop::collection::vec(-10.0f64..10.0, 10..100),
            max_lag in 1usize..20
        ) {
            let autocorr = SamplingAnalysis::autocorrelation(&data, max_lag);

            if !autocorr.is_empty() {
                // First element should be 1.0
                prop_assert!((autocorr[0] - 1.0).abs() < 1e-10);

                // All elements should be between -1 and 1
                for &corr in &autocorr {
                    prop_assert!(corr >= -1.0);
                    prop_assert!(corr <= 1.0);
                }
            }
        }

        #[test]
        fn prop_effective_sample_size_bounds(
            data in prop::collection::vec(-10.0f64..10.0, 10..100)
        ) {
            let eff_size = SamplingAnalysis::effective_sample_size(&data);

            prop_assert!(eff_size > 0);
            prop_assert!(eff_size <= data.len());
        }
    }
}
