//! Random number generation for Monte Carlo simulations
//!
//! Provides high-quality random number generators suitable for statistical calculations.

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};

/// A random number generator for Monte Carlo simulations
pub struct McRng {
    rng: rand::rngs::StdRng,
}

impl McRng {
    /// Creates a new RNG with the given seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Generates a random f64 in the range [0, 1)
    pub fn gen_unit(&mut self) -> f64 {
        self.rng.r#gen()
    }

    /// Generates a random f64 in the range [min, max)
    pub fn gen_range(&mut self, min: f64, max: f64) -> f64 {
        let uniform = Uniform::new(min, max);
        uniform.sample(&mut self.rng)
    }

    /// Generates a random integer in the range [0, n)
    pub fn gen_index(&mut self, n: usize) -> usize {
        self.rng.r#gen_range(0..n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rng_reproducible() {
        // Test that RNG with same seed produces same sequence
        let mut rng1 = McRng::new(12345);
        let mut rng2 = McRng::new(12345);

        for _ in 0..10 {
            assert_eq!(rng1.gen_unit(), rng2.gen_unit());
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        // Test that different seeds produce different sequences
        let mut rng1 = McRng::new(12345);
        let mut rng2 = McRng::new(54321);

        let vals1: Vec<f64> = (0..10).map(|_| rng1.gen_unit()).collect();
        let vals2: Vec<f64> = (0..10).map(|_| rng2.gen_unit()).collect();

        assert_ne!(vals1, vals2);
    }

    #[test]
    fn test_gen_unit_range() {
        // Test that gen_unit() produces values in [0, 1)
        let mut rng = McRng::new(42);

        for _ in 0..1000 {
            let val = rng.gen_unit();
            assert!(val >= 0.0 && val < 1.0, "Value {} out of range [0, 1)", val);
        }
    }

    #[test]
    fn test_gen_range() {
        // Test that gen_range() produces values in specified range
        let mut rng = McRng::new(42);
        let min = -5.0;
        let max = 10.0;

        for _ in 0..1000 {
            let val = rng.gen_range(min, max);
            assert!(
                val >= min && val < max,
                "Value {} out of range [{}, {})",
                val,
                min,
                max
            );
        }
    }

    #[test]
    fn test_gen_index() {
        // Test that gen_index() produces values in [0, n)
        let mut rng = McRng::new(42);
        let n = 10;

        for _ in 0..1000 {
            let idx = rng.gen_index(n);
            assert!(idx < n, "Index {} out of range [0, {})", idx, n);
        }
    }

    #[test]
    fn test_statistical_distribution() {
        // Test that gen_unit() has approximately uniform distribution
        let mut rng = McRng::new(42);
        let n_samples = 10000;
        let n_bins = 10;
        let mut bins = vec![0; n_bins];

        for _ in 0..n_samples {
            let val = rng.gen_unit();
            let bin = (val * n_bins as f64).floor() as usize;
            bins[bin.min(n_bins - 1)] += 1;
        }

        // Each bin should have approximately n_samples/n_bins entries
        let expected = n_samples / n_bins;
        let tolerance = 0.1; // 10% tolerance

        for (i, &count) in bins.iter().enumerate() {
            let relative_error = (count as f64 - expected as f64).abs() / expected as f64;
            assert!(
                relative_error < tolerance,
                "Bin {} has count {} (expected ~{}), relative error: {:.2}%",
                i,
                count,
                expected,
                relative_error * 100.0
            );
        }
    }

    #[test]
    fn test_mean_and_variance() {
        // Test statistical properties of generated numbers
        let mut rng = McRng::new(42);
        let n = 100000;

        let samples: Vec<f64> = (0..n).map(|_| rng.gen_unit()).collect();

        // Calculate mean
        let mean = samples.iter().sum::<f64>() / n as f64;

        // Calculate variance
        let variance = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;

        // For uniform distribution on [0, 1):
        // Expected mean = 0.5
        // Expected variance = 1/12 ≈ 0.0833
        assert_relative_eq!(mean, 0.5, epsilon = 0.01);
        assert_relative_eq!(variance, 1.0 / 12.0, epsilon = 0.01);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_gen_unit_in_range(seed in 0u64..1000) {
            let mut rng = McRng::new(seed);
            for _ in 0..100 {
                let val = rng.gen_unit();
                prop_assert!(val >= 0.0);
                prop_assert!(val < 1.0);
            }
        }

        #[test]
        fn prop_gen_range_in_bounds(
            seed in 0u64..1000,
            min in -100.0..100.0,
            max in -100.0..100.0
        ) {
            prop_assume!(min < max);
            let mut rng = McRng::new(seed);
            for _ in 0..100 {
                let val = rng.gen_range(min, max);
                prop_assert!(val >= min);
                prop_assert!(val < max);
            }
        }

        #[test]
        fn prop_gen_index_in_bounds(
            seed in 0u64..1000,
            n in 1usize..100
        ) {
            let mut rng = McRng::new(seed);
            for _ in 0..100 {
                let idx = rng.gen_index(n);
                prop_assert!(idx < n);
            }
        }

        #[test]
        fn prop_reproducibility(seed in 0u64..1000) {
            let mut rng1 = McRng::new(seed);
            let mut rng2 = McRng::new(seed);

            for _ in 0..10 {
                prop_assert_eq!(rng1.gen_unit(), rng2.gen_unit());
            }
        }
    }
}
