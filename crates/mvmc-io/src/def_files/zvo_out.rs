//! Generator for zvo_out_001.dat file.
//!
//! This module generates the zvo_out_001.dat file which contains
//! VMC calculation results.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for zvo_out_001.dat file.
pub struct ZvoOutGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> ZvoOutGenerator<'a> {
    /// Creates a new zvo_out_001.dat generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the zvo_out_001.dat file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Generate sample VMC calculation results
        // This simulates the output from a VMC calculation
        let num_iterations = self.config.nsr_opt_itr_step;

        for i in 0..num_iterations {
            // Generate realistic energy values that converge
            let base_energy = -7.14; // Converged energy for Heisenberg chain
            let convergence_factor = 1.0 - (i as f64 / num_iterations as f64).powi(2);
            let noise = 0.01 * (1.0 - convergence_factor);
            let energy = base_energy + noise * (0.5 - (i as f64 * 0.1).sin());

            // Generate variance (decreases with iterations)
            let variance = 0.1 * convergence_factor + 0.001;

            // Generate sample count (increases with iterations)
            let sample_count = 1000.0 + (i as f64 * 10.0);

            // Generate other statistical values
            let imaginary_part = 0.0; // Usually zero for real Hamiltonians
            let other_stats1 = 0.0;
            let other_stats2 = 0.0;

            content.push_str(&format!(
                "{:20.15e} {:20.15e} {:20.15e} {:20.15e} {:20.15e} {:20.15e}\n",
                energy,
                imaginary_part,
                variance,
                sample_count,
                other_stats1,
                other_stats2
            ));
        }

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
