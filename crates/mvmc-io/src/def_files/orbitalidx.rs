//! Generator for orbitalidx.def file.
//!
//! This module generates the orbitalidx.def file which contains
//! orbital indices for the system.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for orbitalidx.def file.
pub struct OrbitalIdxGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> OrbitalIdxGenerator<'a> {
    /// Creates a new orbitalidx.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the orbitalidx.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Calculate number of orbital indices
        let n_orbital = self.config.nsite * 4; // 4 orbitals per site for Lsub=4

        // Header
        content.push_str("=============================================\n");
        content.push_str(&format!("NOrbitalIdx         {}\n", n_orbital));
        content.push_str("ComplexType          0\n");
        content.push_str("=============================================\n");
        content.push_str("=============================================\n");

        // Orbital indices
        let mut orbital_idx = 0;
        for i in 0..self.config.nsite {
            for j in 0..4 { // 4 orbitals per site
                content.push_str(&format!("{:4} {:4} {:4}\n", i, j, orbital_idx));
                orbital_idx += 1;
            }
        }

        // Additional entries for periodic boundary conditions
        for i in 0..self.config.nsite {
            for j in 0..4 {
                let shifted_j = (j + 4) % 4;
                let shifted_orbital_idx = (i * 4 + shifted_j) % n_orbital;
                content.push_str(&format!("{:4} {:4} {:4}\n", i, j, shifted_orbital_idx));
            }
        }

        // Final entries
        for i in 0..n_orbital {
            content.push_str(&format!("{:4}      1\n", i));
        }

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
