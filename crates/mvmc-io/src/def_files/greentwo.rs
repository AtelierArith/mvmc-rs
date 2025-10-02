//! Generator for greentwo.def file.
//!
//! This module generates the greentwo.def file which contains two-body
//! Green's function definitions.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for greentwo.def file.
pub struct GreenTwoGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> GreenTwoGenerator<'a> {
    /// Creates a new greentwo.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the greentwo.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Calculate number of two-body Green's functions
        let n_green_two = self.config.nsite * 2 * 2; // 2 spin states, 2 sites per pair

        // Header
        content.push_str("=============================================\n");
        content.push_str(&format!("NCisAjsCktAltDC         {}\n", n_green_two));
        content.push_str("=============================================\n");
        content.push_str("======== Green functions for Sq AND Nq ======\n");
        content.push_str("=============================================\n");

        // Two-body Green's functions
        for i in 0..self.config.nsite {
            for s1 in 0..2 {
                for j in 0..self.config.nsite {
                    for s2 in 0..2 {
                        content.push_str(&format!(
                            "    {}     {}     {}     {}     {}     {}     {}     {}\n",
                            i, s1, i, s1, j, s2, j, s2
                        ));
                    }
                }
            }
        }

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
