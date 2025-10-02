//! Generator for jastrowidx.def file.
//!
//! This module generates the jastrowidx.def file which contains
//! Jastrow factor indices.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for jastrowidx.def file.
pub struct JastrowIdxGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> JastrowIdxGenerator<'a> {
    /// Creates a new jastrowidx.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the jastrowidx.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Calculate number of Jastrow indices
        let n_jastrow = self.config.nsite * (self.config.nsite - 1);

        // Header
        content.push_str("=============================================\n");
        content.push_str(&format!("NJastrowIdx          {}\n", n_jastrow));
        content.push_str("ComplexType          0\n");
        content.push_str("=============================================\n");
        content.push_str("=============================================\n");

        // Jastrow indices for all site pairs
        for i in 0..self.config.nsite {
            for j in 0..self.config.nsite {
                if i != j {
                    content.push_str(&format!("{:4} {:4}      0\n", i, j));
                }
            }
        }

        // Additional entry
        content.push_str("    0      0\n");

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
