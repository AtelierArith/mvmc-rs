//! Generator for gutzwilleridx.def file.
//!
//! This module generates the gutzwilleridx.def file which contains
//! Gutzwiller factor indices.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for gutzwilleridx.def file.
pub struct GutzwillerIdxGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> GutzwillerIdxGenerator<'a> {
    /// Creates a new gutzwilleridx.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the gutzwilleridx.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("=============================================\n");
        content.push_str("NGutzwillerIdx          1\n");
        content.push_str("ComplexType          0\n");
        content.push_str("=============================================\n");
        content.push_str("=============================================\n");

        // Gutzwiller indices for each site
        for i in 0..self.config.nsite {
            content.push_str(&format!("{:4}      0\n", i));
        }

        // Additional entry
        content.push_str("    0      0\n");

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
