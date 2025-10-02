//! Generator for locspn.def file.
//!
//! This module generates the locspn.def file which contains local spin
//! configuration information.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for locspn.def file.
pub struct LocSpnGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> LocSpnGenerator<'a> {
    /// Creates a new locspn.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the locspn.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("================================ \n");
        content.push_str(&format!("NlocalSpin    {}  \n", self.config.nsite));
        content.push_str("================================ \n");
        content.push_str("========i_1LocSpn_0IteElc ====== \n");
        content.push_str("================================ \n");

        // Local spin configuration
        for i in 0..self.config.nsite {
            content.push_str(&format!("{:4}     1\n", i));
        }

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
