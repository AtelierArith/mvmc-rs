//! Generator for trans.def file.
//!
//! This module generates the trans.def file which contains transfer integrals.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for trans.def file.
pub struct TransGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> TransGenerator<'a> {
    /// Creates a new trans.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the trans.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("======================== \n");
        content.push_str("NTransfer       0  \n");
        content.push_str("======================== \n");
        content.push_str("========i_j_s_tijs====== \n");
        content.push_str("======================== \n");

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
