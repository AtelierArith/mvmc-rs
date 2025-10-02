//! Generator for greenone.def file.
//!
//! This module generates the greenone.def file which contains one-body
//! Green's function definitions.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for greenone.def file.
pub struct GreenOneGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> GreenOneGenerator<'a> {
    /// Creates a new greenone.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the greenone.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("===============================\n");
        content.push_str("NCisAjs          2\n");
        content.push_str("===============================\n");
        content.push_str("======== Green functions ======\n");
        content.push_str("===============================\n");

        // One-body Green's functions
        content.push_str("    0     0     0     0\n");
        content.push_str("    0     1     0     1\n");

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
