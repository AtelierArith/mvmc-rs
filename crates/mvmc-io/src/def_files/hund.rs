//! Generator for hund.def file.
//!
//! This module generates the hund.def file which contains Hund coupling
//! parameters.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for hund.def file.
pub struct HundGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> HundGenerator<'a> {
    /// Creates a new hund.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the hund.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("=============================================\n");
        content.push_str(&format!("NHund         {}\n", self.config.nsite));
        content.push_str("=============================================\n");
        content.push_str("=============== Hund coupling ===============\n");
        content.push_str("=============================================\n");

        // Hund coupling for chain lattice
        let j_value = self.config.j.unwrap_or(-0.5);

        if self.config.lattice == "chain" {
            // Nearest neighbor interactions
            for i in 0..self.config.nsite {
                let j = (i + 1) % self.config.nsite;
                content.push_str(&format!("{:4} {:4} {:20.15}\n", i, j, j_value));
            }
        } else if self.config.lattice == "square" {
            // Square lattice interactions
            let lx = (self.config.nsite as f64).sqrt() as usize;
            let ly = lx;

            for i in 0..self.config.nsite {
                let x = i % lx;
                let y = i / lx;

                // Right neighbor
                if x < lx - 1 {
                    let j = y * lx + (x + 1);
                    content.push_str(&format!("{:4} {:4} {:20.15}\n", i, j, j_value));
                }

                // Bottom neighbor
                if y < ly - 1 {
                    let j = (y + 1) * lx + x;
                    content.push_str(&format!("{:4} {:4} {:20.15}\n", i, j, j_value));
                }
            }
        }

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
