//! Generator for qptransidx.def file.
//!
//! This module generates the qptransidx.def file which contains
//! QP transformation indices.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for qptransidx.def file.
pub struct QPTransIdxGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> QPTransIdxGenerator<'a> {
    /// Creates a new qptransidx.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the qptransidx.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Calculate number of QP transformations
        let n_qp_trans = 4; // Default for Lsub=4

        // Header
        content.push_str("=============================================\n");
        content.push_str(&format!("NQPTrans          {}\n", n_qp_trans));
        content.push_str("=============================================\n");
        content.push_str("======== TrIdx_TrWeight_and_TrIdx_i_xi ======\n");
        content.push_str("=============================================\n");

        // QP transformation weights
        for i in 0..n_qp_trans {
            content.push_str(&format!("{}    1.00000\n", i));
        }

        // QP transformation indices
        for tr_idx in 0..n_qp_trans {
            for i in 0..self.config.nsite {
                let j = (i + tr_idx) % self.config.nsite;
                content.push_str(&format!("{:4} {:4} {:4}      1\n", tr_idx, i, j));
            }
        }

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
