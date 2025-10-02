//! Generator for namelist.def file.
//!
//! This module generates the namelist.def file which contains
//! the list of all definition files.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for namelist.def file.
pub struct NamelistGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> NamelistGenerator<'a> {
    /// Creates a new namelist.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the namelist.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // File list
        content.push_str("         ModPara  modpara.def\n");
        content.push_str("         LocSpin  locspn.def\n");
        content.push_str("           Trans  trans.def\n");
        content.push_str("    CoulombInter  coulombinter.def\n");
        content.push_str("            Hund  hund.def\n");
        content.push_str("        Exchange  exchange.def\n");
        content.push_str("        OneBodyG  greenone.def\n");
        content.push_str("        TwoBodyG  greentwo.def\n");
        content.push_str("      Gutzwiller  gutzwilleridx.def\n");
        content.push_str("         Jastrow  jastrowidx.def\n");
        content.push_str("         Orbital  orbitalidx.def\n");
        content.push_str("        TransSym  qptransidx.def\n");

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
