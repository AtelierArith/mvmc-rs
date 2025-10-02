//! Generator for modpara.def file.
//!
//! This module generates the modpara.def file which contains model parameters
//! for the mVMC calculation.

use super::generator::DefFileConfig;
use crate::Result;
use std::path::Path;

/// Generator for modpara.def file.
pub struct ModParaGenerator<'a> {
    config: &'a DefFileConfig,
}

impl<'a> ModParaGenerator<'a> {
    /// Creates a new modpara.def generator.
    pub fn new(config: &'a DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates the modpara.def file.
    pub fn generate<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("--------------------\n");
        content.push_str("Model_Parameters   0\n");
        content.push_str("--------------------\n");
        content.push_str("VMC_Cal_Parameters\n");
        content.push_str("--------------------\n");

        // File headers
        content.push_str(&format!("CDataFileHead  {}\n", self.config.c_data_file_head));
        content.push_str(&format!("CParaFileHead  {}\n", self.config.c_para_file_head));
        content.push_str("--------------------\n");

        // Calculation modes
        content.push_str("NVMCCalMode    0\n");
        content.push_str("NLanczosMode   0\n");
        content.push_str("--------------------\n");

        // Data parameters
        content.push_str(&format!("NDataIdxStart  {}\n", self.config.n_data_idx_start));
        content.push_str(&format!("NDataQtySmp    {}\n", self.config.n_data_qty_smp));
        content.push_str("--------------------\n");

        // System parameters
        content.push_str(&format!("Nsite          {}\n", self.config.nsite));
        content.push_str("Ncond          0    \n");
        content.push_str(&format!("2Sz            {}\n", self.config.two_sz));
        content.push_str("NSPGaussLeg    8\n");
        content.push_str("NSPStot        0\n");
        content.push_str("NMPTrans       -1\n");

        // SR optimization parameters
        content.push_str(&format!("NSROptItrStep  {}\n", self.config.nsr_opt_itr_step));
        content.push_str(&format!("NSROptItrSmp   {}\n", self.config.nsr_opt_itr_smp.unwrap_or(10)));
        content.push_str(&format!("DSROptRedCut   {:.10}\n", self.config.dsr_opt_red_cut.unwrap_or(0.001)));
        content.push_str(&format!("DSROptStaDel   {:.10}\n", self.config.dsr_opt_sta_del.unwrap_or(0.02)));
        content.push_str(&format!("DSROptStepDt   {:.10}\n", self.config.dsr_opt_step_dt.unwrap_or(0.02)));

        // VMC parameters
        content.push_str(&format!("NVMCWarmUp     {}\n", self.config.nvmc_warm_up.unwrap_or(10)));
        content.push_str(&format!("NVMCInterval   {}\n", self.config.nvmc_interval.unwrap_or(1)));
        content.push_str(&format!("NVMCSample     {}\n", self.config.nvmc_sample.unwrap_or(1000)));
        content.push_str("NExUpdatePath  2\n");
        content.push_str(&format!("RndSeed        {}\n", self.config.rnd_seed.unwrap_or(123456789)));
        content.push_str("NSplitSize     1\n");
        content.push_str("NStore         1\n");
        content.push_str("NSRCG          0\n");

        std::fs::write(output_path, content)?;
        Ok(())
    }
}
