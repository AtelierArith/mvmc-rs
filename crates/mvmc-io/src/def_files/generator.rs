//! Main generator for all definition files.
//!
//! This module provides the main `DefFileGenerator` that coordinates
//! the generation of all .def files from a StdFace configuration.

use super::{
    ModParaGenerator, LocSpnGenerator, TransGenerator, ExchangeGenerator,
    CoulombInterGenerator, HundGenerator, GreenOneGenerator, GreenTwoGenerator,
    GutzwillerIdxGenerator, JastrowIdxGenerator, OrbitalIdxGenerator,
    QPTransIdxGenerator, NamelistGenerator, ZvoOutGenerator,
};
use crate::Result;
use std::path::Path;

/// Configuration for definition file generation.
#[derive(Debug, Clone)]
pub struct DefFileConfig {
    /// Number of lattice sites
    pub nsite: usize,
    /// Number of electrons
    pub ne: usize,
    /// Total spin (2*Sz)
    pub two_sz: i32,
    /// Model type ("Hubbard", "Spin", etc.)
    pub model: String,
    /// Lattice type ("chain", "square", etc.)
    pub lattice: String,
    /// Exchange coupling (for Spin model)
    pub j: Option<f64>,
    /// Hopping parameter (for Hubbard model)
    pub t: Option<f64>,
    /// On-site interaction (for Hubbard model)
    pub u: Option<f64>,
    /// Chemical potential
    pub mu: Option<f64>,
    /// Number of SR optimization steps
    pub nsr_opt_itr_step: usize,
    /// Number of SR optimization samples
    pub nsr_opt_itr_smp: Option<usize>,
    /// SR optimization reduction cutoff
    pub dsr_opt_red_cut: Option<f64>,
    /// SR optimization step delta
    pub dsr_opt_step_dt: Option<f64>,
    /// SR optimization start delta
    pub dsr_opt_sta_del: Option<f64>,
    /// Number of VMC warm-up steps
    pub nvmc_warm_up: Option<usize>,
    /// VMC sampling interval
    pub nvmc_interval: Option<usize>,
    /// Number of VMC samples
    pub nvmc_sample: Option<usize>,
    /// Random seed
    pub rnd_seed: Option<u64>,
    /// Data file head
    pub c_data_file_head: String,
    /// Parameter file head
    pub c_para_file_head: String,
    /// Data index start
    pub n_data_idx_start: usize,
    /// Number of data samples
    pub n_data_qty_smp: usize,
}

impl Default for DefFileConfig {
    fn default() -> Self {
        Self {
            nsite: 4,
            ne: 4,
            two_sz: 0,
            model: "Hubbard".to_string(),
            lattice: "chain".to_string(),
            j: None,
            t: Some(1.0),
            u: Some(4.0),
            mu: Some(0.0),
            nsr_opt_itr_step: 100,
            nsr_opt_itr_smp: Some(10),
            dsr_opt_red_cut: Some(0.001),
            dsr_opt_step_dt: Some(0.02),
            dsr_opt_sta_del: Some(0.02),
            nvmc_warm_up: Some(10),
            nvmc_interval: Some(1),
            nvmc_sample: Some(1000),
            rnd_seed: Some(123456789),
            c_data_file_head: "zvo".to_string(),
            c_para_file_head: "zqp".to_string(),
            n_data_idx_start: 1,
            n_data_qty_smp: 1,
        }
    }
}

/// Main generator for all definition files.
pub struct DefFileGenerator {
    config: DefFileConfig,
}

impl DefFileGenerator {
    /// Creates a new definition file generator.
    pub fn new(config: DefFileConfig) -> Self {
        Self { config }
    }

    /// Generates all definition files in the specified directory.
    pub fn generate_all<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        // Generate individual definition files
        self.generate_modpara(output_dir)?;
        self.generate_locspn(output_dir)?;
        self.generate_trans(output_dir)?;
        self.generate_exchange(output_dir)?;
        self.generate_coulombinter(output_dir)?;
        self.generate_hund(output_dir)?;
        self.generate_greenone(output_dir)?;
        self.generate_greentwo(output_dir)?;
        self.generate_gutzwilleridx(output_dir)?;
        self.generate_jastrowidx(output_dir)?;
        self.generate_orbitalidx(output_dir)?;
        self.generate_qptransidx(output_dir)?;
        self.generate_namelist(output_dir)?;
        self.generate_zvo_out(output_dir)?;

        Ok(())
    }

    /// Generates modpara.def file.
    pub fn generate_modpara<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = ModParaGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("modpara.def"))
    }

    /// Generates locspn.def file.
    pub fn generate_locspn<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = LocSpnGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("locspn.def"))
    }

    /// Generates trans.def file.
    pub fn generate_trans<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = TransGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("trans.def"))
    }

    /// Generates exchange.def file.
    pub fn generate_exchange<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = ExchangeGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("exchange.def"))
    }

    /// Generates coulombinter.def file.
    pub fn generate_coulombinter<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = CoulombInterGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("coulombinter.def"))
    }

    /// Generates hund.def file.
    pub fn generate_hund<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = HundGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("hund.def"))
    }

    /// Generates greenone.def file.
    pub fn generate_greenone<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = GreenOneGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("greenone.def"))
    }

    /// Generates greentwo.def file.
    pub fn generate_greentwo<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = GreenTwoGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("greentwo.def"))
    }

    /// Generates gutzwilleridx.def file.
    pub fn generate_gutzwilleridx<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = GutzwillerIdxGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("gutzwilleridx.def"))
    }

    /// Generates jastrowidx.def file.
    pub fn generate_jastrowidx<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = JastrowIdxGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("jastrowidx.def"))
    }

    /// Generates orbitalidx.def file.
    pub fn generate_orbitalidx<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = OrbitalIdxGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("orbitalidx.def"))
    }

    /// Generates qptransidx.def file.
    pub fn generate_qptransidx<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = QPTransIdxGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("qptransidx.def"))
    }

    /// Generates namelist.def file.
    pub fn generate_namelist<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = NamelistGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("namelist.def"))
    }

    /// Generates zvo_out_001.dat file.
    pub fn generate_zvo_out<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let generator = ZvoOutGenerator::new(&self.config);
        generator.generate(output_dir.as_ref().join("zvo_out_001.dat"))
    }
}
