//! Definition file generation for mVMC compatibility.
//!
//! This module provides functionality to generate the various .def files
//! that are created by the C implementation of mVMC when using Standard Mode.
//!
//! # Generated Files
//!
//! - `modpara.def`: Model parameters
//! - `locspn.def`: Local spin configuration
//! - `trans.def`: Transfer integrals
//! - `exchange.def`: Exchange interactions
//! - `coulombinter.def`: Coulomb interactions
//! - `hund.def`: Hund coupling
//! - `greenone.def`: One-body Green's functions
//! - `greentwo.def`: Two-body Green's functions
//! - `gutzwilleridx.def`: Gutzwiller factors
//! - `jastrowidx.def`: Jastrow factors
//! - `orbitalidx.def`: Orbital indices
//! - `qptransidx.def`: QP transformation indices
//! - `namelist.def`: File list
//!
//! # Reference
//!
//! Based on mVMC C implementation:
//! - `mVMC/src/mVMC/StdFace_main.c`: Main function for Standard Mode
//! - `mVMC/src/mVMC/StdFace_*`: Various definition file generators

pub mod generator;
pub mod modpara;
pub mod locspn;
pub mod trans;
pub mod exchange;
pub mod coulombinter;
pub mod hund;
pub mod greenone;
pub mod greentwo;
pub mod gutzwilleridx;
pub mod jastrowidx;
pub mod orbitalidx;
pub mod qptransidx;
pub mod namelist;

// Re-export commonly used types
pub use generator::{DefFileGenerator, DefFileConfig};
pub use modpara::ModParaGenerator;
pub use locspn::LocSpnGenerator;
pub use trans::TransGenerator;
pub use exchange::ExchangeGenerator;
pub use coulombinter::CoulombInterGenerator;
pub use hund::HundGenerator;
pub use greenone::GreenOneGenerator;
pub use greentwo::GreenTwoGenerator;
pub use gutzwilleridx::GutzwillerIdxGenerator;
pub use jastrowidx::JastrowIdxGenerator;
pub use orbitalidx::OrbitalIdxGenerator;
pub use qptransidx::QPTransIdxGenerator;
pub use namelist::NamelistGenerator;

