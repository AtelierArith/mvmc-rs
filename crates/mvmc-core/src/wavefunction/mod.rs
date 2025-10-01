//! Wavefunction representations for VMC calculations
//!
//! This module provides wavefunction structures used in variational Monte Carlo:
//! - Slater determinants for fermion systems
//! - Pfaffian for pairing wavefunctions
//! - Projection operators for spin constraints
//! - RBM (Restricted Boltzmann Machine) corrections

pub mod pfaffian;
pub mod slater;

pub use pfaffian::{PfaffianMatrix, PfaffianWavefunction};
pub use slater::{SlaterDeterminant, SlaterMatrix};
