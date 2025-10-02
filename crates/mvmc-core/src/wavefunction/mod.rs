//! Wavefunction representations for VMC calculations
//!
//! This module provides wavefunction structures used in variational Monte Carlo:
//! - Slater determinants for fermion systems
//! - Pfaffian for pairing wavefunctions
//! - Projection operators for spin constraints
//! - RBM (Restricted Boltzmann Machine) corrections

pub mod combined;
pub mod doublon_holon;
pub mod jastrow;
pub mod pfaffian;
pub mod projection;
pub mod rbm;
pub mod slater;

pub use combined::{AmplitudeResult, CombinedWavefunction};
pub use doublon_holon::{DoublonHolonFactor, DoublonHolonParameters};
pub use jastrow::{JastrowFactor, JastrowParameters};
pub use pfaffian::{PfaffianMatrix, PfaffianWavefunction};
pub use projection::{
    ProjectionCount, ProjectionOperator, log_projection_ratio, log_projection_value,
    make_projection_count, projection_ratio,
};
pub use rbm::{RBMCounter, RBMParameters, RBMWavefunction};
pub use slater::{SlaterDeterminant, SlaterMatrix};
