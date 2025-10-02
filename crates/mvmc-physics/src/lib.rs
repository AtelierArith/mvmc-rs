//! mvmc-physics: Physical models and observables for quantum lattice systems.
//!
//! This crate provides implementations of various quantum lattice models
//! and their associated physical observables for use in variational Monte Carlo
//! calculations.
//!
//! # Examples
//!
//! ```
//! use mvmc_physics::lattice::ChainLattice;
//! use mvmc_physics::hamiltonian::{HubbardHamiltonian, Spin};
//! use mvmc_physics::observables::{EnergyCalculator, Observable};
//!
//! // Create a 1D chain lattice
//! let lattice = ChainLattice::new(6, true).unwrap();
//!
//! // Create a Hubbard Hamiltonian
//! let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
//!
//! // Create an energy calculator
//! let energy_calc = EnergyCalculator::new(Box::new(hamiltonian));
//!
//! // Calculate energy for a configuration
//! let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up, Spin::Down, Spin::Empty];
//! let energy = energy_calc.calculate(&config);
//! ```

pub mod lattice;
pub mod hamiltonian;
pub mod observables;
pub mod wavefunction;

/// Common result type for physics operations.
pub type Result<T> = std::result::Result<T, anyhow::Error>;

// Re-export commonly used types
pub use lattice::{Lattice, ChainLattice, SquareLattice};
pub use hamiltonian::{Hamiltonian, Spin, HubbardHamiltonian, HeisenbergHamiltonian};
pub use observables::{Observable, EnergyCalculator, MagnetizationCalculator, CorrelationCalculator};
pub use wavefunction::{
    SlaterDeterminant, PfaffianWavefunction, CombinedWavefunction,
    ParticleNumberProjector, TotalSpinProjector, MomentumProjector,
    SpatialSymmetryProjector, CombinedProjector, Projector, Wavefunction,
};
