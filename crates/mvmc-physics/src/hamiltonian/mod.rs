//! Hamiltonian implementations for quantum lattice models.
//!
//! This module provides various Hamiltonian implementations for different
//! quantum lattice models such as Hubbard, Heisenberg, and Kondo models.

pub mod hubbard;
pub mod heisenberg;

pub use hubbard::HubbardHamiltonian;
pub use heisenberg::HeisenbergHamiltonian;

use crate::lattice::Lattice;
use num_complex::Complex64;

/// Trait for Hamiltonian implementations.
///
/// This trait defines the common interface for all Hamiltonian types,
/// allowing for generic algorithms that work with any Hamiltonian.
pub trait Hamiltonian: std::fmt::Debug {
    /// Returns the lattice associated with this Hamiltonian.
    fn lattice(&self) -> &dyn Lattice;

    /// Returns the matrix element between two configurations.
    ///
    /// # Arguments
    /// * `config_i` - Initial configuration
    /// * `config_j` - Final configuration
    ///
    /// # Returns
    /// The matrix element <config_i|H|config_j>
    fn matrix_element(&self, config_i: &[Spin], config_j: &[Spin]) -> Complex64;

    /// Returns the diagonal matrix element (expectation value).
    ///
    /// # Arguments
    /// * `config` - Configuration
    ///
    /// # Returns
    /// The diagonal element <config|H|config>
    fn diagonal_element(&self, config: &[Spin]) -> f64;

    /// Returns the total energy of a configuration.
    ///
    /// This is typically the same as `diagonal_element` but may include
    /// additional terms or corrections.
    fn total_energy(&self, config: &[Spin]) -> f64 {
        self.diagonal_element(config)
    }
}

/// Spin states for quantum lattice models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spin {
    /// Up spin
    Up,
    /// Down spin
    Down,
    /// Empty site (for fermionic models)
    Empty,
}

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for Spin {
    type Parameters = ();
    type Strategy = proptest::strategy::Map<std::ops::Range<u8>, fn(u8) -> Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;
        (0..3u8).prop_map(|i| match i {
            0 => Spin::Up,
            1 => Spin::Down,
            _ => Spin::Empty,
        })
    }
}

impl Spin {
    /// Returns the spin value as a number.
    ///
    /// * Up -> 1
    /// * Down -> -1
    /// * Empty -> 0
    pub fn value(&self) -> i32 {
        match self {
            Spin::Up => 1,
            Spin::Down => -1,
            Spin::Empty => 0,
        }
    }

    /// Returns the spin value as a float.
    pub fn value_f64(&self) -> f64 {
        self.value() as f64
    }

    /// Returns true if the site is occupied.
    pub fn is_occupied(&self) -> bool {
        !matches!(self, Spin::Empty)
    }

    /// Returns true if the site is empty.
    pub fn is_empty(&self) -> bool {
        matches!(self, Spin::Empty)
    }
}

/// Error types for Hamiltonian operations.
#[derive(Debug, thiserror::Error)]
pub enum HamiltonianError {
    #[error("Configuration length {config_len} does not match lattice size {lattice_size}")]
    ConfigurationLengthMismatch { config_len: usize, lattice_size: usize },

    #[error("Invalid spin configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Hamiltonian parameter error: {message}")]
    ParameterError { message: String },
}

/// Result type for Hamiltonian operations.
pub type Result<T> = std::result::Result<T, HamiltonianError>;

/// Validates a spin configuration against a lattice.
///
/// # Arguments
/// * `config` - Spin configuration
/// * `lattice` - Lattice structure
///
/// # Returns
/// * `Ok(())` - Configuration is valid
/// * `Err(HamiltonianError)` - Configuration is invalid
pub fn validate_configuration(config: &[Spin], lattice: &dyn Lattice) -> Result<()> {
    if config.len() != lattice.n_sites() {
        return Err(HamiltonianError::ConfigurationLengthMismatch {
            config_len: config.len(),
            lattice_size: lattice.n_sites(),
        });
    }

    Ok(())
}

/// Counts the number of electrons in a configuration.
///
/// # Arguments
/// * `config` - Spin configuration
///
/// # Returns
/// The total number of electrons (occupied sites)
pub fn count_electrons(config: &[Spin]) -> usize {
    config.iter().filter(|&spin| spin.is_occupied()).count()
}

/// Counts the number of up and down spins in a configuration.
///
/// # Arguments
/// * `config` - Spin configuration
///
/// # Returns
/// (number of up spins, number of down spins)
pub fn count_spins(config: &[Spin]) -> (usize, usize) {
    let mut up_count = 0;
    let mut down_count = 0;

    for spin in config {
        match spin {
            Spin::Up => up_count += 1,
            Spin::Down => down_count += 1,
            Spin::Empty => {}
        }
    }

    (up_count, down_count)
}

/// Calculates the total spin Sz of a configuration.
///
/// # Arguments
/// * `config` - Spin configuration
///
/// # Returns
/// The total Sz value (number of up spins - number of down spins)
pub fn total_sz(config: &[Spin]) -> i32 {
    config.iter().map(|spin| spin.value()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ChainLattice;

    #[test]
    fn test_spin_values() {
        assert_eq!(Spin::Up.value(), 1);
        assert_eq!(Spin::Down.value(), -1);
        assert_eq!(Spin::Empty.value(), 0);

        assert_eq!(Spin::Up.value_f64(), 1.0);
        assert_eq!(Spin::Down.value_f64(), -1.0);
        assert_eq!(Spin::Empty.value_f64(), 0.0);
    }

    #[test]
    fn test_spin_occupation() {
        assert!(Spin::Up.is_occupied());
        assert!(Spin::Down.is_occupied());
        assert!(!Spin::Empty.is_occupied());

        assert!(!Spin::Up.is_empty());
        assert!(!Spin::Down.is_empty());
        assert!(Spin::Empty.is_empty());
    }

    #[test]
    fn test_validate_configuration() {
        let lattice = ChainLattice::new(4, true).unwrap();

        // Valid configuration
        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        assert!(validate_configuration(&config, &lattice).is_ok());

        // Invalid configuration (wrong length)
        let config = vec![Spin::Up, Spin::Down];
        assert!(validate_configuration(&config, &lattice).is_err());
    }

    #[test]
    fn test_count_electrons() {
        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        assert_eq!(count_electrons(&config), 3);
    }

    #[test]
    fn test_count_spins() {
        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        assert_eq!(count_spins(&config), (2, 1));
    }

    #[test]
    fn test_total_sz() {
        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        assert_eq!(total_sz(&config), 1); // 2 up - 1 down = 1
    }
}
