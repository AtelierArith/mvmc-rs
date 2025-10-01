//! Lattice structures for quantum lattice models.
//!
//! This module provides various lattice geometries used in quantum lattice models
//! such as Hubbard, Heisenberg, and Kondo models.

pub mod chain;
pub mod square;

pub use chain::ChainLattice;
pub use square::SquareLattice;

/// Trait for lattice structures.
///
/// This trait defines the common interface for all lattice types,
/// allowing for generic algorithms that work with any lattice geometry.
pub trait Lattice: std::fmt::Debug {
    /// Returns the total number of sites in the lattice.
    fn n_sites(&self) -> usize;

    /// Returns the neighbors of a given site.
    ///
    /// # Arguments
    /// * `site` - The site index (0-based)
    ///
    /// # Returns
    /// A vector of neighbor site indices.
    fn neighbors(&self, site: usize) -> Vec<usize>;

    /// Returns the distance between two sites.
    ///
    /// # Arguments
    /// * `i` - First site index
    /// * `j` - Second site index
    ///
    /// # Returns
    /// The distance between the sites.
    fn distance(&self, i: usize, j: usize) -> f64;

    /// Returns the dimension of the lattice.
    fn dimension(&self) -> usize;

    /// Returns the lattice size in each dimension.
    fn size(&self) -> Vec<usize>;
}

/// Error types for lattice operations.
#[derive(Debug, thiserror::Error)]
pub enum LatticeError {
    #[error("Site index {site} is out of bounds for lattice with {n_sites} sites")]
    SiteOutOfBounds { site: usize, n_sites: usize },

    #[error("Invalid lattice parameters: {message}")]
    InvalidParameters { message: String },
}

/// Result type for lattice operations.
pub type Result<T> = std::result::Result<T, LatticeError>;
