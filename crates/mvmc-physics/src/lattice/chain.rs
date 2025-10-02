//! One-dimensional chain lattice implementation.
//!
//! This module provides the `ChainLattice` structure for 1D chain geometries,
//! commonly used in quantum spin chains and 1D Hubbard models.

use super::{Lattice, LatticeError, Result};

/// One-dimensional chain lattice.
///
/// This lattice represents a 1D chain of sites with optional periodic boundary conditions.
/// Each site is connected to its immediate neighbors (left and right).
///
/// # Examples
///
/// ```
/// use mvmc_physics::lattice::{ChainLattice, Lattice};
///
/// // Create a 6-site chain with periodic boundary conditions
/// let lattice = ChainLattice::new(6, true).unwrap();
/// assert_eq!(lattice.n_sites(), 6);
/// assert_eq!(lattice.neighbors(0), vec![5, 1]); // periodic: left neighbor is site 5
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChainLattice {
    /// Length of the chain (number of sites)
    length: usize,
    /// Whether to use periodic boundary conditions
    periodic: bool,
}

impl ChainLattice {
    /// Creates a new chain lattice.
    ///
    /// # Arguments
    /// * `length` - Number of sites in the chain (must be > 0)
    /// * `periodic` - Whether to use periodic boundary conditions
    ///
    /// # Returns
    /// * `Ok(ChainLattice)` - Successfully created lattice
    /// * `Err(LatticeError)` - If length is 0
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::lattice::{ChainLattice, Lattice};
    ///
    /// let lattice = ChainLattice::new(6, true).unwrap();
    /// assert_eq!(lattice.n_sites(), 6);
    /// ```
    pub fn new(length: usize, periodic: bool) -> Result<Self> {
        if length == 0 {
            return Err(LatticeError::InvalidParameters {
                message: "Chain length must be greater than 0".to_string(),
            });
        }

        Ok(Self { length, periodic })
    }

    /// Returns the length of the chain.
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns whether the chain has periodic boundary conditions.
    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// Returns the left neighbor of a site.
    ///
    /// # Arguments
    /// * `site` - The site index
    ///
    /// # Returns
    /// * `Some(usize)` - Left neighbor index if it exists
    /// * `None` - If no left neighbor (open boundary conditions and site is at left end)
    pub fn left_neighbor(&self, site: usize) -> Option<usize> {
        if site >= self.length {
            return None;
        }

        if site == 0 {
            if self.periodic {
                Some(self.length - 1)
            } else {
                None
            }
        } else {
            Some(site - 1)
        }
    }

    /// Returns the right neighbor of a site.
    ///
    /// # Arguments
    /// * `site` - The site index
    ///
    /// # Returns
    /// * `Some(usize)` - Right neighbor index if it exists
    /// * `None` - If no right neighbor (open boundary conditions and site is at right end)
    pub fn right_neighbor(&self, site: usize) -> Option<usize> {
        if site >= self.length {
            return None;
        }

        if site == self.length - 1 {
            if self.periodic {
                Some(0)
            } else {
                None
            }
        } else {
            Some(site + 1)
        }
    }

    /// Returns the Manhattan distance between two sites.
    ///
    /// For a chain lattice, this is simply the absolute difference in site indices,
    /// taking into account periodic boundary conditions.
    pub fn manhattan_distance(&self, i: usize, j: usize) -> usize {
        if i >= self.length || j >= self.length {
            return usize::MAX; // Invalid sites
        }

        if self.periodic {
            let diff = if i > j { i - j } else { j - i };
            let periodic_diff = self.length - diff;
            diff.min(periodic_diff)
        } else {
            if i > j { i - j } else { j - i }
        }
    }
}

impl Lattice for ChainLattice {
    fn n_sites(&self) -> usize {
        self.length
    }

    fn neighbors(&self, site: usize) -> Vec<usize> {
        if site >= self.length {
            return Vec::new();
        }

        let mut neighbors = Vec::new();

        if let Some(left) = self.left_neighbor(site) {
            neighbors.push(left);
        }

        if let Some(right) = self.right_neighbor(site) {
            neighbors.push(right);
        }

        neighbors
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        self.manhattan_distance(i, j) as f64
    }

    fn dimension(&self) -> usize {
        1
    }

    fn size(&self) -> Vec<usize> {
        vec![self.length]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_chain_creation() {
        let lattice = ChainLattice::new(6, true).unwrap();
        assert_eq!(lattice.length(), 6);
        assert!(lattice.is_periodic());
    }

    #[test]
    fn test_chain_creation_zero_length() {
        let result = ChainLattice::new(0, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_neighbors_periodic() {
        let lattice = ChainLattice::new(6, true).unwrap();

        // Site 0: neighbors are 5 (left) and 1 (right)
        let neighbors = lattice.neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&5));
        assert!(neighbors.contains(&1));

        // Site 3: neighbors are 2 (left) and 4 (right)
        let neighbors = lattice.neighbors(3);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&4));
    }

    #[test]
    fn test_neighbors_open() {
        let lattice = ChainLattice::new(6, false).unwrap();

        // Site 0: only right neighbor (1)
        let neighbors = lattice.neighbors(0);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&1));

        // Site 5: only left neighbor (4)
        let neighbors = lattice.neighbors(5);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&4));

        // Site 3: both neighbors (2 and 4)
        let neighbors = lattice.neighbors(3);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&4));
    }

    #[test]
    fn test_neighbors_out_of_bounds() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let neighbors = lattice.neighbors(10);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_distance_periodic() {
        let lattice = ChainLattice::new(6, true).unwrap();

        // Adjacent sites
        assert_abs_diff_eq!(lattice.distance(0, 1), 1.0);
        assert_abs_diff_eq!(lattice.distance(1, 0), 1.0);

        // Sites at opposite ends (periodic)
        assert_abs_diff_eq!(lattice.distance(0, 5), 1.0);
        assert_abs_diff_eq!(lattice.distance(5, 0), 1.0);

        // Same site
        assert_abs_diff_eq!(lattice.distance(3, 3), 0.0);
    }

    #[test]
    fn test_distance_open() {
        let lattice = ChainLattice::new(6, false).unwrap();

        // Adjacent sites
        assert_abs_diff_eq!(lattice.distance(0, 1), 1.0);
        assert_abs_diff_eq!(lattice.distance(1, 0), 1.0);

        // Sites at opposite ends (open)
        assert_abs_diff_eq!(lattice.distance(0, 5), 5.0);
        assert_abs_diff_eq!(lattice.distance(5, 0), 5.0);

        // Same site
        assert_abs_diff_eq!(lattice.distance(3, 3), 0.0);
    }

    #[test]
    fn test_lattice_trait() {
        let lattice = ChainLattice::new(4, true).unwrap();

        assert_eq!(lattice.n_sites(), 4);
        assert_eq!(lattice.dimension(), 1);
        assert_eq!(lattice.size(), vec![4]);
    }

    #[test]
    fn test_left_right_neighbors() {
        let periodic_lattice = ChainLattice::new(4, true).unwrap();
        let open_lattice = ChainLattice::new(4, false).unwrap();

        // Periodic boundary conditions
        assert_eq!(periodic_lattice.left_neighbor(0), Some(3));
        assert_eq!(periodic_lattice.right_neighbor(0), Some(1));
        assert_eq!(periodic_lattice.left_neighbor(3), Some(2));
        assert_eq!(periodic_lattice.right_neighbor(3), Some(0));

        // Open boundary conditions
        assert_eq!(open_lattice.left_neighbor(0), None);
        assert_eq!(open_lattice.right_neighbor(0), Some(1));
        assert_eq!(open_lattice.left_neighbor(3), Some(2));
        assert_eq!(open_lattice.right_neighbor(3), None);
    }

    #[test]
    fn test_manhattan_distance() {
        let lattice = ChainLattice::new(6, true).unwrap();

        // Adjacent sites
        assert_eq!(lattice.manhattan_distance(0, 1), 1);
        assert_eq!(lattice.manhattan_distance(1, 0), 1);

        // Sites at opposite ends (periodic)
        assert_eq!(lattice.manhattan_distance(0, 5), 1);
        assert_eq!(lattice.manhattan_distance(5, 0), 1);

        // Same site
        assert_eq!(lattice.manhattan_distance(3, 3), 0);

        // Invalid sites
        assert_eq!(lattice.manhattan_distance(0, 10), usize::MAX);
    }
}

// Property-based tests using proptest
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_chain_neighbors_count(length in 1usize..100, site in 0usize..100) {
            let lattice = ChainLattice::new(length, true).unwrap();
            let neighbors = lattice.neighbors(site);

            if site < length {
                // Valid site: should have 2 neighbors (periodic) or 1-2 (open)
                prop_assert!(neighbors.len() >= 1);
                prop_assert!(neighbors.len() <= 2);
            } else {
                // Invalid site: should have no neighbors
                prop_assert!(neighbors.is_empty());
            }
        }

        #[test]
        fn prop_chain_distance_symmetric(length in 1usize..100, i in 0usize..100, j in 0usize..100) {
            let lattice = ChainLattice::new(length, true).unwrap();
            let dist_ij = lattice.distance(i, j);
            let dist_ji = lattice.distance(j, i);

            // Distance should be symmetric
            prop_assert!((dist_ij - dist_ji).abs() < 1e-10);
        }

        #[test]
        fn prop_chain_distance_triangle_inequality(length in 1usize..100, i in 0usize..100, j in 0usize..100, k in 0usize..100) {
            let lattice = ChainLattice::new(length, true).unwrap();

            if i < length && j < length && k < length {
                let dist_ij = lattice.distance(i, j);
                let dist_jk = lattice.distance(j, k);
                let dist_ik = lattice.distance(i, k);

                // Triangle inequality: d(i,k) <= d(i,j) + d(j,k)
                prop_assert!(dist_ik <= dist_ij + dist_jk + 1e-10);
            }
        }
    }
}
