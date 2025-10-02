//! Two-dimensional square lattice implementation.
//!
//! This module provides the `SquareLattice` structure for 2D square geometries,
//! commonly used in 2D Hubbard models and spin systems.

use super::{Lattice, LatticeError, Result};

/// Two-dimensional square lattice.
///
/// This lattice represents a 2D square grid of sites with optional periodic boundary conditions.
/// Each site is connected to its nearest neighbors (up, down, left, right).
///
/// # Examples
///
/// ```
/// use mvmc_physics::lattice::{SquareLattice, Lattice};
///
/// // Create a 3x4 square lattice with periodic boundary conditions
/// let lattice = SquareLattice::new(3, 4, true).unwrap();
/// assert_eq!(lattice.n_sites(), 12);
/// assert_eq!(lattice.neighbors(0), vec![9, 1, 3, 2]); // up, right, down, left
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SquareLattice {
    /// Width of the lattice (number of sites in x-direction)
    width: usize,
    /// Height of the lattice (number of sites in y-direction)
    height: usize,
    /// Whether to use periodic boundary conditions
    periodic: bool,
}

impl SquareLattice {
    /// Creates a new square lattice.
    ///
    /// # Arguments
    /// * `width` - Number of sites in x-direction (must be > 0)
    /// * `height` - Number of sites in y-direction (must be > 0)
    /// * `periodic` - Whether to use periodic boundary conditions
    ///
    /// # Returns
    /// * `Ok(SquareLattice)` - Successfully created lattice
    /// * `Err(LatticeError)` - If width or height is 0
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::lattice::{SquareLattice, Lattice};
    ///
    /// let lattice = SquareLattice::new(3, 4, true).unwrap();
    /// assert_eq!(lattice.n_sites(), 12);
    /// ```
    pub fn new(width: usize, height: usize, periodic: bool) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(LatticeError::InvalidParameters {
                message: "Both width and height must be greater than 0".to_string(),
            });
        }

        Ok(Self { width, height, periodic })
    }

    /// Returns the width of the lattice.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the lattice.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns whether the lattice has periodic boundary conditions.
    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// Converts 2D coordinates to 1D site index.
    ///
    /// # Arguments
    /// * `x` - x-coordinate (0-based)
    /// * `y` - y-coordinate (0-based)
    ///
    /// # Returns
    /// * `Some(usize)` - Site index if coordinates are valid
    /// * `None` - If coordinates are out of bounds
    pub fn coords_to_index(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.width || y >= self.height {
            None
        } else {
            Some(y * self.width + x)
        }
    }

    /// Converts 1D site index to 2D coordinates.
    ///
    /// # Arguments
    /// * `index` - Site index (0-based)
    ///
    /// # Returns
    /// * `Some((usize, usize))` - (x, y) coordinates if index is valid
    /// * `None` - If index is out of bounds
    pub fn index_to_coords(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.n_sites() {
            None
        } else {
            Some((index % self.width, index / self.width))
        }
    }

    /// Returns the neighbors of a site in a specific direction.
    ///
    /// # Arguments
    /// * `site` - The site index
    /// * `direction` - Direction (0=up, 1=right, 2=down, 3=left)
    ///
    /// # Returns
    /// * `Some(usize)` - Neighbor site index if it exists
    /// * `None` - If no neighbor in that direction
    pub fn neighbor_in_direction(&self, site: usize, direction: usize) -> Option<usize> {
        if site >= self.n_sites() {
            return None;
        }

        let (x, y) = self.index_to_coords(site)?;

        let (new_x, new_y) = match direction {
            0 => (x, if y == 0 && self.periodic { self.height - 1 } else if y > 0 { y - 1 } else { return None }),
            1 => (if x == self.width - 1 && self.periodic { 0 } else if x < self.width - 1 { x + 1 } else { return None }, y),
            2 => (x, if y == self.height - 1 && self.periodic { 0 } else if y < self.height - 1 { y + 1 } else { return None }),
            3 => (if x == 0 && self.periodic { self.width - 1 } else if x > 0 { x - 1 } else { return None }, y),
            _ => return None,
        };

        self.coords_to_index(new_x, new_y)
    }

    /// Returns the Manhattan distance between two sites.
    ///
    /// This is the sum of the absolute differences in x and y coordinates,
    /// taking into account periodic boundary conditions.
    pub fn manhattan_distance(&self, i: usize, j: usize) -> usize {
        if i >= self.n_sites() || j >= self.n_sites() {
            return usize::MAX; // Invalid sites
        }

        let (x1, y1) = self.index_to_coords(i).unwrap();
        let (x2, y2) = self.index_to_coords(j).unwrap();

        let dx = if self.periodic {
            let diff = if x1 > x2 { x1 - x2 } else { x2 - x1 };
            let periodic_diff = self.width - diff;
            diff.min(periodic_diff)
        } else {
            if x1 > x2 { x1 - x2 } else { x2 - x1 }
        };

        let dy = if self.periodic {
            let diff = if y1 > y2 { y1 - y2 } else { y2 - y1 };
            let periodic_diff = self.height - diff;
            diff.min(periodic_diff)
        } else {
            if y1 > y2 { y1 - y2 } else { y2 - y1 }
        };

        dx + dy
    }
}

impl Lattice for SquareLattice {
    fn n_sites(&self) -> usize {
        self.width * self.height
    }

    fn neighbors(&self, site: usize) -> Vec<usize> {
        if site >= self.n_sites() {
            return Vec::new();
        }

        let mut neighbors = Vec::new();

        // Check all four directions: up, right, down, left
        for direction in 0..4 {
            if let Some(neighbor) = self.neighbor_in_direction(site, direction) {
                neighbors.push(neighbor);
            }
        }

        neighbors
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        self.manhattan_distance(i, j) as f64
    }

    fn dimension(&self) -> usize {
        2
    }

    fn size(&self) -> Vec<usize> {
        vec![self.width, self.height]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_square_creation() {
        let lattice = SquareLattice::new(3, 4, true).unwrap();
        assert_eq!(lattice.width(), 3);
        assert_eq!(lattice.height(), 4);
        assert_eq!(lattice.n_sites(), 12);
        assert!(lattice.is_periodic());
    }

    #[test]
    fn test_square_creation_zero_dimensions() {
        assert!(SquareLattice::new(0, 4, true).is_err());
        assert!(SquareLattice::new(3, 0, true).is_err());
    }

    #[test]
    fn test_coords_conversion() {
        let lattice = SquareLattice::new(3, 4, true).unwrap();

        // Test coords_to_index
        assert_eq!(lattice.coords_to_index(0, 0), Some(0));
        assert_eq!(lattice.coords_to_index(2, 0), Some(2));
        assert_eq!(lattice.coords_to_index(0, 1), Some(3));
        assert_eq!(lattice.coords_to_index(2, 3), Some(11));
        assert_eq!(lattice.coords_to_index(3, 0), None); // out of bounds
        assert_eq!(lattice.coords_to_index(0, 4), None); // out of bounds

        // Test index_to_coords
        assert_eq!(lattice.index_to_coords(0), Some((0, 0)));
        assert_eq!(lattice.index_to_coords(2), Some((2, 0)));
        assert_eq!(lattice.index_to_coords(3), Some((0, 1)));
        assert_eq!(lattice.index_to_coords(11), Some((2, 3)));
        assert_eq!(lattice.index_to_coords(12), None); // out of bounds
    }

    #[test]
    fn test_neighbors_periodic() {
        let lattice = SquareLattice::new(3, 3, true).unwrap();

        // Corner site (0,0) -> index 0
        let neighbors = lattice.neighbors(0);
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&6)); // up (0,2)
        assert!(neighbors.contains(&1)); // right (1,0)
        assert!(neighbors.contains(&3)); // down (0,1)
        assert!(neighbors.contains(&2)); // left (2,0)

        // Center site (1,1) -> index 4
        let neighbors = lattice.neighbors(4);
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&1)); // up (1,0)
        assert!(neighbors.contains(&5)); // right (2,1)
        assert!(neighbors.contains(&7)); // down (1,2)
        assert!(neighbors.contains(&3)); // left (0,1)
    }

    #[test]
    fn test_neighbors_open() {
        let lattice = SquareLattice::new(3, 3, false).unwrap();

        // Corner site (0,0) -> index 0
        let neighbors = lattice.neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1)); // right (1,0)
        assert!(neighbors.contains(&3)); // down (0,1)

        // Center site (1,1) -> index 4
        let neighbors = lattice.neighbors(4);
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&1)); // up (1,0)
        assert!(neighbors.contains(&5)); // right (2,1)
        assert!(neighbors.contains(&7)); // down (1,2)
        assert!(neighbors.contains(&3)); // left (0,1)
    }

    #[test]
    fn test_neighbors_out_of_bounds() {
        let lattice = SquareLattice::new(3, 3, true).unwrap();
        let neighbors = lattice.neighbors(10);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_direction_neighbors() {
        let lattice = SquareLattice::new(3, 3, true).unwrap();

        // Center site (1,1) -> index 4
        assert_eq!(lattice.neighbor_in_direction(4, 0), Some(1)); // up
        assert_eq!(lattice.neighbor_in_direction(4, 1), Some(5)); // right
        assert_eq!(lattice.neighbor_in_direction(4, 2), Some(7)); // down
        assert_eq!(lattice.neighbor_in_direction(4, 3), Some(3)); // left

        // Corner site (0,0) -> index 0 (periodic)
        assert_eq!(lattice.neighbor_in_direction(0, 0), Some(6)); // up (0,2)
        assert_eq!(lattice.neighbor_in_direction(0, 1), Some(1)); // right (1,0)
        assert_eq!(lattice.neighbor_in_direction(0, 2), Some(3)); // down (0,1)
        assert_eq!(lattice.neighbor_in_direction(0, 3), Some(2)); // left (2,0)
    }

    #[test]
    fn test_distance_periodic() {
        let lattice = SquareLattice::new(3, 3, true).unwrap();

        // Adjacent sites
        assert_abs_diff_eq!(lattice.distance(0, 1), 1.0);
        assert_abs_diff_eq!(lattice.distance(0, 3), 1.0);

        // Diagonal sites
        assert_abs_diff_eq!(lattice.distance(0, 4), 2.0);

        // Opposite corners (periodic)
        assert_abs_diff_eq!(lattice.distance(0, 8), 2.0);

        // Same site
        assert_abs_diff_eq!(lattice.distance(4, 4), 0.0);
    }

    #[test]
    fn test_distance_open() {
        let lattice = SquareLattice::new(3, 3, false).unwrap();

        // Adjacent sites
        assert_abs_diff_eq!(lattice.distance(0, 1), 1.0);
        assert_abs_diff_eq!(lattice.distance(0, 3), 1.0);

        // Diagonal sites
        assert_abs_diff_eq!(lattice.distance(0, 4), 2.0);

        // Opposite corners (open)
        assert_abs_diff_eq!(lattice.distance(0, 8), 4.0);

        // Same site
        assert_abs_diff_eq!(lattice.distance(4, 4), 0.0);
    }

    #[test]
    fn test_lattice_trait() {
        let lattice = SquareLattice::new(2, 3, true).unwrap();

        assert_eq!(lattice.n_sites(), 6);
        assert_eq!(lattice.dimension(), 2);
        assert_eq!(lattice.size(), vec![2, 3]);
    }

    #[test]
    fn test_manhattan_distance() {
        let lattice = SquareLattice::new(3, 3, true).unwrap();

        // Adjacent sites
        assert_eq!(lattice.manhattan_distance(0, 1), 1);
        assert_eq!(lattice.manhattan_distance(0, 3), 1);

        // Diagonal sites
        assert_eq!(lattice.manhattan_distance(0, 4), 2);

        // Same site
        assert_eq!(lattice.manhattan_distance(4, 4), 0);

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
        fn prop_square_neighbors_count(width in 1usize..20, height in 1usize..20, site in 0usize..400) {
            let lattice = SquareLattice::new(width, height, true).unwrap();
            let neighbors = lattice.neighbors(site);

            if site < lattice.n_sites() {
                // Valid site: should have 2-4 neighbors depending on position and boundary conditions
                prop_assert!(neighbors.len() >= 2);
                prop_assert!(neighbors.len() <= 4);
            } else {
                // Invalid site: should have no neighbors
                prop_assert!(neighbors.is_empty());
            }
        }

        #[test]
        fn prop_square_distance_symmetric(width in 1usize..20, height in 1usize..20, i in 0usize..400, j in 0usize..400) {
            let lattice = SquareLattice::new(width, height, true).unwrap();
            let dist_ij = lattice.distance(i, j);
            let dist_ji = lattice.distance(j, i);

            // Distance should be symmetric
            prop_assert!((dist_ij - dist_ji).abs() < 1e-10);
        }

        #[test]
        fn prop_coords_roundtrip(width in 1usize..20, height in 1usize..20, x in 0usize..20, y in 0usize..20) {
            let lattice = SquareLattice::new(width, height, true).unwrap();

            if x < width && y < height {
                let index = lattice.coords_to_index(x, y).unwrap();
                let (new_x, new_y) = lattice.index_to_coords(index).unwrap();

                // Round-trip conversion should preserve coordinates
                prop_assert_eq!(x, new_x);
                prop_assert_eq!(y, new_y);
            }
        }
    }
}
