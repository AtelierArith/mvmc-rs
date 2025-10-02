//! Slater determinant wavefunction implementation.
//!
//! This module provides the Slater determinant wavefunction for VMC calculations.
//! The Slater determinant is constructed from single-particle orbitals and
//! represents the many-body wavefunction for fermionic systems.

use crate::Result;
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;

/// Slater determinant wavefunction.
///
/// The Slater determinant is constructed from single-particle orbitals
/// and represents the many-body wavefunction for fermionic systems.
#[derive(Debug, Clone)]
pub struct SlaterDeterminant {
    /// Number of lattice sites
    nsite: usize,
    /// Number of electrons
    ne: usize,
    /// Single-particle orbital coefficients
    /// Shape: (nsite, ne)
    orbitals: Array2<Complex64>,
    /// Cached determinant value
    cached_det: Option<Complex64>,
    /// Cached inverse matrix for efficient updates
    cached_inverse: Option<Array2<Complex64>>,
}

impl SlaterDeterminant {
    /// Creates a new Slater determinant.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `orbitals` - Single-particle orbital coefficients (nsite, ne)
    ///
    /// # Returns
    /// * `Result<Self>` - The constructed Slater determinant
    pub fn new(nsite: usize, ne: usize, orbitals: Array2<Complex64>) -> Result<Self> {
        if orbitals.shape() != [nsite, ne] {
            return Err(anyhow::anyhow!(
                "Orbital matrix shape mismatch: expected ({}, {}), got {:?}",
                nsite, ne, orbitals.shape()
            ));
        }

        if ne > nsite {
            return Err(anyhow::anyhow!(
                "Number of electrons ({}) cannot exceed number of sites ({})",
                ne, nsite
            ));
        }

        Ok(Self {
            nsite,
            ne,
            orbitals,
            cached_det: None,
            cached_inverse: None,
        })
    }

    /// Creates a Slater determinant with random orbitals.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `seed` - Random seed for orbital generation
    ///
    /// # Returns
    /// * `Result<Self>` - The constructed Slater determinant
    pub fn new_random(nsite: usize, ne: usize, seed: u64) -> Result<Self> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut orbitals = Array2::zeros((nsite, ne));

        for i in 0..nsite {
            for j in 0..ne {
                let real = rng.gen_range(-1.0..1.0);
                let imag = rng.gen_range(-1.0..1.0);
                orbitals[[i, j]] = Complex64::new(real, imag);
            }
        }

        Self::new(nsite, ne, orbitals)
    }

    /// Creates a Slater determinant with plane wave orbitals.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Returns
    /// * `Result<Self>` - The constructed Slater determinant
    pub fn new_plane_wave(nsite: usize, ne: usize) -> Result<Self> {
        let mut orbitals = Array2::zeros((nsite, ne));

        for i in 0..nsite {
            for j in 0..ne {
                let k = 2.0 * std::f64::consts::PI * j as f64 / nsite as f64;
                let phase = k * i as f64;
                orbitals[[i, j]] = Complex64::from_polar(1.0, phase);
            }
        }

        Self::new(nsite, ne, orbitals)
    }

    /// Returns the number of lattice sites.
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons.
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns a reference to the orbital matrix.
    pub fn orbitals(&self) -> &Array2<Complex64> {
        &self.orbitals
    }

    /// Returns a mutable reference to the orbital matrix.
    pub fn orbitals_mut(&mut self) -> &mut Array2<Complex64> {
        self.cached_det = None;
        self.cached_inverse = None;
        &mut self.orbitals
    }

    /// Calculates the Slater determinant for a given spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    /// * `Complex64` - The determinant value
    pub fn calculate_determinant(&self, spin_config: &[u8]) -> Complex64 {
        if spin_config.len() != self.nsite {
            return Complex64::new(0.0, 0.0);
        }

        // Extract occupied sites for up and down spins
        let mut up_sites = Vec::new();
        let mut down_sites = Vec::new();

        for (i, &spin) in spin_config.iter().enumerate() {
            match spin {
                1 => up_sites.push(i),      // up spin only
                2 => down_sites.push(i),    // down spin only
                3 => {                      // both spins
                    up_sites.push(i);
                    down_sites.push(i);
                }
                _ => {} // empty site
            }
        }

        // Calculate determinant for up spins
        let up_det = if up_sites.len() > 0 {
            self.calculate_determinant_for_sites(&up_sites)
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Calculate determinant for down spins
        let down_det = if down_sites.len() > 0 {
            self.calculate_determinant_for_sites(&down_sites)
        } else {
            Complex64::new(1.0, 0.0)
        };

        up_det * down_det
    }

    /// Calculates the determinant for a specific set of sites.
    ///
    /// # Arguments
    /// * `sites` - List of occupied sites
    ///
    /// # Returns
    /// * `Complex64` - The determinant value
    fn calculate_determinant_for_sites(&self, sites: &[usize]) -> Complex64 {
        if sites.len() == 0 {
            return Complex64::new(1.0, 0.0);
        }

        if sites.len() > self.ne {
            return Complex64::new(0.0, 0.0);
        }

        // Create submatrix for occupied sites
        let mut submatrix = Array2::zeros((sites.len(), sites.len()));
        for (i, &site) in sites.iter().enumerate() {
            for j in 0..sites.len() {
                if j < self.ne {
                    submatrix[[i, j]] = self.orbitals[[site, j]];
                }
            }
        }

        // Calculate determinant using LU decomposition
        self.determinant_lu(&submatrix)
    }

    /// Calculates determinant using LU decomposition.
    ///
    /// # Arguments
    /// * `matrix` - Square matrix
    ///
    /// # Returns
    /// * `Complex64` - The determinant value
    fn determinant_lu(&self, matrix: &Array2<Complex64>) -> Complex64 {
        let n = matrix.nrows();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if n == 1 {
            return matrix[[0, 0]];
        }

        // Perform LU decomposition
        let mut lu = matrix.clone();
        let mut det = Complex64::new(1.0, 0.0);
        let mut sign = 1;

        for k in 0..n - 1 {
            // Find pivot
            let mut max_row = k;
            let mut max_val = lu[[k, k]].norm();

            for i in k + 1..n {
                let val = lu[[i, k]].norm();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows if necessary
            if max_row != k {
                // Swap rows manually
                for j in k..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_row, j]];
                    lu[[max_row, j]] = temp;
                }
                sign *= -1;
            }

            // Check for singular matrix
            if lu[[k, k]].norm() < 1e-12 {
                return Complex64::new(0.0, 0.0);
            }

            // Perform elimination
            for i in k + 1..n {
                let factor = lu[[i, k]] / lu[[k, k]];
                lu[[i, k]] = factor;
                for j in k + 1..n {
                    let lu_kj = lu[[k, j]];
                    lu[[i, j]] -= factor * lu_kj;
                }
            }
        }

        // Calculate determinant
        for i in 0..n {
            det *= lu[[i, i]];
        }

        if sign == -1 {
            -det
        } else {
            det
        }
    }

    /// Calculates the ratio of determinants after a spin flip.
    ///
    /// # Arguments
    /// * `spin_config` - Current spin configuration
    /// * `flip_site` - Site to flip
    /// * `flip_from` - Current spin state
    /// * `flip_to` - New spin state
    ///
    /// # Returns
    /// * `Complex64` - The ratio of determinants
    pub fn calculate_ratio(&self, spin_config: &[u8], flip_site: usize, _flip_from: u8, flip_to: u8) -> Complex64 {
        // This is a simplified implementation
        // In practice, this would use the Sherman-Morrison formula for efficiency
        let mut new_config = spin_config.to_vec();
        new_config[flip_site] = flip_to;

        let old_det = self.calculate_determinant(spin_config);
        let new_det = self.calculate_determinant(&new_config);

        if old_det.norm() < 1e-12 {
            Complex64::new(0.0, 0.0)
        } else {
            new_det / old_det
        }
    }

    /// Updates the orbital coefficients.
    ///
    /// # Arguments
    /// * `new_orbitals` - New orbital coefficients
    pub fn update_orbitals(&mut self, new_orbitals: Array2<Complex64>) {
        if new_orbitals.shape() == [self.nsite, self.ne] {
            self.orbitals = new_orbitals;
            self.cached_det = None;
            self.cached_inverse = None;
        }
    }

    /// Normalizes the orbitals.
    pub fn normalize(&mut self) {
        for j in 0..self.ne {
            let norm = (0..self.nsite)
                .map(|i| self.orbitals[[i, j]].norm_sqr())
                .sum::<f64>()
                .sqrt();

            if norm > 1e-12 {
                for i in 0..self.nsite {
                    self.orbitals[[i, j]] /= norm;
                }
            }
        }
    }
}

impl fmt::Display for SlaterDeterminant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SlaterDeterminant(nsite={}, ne={})", self.nsite, self.ne)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_slater_determinant_creation() {
        let nsite = 4;
        let ne = 2;
        let orbitals = Array2::from_shape_fn((nsite, ne), |(i, j)| {
            Complex64::new(i as f64 + j as f64, 0.0)
        });

        let slater = SlaterDeterminant::new(nsite, ne, orbitals).unwrap();
        assert_eq!(slater.nsite(), nsite);
        assert_eq!(slater.ne(), ne);
    }

    #[test]
    fn test_determinant_calculation() {
        let nsite = 2;
        let ne = 2;
        let orbitals = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
        ];

        let slater = SlaterDeterminant::new(nsite, ne, orbitals).unwrap();
        let spin_config = vec![1, 2]; // up, down
        let det = slater.calculate_determinant(&spin_config);

        assert!((det.re - 1.0).abs() < 1e-12);
        assert!(det.im.abs() < 1e-12);
    }

    #[test]
    fn test_plane_wave_orbitals() {
        let nsite = 4;
        let ne = 2;
        let slater = SlaterDeterminant::new_plane_wave(nsite, ne).unwrap();

        assert_eq!(slater.nsite(), nsite);
        assert_eq!(slater.ne(), ne);
    }
}

impl crate::wavefunction::Wavefunction for SlaterDeterminant {
    fn calculate(&self, config: &[u8]) -> Result<Complex64> {
        if config.len() != self.nsite {
            return Err(anyhow::anyhow!("Configuration length mismatch"));
        }

        // Convert u8 configuration to spin configuration
        let spin_config = config.iter().map(|&s| {
            match s {
                1 => crate::hamiltonian::Spin::Up,
                2 => crate::hamiltonian::Spin::Down,
                _ => crate::hamiltonian::Spin::Empty,
            }
        }).collect::<Vec<_>>();

        self.calculate_spin(&spin_config)
    }

    fn nsite(&self) -> usize {
        self.nsite
    }

    fn ne(&self) -> usize {
        self.ne
    }
}
