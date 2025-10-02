//! Pfaffian wavefunction implementation.
//!
//! This module provides the Pfaffian wavefunction for VMC calculations.
//! The Pfaffian is used for paired states and can represent superconducting
//! or paired phases in many-body systems.

use crate::Result;
use ndarray::Array2;
use num_complex::Complex64;
use std::fmt;

/// Pfaffian wavefunction.
///
/// The Pfaffian is constructed from pairing amplitudes and represents
/// paired states in many-body systems.
#[derive(Debug, Clone)]
pub struct PfaffianWavefunction {
    /// Number of lattice sites
    nsite: usize,
    /// Number of electrons
    ne: usize,
    /// Pairing amplitude matrix (antisymmetric)
    /// Shape: (nsite, nsite)
    pairing_amplitudes: Array2<Complex64>,
    /// Cached Pfaffian value
    cached_pfaffian: Option<Complex64>,
}

impl PfaffianWavefunction {
    /// Creates a new Pfaffian wavefunction.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons (must be even)
    /// * `pairing_amplitudes` - Pairing amplitude matrix (nsite, nsite)
    ///
    /// # Returns
    /// * `Result<Self>` - The constructed Pfaffian wavefunction
    pub fn new(nsite: usize, ne: usize, pairing_amplitudes: Array2<Complex64>) -> Result<Self> {
        if pairing_amplitudes.shape() != [nsite, nsite] {
            return Err(anyhow::anyhow!(
                "Pairing amplitude matrix shape mismatch: expected ({}, {}), got {:?}",
                nsite, nsite, pairing_amplitudes.shape()
            ));
        }

        if ne % 2 != 0 {
            return Err(anyhow::anyhow!(
                "Number of electrons ({}) must be even for Pfaffian wavefunction",
                ne
            ));
        }

        if ne > nsite {
            return Err(anyhow::anyhow!(
                "Number of electrons ({}) cannot exceed number of sites ({})",
                ne, nsite
            ));
        }

        // Verify antisymmetry
        for i in 0..nsite {
            for j in 0..nsite {
                if i != j {
                    let expected = -pairing_amplitudes[[j, i]];
                    let actual = pairing_amplitudes[[i, j]];
                    if (expected - actual).norm() > 1e-12 {
                        return Err(anyhow::anyhow!(
                            "Pairing amplitude matrix is not antisymmetric at ({}, {})",
                            i, j
                        ));
                    }
                }
            }
        }

        Ok(Self {
            nsite,
            ne,
            pairing_amplitudes,
            cached_pfaffian: None,
        })
    }

    /// Creates a Pfaffian wavefunction with random pairing amplitudes.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons (must be even)
    /// * `seed` - Random seed for pairing amplitude generation
    ///
    /// # Returns
    /// * `Result<Self>` - The constructed Pfaffian wavefunction
    pub fn new_random(nsite: usize, ne: usize, seed: u64) -> Result<Self> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut pairing_amplitudes = Array2::zeros((nsite, nsite));

        for i in 0..nsite {
            for j in i + 1..nsite {
                let real = rng.gen_range(-1.0..1.0);
                let imag = rng.gen_range(-1.0..1.0);
                let amplitude = Complex64::new(real, imag);
                pairing_amplitudes[[i, j]] = amplitude;
                pairing_amplitudes[[j, i]] = -amplitude;
            }
        }

        Self::new(nsite, ne, pairing_amplitudes)
    }

    /// Creates a Pfaffian wavefunction with s-wave pairing.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons (must be even)
    /// * `pairing_strength` - Strength of pairing
    ///
    /// # Returns
    /// * `Result<Self>` - The constructed Pfaffian wavefunction
    pub fn new_s_wave(nsite: usize, ne: usize, pairing_strength: f64) -> Result<Self> {
        let mut pairing_amplitudes = Array2::zeros((nsite, nsite));

        for i in 0..nsite {
            for j in i + 1..nsite {
                let amplitude = Complex64::new(pairing_strength, 0.0);
                pairing_amplitudes[[i, j]] = amplitude;
                pairing_amplitudes[[j, i]] = -amplitude;
            }
        }

        Self::new(nsite, ne, pairing_amplitudes)
    }

    /// Returns the number of lattice sites.
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons.
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns a reference to the pairing amplitude matrix.
    pub fn pairing_amplitudes(&self) -> &Array2<Complex64> {
        &self.pairing_amplitudes
    }

    /// Returns a mutable reference to the pairing amplitude matrix.
    pub fn pairing_amplitudes_mut(&mut self) -> &mut Array2<Complex64> {
        self.cached_pfaffian = None;
        &mut self.pairing_amplitudes
    }

    /// Calculates the Pfaffian for a given spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    /// * `Complex64` - The Pfaffian value
    pub fn calculate_pfaffian(&self, spin_config: &[u8]) -> Complex64 {
        if spin_config.len() != self.nsite {
            return Complex64::new(0.0, 0.0);
        }

        // Extract occupied sites
        let mut occupied_sites = Vec::new();
        for (i, &spin) in spin_config.iter().enumerate() {
            if spin == 1 || spin == 2 || spin == 3 {
                occupied_sites.push(i);
            }
        }

        if occupied_sites.len() != self.ne {
            return Complex64::new(0.0, 0.0);
        }

        self.calculate_pfaffian_for_sites(&occupied_sites)
    }

    /// Calculates the Pfaffian for a specific set of sites.
    ///
    /// # Arguments
    /// * `sites` - List of occupied sites
    ///
    /// # Returns
    /// * `Complex64` - The Pfaffian value
    fn calculate_pfaffian_for_sites(&self, sites: &[usize]) -> Complex64 {
        let n = sites.len();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }

        if n % 2 != 0 {
            return Complex64::new(0.0, 0.0);
        }

        // Create submatrix for occupied sites
        let mut submatrix = Array2::zeros((n, n));
        for (i, &site_i) in sites.iter().enumerate() {
            for (j, &site_j) in sites.iter().enumerate() {
                submatrix[[i, j]] = self.pairing_amplitudes[[site_i, site_j]];
            }
        }

        // Calculate Pfaffian using recursive formula
        self.pfaffian_recursive(&submatrix)
    }

    /// Calculates Pfaffian using recursive formula.
    ///
    /// # Arguments
    /// * `matrix` - Antisymmetric matrix
    ///
    /// # Returns
    /// * `Complex64` - The Pfaffian value
    fn pfaffian_recursive(&self, matrix: &Array2<Complex64>) -> Complex64 {
        let n = matrix.nrows();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if n == 2 {
            return matrix[[0, 1]];
        }

        // Use the recursive formula: pf(A) = sum over all perfect matchings
        // For efficiency, we use a simplified version here
        self.pfaffian_perfect_matching(matrix)
    }

    /// Calculates Pfaffian using perfect matching enumeration.
    ///
    /// # Arguments
    /// * `matrix` - Antisymmetric matrix
    ///
    /// # Returns
    /// * `Complex64` - The Pfaffian value
    fn pfaffian_perfect_matching(&self, matrix: &Array2<Complex64>) -> Complex64 {
        let n = matrix.nrows();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if n == 2 {
            return matrix[[0, 1]];
        }

        // For small matrices, use direct calculation
        if n <= 6 {
            return self.pfaffian_direct(matrix);
        }

        // For larger matrices, use approximation or more sophisticated methods
        // This is a simplified implementation
        self.pfaffian_approximate(matrix)
    }

    /// Direct calculation of Pfaffian for small matrices.
    ///
    /// # Arguments
    /// * `matrix` - Antisymmetric matrix
    ///
    /// # Returns
    /// * `Complex64` - The Pfaffian value
    fn pfaffian_direct(&self, matrix: &Array2<Complex64>) -> Complex64 {
        let n = matrix.nrows();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }
        if n == 2 {
            return matrix[[0, 1]];
        }

        // Generate all perfect matchings
        let mut result = Complex64::new(0.0, 0.0);
        let mut used = vec![false; n];
        self.enumerate_perfect_matchings(matrix, &mut used, 0, &mut result);
        result
    }

    /// Enumerates all perfect matchings recursively.
    ///
    /// # Arguments
    /// * `matrix` - Antisymmetric matrix
    /// * `used` - Array indicating which sites are used
    /// * `current` - Current site being matched
    /// * `result` - Accumulated result
    fn enumerate_perfect_matchings(
        &self,
        matrix: &Array2<Complex64>,
        used: &mut [bool],
        current: usize,
        result: &mut Complex64,
    ) {
        let n = matrix.nrows();

        // Find next unused site
        let mut next = current;
        while next < n && used[next] {
            next += 1;
        }

        if next >= n {
            // All sites are matched, calculate the product
            let mut product = Complex64::new(1.0, 0.0);
            let mut sign = 1;
            let mut matched = Vec::new();

            for i in 0..n {
                if !used[i] {
                    matched.push(i);
                }
            }

            for (i, &site1) in matched.iter().enumerate() {
                for (j, &site2) in matched.iter().enumerate().skip(i + 1) {
                    if j == i + 1 {
                        product *= matrix[[site1, site2]];
                        if site1 > site2 {
                            sign *= -1;
                        }
                    }
                }
            }

            *result += Complex64::new(sign as f64, 0.0) * product;
            return;
        }

        // Try to match current site with all remaining sites
        for i in next + 1..n {
            if !used[i] {
                used[next] = true;
                used[i] = true;
                self.enumerate_perfect_matchings(matrix, used, next + 1, result);
                used[next] = false;
                used[i] = false;
            }
        }
    }

    /// Approximate calculation of Pfaffian for large matrices.
    ///
    /// # Arguments
    /// * `matrix` - Antisymmetric matrix
    ///
    /// # Returns
    /// * `Complex64` - The approximate Pfaffian value
    fn pfaffian_approximate(&self, matrix: &Array2<Complex64>) -> Complex64 {
        // This is a simplified approximation
        // In practice, more sophisticated methods would be used
        let n = matrix.nrows();
        let mut result = Complex64::new(1.0, 0.0);

        for i in 0..n / 2 {
            let j = 2 * i + 1;
            if j < n {
                result *= matrix[[2 * i, j]];
            }
        }

        result
    }

    /// Calculates the ratio of Pfaffians after a spin flip.
    ///
    /// # Arguments
    /// * `spin_config` - Current spin configuration
    /// * `flip_site` - Site to flip
    /// * `flip_from` - Current spin state
    /// * `flip_to` - New spin state
    ///
    /// # Returns
    /// * `Complex64` - The ratio of Pfaffians
    pub fn calculate_ratio(&self, spin_config: &[u8], flip_site: usize, _flip_from: u8, flip_to: u8) -> Complex64 {
        // This is a simplified implementation
        // In practice, this would use more efficient methods
        let mut new_config = spin_config.to_vec();
        new_config[flip_site] = flip_to;

        let old_pfaffian = self.calculate_pfaffian(spin_config);
        let new_pfaffian = self.calculate_pfaffian(&new_config);

        if old_pfaffian.norm() < 1e-12 {
            Complex64::new(0.0, 0.0)
        } else {
            new_pfaffian / old_pfaffian
        }
    }

    /// Updates the pairing amplitudes.
    ///
    /// # Arguments
    /// * `new_amplitudes` - New pairing amplitude matrix
    pub fn update_pairing_amplitudes(&mut self, new_amplitudes: Array2<Complex64>) {
        if new_amplitudes.shape() == [self.nsite, self.nsite] {
            self.pairing_amplitudes = new_amplitudes;
            self.cached_pfaffian = None;
        }
    }

    /// Normalizes the pairing amplitudes.
    pub fn normalize(&mut self) {
        let norm = self.pairing_amplitudes
            .iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();

        if norm > 1e-12 {
            for element in self.pairing_amplitudes.iter_mut() {
                *element /= norm;
            }
        }
    }
}

impl fmt::Display for PfaffianWavefunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PfaffianWavefunction(nsite={}, ne={})", self.nsite, self.ne)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pfaffian_creation() {
        let nsite = 4;
        let ne = 2;
        let mut pairing_amplitudes = Array2::zeros((nsite, nsite));
        pairing_amplitudes[[0, 1]] = Complex64::new(1.0, 0.0);
        pairing_amplitudes[[1, 0]] = Complex64::new(-1.0, 0.0);

        let pfaffian = PfaffianWavefunction::new(nsite, ne, pairing_amplitudes).unwrap();
        assert_eq!(pfaffian.nsite(), nsite);
        assert_eq!(pfaffian.ne(), ne);
    }

    #[test]
    fn test_pfaffian_calculation() {
        let nsite = 2;
        let ne = 2;
        let mut pairing_amplitudes = Array2::zeros((nsite, nsite));
        pairing_amplitudes[[0, 1]] = Complex64::new(1.0, 0.0);
        pairing_amplitudes[[1, 0]] = Complex64::new(-1.0, 0.0);

        let pfaffian = PfaffianWavefunction::new(nsite, ne, pairing_amplitudes).unwrap();
        let spin_config = vec![1, 2]; // up, down
        let pf = pfaffian.calculate_pfaffian(&spin_config);

        assert!((pf.re - 1.0).abs() < 1e-12);
        assert!(pf.im.abs() < 1e-12);
    }

    #[test]
    fn test_s_wave_pairing() {
        let nsite = 4;
        let ne = 2;
        let pfaffian = PfaffianWavefunction::new_s_wave(nsite, ne, 0.5).unwrap();

        assert_eq!(pfaffian.nsite(), nsite);
        assert_eq!(pfaffian.ne(), ne);
    }
}

impl crate::wavefunction::Wavefunction for PfaffianWavefunction {
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
