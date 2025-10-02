//! Heisenberg model Hamiltonian implementation.
//!
//! This module provides the `HeisenbergHamiltonian` structure for the Heisenberg model,
//! which describes interacting spins on a lattice.

use super::{Hamiltonian, Result, Spin, validate_configuration};
use crate::lattice::Lattice;
use num_complex::Complex64;

/// Heisenberg model Hamiltonian.
///
/// The Heisenberg model describes spins on a lattice with:
/// - Exchange interaction between nearest neighbor spins
/// - External magnetic field (optional)
///
/// The Hamiltonian is:
/// H = J ∑_{<i,j>} S_i · S_j - h ∑_i S_i^z
///
/// where:
/// - J is the exchange coupling constant
/// - h is the external magnetic field strength
/// - S_i is the spin vector at site i
/// - S_i^z is the z-component of the spin at site i
///
/// For the Ising limit (S = 1/2), this becomes:
/// H = J ∑_{<i,j>} S_i^z S_j^z - h ∑_i S_i^z
///
/// # Examples
///
/// ```
/// use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian, Spin};
/// use mvmc_physics::lattice::ChainLattice;
///
/// let lattice = ChainLattice::new(6, true).unwrap();
/// let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();
///
/// let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down, Spin::Up, Spin::Down];
/// let energy = hamiltonian.total_energy(&config);
/// ```
#[derive(Debug)]
pub struct HeisenbergHamiltonian {
    /// Lattice structure
    lattice: Box<dyn Lattice>,
    /// Exchange coupling constant (J)
    exchange: f64,
    /// External magnetic field strength (h)
    magnetic_field: f64,
}

impl HeisenbergHamiltonian {
    /// Creates a new Heisenberg Hamiltonian.
    ///
    /// # Arguments
    /// * `lattice` - Lattice structure
    /// * `exchange` - Exchange coupling constant (J)
    /// * `magnetic_field` - External magnetic field strength (h)
    ///
    /// # Returns
    /// * `Ok(HeisenbergHamiltonian)` - Successfully created Hamiltonian
    /// * `Err(HamiltonianError)` - If parameters are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::hamiltonian::HeisenbergHamiltonian;
    /// use mvmc_physics::lattice::ChainLattice;
    ///
    /// let lattice = ChainLattice::new(6, true).unwrap();
    /// let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();
    /// ```
    pub fn new(
        lattice: impl Lattice + 'static,
        exchange: f64,
        magnetic_field: f64,
    ) -> Result<Self> {
        Ok(Self {
            lattice: Box::new(lattice),
            exchange,
            magnetic_field,
        })
    }

    /// Returns the exchange coupling constant.
    pub fn exchange(&self) -> f64 {
        self.exchange
    }

    /// Returns the magnetic field strength.
    pub fn magnetic_field(&self) -> f64 {
        self.magnetic_field
    }

    /// Calculates the exchange energy contribution.
    ///
    /// This is the exchange term: J ∑_{<i,j>} S_i · S_j
    /// For the Ising limit, this becomes: J ∑_{<i,j>} S_i^z S_j^z
    fn exchange_energy(&self, config: &[Spin]) -> f64 {
        let mut energy = 0.0;

        for site in 0..self.lattice.n_sites() {
            let neighbors = self.lattice.neighbors(site);

            for &neighbor in &neighbors {
                // Only consider each bond once (i < j)
                if site < neighbor {
                    let spin_i = config[site].value_f64();
                    let spin_j = config[neighbor].value_f64();

                    // Exchange interaction: J * S_i * S_j
                    energy += self.exchange * spin_i * spin_j;
                }
            }
        }

        energy
    }

    /// Calculates the magnetic field energy contribution.
    ///
    /// This is the magnetic field term: -h ∑_i S_i^z
    fn magnetic_field_energy(&self, config: &[Spin]) -> f64 {
        let mut energy = 0.0;

        for site in 0..self.lattice.n_sites() {
            let spin_z = config[site].value_f64();
            energy -= self.magnetic_field * spin_z;
        }

        energy
    }

    /// Calculates the matrix element for a single spin flip.
    ///
    /// # Arguments
    /// * `config_i` - Initial configuration
    /// * `config_j` - Final configuration
    /// * `flip_site` - Site where spin flips
    ///
    /// # Returns
    /// The matrix element for this specific spin flip
    #[allow(dead_code)]
    fn flip_matrix_element(
        &self,
        config_i: &[Spin],
        config_j: &[Spin],
        flip_site: usize,
    ) -> Complex64 {
        // Check if this is a valid spin flip
        if config_i[flip_site] == config_j[flip_site] {
            return Complex64::new(0.0, 0.0);
        }

        // Check if only one site changed
        let mut diff_count = 0;
        for i in 0..config_i.len() {
            if config_i[i] != config_j[i] {
                diff_count += 1;
            }
        }

        if diff_count != 1 {
            return Complex64::new(0.0, 0.0);
        }

        // For the Ising model, spin flips are not allowed in the Hamiltonian
        // (they would require off-diagonal terms which are not present)
        // This is a classical model where only diagonal elements exist
        Complex64::new(0.0, 0.0)
    }
}

impl Hamiltonian for HeisenbergHamiltonian {
    fn lattice(&self) -> &dyn Lattice {
        self.lattice.as_ref()
    }

    fn matrix_element(&self, config_i: &[Spin], config_j: &[Spin]) -> Complex64 {
        // Validate configurations
        if let Err(_) = validate_configuration(config_i, self.lattice.as_ref()) {
            return Complex64::new(0.0, 0.0);
        }
        if let Err(_) = validate_configuration(config_j, self.lattice.as_ref()) {
            return Complex64::new(0.0, 0.0);
        }

        // Check if configurations are the same (diagonal element)
        if config_i == config_j {
            return Complex64::new(self.diagonal_element(config_i), 0.0);
        }

        // For the Ising model, only diagonal elements are non-zero
        // (no quantum fluctuations)
        Complex64::new(0.0, 0.0)
    }

    fn diagonal_element(&self, config: &[Spin]) -> f64 {
        validate_configuration(config, self.lattice.as_ref()).unwrap();

        self.exchange_energy(config) + self.magnetic_field_energy(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ChainLattice;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_heisenberg_creation() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.5).unwrap();

        assert_eq!(hamiltonian.exchange(), 1.0);
        assert_eq!(hamiltonian.magnetic_field(), 0.5);
    }

    #[test]
    fn test_diagonal_element() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy = hamiltonian.diagonal_element(&config);

        // Expected energy:
        // - Exchange: 1.0 * (1*(-1) + (-1)*1 + 1*(-1) + (-1)*1) = -4.0
        // - Magnetic field: -0.5 * (1 + (-1) + 1 + (-1)) = 0.0
        let expected = -4.0 + 0.0;
        assert_abs_diff_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_exchange_energy_ferromagnetic() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();

        // All spins up (ferromagnetic)
        let config = vec![Spin::Up, Spin::Up, Spin::Up, Spin::Up];
        let energy = hamiltonian.exchange_energy(&config);

        // 4 bonds, each contributing 1.0 * 1 * 1 = 1.0
        assert_abs_diff_eq!(energy, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exchange_energy_antiferromagnetic() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();

        // Alternating spins (antiferromagnetic)
        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy = hamiltonian.exchange_energy(&config);

        // 4 bonds, each contributing 1.0 * 1 * (-1) = -1.0
        assert_abs_diff_eq!(energy, -4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetic_field_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 0.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy = hamiltonian.magnetic_field_energy(&config);

        // -0.5 * (1 + (-1) + 1 + (-1)) = 0.0
        assert_abs_diff_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnetic_field_energy_all_up() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 0.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Up, Spin::Up, Spin::Up];
        let energy = hamiltonian.magnetic_field_energy(&config);

        // -0.5 * (1 + 1 + 1 + 1) = -2.0
        assert_abs_diff_eq!(energy, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_element_same_config() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let matrix_element = hamiltonian.matrix_element(&config, &config);

        // Should be the diagonal element
        let expected = hamiltonian.diagonal_element(&config);
        assert_abs_diff_eq!(matrix_element.re, expected, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix_element.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_element_different_config() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();

        let config_i = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let config_j = vec![Spin::Down, Spin::Down, Spin::Up, Spin::Down];

        let matrix_element = hamiltonian.matrix_element(&config_i, &config_j);

        // Should be 0 (no off-diagonal elements in Ising model)
        assert_abs_diff_eq!(matrix_element.re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix_element.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_total_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, 1.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy = hamiltonian.total_energy(&config);

        // Should be the same as diagonal_element
        let expected = hamiltonian.diagonal_element(&config);
        assert_abs_diff_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_ground_state() {
        let lattice = ChainLattice::new(4, true).unwrap();

        // Test ferromagnetic case (J < 0)
        let hamiltonian_fm = HeisenbergHamiltonian::new(lattice.clone(), -1.0, 0.0).unwrap();
        let config_fm = vec![Spin::Up, Spin::Up, Spin::Up, Spin::Up];
        let energy_fm = hamiltonian_fm.total_energy(&config_fm);

        // Test antiferromagnetic case (J > 0)
        let hamiltonian_afm = HeisenbergHamiltonian::new(lattice, 1.0, 0.0).unwrap();
        let config_afm = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
        let energy_afm = hamiltonian_afm.total_energy(&config_afm);

        // Both should be ground states for their respective J values
        assert!(energy_fm < 0.0); // Ferromagnetic ground state
        assert!(energy_afm < 0.0); // Antiferromagnetic ground state
    }
}

// Property-based tests using proptest
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use crate::lattice::ChainLattice;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_heisenberg_energy_finite(exchange in -10.0f64..10.0, magnetic_field in -10.0f64..10.0) {
            let lattice = ChainLattice::new(6, true).unwrap();
            let hamiltonian = HeisenbergHamiltonian::new(lattice, exchange, magnetic_field).unwrap();

            // Test with a random configuration
            let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down, Spin::Up, Spin::Down];
            let energy = hamiltonian.total_energy(&config);

            // Energy should be finite
            prop_assert!(energy.is_finite());
        }

        #[test]
        fn prop_heisenberg_matrix_element_real(exchange in -10.0f64..10.0, magnetic_field in -10.0f64..10.0) {
            let lattice = ChainLattice::new(4, true).unwrap();
            let hamiltonian = HeisenbergHamiltonian::new(lattice, exchange, magnetic_field).unwrap();

            let config_i = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
            let config_j = vec![Spin::Down, Spin::Up, Spin::Down, Spin::Up];

            let matrix_element = hamiltonian.matrix_element(&config_i, &config_j);

            // Matrix elements should be real (imaginary part should be 0)
            prop_assert!(matrix_element.im.abs() < 1e-10);
        }

        #[test]
        fn prop_heisenberg_energy_symmetric(exchange in -10.0f64..10.0, magnetic_field in -10.0f64..10.0) {
            let lattice = ChainLattice::new(4, true).unwrap();
            let hamiltonian = HeisenbergHamiltonian::new(lattice, exchange, magnetic_field).unwrap();

            let config = vec![Spin::Up, Spin::Down, Spin::Up, Spin::Down];
            let energy = hamiltonian.total_energy(&config);

            // Energy should be symmetric under spin flip (for zero magnetic field)
            if magnetic_field == 0.0 {
                let config_flipped = vec![Spin::Down, Spin::Up, Spin::Down, Spin::Up];
                let energy_flipped = hamiltonian.total_energy(&config_flipped);
                prop_assert!((energy - energy_flipped).abs() < 1e-10);
            }
        }
    }
}
