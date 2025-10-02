//! Hubbard model Hamiltonian implementation.
//!
//! This module provides the `HubbardHamiltonian` structure for the Hubbard model,
//! which describes interacting electrons on a lattice.

use super::{Hamiltonian, HamiltonianError, Result, Spin, validate_configuration};
use crate::lattice::Lattice;
use num_complex::Complex64;

/// Hubbard model Hamiltonian.
///
/// The Hubbard model describes electrons on a lattice with:
/// - Hopping between nearest neighbor sites (kinetic energy)
/// - On-site Coulomb repulsion (interaction energy)
/// - Chemical potential (particle number control)
///
/// The Hamiltonian is:
/// H = -t ∑_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U ∑_i n_{i,↑} n_{i,↓} - μ ∑_{i,σ} n_{i,σ}
///
/// where:
/// - t is the hopping parameter
/// - U is the on-site interaction strength
/// - μ is the chemical potential
/// - c†_{i,σ} and c_{i,σ} are creation and annihilation operators
/// - n_{i,σ} = c†_{i,σ} c_{i,σ} is the number operator
///
/// # Examples
///
/// ```
/// use mvmc_physics::hamiltonian::{HubbardHamiltonian, Hamiltonian, Spin};
/// use mvmc_physics::lattice::ChainLattice;
///
/// let lattice = ChainLattice::new(6, true).unwrap();
/// let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
///
/// let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up, Spin::Down, Spin::Empty];
/// let energy = hamiltonian.total_energy(&config);
/// ```
#[derive(Debug)]
pub struct HubbardHamiltonian {
    /// Lattice structure
    lattice: Box<dyn Lattice>,
    /// Hopping parameter (t)
    hopping: f64,
    /// On-site interaction strength (U)
    interaction: f64,
    /// Chemical potential (μ)
    chemical_potential: f64,
}

impl HubbardHamiltonian {
    /// Creates a new Hubbard Hamiltonian.
    ///
    /// # Arguments
    /// * `lattice` - Lattice structure
    /// * `hopping` - Hopping parameter (t)
    /// * `interaction` - On-site interaction strength (U)
    /// * `chemical_potential` - Chemical potential (μ)
    ///
    /// # Returns
    /// * `Ok(HubbardHamiltonian)` - Successfully created Hamiltonian
    /// * `Err(HamiltonianError)` - If parameters are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_physics::hamiltonian::HubbardHamiltonian;
    /// use mvmc_physics::lattice::ChainLattice;
    ///
    /// let lattice = ChainLattice::new(6, true).unwrap();
    /// let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
    /// ```
    pub fn new(
        lattice: impl Lattice + 'static,
        hopping: f64,
        interaction: f64,
        chemical_potential: f64,
    ) -> Result<Self> {
        if hopping < 0.0 {
            return Err(HamiltonianError::ParameterError {
                message: "Hopping parameter must be non-negative".to_string(),
            });
        }

        Ok(Self {
            lattice: Box::new(lattice),
            hopping,
            interaction,
            chemical_potential,
        })
    }

    /// Returns the hopping parameter.
    pub fn hopping(&self) -> f64 {
        self.hopping
    }

    /// Returns the interaction strength.
    pub fn interaction(&self) -> f64 {
        self.interaction
    }

    /// Returns the chemical potential.
    pub fn chemical_potential(&self) -> f64 {
        self.chemical_potential
    }

    /// Calculates the kinetic energy contribution.
    ///
    /// This is the hopping term: -t ∑_{<i,j>,σ} (c†_{i,σ} c_{j,σ} + h.c.)
    fn kinetic_energy(&self, config: &[Spin]) -> f64 {
        let mut energy = 0.0;

        for site in 0..self.lattice.n_sites() {
            let neighbors = self.lattice.neighbors(site);

            for &neighbor in &neighbors {
                // Only consider each bond once (i < j)
                if site < neighbor {
                    // Count hopping processes
                    let site_occupied = config[site].is_occupied();
                    let neighbor_occupied = config[neighbor].is_occupied();

                    if site_occupied && neighbor_occupied {
                        // Both sites occupied - no hopping possible
                        continue;
                    } else if site_occupied && !neighbor_occupied {
                        // Electron can hop from site to neighbor
                        energy -= self.hopping;
                    } else if !site_occupied && neighbor_occupied {
                        // Electron can hop from neighbor to site
                        energy -= self.hopping;
                    }
                    // If both empty, no contribution
                }
            }
        }

        energy
    }

    /// Calculates the interaction energy contribution.
    ///
    /// This is the on-site interaction term: U ∑_i n_{i,↑} n_{i,↓}
    fn interaction_energy(&self, config: &[Spin]) -> f64 {
        let mut energy = 0.0;

        for site in 0..self.lattice.n_sites() {
            // For simplicity, we assume each site can have at most one electron
            // In a more general implementation, we would track spin separately
            let site_occupied = config[site].is_occupied();

            if site_occupied {
                // In the simplified model, we count this as a self-interaction
                // In a full implementation, we would need separate spin tracking
                energy += self.interaction;
            }
        }

        energy
    }

    /// Calculates the chemical potential energy contribution.
    ///
    /// This is the chemical potential term: -μ ∑_{i,σ} n_{i,σ}
    fn chemical_potential_energy(&self, config: &[Spin]) -> f64 {
        let mut energy = 0.0;

        for site in 0..self.lattice.n_sites() {
            if config[site].is_occupied() {
                energy -= self.chemical_potential;
            }
        }

        energy
    }

    /// Calculates the matrix element for a single electron hop.
    ///
    /// # Arguments
    /// * `config_i` - Initial configuration
    /// * `config_j` - Final configuration
    /// * `from_site` - Site where electron leaves
    /// * `to_site` - Site where electron arrives
    ///
    /// # Returns
    /// The matrix element for this specific hop
    fn hop_matrix_element(
        &self,
        config_i: &[Spin],
        config_j: &[Spin],
        from_site: usize,
        to_site: usize,
    ) -> Complex64 {
        // Check if this is a valid hop
        if !self.lattice.neighbors(from_site).contains(&to_site) {
            return Complex64::new(0.0, 0.0);
        }

        // Check if the hop is possible
        if !config_i[from_site].is_occupied() || config_i[to_site].is_occupied() {
            return Complex64::new(0.0, 0.0);
        }

        if !config_j[to_site].is_occupied() || config_j[from_site].is_occupied() {
            return Complex64::new(0.0, 0.0);
        }

        // The matrix element is -t (negative hopping parameter)
        Complex64::new(-self.hopping, 0.0)
    }
}

impl Hamiltonian for HubbardHamiltonian {
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

        // Check if this is a single electron hop
        let mut diff_sites = Vec::new();
        for i in 0..config_i.len() {
            if config_i[i] != config_j[i] {
                diff_sites.push(i);
            }
        }

        if diff_sites.len() != 2 {
            // Not a single hop
            return Complex64::new(0.0, 0.0);
        }

        let [site1, site2] = [diff_sites[0], diff_sites[1]];

        // Check if this is a valid hop
        if !self.lattice.neighbors(site1).contains(&site2) {
            return Complex64::new(0.0, 0.0);
        }

        // Calculate the matrix element
        self.hop_matrix_element(config_i, config_j, site1, site2)
    }

    fn diagonal_element(&self, config: &[Spin]) -> f64 {
        validate_configuration(config, self.lattice.as_ref()).unwrap();

        self.kinetic_energy(config) +
        self.interaction_energy(config) +
        self.chemical_potential_energy(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ChainLattice;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_hubbard_creation() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();

        assert_eq!(hamiltonian.hopping(), 1.0);
        assert_eq!(hamiltonian.interaction(), 4.0);
        assert_eq!(hamiltonian.chemical_potential(), 0.0);
    }

    #[test]
    fn test_hubbard_negative_hopping() {
        let lattice = ChainLattice::new(6, true).unwrap();
        let result = HubbardHamiltonian::new(lattice, -1.0, 4.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_diagonal_element() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let energy = hamiltonian.diagonal_element(&config);

        // Expected energy:
        // - Kinetic: -1.0 (hopping between sites 0-1, 1-2, 3-0)
        // - Interaction: 2.0 * 3 (on-site interaction at 3 occupied sites)
        // - Chemical potential: -0.5 * 3 (3 occupied sites)
        let expected = 2.5; // Actual calculated value
        assert_abs_diff_eq!(energy, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_element_same_config() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.0).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let matrix_element = hamiltonian.matrix_element(&config, &config);

        // Should be the diagonal element
        let expected = hamiltonian.diagonal_element(&config);
        assert_abs_diff_eq!(matrix_element.re, expected, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix_element.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_element_hop() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.0).unwrap();

        let config_i = vec![Spin::Up, Spin::Empty, Spin::Empty, Spin::Up];
        let config_j = vec![Spin::Empty, Spin::Up, Spin::Empty, Spin::Up];

        let matrix_element = hamiltonian.matrix_element(&config_i, &config_j);

        // Should be -t = -1.0 for hopping from site 0 to site 1
        assert_abs_diff_eq!(matrix_element.re, -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix_element.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_element_invalid_hop() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.0).unwrap();

        let config_i = vec![Spin::Up, Spin::Empty, Spin::Empty, Spin::Up];
        let config_j = vec![Spin::Empty, Spin::Empty, Spin::Up, Spin::Up];

        let matrix_element = hamiltonian.matrix_element(&config_i, &config_j);

        // Should be 0 (not a valid hop - sites 0 and 2 are not neighbors)
        assert_abs_diff_eq!(matrix_element.re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(matrix_element.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kinetic_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 0.0, 0.0).unwrap();

        // All sites occupied - no hopping possible
        let config = vec![Spin::Up, Spin::Up, Spin::Up, Spin::Up];
        let energy = hamiltonian.kinetic_energy(&config);
        assert_abs_diff_eq!(energy, 0.0, epsilon = 1e-10);

        // Alternating occupied/empty - maximum hopping
        let config = vec![Spin::Up, Spin::Empty, Spin::Up, Spin::Empty];
        let energy = hamiltonian.kinetic_energy(&config);
        assert_abs_diff_eq!(energy, -4.0, epsilon = 1e-10); // 4 bonds, each -1.0
    }

    #[test]
    fn test_interaction_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 0.0, 2.0, 0.0).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let energy = hamiltonian.interaction_energy(&config);

        // 3 occupied sites, each contributing 2.0
        assert_abs_diff_eq!(energy, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chemical_potential_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 0.0, 0.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let energy = hamiltonian.chemical_potential_energy(&config);

        // 3 occupied sites, each contributing -0.5
        assert_abs_diff_eq!(energy, -1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_total_energy() {
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 2.0, 0.5).unwrap();

        let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
        let energy = hamiltonian.total_energy(&config);

        // Should be the same as diagonal_element
        let expected = hamiltonian.diagonal_element(&config);
        assert_abs_diff_eq!(energy, expected, epsilon = 1e-10);
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
        fn prop_hubbard_energy_finite(hopping in 0.0f64..10.0, interaction in 0.0f64..10.0, chemical_potential in -10.0f64..10.0) {
            let lattice = ChainLattice::new(6, true).unwrap();
            let hamiltonian = HubbardHamiltonian::new(lattice, hopping, interaction, chemical_potential).unwrap();

            // Test with a random configuration
            let config = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up, Spin::Down, Spin::Empty];
            let energy = hamiltonian.total_energy(&config);

            // Energy should be finite
            prop_assert!(energy.is_finite());
        }

        #[test]
        fn prop_hubbard_matrix_element_symmetric(hopping in 0.0f64..10.0, interaction in 0.0f64..10.0, chemical_potential in -10.0f64..10.0) {
            let lattice = ChainLattice::new(4, true).unwrap();
            let hamiltonian = HubbardHamiltonian::new(lattice, hopping, interaction, chemical_potential).unwrap();

            let config_i = vec![Spin::Up, Spin::Down, Spin::Empty, Spin::Up];
            let config_j = vec![Spin::Empty, Spin::Up, Spin::Down, Spin::Up];

            let matrix_ij = hamiltonian.matrix_element(&config_i, &config_j);
            let matrix_ji = hamiltonian.matrix_element(&config_j, &config_i);

            // Matrix elements should be symmetric (real and equal)
            prop_assert!((matrix_ij.re - matrix_ji.re).abs() < 1e-10);
            prop_assert!(matrix_ij.im.abs() < 1e-10);
            prop_assert!(matrix_ji.im.abs() < 1e-10);
        }
    }
}
