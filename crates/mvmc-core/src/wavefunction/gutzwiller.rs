//! Gutzwiller projection operator for VMC calculations
//!
//! This module provides the Gutzwiller projection operator which projects out
//! doubly occupied sites. The Gutzwiller factor is given by:
//! P_G = exp(-g Σ_i n_{i↑} n_{i↓})
//!
//! where n_{i↑} and n_{i↓} are the occupation numbers for up and down spins
//! at site i, and g is the Gutzwiller parameter.

use crate::error::{Result, VmcError};
use crate::types::SiteCount;
use num_complex::Complex64;
use std::fmt;

/// Gutzwiller projection parameters
///
/// This structure contains the Gutzwiller parameters g_i for each site.
/// The Gutzwiller factor projects out doubly occupied sites.
#[derive(Debug, Clone, PartialEq)]
pub struct GutzwillerParameters {
    /// Number of lattice sites
    n_sites: usize,
    /// Gutzwiller parameters g_i for each site
    parameters: Vec<f64>,
}

impl GutzwillerParameters {
    /// Creates new Gutzwiller parameters
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    ///
    /// # Returns
    ///
    /// New parameters with all g_i = 0.0
    pub fn new(n_sites: SiteCount) -> Self {
        Self {
            n_sites: n_sites.get(),
            parameters: vec![0.0; n_sites.get()],
        }
    }

    /// Creates new Gutzwiller parameters with initial values
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    /// * `initial_params` - Initial parameter values
    ///
    /// # Returns
    ///
    /// New parameters with specified initial values
    pub fn with_parameters(n_sites: SiteCount, initial_params: Vec<f64>) -> Result<Self> {
        if initial_params.len() != n_sites.get() {
            return Err(VmcError::dim_mismatch(n_sites.get(), initial_params.len()));
        }

        Ok(Self {
            n_sites: n_sites.get(),
            parameters: initial_params,
        })
    }

    /// Returns the number of sites
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Returns the number of parameters
    pub fn n_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Gets the parameter for a specific site
    ///
    /// # Arguments
    ///
    /// * `site` - Site index
    ///
    /// # Returns
    ///
    /// The parameter value g_i
    pub fn get_parameter(&self, site: usize) -> Result<f64> {
        if site >= self.n_sites {
            return Err(VmcError::out_of_bounds(site, self.n_sites));
        }

        Ok(self.parameters[site])
    }

    /// Sets the parameter for a specific site
    ///
    /// # Arguments
    ///
    /// * `site` - Site index
    /// * `value` - New parameter value
    pub fn set_parameter(&mut self, site: usize, value: f64) -> Result<()> {
        if site >= self.n_sites {
            return Err(VmcError::out_of_bounds(site, self.n_sites));
        }

        self.parameters[site] = value;
        Ok(())
    }

    /// Updates parameters with SR optimization results
    ///
    /// # Arguments
    ///
    /// * `updates` - Parameter updates from SR optimization
    /// * `learning_rate` - Learning rate for updates
    pub fn update_parameters(&mut self, updates: &[f64], learning_rate: f64) {
        for (i, param) in self.parameters.iter_mut().enumerate() {
            if i < updates.len() {
                *param -= learning_rate * updates[i];
            }
        }
    }

    /// Returns a reference to all parameters
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// Returns a mutable reference to all parameters
    pub fn parameters_mut(&mut self) -> &mut [f64] {
        &mut self.parameters
    }
}

/// Gutzwiller projection operator
///
/// This operator projects out doubly occupied sites in fermionic systems.
/// The Gutzwiller factor is given by:
/// P_G = exp(-g Σ_i n_{i↑} n_{i↓})
///
/// For spin models (ne=0), this operator is not applicable and returns 1.0.
#[derive(Debug, Clone, PartialEq)]
pub struct GutzwillerProjector {
    /// Gutzwiller parameters
    parameters: GutzwillerParameters,
    /// Number of electrons (0 for spin models)
    ne: usize,
}

impl GutzwillerProjector {
    /// Creates a new Gutzwiller projector
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    /// * `ne` - Number of electrons (0 for spin models)
    ///
    /// # Returns
    ///
    /// New projector with all parameters set to zero
    pub fn new(n_sites: SiteCount, ne: usize) -> Self {
        Self {
            parameters: GutzwillerParameters::new(n_sites),
            ne,
        }
    }

    /// Creates a new Gutzwiller projector with initial parameters
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `initial_params` - Initial parameter values
    ///
    /// # Returns
    ///
    /// New projector with specified parameters
    pub fn with_parameters(n_sites: SiteCount, ne: usize, initial_params: Vec<f64>) -> Result<Self> {
        Ok(Self {
            parameters: GutzwillerParameters::with_parameters(n_sites, initial_params)?,
            ne,
        })
    }

    /// Returns the number of sites
    pub fn n_sites(&self) -> usize {
        self.parameters.n_sites()
    }

    /// Returns the number of electrons
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns the number of parameters
    pub fn n_parameters(&self) -> usize {
        self.parameters.n_parameters()
    }

    /// Returns a reference to the parameters
    pub fn parameters(&self) -> &GutzwillerParameters {
        &self.parameters
    }

    /// Returns a mutable reference to the parameters
    pub fn parameters_mut(&mut self) -> &mut GutzwillerParameters {
        &mut self.parameters
    }

    /// Calculates the Gutzwiller projection factor for a given configuration
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    ///
    /// The Gutzwiller projection factor
    ///
    /// # Note
    ///
    /// For spin models (ne=0), this always returns 1.0 since there are no
    /// doubly occupied sites in pure spin systems.
    pub fn calculate_factor(&self, spin_config: &[u8]) -> Result<Complex64> {
        if spin_config.len() != self.n_sites() {
            return Err(VmcError::dim_mismatch(self.n_sites(), spin_config.len()));
        }

        // For spin models (ne=0), Gutzwiller projection is not applicable
        if self.ne == 0 {
            return Ok(Complex64::new(1.0, 0.0));
        }

        // Calculate the Gutzwiller factor: exp(-g Σ_i n_{i↑} n_{i↓})
        let mut log_factor = 0.0;

        for site in 0..self.n_sites() {
            let double_occupancy = self.calculate_double_occupancy(spin_config[site]);
            let g_i = self.parameters.get_parameter(site)?;
            log_factor -= g_i * double_occupancy;
        }

        Ok(Complex64::from_polar(log_factor.exp(), 0.0))
    }

    /// Calculates the Gutzwiller projection factor for a configuration change
    ///
    /// This is more efficient than recalculating the full factor when only
    /// one site changes.
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Current spin configuration
    /// * `flip_site` - Site to flip
    /// * `flip_from` - Current spin state
    /// * `flip_to` - New spin state
    ///
    /// # Returns
    ///
    /// The ratio of new factor to old factor
    pub fn calculate_ratio(&self, _spin_config: &[u8], flip_site: usize, flip_from: u8, flip_to: u8) -> Result<Complex64> {
        if flip_site >= self.n_sites() {
            return Err(VmcError::out_of_bounds(flip_site, self.n_sites()));
        }

        // For spin models (ne=0), Gutzwiller projection is not applicable
        if self.ne == 0 {
            return Ok(Complex64::new(1.0, 0.0));
        }

        let old_double_occupancy = self.calculate_double_occupancy(flip_from);
        let new_double_occupancy = self.calculate_double_occupancy(flip_to);
        let delta_double_occupancy = new_double_occupancy - old_double_occupancy;

        let g_i = self.parameters.get_parameter(flip_site)?;
        let log_ratio = -g_i * delta_double_occupancy;

        Ok(Complex64::from_polar(log_ratio.exp(), 0.0))
    }

    /// Calculates parameter derivatives (O-operators) for SR optimization
    ///
    /// For Gutzwiller parameters: O_k = ∂log(P_G)/∂g_k = -n_{k↑} n_{k↓}
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    ///
    /// Vector of parameter derivatives
    pub fn calculate_parameter_derivatives(&self, spin_config: &[u8]) -> Result<Vec<Complex64>> {
        if spin_config.len() != self.n_sites() {
            return Err(VmcError::dim_mismatch(self.n_sites(), spin_config.len()));
        }

        let mut derivatives = Vec::with_capacity(self.n_parameters());

        for site in 0..self.n_sites() {
            let double_occupancy = self.calculate_double_occupancy(spin_config[site]);
            let derivative = -double_occupancy;
            derivatives.push(Complex64::new(derivative, 0.0));
        }

        Ok(derivatives)
    }

    /// Updates parameters using SR optimization results
    ///
    /// # Arguments
    ///
    /// * `updates` - Parameter updates from SR optimization
    /// * `learning_rate` - Learning rate for updates
    pub fn update_parameters(&mut self, updates: &[f64], learning_rate: f64) {
        self.parameters.update_parameters(updates, learning_rate);
    }

    /// Calculates double occupancy for a given spin state
    ///
    /// # Arguments
    ///
    /// * `spin_state` - Spin state (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    ///
    /// Double occupancy (0 or 1)
    fn calculate_double_occupancy(&self, spin_state: u8) -> f64 {
        match spin_state {
            3 => 1.0,  // both up and down (doubly occupied)
            _ => 0.0,  // empty, up only, or down only
        }
    }

    /// Normalizes the Gutzwiller parameters
    ///
    /// This is a placeholder for future normalization schemes
    pub fn normalize(&mut self) {
        // For now, no normalization is applied
        // In the future, we might want to normalize the parameters
        // to prevent them from growing too large
    }

    /// Adds random noise to parameters for initialization
    ///
    /// # Arguments
    ///
    /// * `noise_scale` - Scale of the random noise
    pub fn add_noise(&mut self, noise_scale: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for param in self.parameters.parameters_mut() {
            let noise = rng.gen_range(-noise_scale..noise_scale);
            *param += noise;
        }
    }
}

impl fmt::Display for GutzwillerProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GutzwillerProjector(n_sites={}, ne={}, n_params={})",
               self.n_sites(), self.ne(), self.n_parameters())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SiteCount;

    #[test]
    fn test_gutzwiller_creation() {
        let nsite = SiteCount::new(4);
        let projector = GutzwillerProjector::new(nsite, 2);

        assert_eq!(projector.n_sites(), 4);
        assert_eq!(projector.ne(), 2);
        assert_eq!(projector.n_parameters(), 4);
    }

    #[test]
    fn test_gutzwiller_factor_fermionic() {
        let nsite = SiteCount::new(2);
        let projector = GutzwillerProjector::new(nsite, 2);

        // Test with no double occupancy
        let spin_config = vec![1, 2]; // up, down
        let factor = projector.calculate_factor(&spin_config).unwrap();
        assert!((factor.norm() - 1.0).abs() < 1e-10);

        // Test with double occupancy
        let spin_config = vec![3, 0]; // both, empty
        let factor = projector.calculate_factor(&spin_config).unwrap();
        assert!((factor.norm() - 1.0).abs() < 1e-10); // With g=0, factor should be 1
    }

    #[test]
    fn test_gutzwiller_factor_spin_model() {
        let nsite = SiteCount::new(2);
        let projector = GutzwillerProjector::new(nsite, 0); // Spin model

        // For spin models, Gutzwiller projection should always return 1.0
        let spin_config = vec![1, 2]; // up, down
        let factor = projector.calculate_factor(&spin_config).unwrap();
        assert!((factor.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gutzwiller_ratio() {
        let nsite = SiteCount::new(2);
        let projector = GutzwillerProjector::new(nsite, 2);

        let spin_config = vec![1, 2]; // up, down
        let ratio = projector.calculate_ratio(&spin_config, 0, 1, 3).unwrap();

        // With g=0, ratio should be 1.0
        assert!((ratio.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gutzwiller_parameter_derivatives() {
        let nsite = SiteCount::new(2);
        let projector = GutzwillerProjector::new(nsite, 2);

        let spin_config = vec![3, 0]; // both, empty
        let derivatives = projector.calculate_parameter_derivatives(&spin_config).unwrap();

        // For site 0: both up and down, so double occupancy = 1, derivative = -1
        // For site 1: empty, so double occupancy = 0, derivative = 0
        assert_eq!(derivatives.len(), 2);
        assert!((derivatives[0].re - (-1.0)).abs() < 1e-10);
        assert!((derivatives[1].re - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gutzwiller_with_parameters() {
        let nsite = SiteCount::new(2);
        let params = vec![0.5, 0.0]; // g_0 = 0.5, g_1 = 0.0
        let projector = GutzwillerProjector::with_parameters(nsite, 2, params).unwrap();

        let spin_config = vec![3, 0]; // both, empty
        let factor = projector.calculate_factor(&spin_config).unwrap();

        // With g_0 = 0.5, double occupancy at site 0 = 1
        // log(P_G) = -0.5 * 1 = -0.5
        // P_G = exp(-0.5)
        let expected = (-0.5_f64).exp();
        assert!((factor.norm() - expected).abs() < 1e-10);
    }
}
