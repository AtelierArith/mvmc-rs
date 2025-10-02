//! Spin Jastrow wavefunction for Heisenberg models
//!
//! This module provides a Jastrow-type wavefunction specifically designed for
//! Heisenberg spin models (ne=0). The wavefunction captures spin-spin correlations
//! through a Jastrow factor of the form:
//! ψ = exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
//!
//! where S_i^z = +1 for spin up, -1 for spin down, and v_{ij} are variational parameters.

use crate::error::{Result, VmcError};
use crate::types::SiteCount;
use num_complex::Complex64;
use std::fmt;

/// Spin Jastrow wavefunction parameters
///
/// This structure contains the variational parameters for the spin Jastrow factor.
/// The parameters v_{ij} control the strength of spin-spin correlations between sites i and j.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinJastrowParameters {
    /// Number of lattice sites
    n_sites: usize,
    /// Jastrow parameters v_{ij} for i < j
    /// Indexed as v[i * (i-1) / 2 + j] for i < j
    parameters: Vec<f64>,
}

impl SpinJastrowParameters {
    /// Creates new spin Jastrow parameters
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    ///
    /// # Returns
    ///
    /// New parameters with all v_{ij} = 0.0
    pub fn new(n_sites: SiteCount) -> Self {
        let n = n_sites.get();
        let n_params = n * (n - 1) / 2; // Number of unique pairs i < j
        Self {
            n_sites: n,
            parameters: vec![0.0; n_params],
        }
    }

    /// Creates new spin Jastrow parameters with initial values
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
        let n = n_sites.get();
        let expected_n_params = n * (n - 1) / 2;

        if initial_params.len() != expected_n_params {
            return Err(VmcError::dim_mismatch(expected_n_params, initial_params.len()));
        }

        Ok(Self {
            n_sites: n,
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

    /// Gets the parameter for a specific site pair
    ///
    /// # Arguments
    ///
    /// * `i` - First site index (must be < j)
    /// * `j` - Second site index (must be > i)
    ///
    /// # Returns
    ///
    /// The parameter value v_{ij}
    pub fn get_parameter(&self, i: usize, j: usize) -> Result<f64> {
        if i >= j || j >= self.n_sites {
            return Err(VmcError::out_of_bounds(j, self.n_sites));
        }

        // Upper triangular indexing: for i < j, idx = i * n - i * (i + 1) / 2 + j - i - 1
        // Safe calculation to avoid overflow
        let n = self.n_sites;
        let idx = i * n - i * (i + 1) / 2 + j - i - 1;
        Ok(self.parameters[idx])
    }

    /// Sets the parameter for a specific site pair
    ///
    /// # Arguments
    ///
    /// * `i` - First site index (must be < j)
    /// * `j` - Second site index (must be > i)
    /// * `value` - New parameter value
    pub fn set_parameter(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        if i >= j || j >= self.n_sites {
            return Err(VmcError::out_of_bounds(j, self.n_sites));
        }

        // Upper triangular indexing: for i < j, idx = i * n - i * (i + 1) / 2 + j - i - 1
        // Safe calculation to avoid overflow
        let n = self.n_sites;
        let idx = i * n - i * (i + 1) / 2 + j - i - 1;
        self.parameters[idx] = value;
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

/// Spin Jastrow wavefunction for Heisenberg models
///
/// This wavefunction is specifically designed for Heisenberg spin models where ne=0.
/// It captures spin-spin correlations through a Jastrow factor:
/// ψ = exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
///
/// where S_i^z = +1 for spin up, -1 for spin down.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinJastrowWavefunction {
    /// Jastrow parameters
    parameters: SpinJastrowParameters,
    /// Whether to use complex parameters (for future extension)
    complex_parameters: bool,
}

impl SpinJastrowWavefunction {
    /// Creates a new spin Jastrow wavefunction
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    ///
    /// # Returns
    ///
    /// New wavefunction with all parameters set to zero
    pub fn new(n_sites: SiteCount) -> Self {
        Self {
            parameters: SpinJastrowParameters::new(n_sites),
            complex_parameters: false,
        }
    }

    /// Creates a new spin Jastrow wavefunction with initial parameters
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    /// * `initial_params` - Initial parameter values
    ///
    /// # Returns
    ///
    /// New wavefunction with specified parameters
    pub fn with_parameters(n_sites: SiteCount, initial_params: Vec<f64>) -> Result<Self> {
        Ok(Self {
            parameters: SpinJastrowParameters::with_parameters(n_sites, initial_params)?,
            complex_parameters: false,
        })
    }

    /// Returns the number of sites
    pub fn n_sites(&self) -> usize {
        self.parameters.n_sites()
    }

    /// Returns the number of parameters
    pub fn n_parameters(&self) -> usize {
        self.parameters.n_parameters()
    }

    /// Returns a reference to the parameters
    pub fn parameters(&self) -> &SpinJastrowParameters {
        &self.parameters
    }

    /// Returns a mutable reference to the parameters
    pub fn parameters_mut(&mut self) -> &mut SpinJastrowParameters {
        &mut self.parameters
    }

    /// Calculates the wavefunction amplitude for a given spin configuration
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down)
    ///
    /// # Returns
    ///
    /// The wavefunction amplitude
    ///
    /// # Note
    ///
    /// For Heisenberg models, we expect only spin up (1) and spin down (2) configurations.
    /// Empty sites (0) are not allowed in pure spin models.
    pub fn calculate_amplitude(&self, spin_config: &[u8]) -> Result<Complex64> {
        if spin_config.len() != self.n_sites() {
            return Err(VmcError::dim_mismatch(self.n_sites(), spin_config.len()));
        }

        // Calculate the Jastrow factor: exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
        let mut log_amplitude = 0.0;

        for i in 0..self.n_sites() {
            for j in (i + 1)..self.n_sites() {
                let s_i = self.spin_value(spin_config[i])?;
                let s_j = self.spin_value(spin_config[j])?;
                let v_ij = self.parameters.get_parameter(i, j)?;

                log_amplitude += v_ij * (s_i as f64) * (s_j as f64);
            }
        }

        Ok(Complex64::from_polar(log_amplitude.exp(), 0.0))
    }

    /// Calculates the wavefunction amplitude for a spin configuration change
    ///
    /// This is more efficient than recalculating the full amplitude when only
    /// one spin is flipped.
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
    /// The ratio of new amplitude to old amplitude
    pub fn calculate_ratio(&self, spin_config: &[u8], flip_site: usize, flip_from: u8, flip_to: u8) -> Result<Complex64> {
        if flip_site >= self.n_sites() {
            return Err(VmcError::out_of_bounds(flip_site, self.n_sites()));
        }

        let s_old = self.spin_value(flip_from)?;
        let s_new = self.spin_value(flip_to)?;
        let delta_s = s_new - s_old;

        // Calculate the change in log amplitude
        let mut delta_log_amplitude = 0.0;

        for j in 0..self.n_sites() {
            if j != flip_site {
                let s_j = self.spin_value(spin_config[j])?;
                let v_ij = if j < flip_site {
                    self.parameters.get_parameter(j, flip_site)?
                } else {
                    self.parameters.get_parameter(flip_site, j)?
                };

                delta_log_amplitude += v_ij * (delta_s as f64) * (s_j as f64);
            }
        }

        Ok(Complex64::from_polar(delta_log_amplitude.exp(), 0.0))
    }

    /// Calculates parameter derivatives (O-operators) for SR optimization
    ///
    /// For Jastrow parameters: O_k = ∂log(ψ)/∂v_k = S_i^z S_j^z
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

        for i in 0..self.n_sites() {
            for j in (i + 1)..self.n_sites() {
                let s_i = self.spin_value(spin_config[i])?;
                let s_j = self.spin_value(spin_config[j])?;
                let derivative = (s_i as f64) * (s_j as f64);
                derivatives.push(Complex64::new(derivative, 0.0));
            }
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

    /// Converts spin configuration to spin value
    ///
    /// # Arguments
    ///
    /// * `spin_state` - Spin state (0: empty, 1: up, 2: down)
    ///
    /// # Returns
    ///
    /// Spin value (+1 for up, -1 for down)
    fn spin_value(&self, spin_state: u8) -> Result<i32> {
        match spin_state {
            1 => Ok(1),   // spin up
            2 => Ok(-1),  // spin down
            0 => Err(VmcError::InvalidConfiguration("Empty sites not allowed in Heisenberg models".to_string())),
            _ => Err(VmcError::InvalidConfiguration("Invalid spin state".to_string())),
        }
    }

    /// Normalizes the wavefunction parameters
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

impl fmt::Display for SpinJastrowWavefunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpinJastrowWavefunction(n_sites={}, n_params={})",
               self.n_sites(), self.n_parameters())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SiteCount;

    #[test]
    fn test_spin_jastrow_creation() {
        let nsite = SiteCount::new(4);
        let wavefunction = SpinJastrowWavefunction::new(nsite);

        assert_eq!(wavefunction.n_sites(), 4);
        assert_eq!(wavefunction.n_parameters(), 6); // 4 choose 2 = 6
    }

    #[test]
    fn test_spin_jastrow_amplitude() {
        let nsite = SiteCount::new(2);
        let wavefunction = SpinJastrowWavefunction::new(nsite);

        // Test with spin up, spin down configuration
        let spin_config = vec![1, 2]; // up, down
        let amplitude = wavefunction.calculate_amplitude(&spin_config).unwrap();

        // With zero parameters, amplitude should be 1.0
        assert!((amplitude.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spin_jastrow_ratio() {
        let nsite = SiteCount::new(2);
        let wavefunction = SpinJastrowWavefunction::new(nsite);

        let spin_config = vec![1, 2]; // up, down
        let ratio = wavefunction.calculate_ratio(&spin_config, 0, 1, 2).unwrap();

        // With zero parameters, ratio should be 1.0
        assert!((ratio.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spin_jastrow_parameter_derivatives() {
        let nsite = SiteCount::new(2);
        let wavefunction = SpinJastrowWavefunction::new(nsite);

        let spin_config = vec![1, 2]; // up, down
        let derivatives = wavefunction.calculate_parameter_derivatives(&spin_config).unwrap();

        // For 2 sites, we have 1 parameter v_{0,1}
        // With S_0 = +1, S_1 = -1, derivative should be (+1) * (-1) = -1
        assert_eq!(derivatives.len(), 1);
        assert!((derivatives[0].re - (-1.0)).abs() < 1e-10);
        assert!(derivatives[0].im.abs() < 1e-10);
    }

    #[test]
    fn test_spin_jastrow_with_parameters() {
        let nsite = SiteCount::new(2);
        let params = vec![0.5]; // v_{0,1} = 0.5
        let wavefunction = SpinJastrowWavefunction::with_parameters(nsite, params).unwrap();

        let spin_config = vec![1, 2]; // up, down
        let amplitude = wavefunction.calculate_amplitude(&spin_config).unwrap();

        // With v_{0,1} = 0.5, S_0 = +1, S_1 = -1
        // log(ψ) = 0.5 * (+1) * (-1) = -0.5
        // ψ = exp(-0.5)
        let expected = (-0.5_f64).exp();
        assert!((amplitude.norm() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_spin_config() {
        let nsite = SiteCount::new(2);
        let wavefunction = SpinJastrowWavefunction::new(nsite);

        // Test with empty site (not allowed in Heisenberg models)
        let spin_config = vec![0, 1]; // empty, up
        let result = wavefunction.calculate_amplitude(&spin_config);
        assert!(result.is_err());
    }
}
