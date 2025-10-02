//! Heisenberg model wavefunction for VMC calculations
//!
//! This module provides a specialized wavefunction for Heisenberg spin models (ne=0).
//! The wavefunction combines spin Jastrow correlations and Gutzwiller projection:
//! ψ = P_G × exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
//!
//! For Heisenberg models, the Gutzwiller projection is not applicable (always 1.0),
//! so the wavefunction reduces to:
//! ψ = exp(Σ_{i<j} v_{ij} S_i^z S_j^z)

use crate::error::{Result, VmcError};
use crate::types::SiteCount;
use crate::wavefunction::{GutzwillerProjector, SpinJastrowWavefunction};
use num_complex::Complex64;
use std::fmt;

/// Heisenberg model wavefunction
///
/// This wavefunction is specifically designed for Heisenberg spin models where ne=0.
/// It combines spin Jastrow correlations with Gutzwiller projection (which is
/// not applicable for spin models and always returns 1.0).
///
/// The wavefunction is given by:
/// ψ = P_G × exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
///
/// where P_G is the Gutzwiller projection factor (1.0 for spin models) and
/// v_{ij} are the Jastrow parameters for spin-spin correlations.
#[derive(Debug, Clone)]
pub struct HeisenbergWavefunction {
    /// Number of lattice sites
    n_sites: usize,
    /// Number of electrons (should be 0 for Heisenberg models)
    ne: usize,
    /// Spin Jastrow wavefunction for spin-spin correlations
    spin_jastrow: SpinJastrowWavefunction,
    /// Gutzwiller projector (not applicable for spin models)
    gutzwiller: GutzwillerProjector,
}

impl HeisenbergWavefunction {
    /// Creates a new Heisenberg wavefunction
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    ///
    /// # Returns
    ///
    /// New wavefunction with small random initial parameters
    pub fn new(n_sites: SiteCount) -> Self {
        let nsite = n_sites.get();
        let mut wavefunction = Self {
            n_sites: nsite,
            ne: 0, // Heisenberg models have no electrons
            spin_jastrow: SpinJastrowWavefunction::new(n_sites),
            gutzwiller: GutzwillerProjector::new(n_sites, 0),
        };

        // Add small random noise to parameters for proper initialization
        wavefunction.add_noise(0.01);

        wavefunction
    }

    /// Creates a new Heisenberg wavefunction with zero parameters (for testing)
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    ///
    /// # Returns
    ///
    /// New wavefunction with zero parameters
    pub fn new_zero_params(n_sites: SiteCount) -> Self {
        let nsite = n_sites.get();
        Self {
            n_sites: nsite,
            ne: 0, // Heisenberg models have no electrons
            spin_jastrow: SpinJastrowWavefunction::new(n_sites),
            gutzwiller: GutzwillerProjector::new(n_sites, 0),
        }
    }

    /// Creates a new Heisenberg wavefunction with initial Jastrow parameters
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of lattice sites
    /// * `jastrow_params` - Initial Jastrow parameters
    ///
    /// # Returns
    ///
    /// New wavefunction with specified Jastrow parameters
    pub fn with_jastrow_parameters(n_sites: SiteCount, jastrow_params: Vec<f64>) -> Result<Self> {
        let nsite = n_sites.get();
        Ok(Self {
            n_sites: nsite,
            ne: 0,
            spin_jastrow: SpinJastrowWavefunction::with_parameters(n_sites, jastrow_params)?,
            gutzwiller: GutzwillerProjector::new(n_sites, 0),
        })
    }

    /// Returns the number of sites
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Returns the number of electrons (always 0 for Heisenberg models)
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns the number of variational parameters
    pub fn n_parameters(&self) -> usize {
        self.spin_jastrow.n_parameters() + self.gutzwiller.n_parameters()
    }

    /// Returns a reference to the spin Jastrow wavefunction
    pub fn spin_jastrow(&self) -> &SpinJastrowWavefunction {
        &self.spin_jastrow
    }

    /// Returns a mutable reference to the spin Jastrow wavefunction
    pub fn spin_jastrow_mut(&mut self) -> &mut SpinJastrowWavefunction {
        &mut self.spin_jastrow
    }

    /// Returns a reference to the Gutzwiller projector
    pub fn gutzwiller(&self) -> &GutzwillerProjector {
        &self.gutzwiller
    }

    /// Returns a mutable reference to the Gutzwiller projector
    pub fn gutzwiller_mut(&mut self) -> &mut GutzwillerProjector {
        &mut self.gutzwiller
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
    /// Empty sites (0) are not allowed in pure spin systems.
    pub fn calculate_amplitude(&self, spin_config: &[u8]) -> Result<Complex64> {
        if spin_config.len() != self.n_sites {
            return Err(VmcError::dim_mismatch(self.n_sites, spin_config.len()));
        }

        // Calculate Jastrow factor: exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
        let jastrow_amplitude = self.spin_jastrow.calculate_amplitude(spin_config)?;

        // Calculate Gutzwiller factor (always 1.0 for spin models)
        let gutzwiller_factor = self.gutzwiller.calculate_factor(spin_config)?;

        // Combine the factors
        Ok(jastrow_amplitude * gutzwiller_factor)
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
        // Calculate Jastrow ratio
        let jastrow_ratio = self.spin_jastrow.calculate_ratio(spin_config, flip_site, flip_from, flip_to)?;

        // Calculate Gutzwiller ratio (always 1.0 for spin models)
        let gutzwiller_ratio = self.gutzwiller.calculate_ratio(spin_config, flip_site, flip_from, flip_to)?;

        // Combine the ratios
        Ok(jastrow_ratio * gutzwiller_ratio)
    }

    /// Calculates parameter derivatives (O-operators) for SR optimization
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    ///
    /// Vector of parameter derivatives
    pub fn calculate_parameter_derivatives(&self, spin_config: &[u8]) -> Result<Vec<Complex64>> {
        let mut derivatives = Vec::new();

        // Jastrow parameter derivatives
        let jastrow_derivs = self.spin_jastrow.calculate_parameter_derivatives(spin_config)?;
        derivatives.extend(jastrow_derivs);

        // Gutzwiller parameter derivatives (always zero for spin models)
        let gutzwiller_derivs = self.gutzwiller.calculate_parameter_derivatives(spin_config)?;
        derivatives.extend(gutzwiller_derivs);

        Ok(derivatives)
    }

    /// Updates parameters using SR optimization results
    ///
    /// # Arguments
    ///
    /// * `updates` - Parameter updates from SR optimization
    /// * `learning_rate` - Learning rate for updates
    pub fn update_parameters(&mut self, updates: &[f64], learning_rate: f64) {
        let n_jastrow_params = self.spin_jastrow.n_parameters();
        let n_gutzwiller_params = self.gutzwiller.n_parameters();

        // Update Jastrow parameters
        if n_jastrow_params > 0 && updates.len() >= n_jastrow_params {
            self.spin_jastrow.update_parameters(&updates[0..n_jastrow_params], learning_rate);
        }

        // Update Gutzwiller parameters (not applicable for spin models)
        if n_gutzwiller_params > 0 && updates.len() >= n_jastrow_params + n_gutzwiller_params {
            self.gutzwiller.update_parameters(&updates[n_jastrow_params..n_jastrow_params + n_gutzwiller_params], learning_rate);
        }
    }

    /// Normalizes the wavefunction parameters
    ///
    /// This applies normalization to all components.
    pub fn normalize(&mut self) {
        self.spin_jastrow.normalize();
        self.gutzwiller.normalize();
    }

    /// Adds random noise to parameters for initialization
    ///
    /// # Arguments
    ///
    /// * `noise_scale` - Scale of the random noise
    pub fn add_noise(&mut self, noise_scale: f64) {
        self.spin_jastrow.add_noise(noise_scale);
        self.gutzwiller.add_noise(noise_scale);
    }

    /// Exports current variational parameters
    ///
    /// # Returns
    ///
    /// Vector of all parameters in the order: [Jastrow parameters, Gutzwiller parameters]
    pub fn export_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        params.extend(self.spin_jastrow.parameters().parameters());
        params.extend(self.gutzwiller.parameters().parameters());
        params
    }

    /// Imports variational parameters
    ///
    /// # Arguments
    ///
    /// * `params` - Parameter values to import
    ///
    /// # Returns
    ///
    /// Error if the number of parameters doesn't match
    pub fn import_parameters(&mut self, params: &[f64]) -> Result<()> {
        let n_jastrow_params = self.spin_jastrow.n_parameters();
        let n_gutzwiller_params = self.gutzwiller.n_parameters();
        let expected_total = n_jastrow_params + n_gutzwiller_params;

        if params.len() != expected_total {
            return Err(VmcError::dim_mismatch(expected_total, params.len()));
        }

        // Import Jastrow parameters
        if n_jastrow_params > 0 {
            let jastrow_params = params[0..n_jastrow_params].to_vec();
            self.spin_jastrow = SpinJastrowWavefunction::with_parameters(
                SiteCount::new(self.n_sites),
                jastrow_params
            )?;
        }

        // Import Gutzwiller parameters
        if n_gutzwiller_params > 0 {
            let gutzwiller_params = params[n_jastrow_params..n_jastrow_params + n_gutzwiller_params].to_vec();
            self.gutzwiller = GutzwillerProjector::with_parameters(
                SiteCount::new(self.n_sites),
                self.ne,
                gutzwiller_params
            )?;
        }

        Ok(())
    }
}

impl fmt::Display for HeisenbergWavefunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HeisenbergWavefunction(n_sites={}, ne={}, n_params={})",
               self.n_sites(), self.ne(), self.n_parameters())
    }
}

impl PartialEq for HeisenbergWavefunction {
    fn eq(&self, other: &Self) -> bool {
        self.n_sites == other.n_sites &&
        self.ne == other.ne &&
        self.spin_jastrow == other.spin_jastrow &&
        self.gutzwiller == other.gutzwiller
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SiteCount;

    #[test]
    fn test_heisenberg_wavefunction_creation() {
        let nsite = SiteCount::new(4);
        let wavefunction = HeisenbergWavefunction::new(nsite);

        assert_eq!(wavefunction.n_sites(), 4);
        assert_eq!(wavefunction.ne(), 0);
        assert!(wavefunction.n_parameters() > 0);
    }

    #[test]
    fn test_heisenberg_wavefunction_amplitude() {
        let nsite = SiteCount::new(2);
        let wavefunction = HeisenbergWavefunction::new_zero_params(nsite);

        // Test with spin up, spin down configuration
        let spin_config = vec![1, 2]; // up, down
        let amplitude = wavefunction.calculate_amplitude(&spin_config).unwrap();

        // With zero parameters, amplitude should be 1.0
        assert!((amplitude.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_heisenberg_wavefunction_ratio() {
        let nsite = SiteCount::new(2);
        let wavefunction = HeisenbergWavefunction::new_zero_params(nsite);

        let spin_config = vec![1, 2]; // up, down
        let ratio = wavefunction.calculate_ratio(&spin_config, 0, 1, 2).unwrap();

        // With zero parameters, ratio should be 1.0
        assert!((ratio.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_heisenberg_wavefunction_parameter_derivatives() {
        let nsite = SiteCount::new(2);
        let wavefunction = HeisenbergWavefunction::new(nsite);

        let spin_config = vec![1, 2]; // up, down
        let derivatives = wavefunction.calculate_parameter_derivatives(&spin_config).unwrap();

        // Should have derivatives for both Jastrow and Gutzwiller parameters
        assert!(derivatives.len() > 0);
    }

    #[test]
    fn test_heisenberg_wavefunction_with_parameters() {
        let nsite = SiteCount::new(2);
        let jastrow_params = vec![0.5]; // v_{0,1} = 0.5
        let wavefunction = HeisenbergWavefunction::with_jastrow_parameters(nsite, jastrow_params).unwrap();

        let spin_config = vec![1, 2]; // up, down
        let amplitude = wavefunction.calculate_amplitude(&spin_config).unwrap();

        // With v_{0,1} = 0.5, S_0 = +1, S_1 = -1
        // log(ψ) = 0.5 * (+1) * (-1) = -0.5
        // ψ = exp(-0.5)
        let expected = (-0.5_f64).exp();
        assert!((amplitude.norm() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_heisenberg_wavefunction_parameter_import_export() {
        let nsite = SiteCount::new(2);
        let mut wavefunction = HeisenbergWavefunction::new(nsite);

        // Set some parameters
        let jastrow_params = vec![0.5];
        wavefunction = HeisenbergWavefunction::with_jastrow_parameters(nsite, jastrow_params).unwrap();

        // Export parameters
        let exported = wavefunction.export_parameters();
        assert!(exported.len() > 0);

        // Import parameters
        let result = wavefunction.import_parameters(&exported);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_spin_config() {
        let nsite = SiteCount::new(2);
        let wavefunction = HeisenbergWavefunction::new(nsite);

        // Test with empty site (not allowed in Heisenberg models)
        let spin_config = vec![0, 1]; // empty, up
        let result = wavefunction.calculate_amplitude(&spin_config);
        assert!(result.is_err());
    }
}
