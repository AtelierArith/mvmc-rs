//! Jastrow factor for electron-electron correlations
//!
//! This module provides Jastrow factors that capture electron-electron correlations
//! in variational Monte Carlo calculations. The Jastrow factor is a multiplicative
//! correction to the Slater determinant that accounts for electron-electron interactions.
//!
//! # Jastrow Factor Types
//!
//! 1. **Gutzwiller Factor**: Prevents double occupancy at the same site
//! 2. **Density-Density Jastrow**: Correlates electron densities at different sites
//! 3. **Spin-Spin Jastrow**: Correlates spin configurations
//!
//! # Mathematical Form
//!
//! The Jastrow factor has the general form:
//! J(x) = exp(∑ᵢⱼ vᵢⱼ nᵢ nⱼ + ∑ᵢⱼ wᵢⱼ Sᵢᶻ Sⱼᶻ + ...)
//!
//! where nᵢ is the electron density at site i, and Sᵢᶻ is the z-component of spin.

use crate::error::{Result, VmcError};
// use crate::types::{ElectronCount, SiteCount}; // Unused for now
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(test)]
use approx::assert_abs_diff_eq;

/// Jastrow factor parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JastrowParameters {
    /// Number of lattice sites
    pub nsite: usize,
    /// Density-density correlation parameters vᵢⱼ
    pub density_density: Vec<Vec<f64>>,
    /// Spin-spin correlation parameters wᵢⱼ
    pub spin_spin: Vec<Vec<f64>>,
    /// Gutzwiller parameter g (for double occupancy penalty)
    pub gutzwiller_g: f64,
    /// Whether to include density-density correlations
    pub include_density_density: bool,
    /// Whether to include spin-spin correlations
    pub include_spin_spin: bool,
    /// Whether to include Gutzwiller factor
    pub include_gutzwiller: bool,
}

/// Jastrow factor for electron-electron correlations
///
/// The Jastrow factor is a multiplicative correction to the Slater determinant
/// that captures electron-electron correlations beyond mean-field theory.
///
/// # References
///
/// - C implementation: `mVMC/src/mVMC/vmcmake.c:MakeJastrow`
/// - Jastrow factor calculation: `mVMC/src/mVMC/vmccal.c:CalculateJastrow`
#[derive(Debug, Clone, PartialEq)]
pub struct JastrowFactor {
    /// Jastrow parameters
    parameters: JastrowParameters,
    /// Precomputed values for efficiency
    precomputed: JastrowPrecomputed,
}

/// Precomputed values for efficient Jastrow factor calculation
#[derive(Debug, Clone, PartialEq)]
struct JastrowPrecomputed {
    /// Maximum number of electrons per site
    max_electrons_per_site: usize,
    /// Whether the system is spin-polarized
    is_spin_polarized: bool,
}

impl JastrowParameters {
    /// Creates new Jastrow parameters
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `gutzwiller_g` - Gutzwiller parameter (penalty for double occupancy)
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::jastrow::JastrowParameters;
    ///
    /// let params = JastrowParameters::new(4, 0.5);
    /// assert_eq!(params.nsite, 4);
    /// assert_eq!(params.gutzwiller_g, 0.5);
    /// ```
    pub fn new(nsite: usize, gutzwiller_g: f64) -> Self {
        Self {
            nsite,
            density_density: vec![vec![0.0; nsite]; nsite],
            spin_spin: vec![vec![0.0; nsite]; nsite],
            gutzwiller_g,
            include_density_density: true,
            include_spin_spin: true,
            include_gutzwiller: true,
        }
    }

    /// Sets density-density correlation parameters
    pub fn with_density_density(mut self, v: Vec<Vec<f64>>) -> Self {
        if v.len() == self.nsite && v.iter().all(|row| row.len() == self.nsite) {
            self.density_density = v;
        }
        self
    }

    /// Sets spin-spin correlation parameters
    pub fn with_spin_spin(mut self, w: Vec<Vec<f64>>) -> Self {
        if w.len() == self.nsite && w.iter().all(|row| row.len() == self.nsite) {
            self.spin_spin = w;
        }
        self
    }

    /// Enables or disables density-density correlations
    pub fn with_density_density_enabled(mut self, enabled: bool) -> Self {
        self.include_density_density = enabled;
        self
    }

    /// Enables or disables spin-spin correlations
    pub fn with_spin_spin_enabled(mut self, enabled: bool) -> Self {
        self.include_spin_spin = enabled;
        self
    }

    /// Enables or disables Gutzwiller factor
    pub fn with_gutzwiller_enabled(mut self, enabled: bool) -> Self {
        self.include_gutzwiller = enabled;
        self
    }

    /// Validates the parameters
    pub fn validate(&self) -> Result<()> {
        if self.nsite == 0 {
            return Err(VmcError::invalid_param("Number of sites must be positive"));
        }

        if self.density_density.len() != self.nsite {
            return Err(VmcError::dim_mismatch(
                self.nsite,
                self.density_density.len(),
            ));
        }

        for (i, row) in self.density_density.iter().enumerate() {
            if row.len() != self.nsite {
                return Err(VmcError::dim_mismatch(self.nsite, row.len()));
            }
            // Check for NaN or infinite values
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(VmcError::invalid_param(&format!(
                        "Invalid density-density parameter at ({}, {})",
                        i, j
                    )));
                }
            }
        }

        if self.spin_spin.len() != self.nsite {
            return Err(VmcError::dim_mismatch(self.nsite, self.spin_spin.len()));
        }

        for (i, row) in self.spin_spin.iter().enumerate() {
            if row.len() != self.nsite {
                return Err(VmcError::dim_mismatch(self.nsite, row.len()));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(VmcError::invalid_param(&format!(
                        "Invalid spin-spin parameter at ({}, {})",
                        i, j
                    )));
                }
            }
        }

        if !self.gutzwiller_g.is_finite() {
            return Err(VmcError::invalid_param("Gutzwiller parameter must be finite"));
        }

        Ok(())
    }
}

impl JastrowFactor {
    /// Creates a new Jastrow factor
    ///
    /// # Arguments
    ///
    /// * `parameters` - Jastrow parameters
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::jastrow::{JastrowFactor, JastrowParameters};
    ///
    /// let params = JastrowParameters::new(4, 0.5);
    /// let jastrow = JastrowFactor::new(params).unwrap();
    /// ```
    pub fn new(parameters: JastrowParameters) -> Result<Self> {
        parameters.validate()?;

        let precomputed = JastrowPrecomputed {
            max_electrons_per_site: 2, // Up to 2 electrons per site (up and down)
            is_spin_polarized: false,  // Will be determined from configuration
        };

        Ok(Self {
            parameters,
            precomputed,
        })
    }

    /// Creates a simple Gutzwiller factor
    ///
    /// This creates a Jastrow factor that only includes the Gutzwiller term,
    /// which penalizes double occupancy at the same site.
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `gutzwiller_g` - Gutzwiller parameter
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::jastrow::JastrowFactor;
    ///
    /// let jastrow = JastrowFactor::gutzwiller(4, 0.5).unwrap();
    /// ```
    pub fn gutzwiller(nsite: usize, gutzwiller_g: f64) -> Result<Self> {
        let parameters = JastrowParameters::new(nsite, gutzwiller_g)
            .with_density_density_enabled(false)
            .with_spin_spin_enabled(false)
            .with_gutzwiller_enabled(true);

        Self::new(parameters)
    }

    /// Calculates the Jastrow factor for a given electron configuration
    ///
    /// The Jastrow factor is calculated as:
    /// J(x) = exp(∑ᵢⱼ vᵢⱼ nᵢ nⱼ + ∑ᵢⱼ wᵢⱼ Sᵢᶻ Sⱼᶻ + g ∑ᵢ nᵢ↑ nᵢ↓)
    ///
    /// where:
    /// - nᵢ is the total electron density at site i
    /// - Sᵢᶻ is the z-component of spin at site i
    /// - nᵢ↑ and nᵢ↓ are the up and down electron densities at site i
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration (0=empty, 1=up, 2=down, 3=both)
    ///
    /// # Returns
    ///
    /// The Jastrow factor value
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmccal.c:CalculateJastrow`
    pub fn calculate_factor(&self, electron_config: &[u8]) -> Result<Complex64> {
        if electron_config.len() != self.parameters.nsite {
            return Err(VmcError::dim_mismatch(
                self.parameters.nsite,
                electron_config.len(),
            ));
        }

        let mut log_jastrow = 0.0;

        // Calculate density-density correlations
        if self.parameters.include_density_density {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    let ni = self.electron_density_at_site(electron_config, i);
                    let nj = self.electron_density_at_site(electron_config, j);
                    let vij = self.parameters.density_density[i][j];
                    log_jastrow += vij * ni * nj;
                }
            }
        }

        // Calculate spin-spin correlations
        if self.parameters.include_spin_spin {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    let szi = self.spin_z_at_site(electron_config, i);
                    let szj = self.spin_z_at_site(electron_config, j);
                    let wij = self.parameters.spin_spin[i][j];
                    log_jastrow += wij * szi * szj;
                }
            }
        }

        // Calculate Gutzwiller factor
        if self.parameters.include_gutzwiller {
            for i in 0..self.parameters.nsite {
                let double_occupancy = self.double_occupancy_at_site(electron_config, i);
                log_jastrow += self.parameters.gutzwiller_g * double_occupancy;
            }
        }

        Ok(Complex64::new(log_jastrow, 0.0).exp())
    }

    /// Calculates the log of the Jastrow factor
    ///
    /// This is more numerically stable than calculating the factor directly.
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The log of the Jastrow factor
    pub fn calculate_log_factor(&self, electron_config: &[u8]) -> Result<f64> {
        if electron_config.len() != self.parameters.nsite {
            return Err(VmcError::dim_mismatch(
                self.parameters.nsite,
                electron_config.len(),
            ));
        }

        let mut log_jastrow = 0.0;

        // Calculate density-density correlations
        if self.parameters.include_density_density {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    let ni = self.electron_density_at_site(electron_config, i);
                    let nj = self.electron_density_at_site(electron_config, j);
                    let vij = self.parameters.density_density[i][j];
                    log_jastrow += vij * ni * nj;
                }
            }
        }

        // Calculate spin-spin correlations
        if self.parameters.include_spin_spin {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    let szi = self.spin_z_at_site(electron_config, i);
                    let szj = self.spin_z_at_site(electron_config, j);
                    let wij = self.parameters.spin_spin[i][j];
                    log_jastrow += wij * szi * szj;
                }
            }
        }

        // Calculate Gutzwiller factor
        if self.parameters.include_gutzwiller {
            for i in 0..self.parameters.nsite {
                let double_occupancy = self.double_occupancy_at_site(electron_config, i);
                log_jastrow += self.parameters.gutzwiller_g * double_occupancy;
            }
        }

        Ok(log_jastrow)
    }

    /// Calculates the ratio of Jastrow factors for two configurations
    ///
    /// This is used in Metropolis acceptance criteria:
    /// ratio = J(new)/J(old)
    ///
    /// # Arguments
    ///
    /// * `new_config` - New electron configuration
    /// * `old_config` - Old electron configuration
    ///
    /// # Returns
    ///
    /// The Jastrow factor ratio
    pub fn calculate_ratio(
        &self,
        new_config: &[u8],
        old_config: &[u8],
    ) -> Result<Complex64> {
        let log_new = self.calculate_log_factor(new_config)?;
        let log_old = self.calculate_log_factor(old_config)?;
        let log_ratio = log_new - log_old;

        Ok(Complex64::new(log_ratio, 0.0).exp())
    }

    /// Returns the electron density at a specific site
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    /// * `site` - Site index
    ///
    /// # Returns
    ///
    /// The electron density (0.0, 1.0, or 2.0)
    fn electron_density_at_site(&self, electron_config: &[u8], site: usize) -> f64 {
        match electron_config[site] {
            0 => 0.0, // Empty
            1 => 1.0, // Up only
            2 => 1.0, // Down only
            3 => 2.0, // Both up and down
            _ => 0.0, // Invalid state
        }
    }

    /// Returns the z-component of spin at a specific site
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    /// * `site` - Site index
    ///
    /// # Returns
    ///
    /// The z-component of spin (-1.0, 0.0, or 1.0)
    fn spin_z_at_site(&self, electron_config: &[u8], site: usize) -> f64 {
        match electron_config[site] {
            0 => 0.0,  // Empty
            1 => 0.5,  // Up only
            2 => -0.5, // Down only
            3 => 0.0,  // Both up and down (singlet)
            _ => 0.0,  // Invalid state
        }
    }

    /// Returns the double occupancy at a specific site
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    /// * `site` - Site index
    ///
    /// # Returns
    ///
    /// The double occupancy (0.0 or 1.0)
    fn double_occupancy_at_site(&self, electron_config: &[u8], site: usize) -> f64 {
        match electron_config[site] {
            3 => 1.0, // Both up and down
            _ => 0.0, // Not double occupied
        }
    }

    /// Returns the Jastrow parameters
    pub fn parameters(&self) -> &JastrowParameters {
        &self.parameters
    }

    /// Returns a mutable reference to the Jastrow parameters
    pub fn parameters_mut(&mut self) -> &mut JastrowParameters {
        &mut self.parameters
    }
}

impl fmt::Display for JastrowFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JastrowFactor(")?;
        write!(f, "nsite={}", self.parameters.nsite)?;
        if self.parameters.include_density_density {
            write!(f, ", density-density")?;
        }
        if self.parameters.include_spin_spin {
            write!(f, ", spin-spin")?;
        }
        if self.parameters.include_gutzwiller {
            write!(f, ", gutzwiller(g={})", self.parameters.gutzwiller_g)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jastrow_parameters_creation() {
        let params = JastrowParameters::new(4, 0.5);
        assert_eq!(params.nsite, 4);
        assert_eq!(params.gutzwiller_g, 0.5);
        assert!(params.include_density_density);
        assert!(params.include_spin_spin);
        assert!(params.include_gutzwiller);
    }

    #[test]
    fn test_jastrow_parameters_validation() {
        let params = JastrowParameters::new(4, 0.5);
        assert!(params.validate().is_ok());

        let invalid_params = JastrowParameters::new(0, 0.5);
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_jastrow_factor_creation() {
        let params = JastrowParameters::new(4, 0.5);
        let jastrow = JastrowFactor::new(params);
        assert!(jastrow.is_ok());
    }

    #[test]
    fn test_gutzwiller_factor() {
        let jastrow = JastrowFactor::gutzwiller(4, 0.5);
        assert!(jastrow.is_ok());

        let jastrow = jastrow.unwrap();
        assert!(jastrow.parameters.include_gutzwiller);
        assert!(!jastrow.parameters.include_density_density);
        assert!(!jastrow.parameters.include_spin_spin);
    }

    #[test]
    fn test_jastrow_factor_calculation() {
        let jastrow = JastrowFactor::gutzwiller(2, 0.5).unwrap();

        // Empty configuration
        let empty_config = vec![0, 0];
        let factor = jastrow.calculate_factor(&empty_config).unwrap();
        assert_abs_diff_eq!(factor.re, 1.0, epsilon = 1e-10);

        // Single electron configuration
        let single_config = vec![1, 0];
        let factor = jastrow.calculate_factor(&single_config).unwrap();
        assert_abs_diff_eq!(factor.re, 1.0, epsilon = 1e-10);

        // Double occupancy configuration
        let double_config = vec![3, 0];
        let factor = jastrow.calculate_factor(&double_config).unwrap();
        let expected = (0.5_f64 * 1.0).exp(); // g * double_occupancy
        assert_abs_diff_eq!(factor.re, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_jastrow_factor_ratio() {
        let jastrow = JastrowFactor::gutzwiller(2, 0.5).unwrap();

        let config1 = vec![1, 0];
        let config2 = vec![3, 0];
        let ratio = jastrow.calculate_ratio(&config2, &config1).unwrap();

        let factor1 = jastrow.calculate_factor(&config1).unwrap();
        let factor2 = jastrow.calculate_factor(&config2).unwrap();
        let expected_ratio = factor2 / factor1;

        assert_abs_diff_eq!(ratio.re, expected_ratio.re, epsilon = 1e-10);
    }

    #[test]
    fn test_electron_density_calculation() {
        let jastrow = JastrowFactor::gutzwiller(4, 0.5).unwrap();

        let config = vec![0, 1, 2, 3];
        assert_eq!(jastrow.electron_density_at_site(&config, 0), 0.0);
        assert_eq!(jastrow.electron_density_at_site(&config, 1), 1.0);
        assert_eq!(jastrow.electron_density_at_site(&config, 2), 1.0);
        assert_eq!(jastrow.electron_density_at_site(&config, 3), 2.0);
    }

    #[test]
    fn test_spin_z_calculation() {
        let jastrow = JastrowFactor::gutzwiller(4, 0.5).unwrap();

        let config = vec![0, 1, 2, 3];
        assert_eq!(jastrow.spin_z_at_site(&config, 0), 0.0);
        assert_eq!(jastrow.spin_z_at_site(&config, 1), 0.5);
        assert_eq!(jastrow.spin_z_at_site(&config, 2), -0.5);
        assert_eq!(jastrow.spin_z_at_site(&config, 3), 0.0);
    }

    #[test]
    fn test_double_occupancy_calculation() {
        let jastrow = JastrowFactor::gutzwiller(4, 0.5).unwrap();

        let config = vec![0, 1, 2, 3];
        assert_eq!(jastrow.double_occupancy_at_site(&config, 0), 0.0);
        assert_eq!(jastrow.double_occupancy_at_site(&config, 1), 0.0);
        assert_eq!(jastrow.double_occupancy_at_site(&config, 2), 0.0);
        assert_eq!(jastrow.double_occupancy_at_site(&config, 3), 1.0);
    }

    #[test]
    fn test_jastrow_factor_display() {
        let jastrow = JastrowFactor::gutzwiller(4, 0.5).unwrap();
        let display = format!("{}", jastrow);
        assert!(display.contains("JastrowFactor"));
        assert!(display.contains("nsite=4"));
        assert!(display.contains("gutzwiller(g=0.5)"));
    }
}
