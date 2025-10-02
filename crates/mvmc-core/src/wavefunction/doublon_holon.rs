//! Doublon-Holon correlation factor for VMC calculations
//!
//! This module provides a correlation factor that captures the correlation
//! between doublons (doubly occupied sites) and holons (empty sites) in
//! strongly correlated electron systems.
//!
//! # Doublon-Holon Correlation
//!
//! In strongly correlated systems, doublons and holons tend to avoid each other
//! due to the strong on-site Coulomb repulsion. This correlation factor captures
//! this effect by penalizing configurations where doublons and holons are close
//! to each other.
//!
//! # Mathematical Form
//!
//! The Doublon-Holon correlation factor has the form:
//! DH(x) = exp(∑ᵢⱼ αᵢⱼ dᵢ hⱼ + ∑ᵢⱼ βᵢⱼ dᵢ dⱼ + ∑ᵢⱼ γᵢⱼ hᵢ hⱼ)
//!
//! where:
//! - dᵢ is 1 if site i is doubly occupied, 0 otherwise
//! - hᵢ is 1 if site i is empty, 0 otherwise
//! - αᵢⱼ is the doublon-holon correlation parameter
//! - βᵢⱼ is the doublon-doublon correlation parameter
//! - γᵢⱼ is the holon-holon correlation parameter

use crate::error::{Result, VmcError};
// use crate::types::{ElectronCount, SiteCount}; // Unused for now
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(test)]
use approx::assert_abs_diff_eq;

/// Doublon-Holon correlation parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoublonHolonParameters {
    /// Number of lattice sites
    pub nsite: usize,
    /// Doublon-holon correlation parameters αᵢⱼ
    pub doublon_holon: Vec<Vec<f64>>,
    /// Doublon-doublon correlation parameters βᵢⱼ
    pub doublon_doublon: Vec<Vec<f64>>,
    /// Holon-holon correlation parameters γᵢⱼ
    pub holon_holon: Vec<Vec<f64>>,
    /// Whether to include doublon-holon correlations
    pub include_doublon_holon: bool,
    /// Whether to include doublon-doublon correlations
    pub include_doublon_doublon: bool,
    /// Whether to include holon-holon correlations
    pub include_holon_holon: bool,
    /// Maximum correlation distance (for efficiency)
    pub max_distance: Option<usize>,
}

/// Doublon-Holon correlation factor
///
/// This factor captures the correlation between doublons and holons in
/// strongly correlated electron systems. It penalizes configurations where
/// doublons and holons are close to each other.
///
/// # References
///
/// - C implementation: `mVMC/src/mVMC/vmcmake.c:MakeDoublonHolon`
/// - Correlation calculation: `mVMC/src/mVMC/vmccal.c:CalculateDoublonHolon`
#[derive(Debug, Clone, PartialEq)]
pub struct DoublonHolonFactor {
    /// Doublon-Holon parameters
    parameters: DoublonHolonParameters,
    /// Precomputed values for efficiency
    precomputed: DoublonHolonPrecomputed,
}

/// Precomputed values for efficient Doublon-Holon factor calculation
#[derive(Debug, Clone, PartialEq)]
struct DoublonHolonPrecomputed {
    /// Maximum correlation distance
    max_distance: usize,
    /// Whether to use distance cutoff
    use_distance_cutoff: bool,
}

impl DoublonHolonParameters {
    /// Creates new Doublon-Holon parameters
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::doublon_holon::DoublonHolonParameters;
    ///
    /// let params = DoublonHolonParameters::new(4);
    /// assert_eq!(params.nsite, 4);
    /// ```
    pub fn new(nsite: usize) -> Self {
        Self {
            nsite,
            doublon_holon: vec![vec![0.0; nsite]; nsite],
            doublon_doublon: vec![vec![0.0; nsite]; nsite],
            holon_holon: vec![vec![0.0; nsite]; nsite],
            include_doublon_holon: true,
            include_doublon_doublon: true,
            include_holon_holon: true,
            max_distance: None,
        }
    }

    /// Sets doublon-holon correlation parameters
    pub fn with_doublon_holon(mut self, alpha: Vec<Vec<f64>>) -> Self {
        if alpha.len() == self.nsite && alpha.iter().all(|row| row.len() == self.nsite) {
            self.doublon_holon = alpha;
        }
        self
    }

    /// Sets doublon-doublon correlation parameters
    pub fn with_doublon_doublon(mut self, beta: Vec<Vec<f64>>) -> Self {
        if beta.len() == self.nsite && beta.iter().all(|row| row.len() == self.nsite) {
            self.doublon_doublon = beta;
        }
        self
    }

    /// Sets holon-holon correlation parameters
    pub fn with_holon_holon(mut self, gamma: Vec<Vec<f64>>) -> Self {
        if gamma.len() == self.nsite && gamma.iter().all(|row| row.len() == self.nsite) {
            self.holon_holon = gamma;
        }
        self
    }

    /// Enables or disables doublon-holon correlations
    pub fn with_doublon_holon_enabled(mut self, enabled: bool) -> Self {
        self.include_doublon_holon = enabled;
        self
    }

    /// Enables or disables doublon-doublon correlations
    pub fn with_doublon_doublon_enabled(mut self, enabled: bool) -> Self {
        self.include_doublon_doublon = enabled;
        self
    }

    /// Enables or disables holon-holon correlations
    pub fn with_holon_holon_enabled(mut self, enabled: bool) -> Self {
        self.include_holon_holon = enabled;
        self
    }

    /// Sets the maximum correlation distance
    pub fn with_max_distance(mut self, max_distance: Option<usize>) -> Self {
        self.max_distance = max_distance;
        self
    }

    /// Validates the parameters
    pub fn validate(&self) -> Result<()> {
        if self.nsite == 0 {
            return Err(VmcError::invalid_param("Number of sites must be positive"));
        }

        // Validate doublon-holon parameters
        if self.doublon_holon.len() != self.nsite {
            return Err(VmcError::dim_mismatch(
                self.nsite,
                self.doublon_holon.len(),
            ));
        }

        for (i, row) in self.doublon_holon.iter().enumerate() {
            if row.len() != self.nsite {
                return Err(VmcError::dim_mismatch(self.nsite, row.len()));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(VmcError::invalid_param(&format!(
                        "Invalid doublon-holon parameter at ({}, {})",
                        i, j
                    )));
                }
            }
        }

        // Validate doublon-doublon parameters
        if self.doublon_doublon.len() != self.nsite {
            return Err(VmcError::dim_mismatch(
                self.nsite,
                self.doublon_doublon.len(),
            ));
        }

        for (i, row) in self.doublon_doublon.iter().enumerate() {
            if row.len() != self.nsite {
                return Err(VmcError::dim_mismatch(self.nsite, row.len()));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(VmcError::invalid_param(&format!(
                        "Invalid doublon-doublon parameter at ({}, {})",
                        i, j
                    )));
                }
            }
        }

        // Validate holon-holon parameters
        if self.holon_holon.len() != self.nsite {
            return Err(VmcError::dim_mismatch(
                self.nsite,
                self.holon_holon.len(),
            ));
        }

        for (i, row) in self.holon_holon.iter().enumerate() {
            if row.len() != self.nsite {
                return Err(VmcError::dim_mismatch(self.nsite, row.len()));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(VmcError::invalid_param(&format!(
                        "Invalid holon-holon parameter at ({}, {})",
                        i, j
                    )));
                }
            }
        }

        Ok(())
    }
}

impl DoublonHolonFactor {
    /// Creates a new Doublon-Holon factor
    ///
    /// # Arguments
    ///
    /// * `parameters` - Doublon-Holon parameters
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::doublon_holon::{DoublonHolonFactor, DoublonHolonParameters};
    ///
    /// let params = DoublonHolonParameters::new(4);
    /// let dh_factor = DoublonHolonFactor::new(params).unwrap();
    /// ```
    pub fn new(parameters: DoublonHolonParameters) -> Result<Self> {
        parameters.validate()?;

        let max_distance = parameters.max_distance.unwrap_or(parameters.nsite);
        let precomputed = DoublonHolonPrecomputed {
            max_distance,
            use_distance_cutoff: parameters.max_distance.is_some(),
        };

        Ok(Self {
            parameters,
            precomputed,
        })
    }

    /// Creates a simple doublon-holon correlation factor
    ///
    /// This creates a factor that only includes doublon-holon correlations
    /// with a simple distance-dependent form.
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `strength` - Correlation strength
    /// * `max_distance` - Maximum correlation distance
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::doublon_holon::DoublonHolonFactor;
    ///
    /// let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2)).unwrap();
    /// ```
    pub fn simple(
        nsite: usize,
        strength: f64,
        max_distance: Option<usize>,
    ) -> Result<Self> {
        let mut parameters = DoublonHolonParameters::new(nsite)
            .with_doublon_doublon_enabled(false)
            .with_holon_holon_enabled(false)
            .with_max_distance(max_distance);

        // Set simple distance-dependent parameters
        for i in 0..nsite {
            for j in 0..nsite {
                if i != j {
                    let distance = self::distance_1d(i, j, nsite);
                    if let Some(max_dist) = max_distance {
                        if distance <= max_dist {
                            parameters.doublon_holon[i][j] = strength / (distance as f64 + 1.0);
                        }
                    } else {
                        parameters.doublon_holon[i][j] = strength / (distance as f64 + 1.0);
                    }
                }
            }
        }

        Self::new(parameters)
    }

    /// Calculates the Doublon-Holon factor for a given electron configuration
    ///
    /// The factor is calculated as:
    /// DH(x) = exp(∑ᵢⱼ αᵢⱼ dᵢ hⱼ + ∑ᵢⱼ βᵢⱼ dᵢ dⱼ + ∑ᵢⱼ γᵢⱼ hᵢ hⱼ)
    ///
    /// where:
    /// - dᵢ is 1 if site i is doubly occupied, 0 otherwise
    /// - hᵢ is 1 if site i is empty, 0 otherwise
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration (0=empty, 1=up, 2=down, 3=both)
    ///
    /// # Returns
    ///
    /// The Doublon-Holon factor value
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmccal.c:CalculateDoublonHolon`
    pub fn calculate_factor(&self, electron_config: &[u8]) -> Result<Complex64> {
        if electron_config.len() != self.parameters.nsite {
            return Err(VmcError::dim_mismatch(
                self.parameters.nsite,
                electron_config.len(),
            ));
        }

        let mut log_dh = 0.0;

        // Calculate doublon-holon correlations
        if self.parameters.include_doublon_holon {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    if self.should_calculate_correlation(i, j) {
                        let di = self.is_doublon(electron_config, i);
                        let hj = self.is_holon(electron_config, j);
                        let alpha_ij = self.parameters.doublon_holon[i][j];
                        log_dh += alpha_ij * di * hj;
                    }
                }
            }
        }

        // Calculate doublon-doublon correlations
        if self.parameters.include_doublon_doublon {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    if self.should_calculate_correlation(i, j) {
                        let di = self.is_doublon(electron_config, i);
                        let dj = self.is_doublon(electron_config, j);
                        let beta_ij = self.parameters.doublon_doublon[i][j];
                        log_dh += beta_ij * di * dj;
                    }
                }
            }
        }

        // Calculate holon-holon correlations
        if self.parameters.include_holon_holon {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    if self.should_calculate_correlation(i, j) {
                        let hi = self.is_holon(electron_config, i);
                        let hj = self.is_holon(electron_config, j);
                        let gamma_ij = self.parameters.holon_holon[i][j];
                        log_dh += gamma_ij * hi * hj;
                    }
                }
            }
        }

        Ok(Complex64::new(log_dh, 0.0).exp())
    }

    /// Calculates the log of the Doublon-Holon factor
    ///
    /// This is more numerically stable than calculating the factor directly.
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The log of the Doublon-Holon factor
    pub fn calculate_log_factor(&self, electron_config: &[u8]) -> Result<f64> {
        if electron_config.len() != self.parameters.nsite {
            return Err(VmcError::dim_mismatch(
                self.parameters.nsite,
                electron_config.len(),
            ));
        }

        let mut log_dh = 0.0;

        // Calculate doublon-holon correlations
        if self.parameters.include_doublon_holon {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    if self.should_calculate_correlation(i, j) {
                        let di = self.is_doublon(electron_config, i);
                        let hj = self.is_holon(electron_config, j);
                        let alpha_ij = self.parameters.doublon_holon[i][j];
                        log_dh += alpha_ij * di * hj;
                    }
                }
            }
        }

        // Calculate doublon-doublon correlations
        if self.parameters.include_doublon_doublon {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    if self.should_calculate_correlation(i, j) {
                        let di = self.is_doublon(electron_config, i);
                        let dj = self.is_doublon(electron_config, j);
                        let beta_ij = self.parameters.doublon_doublon[i][j];
                        log_dh += beta_ij * di * dj;
                    }
                }
            }
        }

        // Calculate holon-holon correlations
        if self.parameters.include_holon_holon {
            for i in 0..self.parameters.nsite {
                for j in 0..self.parameters.nsite {
                    if self.should_calculate_correlation(i, j) {
                        let hi = self.is_holon(electron_config, i);
                        let hj = self.is_holon(electron_config, j);
                        let gamma_ij = self.parameters.holon_holon[i][j];
                        log_dh += gamma_ij * hi * hj;
                    }
                }
            }
        }

        Ok(log_dh)
    }

    /// Calculates the ratio of Doublon-Holon factors for two configurations
    ///
    /// This is used in Metropolis acceptance criteria:
    /// ratio = DH(new)/DH(old)
    ///
    /// # Arguments
    ///
    /// * `new_config` - New electron configuration
    /// * `old_config` - Old electron configuration
    ///
    /// # Returns
    ///
    /// The Doublon-Holon factor ratio
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

    /// Checks if a site is doubly occupied
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    /// * `site` - Site index
    ///
    /// # Returns
    ///
    /// 1.0 if the site is doubly occupied, 0.0 otherwise
    fn is_doublon(&self, electron_config: &[u8], site: usize) -> f64 {
        match electron_config[site] {
            3 => 1.0, // Both up and down
            _ => 0.0, // Not doubly occupied
        }
    }

    /// Checks if a site is empty (holon)
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    /// * `site` - Site index
    ///
    /// # Returns
    ///
    /// 1.0 if the site is empty, 0.0 otherwise
    fn is_holon(&self, electron_config: &[u8], site: usize) -> f64 {
        match electron_config[site] {
            0 => 1.0, // Empty
            _ => 0.0, // Not empty
        }
    }

    /// Determines whether to calculate correlation between two sites
    ///
    /// This includes distance cutoff if specified.
    ///
    /// # Arguments
    ///
    /// * `i` - First site index
    /// * `j` - Second site index
    ///
    /// # Returns
    ///
    /// True if correlation should be calculated
    fn should_calculate_correlation(&self, i: usize, j: usize) -> bool {
        if i == j {
            return false; // No self-correlation
        }

        if self.precomputed.use_distance_cutoff {
            let distance = distance_1d(i, j, self.parameters.nsite);
            distance <= self.precomputed.max_distance
        } else {
            true
        }
    }

    /// Returns the Doublon-Holon parameters
    pub fn parameters(&self) -> &DoublonHolonParameters {
        &self.parameters
    }

    /// Returns a mutable reference to the Doublon-Holon parameters
    pub fn parameters_mut(&mut self) -> &mut DoublonHolonParameters {
        &mut self.parameters
    }
}

/// Calculates the 1D distance between two sites
///
/// This assumes a 1D chain with periodic boundary conditions.
///
/// # Arguments
///
/// * `i` - First site index
/// * `j` - Second site index
/// * `nsite` - Total number of sites
///
/// # Returns
///
/// The minimum distance between the sites
fn distance_1d(i: usize, j: usize, nsite: usize) -> usize {
    let diff = if i > j { i - j } else { j - i };
    std::cmp::min(diff, nsite - diff)
}

impl fmt::Display for DoublonHolonFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DoublonHolonFactor(")?;
        write!(f, "nsite={}", self.parameters.nsite)?;
        if self.parameters.include_doublon_holon {
            write!(f, ", doublon-holon")?;
        }
        if self.parameters.include_doublon_doublon {
            write!(f, ", doublon-doublon")?;
        }
        if self.parameters.include_holon_holon {
            write!(f, ", holon-holon")?;
        }
        if let Some(max_dist) = self.parameters.max_distance {
            write!(f, ", max_distance={}", max_dist)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doublon_holon_parameters_creation() {
        let params = DoublonHolonParameters::new(4);
        assert_eq!(params.nsite, 4);
        assert!(params.include_doublon_holon);
        assert!(params.include_doublon_doublon);
        assert!(params.include_holon_holon);
    }

    #[test]
    fn test_doublon_holon_parameters_validation() {
        let params = DoublonHolonParameters::new(4);
        assert!(params.validate().is_ok());

        let invalid_params = DoublonHolonParameters::new(0);
        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_doublon_holon_factor_creation() {
        let params = DoublonHolonParameters::new(4);
        let dh_factor = DoublonHolonFactor::new(params);
        assert!(dh_factor.is_ok());
    }

    #[test]
    fn test_simple_doublon_holon_factor() {
        let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2));
        assert!(dh_factor.is_ok());

        let dh_factor = dh_factor.unwrap();
        assert!(dh_factor.parameters.include_doublon_holon);
        assert!(!dh_factor.parameters.include_doublon_doublon);
        assert!(!dh_factor.parameters.include_holon_holon);
    }

    #[test]
    fn test_doublon_holon_factor_calculation() {
        let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2)).unwrap();

        // Empty configuration (all holons)
        let empty_config = vec![0, 0, 0, 0];
        let factor = dh_factor.calculate_factor(&empty_config).unwrap();
        assert_abs_diff_eq!(factor.re, 1.0, epsilon = 1e-10);

        // Mixed configuration (doublon and holon)
        let mixed_config = vec![3, 0, 1, 2];
        let factor = dh_factor.calculate_factor(&mixed_config).unwrap();
        assert!(factor.re > 0.0);
    }

    #[test]
    fn test_doublon_holon_factor_ratio() {
        let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2)).unwrap();

        let config1 = vec![3, 0, 1, 2];
        let config2 = vec![0, 3, 1, 2];
        let ratio = dh_factor.calculate_ratio(&config2, &config1).unwrap();

        let factor1 = dh_factor.calculate_factor(&config1).unwrap();
        let factor2 = dh_factor.calculate_factor(&config2).unwrap();
        let expected_ratio = factor2 / factor1;

        assert_abs_diff_eq!(ratio.re, expected_ratio.re, epsilon = 1e-10);
    }

    #[test]
    fn test_doublon_detection() {
        let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2)).unwrap();

        let config = vec![0, 1, 2, 3];
        assert_eq!(dh_factor.is_doublon(&config, 0), 0.0);
        assert_eq!(dh_factor.is_doublon(&config, 1), 0.0);
        assert_eq!(dh_factor.is_doublon(&config, 2), 0.0);
        assert_eq!(dh_factor.is_doublon(&config, 3), 1.0);
    }

    #[test]
    fn test_holon_detection() {
        let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2)).unwrap();

        let config = vec![0, 1, 2, 3];
        assert_eq!(dh_factor.is_holon(&config, 0), 1.0);
        assert_eq!(dh_factor.is_holon(&config, 1), 0.0);
        assert_eq!(dh_factor.is_holon(&config, 2), 0.0);
        assert_eq!(dh_factor.is_holon(&config, 3), 0.0);
    }

    #[test]
    fn test_distance_calculation() {
        assert_eq!(distance_1d(0, 1, 4), 1);
        assert_eq!(distance_1d(0, 3, 4), 1);
        assert_eq!(distance_1d(1, 3, 4), 2);
        assert_eq!(distance_1d(0, 2, 4), 2);
    }

    #[test]
    fn test_doublon_holon_factor_display() {
        let dh_factor = DoublonHolonFactor::simple(4, 0.1, Some(2)).unwrap();
        let display = format!("{}", dh_factor);
        assert!(display.contains("DoublonHolonFactor"));
        assert!(display.contains("nsite=4"));
        assert!(display.contains("doublon-holon"));
        assert!(display.contains("max_distance=2"));
    }
}
