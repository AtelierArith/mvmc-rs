//! Parameter validation utilities
//!
//! This module provides validation functionality for VMC parameters.

use crate::config::parameters::{MonteCarloParameters, SRParameters, VmcParameters};
use crate::{Result, VmcError};

/// Parameter validator
///
/// Provides validation methods for VMC parameters beyond the basic
/// validation in each parameter struct.
pub struct ParameterValidator;

impl ParameterValidator {
    /// Validates VmcParameters comprehensively
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::config::{VmcParameters, ParameterValidator};
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, CalcMode, LanczosMode, RandomSeed};
    /// use mvmc_core::config::{SRParameters, MonteCarloParameters};
    ///
    /// let params = VmcParameters::new(
    ///     SiteCount::new(4),
    ///     ElectronCount::new(2),
    ///     TwoSz::new(0),
    ///     CalcMode::Optimization,
    ///     LanczosMode::None,
    ///     RandomSeed::new(12345),
    ///     SRParameters::default(),
    ///     MonteCarloParameters::default(),
    /// );
    ///
    /// assert!(ParameterValidator::validate_vmc_params(&params).is_ok());
    /// ```
    pub fn validate_vmc_params(params: &VmcParameters) -> Result<()> {
        // Delegate to the parameter's own validate method
        params.validate()?;

        // Additional cross-parameter validations can be added here
        // For example, check if sample count is sufficient for the number of sites
        if params.mc_params.num_samples < params.nsite.get() {
            return Err(VmcError::invalid_config(format!(
                "Number of samples ({}) should be at least equal to number of sites ({})",
                params.mc_params.num_samples,
                params.nsite.get()
            )));
        }

        Ok(())
    }

    /// Validates SR parameters
    pub fn validate_sr_params(params: &SRParameters) -> Result<()> {
        params.validate()
    }

    /// Validates Monte Carlo parameters
    pub fn validate_mc_params(params: &MonteCarloParameters) -> Result<()> {
        params.validate()
    }

    /// Validates that spin configuration is physically reasonable
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::config::ParameterValidator;
    /// use mvmc_core::types::{ElectronCount, TwoSz};
    ///
    /// let ne = ElectronCount::new(4);
    /// let two_sz = TwoSz::new(0);
    /// assert!(ParameterValidator::validate_spin_config(ne, two_sz).is_ok());
    ///
    /// let two_sz_invalid = TwoSz::new(10); // |2*Sz| > Ne
    /// assert!(ParameterValidator::validate_spin_config(ne, two_sz_invalid).is_err());
    /// ```
    pub fn validate_spin_config(
        ne: crate::types::ElectronCount,
        two_sz: crate::types::TwoSz,
    ) -> Result<()> {
        let ne_val = ne.get() as i32;
        let two_sz_val = two_sz.get();

        // Check if |2*Sz| <= Ne
        if two_sz_val.abs() > ne_val {
            return Err(VmcError::invalid_config(format!(
                "Invalid spin configuration: |2*Sz| = {} > Ne = {}",
                two_sz_val.abs(),
                ne_val
            )));
        }

        // Check if (Ne - 2*Sz) is even (must have integer number of down spins)
        if (ne_val - two_sz_val) % 2 != 0 {
            return Err(VmcError::invalid_config(format!(
                "Invalid spin configuration: (Ne - 2*Sz) = ({} - {}) must be even",
                ne_val, two_sz_val
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CalcMode, ElectronCount, LanczosMode, RandomSeed, SiteCount, TwoSz};

    #[test]
    fn test_validate_vmc_params_valid() {
        let params = VmcParameters::new(
            SiteCount::new(4),
            ElectronCount::new(2),
            TwoSz::new(0),
            CalcMode::Optimization,
            LanczosMode::None,
            RandomSeed::new(12345),
            SRParameters::default(),
            MonteCarloParameters::default(),
        );

        assert!(ParameterValidator::validate_vmc_params(&params).is_ok());
    }

    #[test]
    fn test_validate_vmc_params_insufficient_samples() {
        let params = VmcParameters::new(
            SiteCount::new(1000), // Large site count
            ElectronCount::new(500),
            TwoSz::new(0),
            CalcMode::Optimization,
            LanczosMode::None,
            RandomSeed::new(12345),
            SRParameters::default(),
            MonteCarloParameters::new(100, 1, 100, false, 1), // Only 100 samples
        );

        assert!(ParameterValidator::validate_vmc_params(&params).is_err());
    }

    #[test]
    fn test_validate_spin_config_valid() {
        // Ne = 4, 2*Sz = 0 (2 up, 2 down)
        assert!(
            ParameterValidator::validate_spin_config(ElectronCount::new(4), TwoSz::new(0))
                .is_ok()
        );

        // Ne = 4, 2*Sz = 2 (3 up, 1 down)
        assert!(
            ParameterValidator::validate_spin_config(ElectronCount::new(4), TwoSz::new(2))
                .is_ok()
        );

        // Ne = 4, 2*Sz = -2 (1 up, 3 down)
        assert!(
            ParameterValidator::validate_spin_config(ElectronCount::new(4), TwoSz::new(-2))
                .is_ok()
        );
    }

    #[test]
    fn test_validate_spin_config_too_large() {
        // Ne = 4, 2*Sz = 6 (impossible: would need 5 up electrons)
        assert!(
            ParameterValidator::validate_spin_config(ElectronCount::new(4), TwoSz::new(6))
                .is_err()
        );
    }

    #[test]
    fn test_validate_spin_config_odd_parity() {
        // Ne = 4, 2*Sz = 1 (would need 2.5 up and 1.5 down electrons)
        assert!(
            ParameterValidator::validate_spin_config(ElectronCount::new(4), TwoSz::new(1))
                .is_err()
        );
    }

    #[test]
    fn test_validate_sr_params() {
        let params = SRParameters::default();
        assert!(ParameterValidator::validate_sr_params(&params).is_ok());
    }

    #[test]
    fn test_validate_mc_params() {
        let params = MonteCarloParameters::default();
        assert!(ParameterValidator::validate_mc_params(&params).is_ok());
    }
}
