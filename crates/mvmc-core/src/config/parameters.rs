//! VMC calculation parameters
//!
//! This module defines parameter structures for variational Monte Carlo calculations.

use crate::types::{CalcMode, ElectronCount, LanczosMode, RandomSeed, SiteCount, TwoSz};
use crate::{Result, VmcError};

/// Main VMC calculation parameters
///
/// This structure contains all parameters needed for a VMC calculation,
/// corresponding to the variables in the C implementation's global.h
#[derive(Debug, Clone, PartialEq)]
pub struct VmcParameters {
    /// Number of lattice sites
    pub nsite: SiteCount,

    /// Number of electrons
    pub ne: ElectronCount,

    /// Spin quantum number (2*Sz)
    pub two_sz: TwoSz,

    /// Calculation mode (optimization or expectation)
    pub calc_mode: CalcMode,

    /// Lanczos mode
    pub lanczos_mode: LanczosMode,

    /// Random seed
    pub random_seed: RandomSeed,

    /// SR (Stochastic Reconfiguration) parameters
    pub sr_params: SRParameters,

    /// Monte Carlo sampling parameters
    pub mc_params: MonteCarloParameters,
}

impl VmcParameters {
    /// Creates a new VmcParameters with the given values
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::config::VmcParameters;
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
    /// assert_eq!(params.nsite.get(), 4);
    /// assert_eq!(params.ne.get(), 2);
    /// ```
    pub fn new(
        nsite: SiteCount,
        ne: ElectronCount,
        two_sz: TwoSz,
        calc_mode: CalcMode,
        lanczos_mode: LanczosMode,
        random_seed: RandomSeed,
        sr_params: SRParameters,
        mc_params: MonteCarloParameters,
    ) -> Self {
        Self {
            nsite,
            ne,
            two_sz,
            calc_mode,
            lanczos_mode,
            random_seed,
            sr_params,
            mc_params,
        }
    }

    /// Creates a builder for VmcParameters
    pub fn builder() -> VmcParametersBuilder {
        VmcParametersBuilder::default()
    }

    /// Validates the parameters
    ///
    /// Returns an error if the parameters are inconsistent or invalid.
    pub fn validate(&self) -> Result<()> {
        // Check electron count doesn't exceed site count
        if self.ne.get() > self.nsite.get() * 2 {
            return Err(VmcError::invalid_config(format!(
                "Too many electrons: {} > 2 * {}",
                self.ne.get(),
                self.nsite.get()
            )));
        }

        // Validate SR parameters
        self.sr_params.validate()?;

        // Validate MC parameters
        self.mc_params.validate()?;

        Ok(())
    }
}

/// Stochastic Reconfiguration (SR) method parameters
///
/// Parameters for the SR optimization algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct SRParameters {
    /// Number of SR iteration steps (NSROptItrStep)
    pub iteration_steps: usize,

    /// Number of steps for calculating average values (NSROptItrSmp)
    pub iteration_sample: usize,

    /// Number of steps with fixed samples (NSROptFixSmp)
    pub fixed_sample_steps: usize,

    /// Reduction cutoff for truncating redundant directions (DSROptRedCut)
    pub reduction_cutoff: f64,

    /// Stabilizing factor for diagonal element modification (DSROptStaDel)
    pub stability_delta: f64,

    /// Step width of the SR method (DSROptStepDt)
    pub step_size: f64,

    /// Maximum iterations in SR-CG method (NSROptCGMaxIter)
    pub cg_max_iterations: usize,

    /// Tolerance for SR-CG method (DSROptCGTol)
    pub cg_tolerance: f64,
}

impl SRParameters {
    /// Creates SR parameters with default values recommended for typical calculations
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::config::SRParameters;
    ///
    /// let params = SRParameters::default();
    /// assert_eq!(params.iteration_steps, 1000);
    /// ```
    pub fn new(
        iteration_steps: usize,
        iteration_sample: usize,
        fixed_sample_steps: usize,
        reduction_cutoff: f64,
        stability_delta: f64,
        step_size: f64,
        cg_max_iterations: usize,
        cg_tolerance: f64,
    ) -> Self {
        Self {
            iteration_steps,
            iteration_sample,
            fixed_sample_steps,
            reduction_cutoff,
            stability_delta,
            step_size,
            cg_max_iterations,
            cg_tolerance,
        }
    }

    /// Validates the SR parameters
    pub fn validate(&self) -> Result<()> {
        if self.iteration_steps == 0 {
            return Err(VmcError::invalid_param("iteration_steps must be > 0"));
        }

        if self.reduction_cutoff < 0.0 {
            return Err(VmcError::invalid_param("reduction_cutoff must be >= 0"));
        }

        if self.step_size <= 0.0 {
            return Err(VmcError::invalid_param("step_size must be > 0"));
        }

        if self.cg_tolerance <= 0.0 {
            return Err(VmcError::invalid_param("cg_tolerance must be > 0"));
        }

        Ok(())
    }
}

impl Default for SRParameters {
    /// Default SR parameters based on typical mVMC usage
    fn default() -> Self {
        Self {
            iteration_steps: 1000,
            iteration_sample: 100,
            fixed_sample_steps: 1,
            reduction_cutoff: 1e-8,
            stability_delta: 1e-2,
            step_size: 3e-3,
            cg_max_iterations: 1000,
            cg_tolerance: 1e-10,
        }
    }
}

/// Monte Carlo sampling parameters
///
/// Parameters controlling the Monte Carlo sampling process.
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloParameters {
    /// Number of warm-up steps (NVMCWarmUp)
    pub warmup_steps: usize,

    /// Sampling interval in Monte Carlo steps (NVMCInterval)
    pub sampling_interval: usize,

    /// Number of samples to collect (NVMCSample)
    pub num_samples: usize,

    /// Enable exchange hopping updates (NExUpdatePath)
    pub exchange_update: bool,

    /// Block size for Pfaffian update (NBlockUpdateSize)
    pub block_update_size: usize,
}

impl MonteCarloParameters {
    /// Creates new Monte Carlo parameters
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::config::MonteCarloParameters;
    ///
    /// let params = MonteCarloParameters::new(100, 1, 1000, false, 1);
    /// assert_eq!(params.num_samples, 1000);
    /// ```
    pub fn new(
        warmup_steps: usize,
        sampling_interval: usize,
        num_samples: usize,
        exchange_update: bool,
        block_update_size: usize,
    ) -> Self {
        Self {
            warmup_steps,
            sampling_interval,
            num_samples,
            exchange_update,
            block_update_size,
        }
    }

    /// Validates the Monte Carlo parameters
    pub fn validate(&self) -> Result<()> {
        if self.num_samples == 0 {
            return Err(VmcError::invalid_param("num_samples must be > 0"));
        }

        if self.sampling_interval == 0 {
            return Err(VmcError::invalid_param("sampling_interval must be > 0"));
        }

        if self.block_update_size == 0 {
            return Err(VmcError::invalid_param("block_update_size must be > 0"));
        }

        Ok(())
    }

    /// Returns the total number of Monte Carlo steps needed
    pub fn total_steps(&self) -> usize {
        self.warmup_steps + self.num_samples * self.sampling_interval
    }
}

impl Default for MonteCarloParameters {
    /// Default Monte Carlo parameters based on typical mVMC usage
    fn default() -> Self {
        Self {
            warmup_steps: 100,
            sampling_interval: 1,
            num_samples: 1000,
            exchange_update: false,
            block_update_size: 1,
        }
    }
}

/// Builder for VmcParameters
#[derive(Default)]
pub struct VmcParametersBuilder {
    nsite: Option<SiteCount>,
    ne: Option<ElectronCount>,
    two_sz: Option<TwoSz>,
    calc_mode: Option<CalcMode>,
    lanczos_mode: Option<LanczosMode>,
    random_seed: Option<RandomSeed>,
    sr_params: Option<SRParameters>,
    mc_params: Option<MonteCarloParameters>,
}

impl VmcParametersBuilder {
    pub fn nsite(mut self, nsite: SiteCount) -> Self {
        self.nsite = Some(nsite);
        self
    }

    pub fn ne(mut self, ne: ElectronCount) -> Self {
        self.ne = Some(ne);
        self
    }

    pub fn two_sz(mut self, two_sz: TwoSz) -> Self {
        self.two_sz = Some(two_sz);
        self
    }

    pub fn calc_mode(mut self, calc_mode: CalcMode) -> Self {
        self.calc_mode = Some(calc_mode);
        self
    }

    pub fn lanczos_mode(mut self, lanczos_mode: LanczosMode) -> Self {
        self.lanczos_mode = Some(lanczos_mode);
        self
    }

    pub fn random_seed(mut self, random_seed: RandomSeed) -> Self {
        self.random_seed = Some(random_seed);
        self
    }

    pub fn sr_params(mut self, sr_params: SRParameters) -> Self {
        self.sr_params = Some(sr_params);
        self
    }

    pub fn mc_params(mut self, mc_params: MonteCarloParameters) -> Self {
        self.mc_params = Some(mc_params);
        self
    }

    pub fn build(self) -> Result<VmcParameters> {
        let params = VmcParameters {
            nsite: self
                .nsite
                .ok_or_else(|| VmcError::invalid_param("nsite is required"))?,
            ne: self
                .ne
                .ok_or_else(|| VmcError::invalid_param("ne is required"))?,
            two_sz: self.two_sz.unwrap_or(TwoSz::new(0)),
            calc_mode: self.calc_mode.unwrap_or(CalcMode::Optimization),
            lanczos_mode: self.lanczos_mode.unwrap_or(LanczosMode::None),
            random_seed: self.random_seed.unwrap_or(RandomSeed::new(123456789)),
            sr_params: self.sr_params.unwrap_or_default(),
            mc_params: self.mc_params.unwrap_or_default(),
        };

        params.validate()?;
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmc_parameters_creation() {
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

        assert_eq!(params.nsite.get(), 4);
        assert_eq!(params.ne.get(), 2);
        assert_eq!(params.two_sz.get(), 0);
        assert_eq!(params.calc_mode, CalcMode::Optimization);
    }

    #[test]
    fn test_vmc_parameters_validation() {
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

        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_vmc_parameters_too_many_electrons() {
        let params = VmcParameters::new(
            SiteCount::new(4),
            ElectronCount::new(10), // Too many!
            TwoSz::new(0),
            CalcMode::Optimization,
            LanczosMode::None,
            RandomSeed::new(12345),
            SRParameters::default(),
            MonteCarloParameters::default(),
        );

        assert!(params.validate().is_err());
    }

    #[test]
    fn test_sr_parameters_default() {
        let params = SRParameters::default();
        assert_eq!(params.iteration_steps, 1000);
        assert_eq!(params.step_size, 3e-3);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_sr_parameters_validation() {
        let mut params = SRParameters::default();

        // Valid parameters
        assert!(params.validate().is_ok());

        // Zero iteration steps
        params.iteration_steps = 0;
        assert!(params.validate().is_err());
        params.iteration_steps = 1000;

        // Negative reduction cutoff
        params.reduction_cutoff = -1.0;
        assert!(params.validate().is_err());
        params.reduction_cutoff = 1e-8;

        // Zero step size
        params.step_size = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_mc_parameters_default() {
        let params = MonteCarloParameters::default();
        assert_eq!(params.warmup_steps, 100);
        assert_eq!(params.num_samples, 1000);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_mc_parameters_total_steps() {
        let params = MonteCarloParameters::new(100, 2, 500, false, 1);
        assert_eq!(params.total_steps(), 100 + 500 * 2);
    }

    #[test]
    fn test_mc_parameters_validation() {
        let mut params = MonteCarloParameters::default();

        // Valid parameters
        assert!(params.validate().is_ok());

        // Zero samples
        params.num_samples = 0;
        assert!(params.validate().is_err());
        params.num_samples = 1000;

        // Zero interval
        params.sampling_interval = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let params = VmcParameters::builder()
            .nsite(SiteCount::new(6))
            .ne(ElectronCount::new(4))
            .two_sz(TwoSz::new(0))
            .calc_mode(CalcMode::Expectation)
            .random_seed(RandomSeed::new(54321))
            .build()
            .unwrap();

        assert_eq!(params.nsite.get(), 6);
        assert_eq!(params.ne.get(), 4);
        assert_eq!(params.calc_mode, CalcMode::Expectation);
    }

    #[test]
    fn test_builder_missing_required_fields() {
        let result = VmcParameters::builder()
            .nsite(SiteCount::new(6))
            // Missing ne
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_defaults() {
        let params = VmcParameters::builder()
            .nsite(SiteCount::new(4))
            .ne(ElectronCount::new(2))
            .build()
            .unwrap();

        // Should use default values
        assert_eq!(params.two_sz.get(), 0);
        assert_eq!(params.calc_mode, CalcMode::Optimization);
        assert_eq!(params.lanczos_mode, LanczosMode::None);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_valid_electron_count(
            nsite in 1usize..100,
            ne in 0usize..200,
        ) {
            let params = VmcParameters::new(
                SiteCount::new(nsite),
                ElectronCount::new(ne),
                TwoSz::new(0),
                CalcMode::Optimization,
                LanczosMode::None,
                RandomSeed::new(12345),
                SRParameters::default(),
                MonteCarloParameters::default(),
            );

            let is_valid = ne <= nsite * 2;
            prop_assert_eq!(params.validate().is_ok(), is_valid);
        }

        #[test]
        fn prop_mc_total_steps(
            warmup in 0usize..1000,
            interval in 1usize..100,
            samples in 1usize..1000,
        ) {
            let params = MonteCarloParameters::new(
                warmup, interval, samples, false, 1
            );
            let expected = warmup + interval * samples;
            prop_assert_eq!(params.total_steps(), expected);
        }
    }
}
