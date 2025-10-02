//! Metropolis sampling algorithm for variational Monte Carlo calculations
//!
//! This module implements the Metropolis sampling algorithm for VMC calculations.
//! It corresponds to the main VMC calculation loop in the C implementation.
//! Reference: mVMC/src/mVMC/vmccal.c
//!
//! The Metropolis algorithm is used to sample electron configurations
//! according to the probability distribution |ψ|², where ψ is the trial wavefunction.

use crate::{Result, VmcError};
use crate::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
use crate::wavefunction::{
    SlaterDeterminant, PfaffianWavefunction, ProjectionOperator, ProjectionCount,
    make_projection_count,
};
use num_complex::Complex64;

/// Electron configuration for VMC calculations
///
/// This structure represents the current electron configuration in the lattice.
/// It corresponds to the electron arrays in the C implementation.
/// Reference: mVMC/src/mVMC/vmccal.c (eleIdx, eleCfg, eleNum)
///
/// # Structure
///
/// - `ele_idx`: Electron indices (which electrons are present)
/// - `ele_cfg`: Electron configuration (up/down spin for each site)
/// - `ele_num`: Electron numbers (n_up, n_down for each site)
///
/// # Examples
///
/// ```
/// use mvmc_core::monte_carlo::ElectronConfiguration;
/// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let two_sz = TwoSz::new(0);
/// let config = ElectronConfiguration::new(nsite, ne, two_sz);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ElectronConfiguration {
    /// Number of lattice sites
    nsite: usize,

    /// Number of electrons
    ne: usize,

    /// Spin quantum number (2*Sz)
    two_sz: i32,

    /// Electron indices (which electrons are present)
    /// Reference: mVMC/src/mVMC/vmccal.c (eleIdx)
    ele_idx: Vec<usize>,

    /// Electron configuration (up/down spin for each site)
    /// Reference: mVMC/src/mVMC/vmccal.c (eleCfg)
    /// ele_cfg[i] = 0: empty, 1: up, 2: down, 3: both
    ele_cfg: Vec<usize>,

    /// Electron numbers (n_up, n_down for each site)
    /// Reference: mVMC/src/mVMC/vmccal.c (eleNum)
    /// ele_num[i] = n_up[i], ele_num[i + nsite] = n_down[i]
    ele_num: Vec<i32>,
}

/// Metropolis sampling step result
///
/// This structure contains the result of a single Metropolis step.
#[derive(Debug, Clone, PartialEq)]
pub struct MetropolisStep {
    /// Whether the proposed move was accepted
    pub accepted: bool,

    /// The proposed electron configuration
    pub proposed_config: ElectronConfiguration,

    /// The acceptance probability
    pub acceptance_prob: f64,

    /// The wavefunction amplitude ratio
    pub amplitude_ratio: Complex64,
}

/// Metropolis sampler for VMC calculations
///
/// This structure manages the Metropolis sampling process.
/// It corresponds to the main VMC calculation loop in the C implementation.
/// Reference: mVMC/src/mVMC/vmccal.c
///
/// # Examples
///
/// ```
/// use mvmc_core::monte_carlo::MetropolisSampler;
/// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
///
/// let nsite = SiteCount::new(4);
/// let ne = ElectronCount::new(2);
/// let two_sz = TwoSz::new(0);
/// let seed = RandomSeed::new(12345);
/// let sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);
/// ```
#[derive(Debug, Clone)]
pub struct MetropolisSampler {
    /// Number of lattice sites
    nsite: usize,

    /// Number of electrons
    ne: usize,

    /// Spin quantum number (2*Sz)
    two_sz: i32,

    /// Random seed
    seed: u64,

    /// Current electron configuration
    current_config: ElectronConfiguration,

    /// Slater determinant wavefunction
    slater_wfn: SlaterDeterminant,

    /// Pfaffian wavefunction (if used)
    pfaffian_wfn: Option<PfaffianWavefunction>,

    /// Projection operator
    proj_op: ProjectionOperator,

    /// Current projection count
    proj_cnt: ProjectionCount,

    /// Random number generator state
    rng_state: u64,
}

impl ElectronConfiguration {
    /// Creates a new electron configuration
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `two_sz` - Spin quantum number (2*Sz)
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::ElectronConfiguration;
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let two_sz = TwoSz::new(0);
    /// let config = ElectronConfiguration::new(nsite, ne, two_sz);
    /// ```
    pub fn new(nsite: SiteCount, ne: ElectronCount, two_sz: TwoSz) -> Self {
        let nsite_val = nsite.get();
        let ne_val = ne.get();
        let two_sz_val = two_sz.get();

        // Initialize with empty configuration
        let ele_idx = vec![0; ne_val];
        let ele_cfg = vec![0; nsite_val];
        let ele_num = vec![0; 2 * nsite_val]; // n_up, n_down for each site

        Self {
            nsite: nsite_val,
            ne: ne_val,
            two_sz: two_sz_val,
            ele_idx,
            ele_cfg,
            ele_num,
        }
    }

    /// Returns the number of lattice sites
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns the spin quantum number
    pub fn two_sz(&self) -> i32 {
        self.two_sz
    }

    /// Returns the electron configuration array
    pub fn ele_cfg(&self) -> &[usize] {
        &self.ele_cfg
    }

    /// Returns the electron numbers array
    pub fn ele_num(&self) -> &[i32] {
        &self.ele_num
    }

    /// Returns the electron indices array
    pub fn ele_idx(&self) -> &[usize] {
        &self.ele_idx
    }

    /// Returns the number of up-spin electrons at a given site
    pub fn electron_number_up(&self, site: usize) -> usize {
        if site >= self.nsite {
            return 0;
        }
        self.ele_num[site] as usize
    }

    /// Returns the number of down-spin electrons at a given site
    pub fn electron_number_down(&self, site: usize) -> usize {
        if site >= self.nsite {
            return 0;
        }
        self.ele_num[site + self.nsite] as usize
    }

    /// Sets the electron configuration
    ///
    /// # Arguments
    ///
    /// * `ele_cfg` - Electron configuration (up/down spin for each site)
    /// * `ele_num` - Electron numbers (n_up, n_down for each site)
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn set_configuration(&mut self, ele_cfg: &[usize], ele_num: &[i32]) -> Result<()> {
        if ele_cfg.len() != self.nsite {
            return Err(VmcError::dim_mismatch(self.nsite, ele_cfg.len()));
        }
        if ele_num.len() != 2 * self.nsite {
            return Err(VmcError::dim_mismatch(2 * self.nsite, ele_num.len()));
        }

        self.ele_cfg.copy_from_slice(ele_cfg);
        self.ele_num.copy_from_slice(ele_num);

        // Update electron indices
        self.update_electron_indices()?;

        Ok(())
    }

    /// Updates electron indices based on current configuration
    fn update_electron_indices(&mut self) -> Result<()> {
        let mut idx = 0;

        for site in 0..self.nsite {
            let n_up = self.ele_num[site] as usize;
            let n_down = self.ele_num[site + self.nsite] as usize;

            for _ in 0..n_up {
                if idx >= self.ne {
                    return Err(VmcError::invalid_config("Too many electrons"));
                }
                self.ele_idx[idx] = site;
                idx += 1;
            }

            for _ in 0..n_down {
                if idx >= self.ne {
                    return Err(VmcError::invalid_config("Too many electrons"));
                }
                self.ele_idx[idx] = site + self.nsite; // Down spin offset
                idx += 1;
            }
        }

        // Fill remaining indices with zeros
        for i in idx..self.ne {
            self.ele_idx[i] = 0;
        }

        Ok(())
    }

    /// Proposes a random electron hop
    ///
    /// # Arguments
    ///
    /// * `rng_state` - Random number generator state
    ///
    /// # Returns
    ///
    /// A new configuration with a proposed electron hop
    pub fn propose_hop(&self, rng_state: &mut u64) -> Result<Self> {
        let mut new_config = self.clone();

        // If no electrons, return unchanged
        if self.ne == 0 {
            return Ok(new_config);
        }

        // Count total electrons
        let total_electrons: i32 = self.ele_num.iter().sum();
        if total_electrons == 0 {
            return Ok(new_config);
        }

        // Choose a random electron to move
        let electron_idx = (*rng_state % total_electrons as u64) as usize;

        // Find the electron to move
        let mut electron_count = 0;
        let mut source_site = 0;
        let mut is_up_spin = true;

        for site in 0..self.nsite {
            let n_up = self.ele_num[site] as usize;
            if electron_count + n_up > electron_idx {
                source_site = site;
                is_up_spin = true;
                break;
            }
            electron_count += n_up;

            let n_down = self.ele_num[site + self.nsite] as usize;
            if electron_count + n_down > electron_idx {
                source_site = site;
                is_up_spin = false;
                break;
            }
            electron_count += n_down;
        }

        // Choose a random destination site
        let new_site = (*rng_state / total_electrons as u64) % self.nsite as u64;
        let new_site = new_site as usize;

        // Update RNG state
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);

        // Perform the hop
        if is_up_spin {
            // Moving up-spin electron
            if new_config.ele_num[source_site] > 0 {
                new_config.ele_num[source_site] -= 1;
                new_config.ele_num[new_site] += 1;
            }
        } else {
            // Moving down-spin electron
            if new_config.ele_num[source_site + self.nsite] > 0 {
                new_config.ele_num[source_site + self.nsite] -= 1;
                new_config.ele_num[new_site + self.nsite] += 1;
            }
        }

        // Update electron indices
        new_config.update_electron_indices()?;

        Ok(new_config)
    }
}

impl MetropolisSampler {
    /// Creates a new Metropolis sampler
    ///
    /// # Arguments
    ///
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `two_sz` - Spin quantum number (2*Sz)
    /// * `seed` - Random seed
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::MetropolisSampler;
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let two_sz = TwoSz::new(0);
    /// let seed = RandomSeed::new(12345);
    /// let sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);
    /// ```
    pub fn new(nsite: SiteCount, ne: ElectronCount, two_sz: TwoSz, seed: RandomSeed) -> Self {
        let nsite_val = nsite.get();
        let ne_val = ne.get();
        let two_sz_val = two_sz.get();
        let seed_val = seed.get();

        let mut current_config = ElectronConfiguration::new(nsite, ne, two_sz);

        // Initialize with a simple configuration if we have electrons
        if ne_val > 0 {
            let mut ele_cfg = vec![0; nsite_val];
            let mut ele_num = vec![0; 2 * nsite_val];

            // Place electrons in a simple pattern
            let n_up = (ne_val + 1) / 2;
            let n_down = ne_val / 2;

            for i in 0..n_up.min(nsite_val) {
                ele_cfg[i] = 1;
                ele_num[i] = 1;
            }

            for i in 0..n_down.min(nsite_val) {
                if ele_cfg[i] == 1 {
                    ele_cfg[i] = 3; // Both up and down
                } else {
                    ele_cfg[i] = 2;
                }
                ele_num[i + nsite_val] = 1;
            }

            current_config.set_configuration(&ele_cfg, &ele_num).unwrap();
        }

        let slater_wfn = SlaterDeterminant::new(nsite, ne);
        let proj_op = ProjectionOperator::new(nsite, ne);
        let proj_cnt = ProjectionCount::new(proj_op.n_proj());

        Self {
            nsite: nsite_val,
            ne: ne_val,
            two_sz: two_sz_val,
            seed: seed_val,
            current_config,
            slater_wfn,
            pfaffian_wfn: None,
            proj_op,
            proj_cnt,
            rng_state: seed_val,
        }
    }

    /// Returns the current electron configuration
    pub fn current_config(&self) -> &ElectronConfiguration {
        &self.current_config
    }

    /// Returns sampling statistics
    pub fn statistics(&self) -> crate::monte_carlo::SamplingStatistics {
        crate::monte_carlo::SamplingStatistics::default()
    }

    /// Performs a single Metropolis step
    ///
    /// # Returns
    ///
    /// The result of the Metropolis step
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::MetropolisSampler;
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let two_sz = TwoSz::new(0);
    /// let seed = RandomSeed::new(12345);
    /// let mut sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);
    ///
    /// let step = sampler.metropolis_step().unwrap();
    /// ```
    pub fn metropolis_step(&mut self) -> Result<MetropolisStep> {
        // Propose a new configuration
        let proposed_config = self.current_config.propose_hop(&mut self.rng_state)?;

        // Calculate wavefunction amplitude ratio
        let current_amplitude = self.calculate_amplitude(&self.current_config)?;
        let proposed_amplitude = self.calculate_amplitude(&proposed_config)?;

        // Avoid division by zero and ensure finite values
        let amplitude_ratio = if current_amplitude.norm() < 1e-12 {
            Complex64::new(1.0, 0.0) // Default to 1 if current amplitude is too small
        } else {
            proposed_amplitude / current_amplitude
        };

        // Ensure the ratio is finite
        let amplitude_ratio = if amplitude_ratio.is_finite() {
            amplitude_ratio
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Calculate acceptance probability
        let acceptance_prob = (amplitude_ratio.norm().powi(2)).min(1.0);

        // Generate random number for acceptance
        let random_num = self.generate_random();

        // Accept or reject
        let accepted = random_num < acceptance_prob;

        if accepted {
            self.current_config = proposed_config.clone();
            // Update projection count
            make_projection_count(&self.proj_op, &mut self.proj_cnt, &self.current_config.ele_num)?;
        }

        Ok(MetropolisStep {
            accepted,
            proposed_config,
            acceptance_prob,
            amplitude_ratio,
        })
    }

    /// Calculates the wavefunction amplitude for a given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The wavefunction amplitude
    fn calculate_amplitude(&self, config: &ElectronConfiguration) -> Result<Complex64> {
        // Calculate Slater determinant amplitude
        let slater_amplitude = self.slater_wfn.amplitude();

        // Calculate projection amplitude
        let proj_amplitude = self.calculate_projection_amplitude(config)?;

        // Total amplitude is the product
        Ok(slater_amplitude * proj_amplitude)
    }

    /// Calculates the projection amplitude for a given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The projection amplitude
    fn calculate_projection_amplitude(&self, config: &ElectronConfiguration) -> Result<Complex64> {
        // Create temporary projection count
        let mut temp_proj_cnt = ProjectionCount::new(self.proj_op.n_proj());
        make_projection_count(&self.proj_op, &mut temp_proj_cnt, &config.ele_num)?;

        // Calculate projection value
        let mut proj_value = Complex64::new(0.0, 0.0);
        for i in 0..self.proj_op.n_proj() {
            if let (Ok(param), Ok(count)) = (self.proj_op.get_parameter(i), temp_proj_cnt.get(i)) {
                proj_value += param * count as f64;
            }
        }

        Ok(proj_value.exp())
    }

    /// Generates a random number between 0 and 1
    fn generate_random(&mut self) -> f64 {
        // Simple linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Runs Metropolis sampling for a specified number of steps
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Number of Metropolis steps
    ///
    /// # Returns
    ///
    /// Vector of Metropolis step results
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::MetropolisSampler;
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let two_sz = TwoSz::new(0);
    /// let seed = RandomSeed::new(12345);
    /// let mut sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);
    ///
    /// let steps = sampler.run_sampling(1000).unwrap();
    /// assert_eq!(steps.len(), 1000);
    /// ```
    pub fn run_sampling(&mut self, n_steps: usize) -> Result<Vec<MetropolisStep>> {
        let mut steps = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let step = self.metropolis_step()?;
            steps.push(step);
        }

        Ok(steps)
    }

    /// Returns the acceptance rate from a sequence of steps
    ///
    /// # Arguments
    ///
    /// * `steps` - Sequence of Metropolis steps
    ///
    /// # Returns
    ///
    /// The acceptance rate (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::monte_carlo::MetropolisSampler;
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
    ///
    /// let nsite = SiteCount::new(4);
    /// let ne = ElectronCount::new(2);
    /// let two_sz = TwoSz::new(0);
    /// let seed = RandomSeed::new(12345);
    /// let mut sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);
    ///
    /// let steps = sampler.run_sampling(1000).unwrap();
    /// let acceptance_rate = sampler.acceptance_rate(&steps);
    /// println!("Acceptance rate: {:.2}%", acceptance_rate * 100.0);
    /// ```
    pub fn acceptance_rate(&self, steps: &[MetropolisStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }

        let accepted_count = steps.iter().filter(|step| step.accepted).count();
        accepted_count as f64 / steps.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};

    #[test]
    fn test_electron_configuration_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let two_sz = TwoSz::new(0);
        let config = ElectronConfiguration::new(nsite, ne, two_sz);

        assert_eq!(config.nsite(), 4);
        assert_eq!(config.ne(), 2);
        assert_eq!(config.two_sz(), 0);
    }

    #[test]
    fn test_electron_configuration_set_config() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let two_sz = TwoSz::new(1);
        let mut config = ElectronConfiguration::new(nsite, ne, two_sz);

        // Set configuration: one up-spin electron on site 0
        let ele_cfg = vec![1, 0]; // up on site 0, empty on site 1
        let ele_num = vec![1, 0, 0, 0]; // n_up[0]=1, n_up[1]=0, n_down[0]=0, n_down[1]=0

        config.set_configuration(&ele_cfg, &ele_num).unwrap();

        assert_eq!(config.ele_cfg(), &[1, 0]);
        assert_eq!(config.ele_num(), &[1, 0, 0, 0]);
    }

    #[test]
    fn test_electron_configuration_invalid_config() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);
        let two_sz = TwoSz::new(1);
        let mut config = ElectronConfiguration::new(nsite, ne, two_sz);

        // Wrong size configuration
        let ele_cfg = vec![1]; // Should be size 2
        let ele_num = vec![1, 0, 0, 0];

        let result = config.set_configuration(&ele_cfg, &ele_num);
        assert!(result.is_err());
    }

    #[test]
    fn test_metropolis_sampler_creation() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let two_sz = TwoSz::new(0);
        let seed = RandomSeed::new(12345);
        let sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);

        assert_eq!(sampler.current_config().nsite(), 4);
        assert_eq!(sampler.current_config().ne(), 2);
    }

    #[test]
    fn test_metropolis_step() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let two_sz = TwoSz::new(0);
        let seed = RandomSeed::new(12345);
        let mut sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);

        let step = sampler.metropolis_step().unwrap();

        // Step should be valid (either accepted or rejected)
        assert!(step.acceptance_prob >= 0.0);
        assert!(step.acceptance_prob <= 1.0);
    }

    #[test]
    fn test_metropolis_sampling() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let two_sz = TwoSz::new(0);
        let seed = RandomSeed::new(12345);
        let mut sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);

        let steps = sampler.run_sampling(100).unwrap();

        assert_eq!(steps.len(), 100);

        // Check that all steps are valid
        for step in &steps {
            assert!(step.acceptance_prob >= 0.0);
            assert!(step.acceptance_prob <= 1.0);
        }
    }

    #[test]
    fn test_acceptance_rate() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let two_sz = TwoSz::new(0);
        let seed = RandomSeed::new(12345);
        let mut sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);

        let steps = sampler.run_sampling(1000).unwrap();
        let acceptance_rate = sampler.acceptance_rate(&steps);

        assert!(acceptance_rate >= 0.0);
        assert!(acceptance_rate <= 1.0);
    }

    #[test]
    fn test_empty_steps_acceptance_rate() {
        let nsite = SiteCount::new(4);
        let ne = ElectronCount::new(2);
        let two_sz = TwoSz::new(0);
        let seed = RandomSeed::new(12345);
        let sampler = MetropolisSampler::new(nsite, ne, two_sz, seed);

        let steps = vec![];
        let acceptance_rate = sampler.acceptance_rate(&steps);

        assert_eq!(acceptance_rate, 0.0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount, TwoSz, RandomSeed};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_metropolis_step_valid(
            nsite in 1usize..10,
            ne in 0usize..5,
            two_sz in -2i32..3,
            seed in 0u64..10000
        ) {
            if ne <= nsite * 2 { // Reasonable electron count
                let nsite_val = SiteCount::new(nsite);
                let ne_val = ElectronCount::new(ne);
                let two_sz_val = TwoSz::new(two_sz);
                let seed_val = RandomSeed::new(seed);
                let mut sampler = MetropolisSampler::new(nsite_val, ne_val, two_sz_val, seed_val);

                let step = sampler.metropolis_step().unwrap();

                // Acceptance probability should be valid
                prop_assert!(step.acceptance_prob >= 0.0);
                prop_assert!(step.acceptance_prob <= 1.0);

                // Amplitude ratio should be finite
                prop_assert!(step.amplitude_ratio.is_finite());
            }
        }

        #[test]
        fn prop_acceptance_rate_bounds(
            nsite in 1usize..10,
            ne in 0usize..5,
            two_sz in -2i32..3,
            seed in 0u64..10000,
            n_steps in 1usize..100
        ) {
            if ne <= nsite * 2 {
                let nsite_val = SiteCount::new(nsite);
                let ne_val = ElectronCount::new(ne);
                let two_sz_val = TwoSz::new(two_sz);
                let seed_val = RandomSeed::new(seed);
                let mut sampler = MetropolisSampler::new(nsite_val, ne_val, two_sz_val, seed_val);

                let steps = sampler.run_sampling(n_steps).unwrap();
                let acceptance_rate = sampler.acceptance_rate(&steps);

                prop_assert!(acceptance_rate >= 0.0);
                prop_assert!(acceptance_rate <= 1.0);
            }
        }
    }
}
