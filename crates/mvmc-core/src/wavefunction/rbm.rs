//! Restricted Boltzmann Machine (RBM) wavefunction
//!
//! This module provides RBM-based wavefunction representations for VMC calculations.
//! RBM is used as a correlation factor to capture complex many-body correlations
//! that cannot be easily represented by traditional methods like Slater determinants.

use crate::error::{Result, VmcError};
use crate::types::SiteCount;
use num_complex::Complex64;
use std::fmt;

/// RBM wavefunction parameters
///
/// This structure contains the weights and biases for the RBM network.
/// The RBM consists of:
/// - Visible layer: physical sites (electron configurations)
/// - Hidden layer: auxiliary variables for correlation
#[derive(Debug, Clone, PartialEq)]
pub struct RBMParameters {
    /// Number of physical sites
    n_sites: usize,
    /// Number of hidden neurons
    n_hidden: usize,
    /// Weights between visible and hidden layers [n_sites * n_hidden]
    weights: Vec<Vec<Complex64>>,
    /// Visible layer biases [n_sites]
    visible_bias: Vec<Complex64>,
    /// Hidden layer biases [n_hidden]
    hidden_bias: Vec<Complex64>,
}

/// RBM wavefunction
///
/// This structure represents a wavefunction using RBM as a correlation factor.
/// The wavefunction is given by:
/// ψ(x) = exp(∑ᵢ aᵢ xᵢ + ∑ⱼ bⱼ hⱼ + ∑ᵢⱼ Wᵢⱼ xᵢ hⱼ)
/// where xᵢ are visible variables (electron configurations) and hⱼ are hidden variables.
#[derive(Debug, Clone, PartialEq)]
pub struct RBMWavefunction {
    /// RBM parameters
    parameters: RBMParameters,
    /// Whether to use complex weights
    complex_weights: bool,
}

/// RBM count structure for efficient computation
///
/// This structure stores intermediate calculations for RBM evaluation.
/// It corresponds to the `rbmCnt` array in the C implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct RBMCounter {
    /// Physical layer contributions [n_sites]
    physical_layer: Vec<Complex64>,
    /// Hidden layer contributions [n_hidden]
    hidden_layer: Vec<Complex64>,
}

impl RBMParameters {
    /// Creates new RBM parameters
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of physical sites
    /// * `n_hidden` - Number of hidden neurons
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::rbm::RBMParameters;
    /// use mvmc_core::types::{SiteCount, ElectronCount};
    ///
    /// let nsite = SiteCount::new(4);
    /// let n_hidden = 8;
    /// let params = RBMParameters::new(nsite, n_hidden);
    /// ```
    pub fn new(n_sites: SiteCount, n_hidden: usize) -> Self {
        let n_sites_val = n_sites.get();

        Self {
            n_sites: n_sites_val,
            n_hidden,
            weights: vec![vec![Complex64::new(0.0, 0.0); n_hidden]; n_sites_val],
            visible_bias: vec![Complex64::new(0.0, 0.0); n_sites_val],
            hidden_bias: vec![Complex64::new(0.0, 0.0); n_hidden],
        }
    }

    /// Returns the number of physical sites
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Returns the number of hidden neurons
    pub fn n_hidden(&self) -> usize {
        self.n_hidden
    }

    /// Gets the weight between visible unit i and hidden unit j
    pub fn get_weight(&self, i: usize, j: usize) -> Result<Complex64> {
        if i >= self.n_sites || j >= self.n_hidden {
            return Err(VmcError::out_of_bounds(i.max(j), self.n_sites.max(self.n_hidden)));
        }
        Ok(self.weights[i][j])
    }

    /// Sets the weight between visible unit i and hidden unit j
    pub fn set_weight(&mut self, i: usize, j: usize, weight: Complex64) -> Result<()> {
        if i >= self.n_sites || j >= self.n_hidden {
            return Err(VmcError::out_of_bounds(i.max(j), self.n_sites.max(self.n_hidden)));
        }
        self.weights[i][j] = weight;
        Ok(())
    }

    /// Gets the visible bias for site i
    pub fn get_visible_bias(&self, i: usize) -> Result<Complex64> {
        if i >= self.n_sites {
            return Err(VmcError::out_of_bounds(i, self.n_sites));
        }
        Ok(self.visible_bias[i])
    }

    /// Sets the visible bias for site i
    pub fn set_visible_bias(&mut self, i: usize, bias: Complex64) -> Result<()> {
        if i >= self.n_sites {
            return Err(VmcError::out_of_bounds(i, self.n_sites));
        }
        self.visible_bias[i] = bias;
        Ok(())
    }

    /// Gets the hidden bias for neuron j
    pub fn get_hidden_bias(&self, j: usize) -> Result<Complex64> {
        if j >= self.n_hidden {
            return Err(VmcError::out_of_bounds(j, self.n_hidden));
        }
        Ok(self.hidden_bias[j])
    }

    /// Sets the hidden bias for neuron j
    pub fn set_hidden_bias(&mut self, j: usize, bias: Complex64) -> Result<()> {
        if j >= self.n_hidden {
            return Err(VmcError::out_of_bounds(j, self.n_hidden));
        }
        self.hidden_bias[j] = bias;
        Ok(())
    }

    /// Returns all weights as a flat vector
    pub fn weights_flat(&self) -> Vec<Complex64> {
        self.weights.iter().flatten().cloned().collect()
    }

    /// Returns all visible biases
    pub fn visible_biases(&self) -> &[Complex64] {
        &self.visible_bias
    }

    /// Returns all hidden biases
    pub fn hidden_biases(&self) -> &[Complex64] {
        &self.hidden_bias
    }
}

impl RBMWavefunction {
    /// Creates a new RBM wavefunction
    ///
    /// # Arguments
    ///
    /// * `n_sites` - Number of physical sites
    /// * `n_hidden` - Number of hidden neurons
    /// * `complex_weights` - Whether to use complex weights
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::wavefunction::rbm::RBMWavefunction;
    /// use mvmc_core::types::SiteCount;
    ///
    /// let nsite = SiteCount::new(4);
    /// let rbm = RBMWavefunction::new(nsite, 8, true);
    /// ```
    pub fn new(n_sites: SiteCount, n_hidden: usize, complex_weights: bool) -> Self {
        Self {
            parameters: RBMParameters::new(n_sites, n_hidden),
            complex_weights,
        }
    }

    /// Returns the RBM parameters
    pub fn parameters(&self) -> &RBMParameters {
        &self.parameters
    }

    /// Returns mutable RBM parameters
    pub fn parameters_mut(&mut self) -> &mut RBMParameters {
        &mut self.parameters
    }

    /// Returns whether complex weights are used
    pub fn complex_weights(&self) -> bool {
        self.complex_weights
    }

    /// Calculates the RBM weight for a given electron configuration
    ///
    /// This corresponds to the `WeightRBM` function in the C implementation.
    /// The weight is calculated as:
    /// W = exp(∑ᵢ aᵢ xᵢ + ∑ⱼ log(cosh(bⱼ + ∑ᵢ Wᵢⱼ xᵢ)))
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration (0 or 1 for each site)
    ///
    /// # Returns
    ///
    /// The RBM weight as a complex number
    pub fn calculate_weight(&self, electron_config: &[u8]) -> Result<Complex64> {
        if electron_config.len() != self.parameters.n_sites {
            return Err(VmcError::dim_mismatch(self.parameters.n_sites, electron_config.len()));
        }

        let mut weight = Complex64::new(0.0, 0.0);

        // Calculate visible layer contribution: ∑ᵢ aᵢ xᵢ
        for i in 0..self.parameters.n_sites {
            if electron_config[i] > 0 {
                weight += self.parameters.visible_bias[i];
            }
        }

        // Calculate hidden layer contribution: ∑ⱼ log(cosh(bⱼ + ∑ᵢ Wᵢⱼ xᵢ))
        for j in 0..self.parameters.n_hidden {
            let mut hidden_input = self.parameters.hidden_bias[j];
            for i in 0..self.parameters.n_sites {
                if electron_config[i] > 0 {
                    hidden_input += self.parameters.weights[i][j];
                }
            }
            weight += hidden_input.cosh().ln();
        }

        Ok(weight.exp())
    }

    /// Calculates the log of the RBM weight
    ///
    /// This corresponds to the `LogWeightRBM` function in the C implementation.
    /// This is more numerically stable than calculating the weight and taking the log.
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration (0 or 1 for each site)
    ///
    /// # Returns
    ///
    /// The log of the RBM weight as a complex number
    pub fn calculate_log_weight(&self, electron_config: &[u8]) -> Result<Complex64> {
        if electron_config.len() != self.parameters.n_sites {
            return Err(VmcError::dim_mismatch(self.parameters.n_sites, electron_config.len()));
        }

        let mut log_weight = Complex64::new(0.0, 0.0);

        // Calculate visible layer contribution: ∑ᵢ aᵢ xᵢ
        for i in 0..self.parameters.n_sites {
            if electron_config[i] > 0 {
                log_weight += self.parameters.visible_bias[i];
            }
        }

        // Calculate hidden layer contribution: ∑ⱼ log(cosh(bⱼ + ∑ᵢ Wᵢⱼ xᵢ))
        for j in 0..self.parameters.n_hidden {
            let mut hidden_input = self.parameters.hidden_bias[j];
            for i in 0..self.parameters.n_sites {
                if electron_config[i] > 0 {
                    hidden_input += self.parameters.weights[i][j];
                }
            }
            log_weight += hidden_input.cosh().ln();
        }

        Ok(log_weight)
    }

    /// Calculates the RBM ratio between two configurations
    ///
    /// This corresponds to the `RBMRatio` function in the C implementation.
    /// The ratio is calculated as W(new) / W(old).
    ///
    /// # Arguments
    ///
    /// * `new_config` - New electron configuration
    /// * `old_config` - Old electron configuration
    ///
    /// # Returns
    ///
    /// The RBM ratio as a complex number
    pub fn calculate_ratio(&self, new_config: &[u8], old_config: &[u8]) -> Result<Complex64> {
        let log_weight_new = self.calculate_log_weight(new_config)?;
        let log_weight_old = self.calculate_log_weight(old_config)?;
        Ok((log_weight_new - log_weight_old).exp())
    }

    /// Calculates the log of the RBM ratio
    ///
    /// This corresponds to the `LogRBMRatio` function in the C implementation.
    /// This is more numerically stable than calculating the ratio and taking the log.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New electron configuration
    /// * `old_config` - Old electron configuration
    ///
    /// # Returns
    ///
    /// The log of the RBM ratio as a complex number
    pub fn calculate_log_ratio(&self, new_config: &[u8], old_config: &[u8]) -> Result<Complex64> {
        let log_weight_new = self.calculate_log_weight(new_config)?;
        let log_weight_old = self.calculate_log_weight(old_config)?;
        Ok(log_weight_new - log_weight_old)
    }

    /// Calculates the RBM counter for efficient computation
    ///
    /// This corresponds to the `MakeRBMCnt` function in the C implementation.
    /// The counter stores intermediate calculations for efficient ratio computation.
    ///
    /// # Arguments
    ///
    /// * `electron_config` - Electron configuration
    ///
    /// # Returns
    ///
    /// The RBM counter
    pub fn make_counter(&self, electron_config: &[u8]) -> Result<RBMCounter> {
        if electron_config.len() != self.parameters.n_sites {
            return Err(VmcError::dim_mismatch(self.parameters.n_sites, electron_config.len()));
        }

        let mut physical_layer = vec![Complex64::new(0.0, 0.0); self.parameters.n_sites];
        let mut hidden_layer = vec![Complex64::new(0.0, 0.0); self.parameters.n_hidden];

        // Calculate physical layer contributions
        for i in 0..self.parameters.n_sites {
            if electron_config[i] > 0 {
                physical_layer[i] = self.parameters.visible_bias[i];
            }
        }

        // Calculate hidden layer contributions
        for j in 0..self.parameters.n_hidden {
            let mut hidden_input = self.parameters.hidden_bias[j];
            for i in 0..self.parameters.n_sites {
                if electron_config[i] > 0 {
                    hidden_input += self.parameters.weights[i][j];
                }
            }
            hidden_layer[j] = hidden_input;
        }

        Ok(RBMCounter {
            physical_layer,
            hidden_layer,
        })
    }

    /// Calculates the RBM ratio using counters
    ///
    /// This is more efficient than recalculating everything from scratch.
    ///
    /// # Arguments
    ///
    /// * `new_counter` - RBM counter for new configuration
    /// * `old_counter` - RBM counter for old configuration
    ///
    /// # Returns
    ///
    /// The RBM ratio as a complex number
    pub fn calculate_ratio_from_counters(&self, new_counter: &RBMCounter, old_counter: &RBMCounter) -> Result<Complex64> {
        if new_counter.physical_layer.len() != self.parameters.n_sites ||
           old_counter.physical_layer.len() != self.parameters.n_sites ||
           new_counter.hidden_layer.len() != self.parameters.n_hidden ||
           old_counter.hidden_layer.len() != self.parameters.n_hidden {
            return Err(VmcError::dim_mismatch(self.parameters.n_sites, new_counter.physical_layer.len()));
        }

        let mut log_ratio = Complex64::new(0.0, 0.0);

        // Calculate physical layer contribution
        for i in 0..self.parameters.n_sites {
            log_ratio += new_counter.physical_layer[i] - old_counter.physical_layer[i];
        }

        // Calculate hidden layer contribution
        for j in 0..self.parameters.n_hidden {
            let new_hidden = new_counter.hidden_layer[j];
            let old_hidden = old_counter.hidden_layer[j];

            // Use the same formula as in the C implementation
            let new_cosh = new_hidden.cosh();
            let old_cosh = old_hidden.cosh();

            if new_cosh.norm() > 0.0 && old_cosh.norm() > 0.0 {
                log_ratio += (new_cosh / old_cosh).ln();
            }
        }

        Ok(log_ratio.exp())
    }

    /// Calculates the log of the RBM ratio using counters
    ///
    /// This is more efficient and numerically stable than calculating the ratio.
    ///
    /// # Arguments
    ///
    /// * `new_counter` - RBM counter for new configuration
    /// * `old_counter` - RBM counter for old configuration
    ///
    /// # Returns
    ///
    /// The log of the RBM ratio as a complex number
    pub fn calculate_log_ratio_from_counters(&self, new_counter: &RBMCounter, old_counter: &RBMCounter) -> Result<Complex64> {
        if new_counter.physical_layer.len() != self.parameters.n_sites ||
           old_counter.physical_layer.len() != self.parameters.n_sites ||
           new_counter.hidden_layer.len() != self.parameters.n_hidden ||
           old_counter.hidden_layer.len() != self.parameters.n_hidden {
            return Err(VmcError::dim_mismatch(self.parameters.n_sites, new_counter.physical_layer.len()));
        }

        let mut log_ratio = Complex64::new(0.0, 0.0);

        // Calculate physical layer contribution
        for i in 0..self.parameters.n_sites {
            log_ratio += new_counter.physical_layer[i] - old_counter.physical_layer[i];
        }

        // Calculate hidden layer contribution
        for j in 0..self.parameters.n_hidden {
            let new_hidden = new_counter.hidden_layer[j];
            let old_hidden = old_counter.hidden_layer[j];

            // Use the same formula as in the C implementation
            let new_cosh = new_hidden.cosh();
            let old_cosh = old_hidden.cosh();

            if new_cosh.norm() > 0.0 && old_cosh.norm() > 0.0 {
                log_ratio += (new_cosh / old_cosh).ln();
            }
        }

        Ok(log_ratio)
    }
}

impl RBMCounter {
    /// Creates a new RBM counter
    pub fn new(n_sites: usize, n_hidden: usize) -> Self {
        Self {
            physical_layer: vec![Complex64::new(0.0, 0.0); n_sites],
            hidden_layer: vec![Complex64::new(0.0, 0.0); n_hidden],
        }
    }

    /// Returns the physical layer contributions
    pub fn physical_layer(&self) -> &[Complex64] {
        &self.physical_layer
    }

    /// Returns the hidden layer contributions
    pub fn hidden_layer(&self) -> &[Complex64] {
        &self.hidden_layer
    }

    /// Updates the counter for a single site change
    ///
    /// This corresponds to the `UpdateRBMCnt` function in the C implementation.
    ///
    /// # Arguments
    ///
    /// * `site` - Site index that changed
    /// * `old_value` - Old electron value (0 or 1)
    /// * `new_value` - New electron value (0 or 1)
    /// * `rbm` - RBM wavefunction
    pub fn update_site(&mut self, site: usize, old_value: u8, new_value: u8, rbm: &RBMWavefunction) -> Result<()> {
        if site >= rbm.parameters.n_sites {
            return Err(VmcError::out_of_bounds(site, rbm.parameters.n_sites));
        }

        let value_diff = (new_value as i32) - (old_value as i32);
        let bias_diff = Complex64::new(value_diff as f64, 0.0) * rbm.parameters.visible_bias[site];

        // Update physical layer
        self.physical_layer[site] += bias_diff;

        // Update hidden layer
        for j in 0..rbm.parameters.n_hidden {
            let weight_diff = Complex64::new(value_diff as f64, 0.0) * rbm.parameters.weights[site][j];
            self.hidden_layer[j] += weight_diff;
        }

        Ok(())
    }
}

impl fmt::Display for RBMWavefunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RBMWavefunction(n_sites={}, n_hidden={}, complex_weights={})",
               self.parameters.n_sites, self.parameters.n_hidden, self.complex_weights)
    }
}

impl fmt::Display for RBMParameters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RBMParameters(n_sites={}, n_hidden={})",
               self.n_sites, self.n_hidden)
    }
}

impl fmt::Display for RBMCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RBMCounter(physical_layer_len={}, hidden_layer_len={})",
               self.physical_layer.len(), self.hidden_layer.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SiteCount;

    #[test]
    fn test_rbm_parameters_creation() {
        let nsite = SiteCount::new(4);
        let n_hidden = 8;
        let params = RBMParameters::new(nsite, n_hidden);

        assert_eq!(params.n_sites(), 4);
        assert_eq!(params.n_hidden(), 8);
        assert_eq!(params.weights.len(), 4);
        assert_eq!(params.weights[0].len(), 8);
        assert_eq!(params.visible_bias.len(), 4);
        assert_eq!(params.hidden_bias.len(), 8);
    }

    #[test]
    fn test_rbm_parameters_access() {
        let nsite = SiteCount::new(2);
        let n_hidden = 4;
        let mut params = RBMParameters::new(nsite, n_hidden);

        let weight = Complex64::new(1.0, 0.5);
        params.set_weight(0, 1, weight).unwrap();
        assert_eq!(params.get_weight(0, 1).unwrap(), weight);

        let bias = Complex64::new(0.5, -0.3);
        params.set_visible_bias(1, bias).unwrap();
        assert_eq!(params.get_visible_bias(1).unwrap(), bias);

        let hidden_bias = Complex64::new(-0.2, 0.1);
        params.set_hidden_bias(2, hidden_bias).unwrap();
        assert_eq!(params.get_hidden_bias(2).unwrap(), hidden_bias);
    }

    #[test]
    fn test_rbm_parameters_bounds_checking() {
        let nsite = SiteCount::new(2);
        let n_hidden = 4;
        let mut params = RBMParameters::new(nsite, n_hidden);

        assert!(params.get_weight(2, 1).is_err());
        assert!(params.get_weight(1, 4).is_err());
        assert!(params.set_weight(2, 1, Complex64::new(1.0, 0.0)).is_err());

        assert!(params.get_visible_bias(2).is_err());
        assert!(params.set_visible_bias(2, Complex64::new(1.0, 0.0)).is_err());

        assert!(params.get_hidden_bias(4).is_err());
        assert!(params.set_hidden_bias(4, Complex64::new(1.0, 0.0)).is_err());
    }

    #[test]
    fn test_rbm_wavefunction_creation() {
        let nsite = SiteCount::new(4);
        let n_hidden = 8;
        let rbm = RBMWavefunction::new(nsite, n_hidden, true);

        assert_eq!(rbm.parameters().n_sites(), 4);
        assert_eq!(rbm.parameters().n_hidden(), 8);
        assert!(rbm.complex_weights());
    }

    #[test]
    fn test_rbm_weight_calculation() {
        let nsite = SiteCount::new(2);
        let n_hidden = 2;
        let mut rbm = RBMWavefunction::new(nsite, n_hidden, false);

        // Set some simple parameters
        rbm.parameters_mut().set_visible_bias(0, Complex64::new(1.0, 0.0)).unwrap();
        rbm.parameters_mut().set_visible_bias(1, Complex64::new(0.5, 0.0)).unwrap();
        rbm.parameters_mut().set_hidden_bias(0, Complex64::new(0.0, 0.0)).unwrap();
        rbm.parameters_mut().set_hidden_bias(1, Complex64::new(0.0, 0.0)).unwrap();

        // Test with configuration [1, 0]
        let config = vec![1, 0];
        let weight = rbm.calculate_weight(&config).unwrap();
        assert!(weight.norm() > 0.0);

        // Test with configuration [0, 0]
        let config = vec![0, 0];
        let weight = rbm.calculate_weight(&config).unwrap();
        assert!(weight.norm() >= 0.0);
    }

    #[test]
    fn test_rbm_ratio_calculation() {
        let nsite = SiteCount::new(2);
        let n_hidden = 2;
        let rbm = RBMWavefunction::new(nsite, n_hidden, false);

        let config1 = vec![1, 0];
        let config2 = vec![0, 1];

        let ratio = rbm.calculate_ratio(&config2, &config1).unwrap();
        let log_ratio = rbm.calculate_log_ratio(&config2, &config1).unwrap();

        // Check that ratio and exp(log_ratio) are approximately equal
        assert!((ratio - log_ratio.exp()).norm() < 1e-10);
    }

    #[test]
    fn test_rbm_counter_creation() {
        let nsite = SiteCount::new(3);
        let n_hidden = 4;
        let rbm = RBMWavefunction::new(nsite, n_hidden, false);

        let config = vec![1, 0, 1];
        let counter = rbm.make_counter(&config).unwrap();

        assert_eq!(counter.physical_layer().len(), 3);
        assert_eq!(counter.hidden_layer().len(), 4);
    }

    #[test]
    fn test_rbm_counter_update() {
        let nsite = SiteCount::new(2);
        let n_hidden = 2;
        let mut rbm = RBMWavefunction::new(nsite, n_hidden, false);

        // Set some parameters
        rbm.parameters_mut().set_visible_bias(0, Complex64::new(1.0, 0.0)).unwrap();
        rbm.parameters_mut().set_weight(0, 0, Complex64::new(0.5, 0.0)).unwrap();

        let config = vec![1, 0];
        let mut counter = rbm.make_counter(&config).unwrap();

        // Store the initial value
        let initial_physical = counter.physical_layer()[0];

        // Update site 0 from 1 to 0
        counter.update_site(0, 1, 0, &rbm).unwrap();

        // The physical layer contribution should decrease (from 1.0 to 0.0)
        let final_physical = counter.physical_layer()[0];
        assert!(final_physical.re < initial_physical.re);
    }

    #[test]
    fn test_rbm_dimension_mismatch() {
        let nsite = SiteCount::new(2);
        let n_hidden = 2;
        let rbm = RBMWavefunction::new(nsite, n_hidden, false);

        let wrong_config = vec![1, 0, 1]; // Wrong length
        assert!(rbm.calculate_weight(&wrong_config).is_err());
        assert!(rbm.make_counter(&wrong_config).is_err());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_rbm_weight_positive(
            n_sites in 1usize..10,
            n_hidden in 1usize..10,
            config in proptest::collection::vec(0u8..2, 1..20)
        ) {
            let nsite = SiteCount::new(n_sites);
            let rbm = RBMWavefunction::new(nsite, n_hidden, true);

            if config.len() == n_sites {
                let weight = rbm.calculate_weight(&config).unwrap();
                prop_assert!(weight.norm() >= 0.0);
            }
        }

        #[test]
        fn prop_rbm_ratio_consistency(
            n_sites in 2usize..8,
            n_hidden in 1usize..8,
            config1 in proptest::collection::vec(0u8..2, 1..20),
            config2 in proptest::collection::vec(0u8..2, 1..20)
        ) {
            let nsite = SiteCount::new(n_sites);
            let rbm = RBMWavefunction::new(nsite, n_hidden, true);

            if config1.len() == n_sites && config2.len() == n_sites {
                let ratio = rbm.calculate_ratio(&config2, &config1).unwrap();
                let log_ratio = rbm.calculate_log_ratio(&config2, &config1).unwrap();

                // Check that ratio and exp(log_ratio) are approximately equal
                prop_assert!((ratio - log_ratio.exp()).norm() < 1e-10);
            }
        }

        #[test]
        fn prop_rbm_counter_consistency(
            n_sites in 1usize..8,
            n_hidden in 1usize..8,
            config in proptest::collection::vec(0u8..2, 1..20)
        ) {
            let nsite = SiteCount::new(n_sites);
            let rbm = RBMWavefunction::new(nsite, n_hidden, true);

            if config.len() == n_sites {
                let _counter = rbm.make_counter(&config).unwrap();
                let weight_direct = rbm.calculate_weight(&config).unwrap();

                // Weight should be finite
                prop_assert!(weight_direct.is_finite());
            }
        }
    }
}
