//! VMC calculation engine
//!
//! This module provides the main engine for VMC calculations, integrating:
//! - Wavefunction (Slater + Pfaffian + Projection + RBM)
//! - Monte Carlo sampling (Metropolis algorithm)
//! - Optimization algorithms (SR method)
//! - Observable calculation

use crate::config::VmcParameters;
use crate::error::Result;
use crate::monte_carlo::{
    MetropolisSampler, ObservableCalculator, PhysicalObservables, SamplingStatistics,
};
use crate::optimization::SROptimizer;
use crate::types::{CalcMode, ElectronCount, RandomSeed, SiteCount};
use crate::wavefunction::CombinedWavefunction;
use mvmc_physics::wavefunction::CombinedWavefunction as PhysicsCombinedWavefunction;
use mvmc_physics::hamiltonian::Hamiltonian;
use mvmc_physics::hamiltonian::Spin;
use num_complex::Complex64;
use std::fmt;

/// Result of a single VMC iteration
#[derive(Debug, Clone)]
pub struct VmcIterationResult {
    /// Energy value
    pub energy: Complex64,
    /// Energy variance
    pub variance: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Physical observables
    pub observables: PhysicalObservables,
}

/// VMC calculation engine
///
/// This is the main engine that orchestrates the entire VMC calculation process.
///
/// # Components
///
/// - **Wavefunction**: Combined wavefunction (Slater + Pfaffian + Projection + RBM)
/// - **Sampler**: Metropolis Monte Carlo sampler
/// - **Optimizer**: SR (Stochastic Reconfiguration) optimizer
/// - **Calculator**: Observable calculator
/// - **Hamiltonian**: Physical Hamiltonian for energy calculations
///
/// # References
///
/// - C implementation: `mVMC/src/mVMC/vmcmain.c:VMCParaOpt`, `vmccal.c`
#[derive(Debug)]
pub struct VmcEngine {
    /// VMC parameters
    params: VmcParameters,
    /// Combined wavefunction
    wavefunction: PhysicsCombinedWavefunction,
    /// Metropolis sampler
    sampler: MetropolisSampler,
    /// SR optimizer (for optimization mode)
    optimizer: Option<SROptimizer>,
    /// Observable calculator
    calculator: ObservableCalculator,
    /// Physical Hamiltonian
    hamiltonian: Box<dyn Hamiltonian>,
    /// Current optimization iteration
    current_iteration: usize,
}

/// Result of a VMC calculation
#[derive(Debug, Clone)]
pub struct VmcResult {
    /// Calculation mode
    pub mode: CalcMode,
    /// Energy and error
    pub energy: Complex64,
    pub energy_error: f64,
    /// Physical observables
    pub observables: PhysicalObservables,
    /// Sampling statistics
    pub statistics: SamplingStatistics,
    /// Optimization result (if in optimization mode)
    pub optimization: Option<OptimizationResult>,
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Number of iterations performed
    pub iterations: usize,
    /// Energy history
    pub energy_history: Vec<Complex64>,
    /// Convergence achieved
    pub converged: bool,
    /// Final parameter updates
    pub parameter_updates: Vec<f64>,
}

/// Result of expectation value calculation
#[derive(Debug, Clone)]
pub struct ExpectationResult {
    /// Number of samples
    pub num_samples: usize,
    /// Energy and error
    pub energy: Complex64,
    pub energy_error: f64,
    /// Physical observables
    pub observables: PhysicalObservables,
}

impl VmcEngine {
    /// Creates a new VMC engine
    ///
    /// # Arguments
    ///
    /// * `params` - VMC parameters
    /// * `wavefunction` - Combined wavefunction
    /// * `hamiltonian` - Physical Hamiltonian
    ///
    /// # Examples
    ///
    /// ```
    /// use mvmc_core::vmc::VmcEngine;
    /// use mvmc_core::config::{VmcParameters, SRParameters, MonteCarloParameters};
    /// use mvmc_core::wavefunction::CombinedWavefunction;
    /// use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, CalcMode, LanczosMode, RandomSeed};
    /// use mvmc_physics::hamiltonian::{HubbardHamiltonian, Hamiltonian};
    /// use mvmc_physics::lattice::ChainLattice;
    ///
    /// let sr_params = SRParameters::new(1000, 100, 100, 1e-6, 1e-6, 0.1, 100, 1e-6);
    /// let mc_params = MonteCarloParameters::new(100, 1, 1000, false, 1);
    /// let params = VmcParameters::new(
    ///     SiteCount::new(4),
    ///     ElectronCount::new(2),
    ///     TwoSz::new(0),
    ///     CalcMode::Optimization,
    ///     LanczosMode::None,
    ///     RandomSeed::new(12345),
    ///     sr_params,
    ///     mc_params,
    /// );
    ///
    /// let wavefunction = CombinedWavefunction::new(
    ///     SiteCount::new(4),
    ///     ElectronCount::new(2),
    /// );
    ///
    /// let lattice = ChainLattice::new(4, true).unwrap();
    /// let hamiltonian = HubbardHamiltonian::new(lattice, 1.0, 4.0, 0.0).unwrap();
    ///
    /// let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();
    /// ```
    pub fn new(params: VmcParameters, wavefunction: PhysicsCombinedWavefunction, hamiltonian: Box<dyn Hamiltonian>) -> Result<Self> {
        // Validate parameters
        params.validate()?;

        let nsite = SiteCount::new(wavefunction.nsite());
        let ne = ElectronCount::new(wavefunction.ne());

        // Create Metropolis sampler
        let sampler = MetropolisSampler::new(
            nsite,
            ne,
            params.two_sz,
            RandomSeed::new(params.random_seed.get()),
        );

        // Create optimizer if in optimization mode
        let optimizer = if params.calc_mode == CalcMode::Optimization {
            Some(SROptimizer::new(nsite, ne))
        } else {
            None
        };

        // Create observable calculator
        let calculator = ObservableCalculator::new(nsite, ne);

        Ok(Self {
            params,
            wavefunction,
            sampler,
            optimizer,
            calculator,
            hamiltonian,
            current_iteration: 0,
        })
    }

    /// Runs the VMC calculation
    ///
    /// This is the main entry point for running a VMC calculation.
    /// It will either run optimization or expectation value calculation
    /// depending on the calculation mode.
    ///
    /// # Returns
    ///
    /// The VMC result containing energy, observables, and statistics
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmcmain.c:VMCParaOpt`, `VMCParaOptMain`
    pub fn run<F>(&mut self, output_callback: Option<F>) -> Result<VmcResult>
    where
        F: FnMut(usize, Complex64, &SamplingStatistics),
    {
        match self.params.calc_mode {
            CalcMode::Optimization => self.run_optimization(),
            CalcMode::Expectation => self.run_expectation(output_callback),
        }
    }

    /// Runs optimization calculation
    ///
    /// This performs variational parameter optimization using the SR method.
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmcmain.c:VMCParaOpt`
    fn run_optimization(&mut self) -> Result<VmcResult> {
        let sr_params = self.params.sr_params.clone();
        let mc_params = self.params.mc_params.clone();

        let mut energy_history = Vec::new();
        let mut converged = false;

        // Main optimization loop
        for iteration in 0..sr_params.iteration_steps {
            self.current_iteration = iteration;

            // Warmup phase
            self.warmup(mc_params.warmup_steps)?;

            // Sampling phase
            let (energy, _observables, _statistics) = self.sample_and_calculate::<fn(usize, Complex64, &SamplingStatistics)>(
                mc_params.num_samples,
                mc_params.sampling_interval,
                None,
            )?;

            energy_history.push(energy);

            // Check convergence (simple criterion for now)
            if iteration > 10 {
                let recent_energies: Vec<_> = energy_history.iter().rev().take(5).collect();
                let avg_energy: Complex64 = recent_energies.iter().map(|&&e| e).sum::<Complex64>()
                    / Complex64::new(recent_energies.len() as f64, 0.0);
                let variance: f64 = recent_energies
                    .iter()
                    .map(|&&e| (e - avg_energy).norm())
                    .sum::<f64>()
                    / recent_energies.len() as f64;

                if variance < 1e-6 {
                    converged = true;
                    break;
                }
            }

            // Update parameters using SR method
            if let Some(ref mut optimizer) = self.optimizer {
                // This is a simplified version - in full implementation,
                // we would calculate SR matrix and force vector from samples
                let _result = optimizer.optimize()?;

                // Apply parameter updates to wavefunction
                // (This is a placeholder - full implementation would update wavefunction parameters)
            }
        }

        let (final_energy, final_observables, final_statistics) = self.sample_and_calculate::<fn(usize, Complex64, &SamplingStatistics)>(
            mc_params.num_samples * 10, // Use more samples for final result
            mc_params.sampling_interval,
            None,
        )?;

        let optimization_result = OptimizationResult {
            iterations: self.current_iteration + 1,
            energy_history,
            converged,
            parameter_updates: vec![],
        };

        Ok(VmcResult {
            mode: CalcMode::Optimization,
            energy: final_energy,
            energy_error: 0.0, // TODO: Calculate error
            observables: final_observables,
            statistics: final_statistics,
            optimization: Some(optimization_result),
        })
    }

    /// Runs expectation value calculation
    ///
    /// This performs a single calculation of expectation values without optimization.
    ///
    /// # References
    ///
    /// - C implementation: `mVMC/src/mVMC/vmcmain.c:VMCParaOpt` (NVMCCalMode == 1)
    fn run_expectation<F>(&mut self, output_callback: Option<F>) -> Result<VmcResult>
    where
        F: FnMut(usize, Complex64, &SamplingStatistics),
    {
        let mc_params = self.params.mc_params.clone();

        // Warmup phase
        self.warmup(mc_params.warmup_steps)?;

        // Sampling phase
        let (energy, observables, statistics) = self.sample_and_calculate(
            mc_params.num_samples,
            mc_params.sampling_interval,
            output_callback,
        )?;

        Ok(VmcResult {
            mode: CalcMode::Expectation,
            energy,
            energy_error: 0.0, // TODO: Calculate error
            observables,
            statistics,
            optimization: None,
        })
    }

    /// Warmup phase
    ///
    /// Performs warmup Monte Carlo steps to equilibrate the system.
    fn warmup(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            let _step = self.sampler.metropolis_step();
            // Don't collect statistics during warmup
        }
        Ok(())
    }

    /// Sampling and calculation phase
    ///
    /// Performs Monte Carlo sampling and calculates observables.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to collect
    /// * `interval` - Sampling interval (steps between samples)
    /// * `output_callback` - Optional callback for bin-by-bin output
    ///
    /// # Returns
    ///
    /// Tuple of (energy, observables, statistics)
    fn sample_and_calculate<F>(
        &mut self,
        num_samples: usize,
        interval: usize,
        mut output_callback: Option<F>,
    ) -> Result<(Complex64, PhysicalObservables, SamplingStatistics)>
    where
        F: FnMut(usize, Complex64, &SamplingStatistics),
    {
        // Reset calculator
        self.calculator.reset();

        let mut energy_sum = Complex64::new(0.0, 0.0);
        let mut energy_squared_sum = Complex64::new(0.0, 0.0);
        let mut samples_collected = 0;

        for sample_idx in 0..num_samples {
            // Perform interval steps
            for _ in 0..interval {
                let _step = self.sampler.metropolis_step();
            }

            // Collect sample
            let step = self.sampler.metropolis_step()?;
            let config = self.sampler.current_config();

            // Convert electron configuration to spin configuration
            let spin_config = self.electron_config_to_spin_config(&config);

            // Calculate local energy using Hamiltonian
            let local_energy = self.calculate_local_energy(&spin_config)?;
            energy_sum += local_energy;
            energy_squared_sum += local_energy * local_energy;

            // Add sample to calculator
            self.calculator.add_sample(&config, &step, local_energy.re);
            samples_collected += 1;

            // Call output callback if provided
            if let Some(ref mut callback) = output_callback {
                let statistics = self.sampler.statistics();
                callback(sample_idx, local_energy, &statistics);
            }
        }

        // Calculate average energy and variance
        let avg_energy = energy_sum / Complex64::new(samples_collected as f64, 0.0);
        let avg_energy_squared = energy_squared_sum / Complex64::new(samples_collected as f64, 0.0);
        let energy_variance = avg_energy_squared - avg_energy * avg_energy;
        let _energy_error = (energy_variance.re / samples_collected as f64).sqrt();

        // Calculate observables
        let observables = self.calculator.calculate_observables();

        // Get sampling statistics
        let statistics = self.sampler.statistics();

        Ok((avg_energy, observables, statistics))
    }

    /// Returns the current wavefunction
    pub fn wavefunction(&self) -> &PhysicsCombinedWavefunction {
        &self.wavefunction
    }

    /// Returns a mutable reference to the wavefunction
    pub fn wavefunction_mut(&mut self) -> &mut PhysicsCombinedWavefunction {
        &mut self.wavefunction
    }

    /// Returns the current sampler
    pub fn sampler(&self) -> &MetropolisSampler {
        &self.sampler
    }

    /// Returns a mutable reference to the sampler
    pub fn sampler_mut(&mut self) -> &mut MetropolisSampler {
        &mut self.sampler
    }

    /// Returns the current parameters
    pub fn params(&self) -> &VmcParameters {
        &self.params
    }

    /// Returns the current iteration
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Converts electron configuration to spin configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Electron configuration
    ///
    /// # Returns
    ///
    /// Spin configuration for Hamiltonian calculation
    fn electron_config_to_spin_config(&self, config: &crate::monte_carlo::ElectronConfiguration) -> Vec<Spin> {
        let mut spin_config = Vec::with_capacity(config.nsite());

        for site in 0..config.nsite() {
            let n_up = config.electron_number_up(site);
            let n_down = config.electron_number_down(site);

            match (n_up, n_down) {
                (1, 0) => spin_config.push(Spin::Up),
                (0, 1) => spin_config.push(Spin::Down),
                (1, 1) => spin_config.push(Spin::Up), // Both spins present, use Up as representative
                (0, 0) => spin_config.push(Spin::Empty),
                _ => spin_config.push(Spin::Empty), // Invalid state, treat as empty
            }
        }

        spin_config
    }

    /// Calculates local energy for a given spin configuration
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    ///
    /// Local energy as complex number
    fn calculate_local_energy(&self, spin_config: &[Spin]) -> Result<Complex64> {
        // Calculate diagonal element (expectation value)
        let diagonal_energy = self.hamiltonian.diagonal_element(spin_config);

        // For VMC, we need the local energy which includes off-diagonal terms
        // This is a simplified implementation - full VMC would require
        // calculating the ratio of wavefunction amplitudes
        let local_energy = self.calculate_vmc_local_energy(spin_config)?;

        Ok(Complex64::new(local_energy, 0.0))
    }

    /// Runs a single VMC iteration (for optimization mode)
    ///
    /// This method performs one iteration of the VMC calculation,
    /// including sampling, energy calculation, and parameter updates.
    ///
    /// # Returns
    /// * `Result<VmcIterationResult>` - The iteration result
    pub fn run_single_iteration(&mut self) -> Result<VmcIterationResult> {
        // 1. Monte Carlo sampling
        let sampling_result = self.sampler.sample(&self.wavefunction)?;

        // 2. Calculate energy and observables
        let energy = self.calculate_energy(&sampling_result.configurations)?;
        let observables = self.calculator.calculate_observables();

        // 3. Calculate variance
        let variance = self.calculate_variance(&sampling_result.configurations, energy)?;

        // 4. Update parameters if in optimization mode
        // Note: This is a placeholder - full optimization implementation would be needed
        // if let Some(ref mut optimizer) = self.optimizer {
        //     optimizer.update_parameters(&self.wavefunction, &sampling_result)?;
        // }

        Ok(VmcIterationResult {
            energy,
            variance,
            sample_count: sampling_result.configurations.len(),
            acceptance_rate: sampling_result.acceptance_rate,
            observables,
        })
    }

    /// Calculates the total energy for a set of configurations
    ///
    /// # Arguments
    /// * `configurations` - List of electron configurations
    ///
    /// # Returns
    /// * `Result<Complex64>` - The total energy
    fn calculate_energy(&self, configurations: &[crate::monte_carlo::ElectronConfiguration]) -> Result<Complex64> {
        let mut total_energy = Complex64::new(0.0, 0.0);
        let mut total_weight = 0.0;

        for config in configurations {
            let spin_config = self.electron_config_to_spin_config(config);
            let local_energy = self.calculate_local_energy(&spin_config)?;
            let weight = self.wavefunction.calculate(&self.spin_config_to_u8(&spin_config)).norm();

            total_energy += local_energy * weight;
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            Ok(total_energy / total_weight)
        } else {
            Ok(Complex64::new(0.0, 0.0))
        }
    }

    /// Calculates the variance of the energy
    ///
    /// # Arguments
    /// * `configurations` - List of electron configurations
    /// * `mean_energy` - Mean energy value
    ///
    /// # Returns
    /// * `Result<f64>` - The variance
    fn calculate_variance(&self, configurations: &[crate::monte_carlo::ElectronConfiguration], mean_energy: Complex64) -> Result<f64> {
        let mut total_variance = 0.0;
        let mut total_weight = 0.0;

        for config in configurations {
            let spin_config = self.electron_config_to_spin_config(config);
            let local_energy = self.calculate_local_energy(&spin_config)?;
            let weight = self.wavefunction.calculate(&self.spin_config_to_u8(&spin_config)).norm();

            let energy_diff = local_energy - mean_energy;
            total_variance += energy_diff.norm_sqr() * weight;
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            Ok(total_variance / total_weight)
        } else {
            Ok(0.0)
        }
    }

    /// Converts spin configuration to u8 array for wavefunction calculation
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Vec<u8>` - u8 array representation
    fn spin_config_to_u8(&self, spin_config: &[Spin]) -> Vec<u8> {
        spin_config.iter().map(|spin| {
            match spin {
                Spin::Up => 1,
                Spin::Down => 2,
                Spin::Empty => 0,
            }
        }).collect()
    }

    /// Calculates VMC local energy including off-diagonal terms
    ///
    /// # Arguments
    ///
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    ///
    /// VMC local energy
    fn calculate_vmc_local_energy(&self, spin_config: &[Spin]) -> Result<f64> {
        // Start with diagonal energy
        let mut local_energy = self.hamiltonian.diagonal_element(spin_config);

        // Add off-diagonal contributions
        // This is a simplified implementation - full VMC would require
        // calculating wavefunction amplitude ratios for all possible transitions
        let nsite = spin_config.len();

        for i in 0..nsite {
            for j in 0..nsite {
                if i != j {
                    // Check if sites are neighbors
                    if self.hamiltonian.lattice().neighbors(i).contains(&j) {
                        // Calculate matrix element for this transition
                        let mut new_config = spin_config.to_vec();
                        new_config.swap(i, j);

                        let matrix_element = self.hamiltonian.matrix_element(spin_config, &new_config);

                        // Calculate wavefunction amplitude ratio
                        // This is a placeholder - full implementation would use actual wavefunction
                        let amplitude_ratio = 1.0; // Placeholder

                        local_energy += (matrix_element * amplitude_ratio).re;
                    }
                }
            }
        }

        Ok(local_energy)
    }

    /// Calculates the magnetization for a given configuration
    ///
    /// # Arguments
    /// * `configurations` - List of electron configurations
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated magnetization
    fn calculate_magnetization(&self, configurations: &[crate::monte_carlo::ElectronConfiguration]) -> Result<f64> {
        if configurations.is_empty() {
            return Ok(0.0);
        }

        let mut total_magnetization = 0.0;
        let mut total_weight = 0.0;

        for config in configurations {
            // Convert electron configuration to spin configuration
            let spin_config = self.electron_config_to_spin_config(config);

            // Calculate the magnetization for this configuration
            let magnetization = self.calculate_magnetization_single(&spin_config)?;

            // Calculate the wave function amplitude
            let psi = self.wavefunction.calculate(&self.spin_config_to_u8(&spin_config));
            let weight = psi.norm_sqr();

            total_magnetization += magnetization * weight;
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            Ok(total_magnetization / total_weight)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates the magnetization for a single configuration
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated magnetization
    fn calculate_magnetization_single(&self, spin_config: &[Spin]) -> Result<f64> {
        let magnetization = spin_config.iter().map(|spin| spin.value_f64()).sum();
        Ok(magnetization)
    }

    /// Calculates the spin-spin correlation for a given configuration
    ///
    /// # Arguments
    /// * `configurations` - List of electron configurations
    /// * `site1` - First site index
    /// * `site2` - Second site index
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated correlation
    fn calculate_spin_spin_correlation(
        &self,
        configurations: &[crate::monte_carlo::ElectronConfiguration],
        site1: usize,
        site2: usize,
    ) -> Result<f64> {
        if configurations.is_empty() {
            return Ok(0.0);
        }

        let mut total_correlation = 0.0;
        let mut total_weight = 0.0;

        for config in configurations {
            // Convert electron configuration to spin configuration
            let spin_config = self.electron_config_to_spin_config(config);

            // Calculate the correlation for this configuration
            let correlation = self.calculate_spin_spin_correlation_single(&spin_config, site1, site2)?;

            // Calculate the wave function amplitude
            let psi = self.wavefunction.calculate(&self.spin_config_to_u8(&spin_config));
            let weight = psi.norm_sqr();

            total_correlation += correlation * weight;
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            Ok(total_correlation / total_weight)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates the spin-spin correlation for a single configuration
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    /// * `site1` - First site index
    /// * `site2` - Second site index
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated correlation
    fn calculate_spin_spin_correlation_single(
        &self,
        spin_config: &[Spin],
        site1: usize,
        site2: usize,
    ) -> Result<f64> {
        if site1 >= spin_config.len() || site2 >= spin_config.len() {
            return Ok(0.0);
        }

        let spin1 = spin_config[site1].value_f64();
        let spin2 = spin_config[site2].value_f64();
        let correlation = spin1 * spin2;

        Ok(correlation)
    }

    /// Calculates the structure factor for a given configuration
    ///
    /// # Arguments
    /// * `configurations` - List of electron configurations
    /// * `momentum` - Momentum vector
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated structure factor
    fn calculate_structure_factor(
        &self,
        configurations: &[crate::monte_carlo::ElectronConfiguration],
        momentum: &[f64],
    ) -> Result<f64> {
        if configurations.is_empty() {
            return Ok(0.0);
        }

        let mut total_structure_factor = 0.0;
        let mut total_weight = 0.0;

        for config in configurations {
            // Convert electron configuration to spin configuration
            let spin_config = self.electron_config_to_spin_config(config);

            // Calculate the structure factor for this configuration
            let structure_factor = self.calculate_structure_factor_single(&spin_config, momentum)?;

            // Calculate the wave function amplitude
            let psi = self.wavefunction.calculate(&self.spin_config_to_u8(&spin_config));
            let weight = psi.norm_sqr();

            total_structure_factor += structure_factor * weight;
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            Ok(total_structure_factor / total_weight)
        } else {
            Ok(0.0)
        }
    }

    /// Calculates the structure factor for a single configuration
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    /// * `momentum` - Momentum vector
    ///
    /// # Returns
    /// * `Result<f64>` - The calculated structure factor
    fn calculate_structure_factor_single(
        &self,
        spin_config: &[Spin],
        momentum: &[f64],
    ) -> Result<f64> {
        if momentum.len() != self.hamiltonian.lattice().dimension() {
            return Ok(0.0);
        }

        let mut structure_factor = 0.0;
        let n_sites = spin_config.len();

        for i in 0..n_sites {
            for j in 0..n_sites {
                let spin_i = spin_config[i].value_f64();
                let spin_j = spin_config[j].value_f64();

                // Calculate the phase factor
                let mut phase = 0.0;
                for (dim, &k) in momentum.iter().enumerate() {
                    let size = self.hamiltonian.lattice().size()[dim];
                    let pos_i = (i % size) as f64;
                    let pos_j = (j % size) as f64;
                    phase += k * (pos_i - pos_j);
                }

                structure_factor += spin_i * spin_j * (2.0 * std::f64::consts::PI * phase).cos();
            }
        }

        Ok(structure_factor / n_sites as f64)
    }
}

impl fmt::Display for VmcEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VmcEngine(mode={:?}, nsite={}, ne={}, iteration={})",
            self.params.calc_mode,
            self.wavefunction.nsite(),
            self.wavefunction.ne(),
            self.current_iteration
        )
    }
}

impl fmt::Display for VmcResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VMC Calculation Result")?;
        writeln!(f, "  Mode: {:?}", self.mode)?;
        writeln!(f, "  Energy: {} ± {}", self.energy, self.energy_error)?;
        writeln!(f, "  Acceptance rate: {:.2}%", self.statistics.acceptance_rate * 100.0)?;
        if let Some(ref opt) = self.optimization {
            writeln!(f, "  Optimization:")?;
            writeln!(f, "    Iterations: {}", opt.iterations)?;
            writeln!(f, "    Converged: {}", opt.converged)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LanczosMode, RandomSeed, TwoSz};
    use crate::wavefunction::SlaterDeterminant;

    #[test]
    fn test_vmc_engine_creation() {
        let params = VmcParameters::builder()
            .nsite(SiteCount::new(4))
            .ne(ElectronCount::new(2))
            .two_sz(TwoSz::new(0))
            .calc_mode(CalcMode::Optimization)
            .lanczos_mode(LanczosMode::None)
            .random_seed(RandomSeed::new(12345))
            .build()
            .unwrap();

        let wavefunction = CombinedWavefunction::new(SiteCount::new(4), ElectronCount::new(2));

        let engine = VmcEngine::new(params, wavefunction);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert_eq!(engine.current_iteration(), 0);
        assert!(engine.optimizer.is_some());
    }

    #[test]
    fn test_vmc_engine_expectation_mode() {
        let params = VmcParameters::builder()
            .nsite(SiteCount::new(4))
            .ne(ElectronCount::new(2))
            .two_sz(TwoSz::new(0))
            .calc_mode(CalcMode::Expectation)
            .lanczos_mode(LanczosMode::None)
            .random_seed(RandomSeed::new(12345))
            .build()
            .unwrap();

        let wavefunction = CombinedWavefunction::new(SiteCount::new(4), ElectronCount::new(2));

        let engine = VmcEngine::new(params, wavefunction);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert!(engine.optimizer.is_none());
    }

    #[test]
    fn test_vmc_engine_with_slater() {
        let nsite = SiteCount::new(2);
        let ne = ElectronCount::new(1);

        let params = VmcParameters::builder()
            .nsite(nsite)
            .ne(ne)
            .two_sz(TwoSz::new(1))
            .calc_mode(CalcMode::Expectation)
            .lanczos_mode(LanczosMode::None)
            .random_seed(RandomSeed::new(12345))
            .build()
            .unwrap();

        let slater = SlaterDeterminant::new(nsite, ne);
        let wavefunction = CombinedWavefunction::new(nsite, ne).with_slater(slater);

        let engine = VmcEngine::new(params, wavefunction);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_vmc_result_display() {
        let result = VmcResult {
            mode: CalcMode::Expectation,
            energy: Complex64::new(-1.5, 0.0),
            energy_error: 0.01,
            observables: PhysicalObservables::new(4),
            statistics: SamplingStatistics::default(),
            optimization: None,
        };

        let display = format!("{}", result);
        assert!(display.contains("VMC Calculation Result"));
        assert!(display.contains("Energy"));
    }
}

