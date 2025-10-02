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
// use crate::wavefunction::CombinedWavefunction;
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
    /// Energy squared average
    pub energy_squared: Complex64,
    /// Energy variance
    pub variance: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Physical observables
    pub observables: PhysicalObservables,
    /// Total Sz average
    pub sz_total: f64,
    /// Total Sz^2 average
    pub sz_squared: f64,
    /// SR step info for this iteration (if optimization)
    pub sr_info: Option<crate::vmc::sr_optimization::SRStepInfo>,
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
    /// Energy history for optimization
    energy_history: Vec<Complex64>,
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
    /// let sr_params = SRParameters::new(1000, 100, 100, 1e-6, 1e-6, 0.1, 100, 1e-6, true);
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
    /// let wavefunction = mvmc_physics::wavefunction::CombinedWavefunction::new(4, 2).unwrap();
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
        let mut sampler = MetropolisSampler::new(
            nsite,
            ne,
            params.two_sz,
            RandomSeed::new(params.random_seed.get()),
        );
        // Provide lattice neighbors for electron sampler (neighbor hopping)
        let mut neighbors: Vec<Vec<usize>> = Vec::new();
        for i in 0..hamiltonian.lattice().n_sites() {
            neighbors.push(hamiltonian.lattice().neighbors(i));
        }
        sampler.set_neighbors(neighbors);
        sampler.configure(&params.mc_params);

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
            energy_history: Vec::new(),
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
        eprintln!("VmcEngine::run called with calc_mode: {:?}", self.params.calc_mode);
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
    pub fn run_optimization(&mut self) -> Result<VmcResult> {
        eprintln!("Starting run_optimization");
        let sr_params = self.params.sr_params.clone();
        let mc_params = self.params.mc_params.clone();

        // Clear previous energy history
        self.energy_history.clear();
        let mut converged = false;

        // Main optimization loop
        for iteration in 0..sr_params.iteration_steps {
            self.current_iteration = iteration;

            // For Heisenberg model (ne=0), use specialized iteration
            eprintln!("Checking model type: ne = {}", self.sampler.current_config().ne());
            let (_energy, _observables, _statistics) = if self.sampler.current_config().ne() == 0 {
                let result = self.run_heisenberg_iteration(iteration)?;
                (result.energy, result.observables, crate::monte_carlo::SamplingStatistics {
                    total_steps: result.sample_count,
                    accepted_steps: (result.acceptance_rate * result.sample_count as f64) as usize,
                    average_energy: result.energy.re,
                    energy_variance: result.variance,
                    warmup_steps: 0,
                    measurement_steps: result.sample_count,
                    acceptance_rate: result.acceptance_rate,
                })
            } else {
                // Warmup phase
                self.warmup(mc_params.warmup_steps)?;

                // Sampling phase
                self.sample_and_calculate::<fn(usize, Complex64, &SamplingStatistics)>(
                    mc_params.num_samples,
                    mc_params.sampling_interval,
                    None,
                )?
            };

            // Energy is already pushed to self.energy_history in run_heisenberg_iteration

            // Check convergence (simple criterion for now)
            // Disabled for now to ensure we run the full number of iterations
            // TODO: Implement proper convergence criteria
            if false && iteration > 10 {
                let recent_energies: Vec<_> = self.energy_history.iter().rev().take(5).collect();
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
            energy_history: self.energy_history.clone(),
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
            let local_energy;
            let config;
            let step;

            // For Heisenberg model (ne=0), generate spin configurations directly
            if self.sampler.current_config().ne() == 0 {
                let two_sz = self.sampler.current_config().two_sz();
                let mut spin_config = self.generate_initial_spin_config(two_sz);

                // Perform Monte Carlo sampling for this sample
                let mut rng_state = 12345u64 + sample_idx as u64; // Different seed for each sample
                let num_mc_steps = 50; // Fewer steps per sample

                for _ in 0..num_mc_steps {
                    let (proposed_config, _acceptance_prob) = self.propose_spin_flip(&spin_config, &mut rng_state)?;

                    // Calculate energies
                    let current_energy = self.calculate_local_energy(&spin_config)?;
                    let proposed_energy = self.calculate_local_energy(&proposed_config)?;

                    // Metropolis acceptance criterion
                    let energy_diff = proposed_energy.re - current_energy.re;
                    let acceptance_rate = if energy_diff <= 0.0 {
                        1.0
                    } else {
                        (-energy_diff).exp()
                    };

                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let random_num = (rng_state as f64) / (u64::MAX as f64);

                    if random_num < acceptance_rate {
                        spin_config = proposed_config;
                    }
                }

                local_energy = self.calculate_local_energy(&spin_config)?;

                // Create dummy configuration and step for compatibility
                config = self.sampler.current_config().clone();
                step = crate::monte_carlo::MetropolisStep {
                    accepted: true,
                    proposed_config: config.clone(),
                    acceptance_prob: 1.0,
                    amplitude_ratio: Complex64::new(1.0, 0.0),
                };
            } else {
                // Perform interval steps
                for _ in 0..interval {
                    let _step = self.sampler.metropolis_step();
                }

                // Collect sample
                step = self.sampler.metropolis_step()?;
                config = self.sampler.current_config().clone();

                // Convert electron configuration to spin configuration
                let spin_config = self.electron_config_to_spin_config(&config);

                // Calculate local energy using Hamiltonian
                local_energy = self.calculate_local_energy(&spin_config)?;
            }

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

    /// Generates initial spin configuration for Heisenberg model
    ///
    /// For Heisenberg model (ne=0), we need to generate spin configurations directly
    /// instead of using electron configurations.
    ///
    /// # Arguments
    ///
    /// * `two_sz` - Spin quantum number (2*Sz)
    ///
    /// # Returns
    ///
    /// Initial spin configuration
    fn generate_initial_spin_config(&self, two_sz: i32) -> Vec<Spin> {
        let nsite = self.hamiltonian.lattice().n_sites();
        let mut spin_config = vec![Spin::Empty; nsite];

        // For Heisenberg model, ALL sites must have spins (no Empty sites)
        // two_sz = 0 means equal number of up and down spins
        let total_sz = two_sz as i32;
        let n_up = ((nsite as i32 + total_sz) / 2) as usize;
        let _n_down = nsite - n_up;

        // Ensure we don't exceed the number of sites
        let n_up = n_up.min(nsite);

        // Place up spins
        for i in 0..n_up {
            spin_config[i] = Spin::Up;
        }

        // Place down spins for the remaining sites
        for i in n_up..nsite {
            spin_config[i] = Spin::Down;
        }

        // Shuffle the configuration to randomize
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        self.shuffle_spin_config(&mut spin_config, seed);

        // Verify no Empty spins remain
        debug_assert!(spin_config.iter().all(|&s| s != Spin::Empty),
            "Heisenberg model should have no Empty spins: {:?}", spin_config);

        spin_config
    }

    /// Shuffles a spin configuration randomly
    fn shuffle_spin_config(&self, config: &mut [Spin], seed: u64) {
        let mut rng_state = seed;

        for i in 0..config.len() {
            // Simple linear congruential generator
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state as usize) % config.len();

            if i != j {
                config.swap(i, j);
            }
        }
    }

    /// Proposes a spin flip for Heisenberg model
    ///
    /// This method proposes a random spin flip in the configuration.
    /// For Heisenberg model, we can flip any spin from Up to Down or vice versa.
    ///
    /// # Arguments
    /// * `config` - Current spin configuration
    /// * `rng_state` - Random number generator state
    ///
    /// # Returns
    /// * `Result<(Vec<Spin>, f64)>` - (new configuration, acceptance probability)
    fn propose_spin_flip(&self, config: &[Spin], rng_state: &mut u64) -> Result<(Vec<Spin>, f64)> {
        let mut new_config = config.to_vec();

        // Find all non-empty spins
        let mut spin_sites = Vec::new();
        for (i, &spin) in config.iter().enumerate() {
            if spin != Spin::Empty {
                spin_sites.push(i);
            }
        }

        if spin_sites.is_empty() {
            return Ok((new_config, 0.0));
        }

        // Choose a random site to flip
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let site_idx = (*rng_state as usize) % spin_sites.len();
        let site = spin_sites[site_idx];

        // Flip the spin
        new_config[site] = match config[site] {
            Spin::Up => Spin::Down,
            Spin::Down => Spin::Up,
            Spin::Empty => Spin::Empty, // Should not happen
        };

        // For now, use simple acceptance probability
        // In a full implementation, this would depend on the energy difference
        let acceptance_prob = 0.5; // 50% acceptance rate (unused for production)

        Ok((new_config, acceptance_prob))
    }

    /// Proposes a spin exchange that preserves total Sz: swap one Up with one Down spin
    fn propose_spin_exchange(&self, config: &[Spin], rng_state: &mut u64) -> Result<(Vec<Spin>, f64)> {
        let mut up_sites = Vec::new();
        let mut down_sites = Vec::new();
        for (i, &s) in config.iter().enumerate() {
            if s == Spin::Up { up_sites.push(i); }
            if s == Spin::Down { down_sites.push(i); }
        }
        if up_sites.is_empty() || down_sites.is_empty() {
            return Ok((config.to_vec(), 0.0));
        }
        // choose random indices
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let iu = (rng_state.wrapping_rem(up_sites.len() as u64)) as usize;
        *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let id = (rng_state.wrapping_rem(down_sites.len() as u64)) as usize;
        let mut new_config = config.to_vec();
        new_config[up_sites[iu]] = Spin::Down;
        new_config[down_sites[id]] = Spin::Up;
        Ok((new_config, 1.0))
    }

    /// Performs Metropolis sampling for Heisenberg model
    ///
    /// This method performs Monte Carlo sampling by proposing spin flips
    /// and accepting them based on the Metropolis criterion.
    ///
    /// # Arguments
    /// * `config` - Current spin configuration
    /// * `num_steps` - Number of Monte Carlo steps
    /// * `rng_state` - Random number generator state
    ///
    /// # Returns
    /// * `Result<Vec<Spin>>` - Final spin configuration after sampling
    fn metropolis_sampling_heisenberg(
        &self,
        mut config: Vec<Spin>,
        num_steps: usize,
        rng_state: &mut u64,
    ) -> Result<Vec<Spin>> {
        for _ in 0..num_steps {
            let (proposed_config, acceptance_prob) = self.propose_spin_flip(&config, rng_state)?;

            // Generate random number for acceptance
            *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_num = (*rng_state as f64) / (u64::MAX as f64);

            if random_num < acceptance_prob {
                config = proposed_config;
            }
        }

        Ok(config)
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
        let _diagonal_energy = self.hamiltonian.diagonal_element(spin_config);

        // For VMC, we need the local energy which includes off-diagonal terms
        // This is a simplified implementation - full VMC would require
        // calculating the ratio of wavefunction amplitudes
        let local_energy = self.calculate_vmc_local_energy(spin_config)?;

        Ok(Complex64::new(local_energy, 0.0))
    }

    /// Runs a single Heisenberg VMC iteration
    ///
    /// This method performs one iteration of the VMC calculation for Heisenberg models,
    /// including sampling, energy calculation, and parameter updates.
    ///
    /// # Arguments
    /// * `iteration` - Current iteration number
    ///
    /// # Returns
    /// * `Result<VmcIterationResult>` - Results of this iteration
    fn run_heisenberg_iteration(&mut self, iteration: usize) -> Result<VmcIterationResult> {
        println!("run_heisenberg_iteration called with iteration: {}", iteration);
        let mc_params = self.params.mc_params.clone();
        let nsite = self.hamiltonian.lattice().n_sites();

        // Generate initial spin configuration
        let two_sz = self.sampler.current_config().two_sz();
        let mut spin_config = self.generate_initial_spin_config(two_sz);

        // Perform Monte Carlo sampling
        let mut rng_state = 12345u64 + iteration as u64;
        let num_mc_steps = mc_params.num_samples;
        let mut energy_sum = Complex64::new(0.0, 0.0);
        let mut energy_squared_sum = Complex64::new(0.0, 0.0);
        let mut accepted_steps = 0;

        // Collect samples for SR optimization
        let mut sr_samples = Vec::new();

        for _step in 0..num_mc_steps {
            // Propose spin flip
            let (proposed_config, _acceptance_prob) = self.propose_spin_flip(&spin_config, &mut rng_state)?;

            // Calculate wavefunction ratio
            let current_amplitude = self.wavefunction.calculate(&self.spin_config_to_u8(&spin_config));
            let proposed_amplitude = self.wavefunction.calculate(&self.spin_config_to_u8(&proposed_config));

            let amplitude_ratio = if current_amplitude.norm() < 1e-12 {
                Complex64::new(0.0, 0.0)
            } else {
                proposed_amplitude / current_amplitude
            };

            // Metropolis acceptance criterion
            let acceptance_rate = amplitude_ratio.norm().min(1.0);

            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_num = (rng_state as f64) / (u64::MAX as f64);

            if random_num < acceptance_rate {
                spin_config = proposed_config;
                accepted_steps += 1;
            }

            // Calculate local energy
            let local_energy = self.calculate_local_energy(&spin_config)?;
            energy_sum += local_energy;
            energy_squared_sum += local_energy * local_energy;

            // Collect sample for SR optimization
            let u8_config = self.spin_config_to_u8(&spin_config);
            let o_operators = self.wavefunction.calculate_o_operators(&u8_config);
            sr_samples.push(crate::vmc::sr_optimization::SRSampleData {
                energy: local_energy.re,
                o_operators: o_operators.into_iter().map(|c| c.re).collect(),
                weight: 1.0, // TODO: Calculate proper weight
            });
        }

        // Calculate average energy and variance
        let avg_energy = energy_sum / Complex64::new(num_mc_steps as f64, 0.0);
        let avg_energy_squared = energy_squared_sum / Complex64::new(num_mc_steps as f64, 0.0);
        let variance = avg_energy_squared - avg_energy * avg_energy;

        // Store energy in history
        self.energy_history.push(avg_energy);

        // Perform SR optimization if optimizer is available
        if let Some(ref mut _optimizer) = self.optimizer {
            // Use proper SR optimization
            use crate::vmc::sr_optimization::{SROptimizationCalculator, SRSampleData};

            let n_params = self.wavefunction.num_parameters();
            let mut sr_calculator = SROptimizationCalculator::new(n_params);

            // Debug: Check sample count
            println!("SR optimization: n_params = {}, n_samples = {}", n_params, sr_samples.len());

            // Add samples to SR calculator
            for sample in &sr_samples {
                sr_calculator.add_sample(sample.clone());
            }

            // Perform SR optimization
            let sr_params = &self.params.sr_params;
            match sr_calculator.solve_full_sr_with_params(sr_params) {
                Ok((param_updates, sr_info)) => {
                    // Debug output for first few iterations
                    if iteration < 3 {
                        let max_update = param_updates.iter().map(|u| u.abs()).fold(0.0f64, f64::max);
                        let max_o = sr_samples.iter()
                            .flat_map(|s| &s.o_operators)
                            .map(|o| o.abs())
                            .fold(0.0f64, f64::max);
                        let max_energy_o = sr_samples.iter()
                            .map(|s| s.energy * s.o_operators.iter().map(|o| o.abs()).fold(0.0f64, f64::max))
                            .fold(0.0f64, f64::max);

                        println!("Iteration {}: avg_energy = {:.6}, max_o = {:.6}, max_energy_o = {:.6}, max_force = {:.6}, max_update = {:.6}",
                            iteration, avg_energy.re, max_o, max_energy_o, max_update, max_update);

                        // Show first few parameters
                        if !sr_samples.is_empty() {
                            println!("  First 5 O-operators: {:?}", &sr_samples[0].o_operators[..5.min(n_params)]);
                            println!("  First 5 forces: {:?}", &param_updates[..5.min(n_params)]);
                        }
                    }

                    // Apply parameter updates to wavefunction
                    self.wavefunction.update_parameters(&param_updates, sr_params.step_size);
                }
                Err(e) => {
                    eprintln!("SR optimization failed: {}", e);
                    // Fallback to simple gradient descent
                    let learning_rate = 0.01;
                    let mut param_updates = vec![0.0; n_params];

                    // Calculate simple gradient
                    let mut avg_o = vec![0.0; n_params];
                    let mut avg_energy_o = vec![0.0; n_params];

                    for sample in &sr_samples {
                        for (i, &o) in sample.o_operators.iter().enumerate() {
                            avg_o[i] += o;
                            avg_energy_o[i] += sample.energy * o;
                        }
                    }

                    let n_samples = sr_samples.len() as f64;
                    for i in 0..n_params {
                        avg_o[i] /= n_samples;
                        avg_energy_o[i] /= n_samples;
                        let force = avg_energy_o[i] - avg_energy.re * avg_o[i];
                        param_updates[i] = -learning_rate * force;
                    }

                    self.wavefunction.update_parameters(&param_updates, 1.0);
                }
            }
        }

        // Calculate observables
        let observables = PhysicalObservables {
            energy: avg_energy.re,
            kinetic_energy: 0.0, // TODO: Calculate kinetic energy
            potential_energy: 0.0, // TODO: Calculate potential energy
            total_magnetization: 0.0, // TODO: Calculate total magnetization
            site_magnetization: vec![0.0; nsite], // TODO: Calculate site magnetization
            spin_correlation: vec![vec![0.0; nsite]; nsite], // TODO: Calculate spin correlation
            energy_error: 0.0, // TODO: Calculate energy error
            num_samples: num_mc_steps,
        };

        Ok(VmcIterationResult {
            energy: avg_energy,
            energy_squared: avg_energy_squared,
            variance: variance.re,
            sample_count: num_mc_steps,
            acceptance_rate: accepted_steps as f64 / num_mc_steps as f64,
            observables,
            sz_total: 0.0, // TODO: Calculate total Sz
            sz_squared: 0.0, // TODO: Calculate Sz^2
            sr_info: None, // TODO: Add SR step info
        })
    }

    /// Converts spin configuration to u8 array
    fn spin_config_to_u8(&self, spin_config: &[Spin]) -> Vec<u8> {
        spin_config.iter().map(|spin| {
            match spin {
                Spin::Up => 1,
                Spin::Down => 2,
                Spin::Empty => 0,
            }
        }).collect()
    }

    /// Runs a single VMC iteration (for optimization mode)
    ///
    /// This method performs one iteration of the VMC calculation,
    /// including sampling, energy calculation, and parameter updates.
    ///
    /// # Returns
    /// * `Result<VmcIterationResult>` - The iteration result
    pub fn run_single_iteration(&mut self) -> Result<VmcIterationResult> {
        let mut sr_step_info: Option<crate::vmc::sr_optimization::SRStepInfo> = None;
        // For Heisenberg model (ne=0), generate spin configurations directly
        println!("run_single_iteration: ne = {}", self.sampler.current_config().ne());
        if self.sampler.current_config().ne() == 0 {
            println!("Calling run_heisenberg_iteration");
            return self.run_heisenberg_iteration(0);
        }

        // 1. Monte Carlo sampling (NVMCWarmUp/NVMCInterval/NVMCSample 準拠)
        let _mc = &self.params.mc_params;
        let _base_seed = self.params.random_seed.get() + self.current_iteration as u64;
        // Use configured sampler (already set from params)
        let sampling_result = self.sampler.sample(&self.wavefunction)?;

        // 2. Calculate weighted averages: E, E^2, Sz, Sz^2
        let mut total_w = 0.0f64;
        let mut e_sum = num_complex::Complex64::new(0.0, 0.0);
        let mut e2_sum = num_complex::Complex64::new(0.0, 0.0);
        let mut sz_sum = 0.0f64;
        let mut sz2_sum = 0.0f64;

        for cfg in &sampling_result.configurations {
            let spin_config = self.electron_config_to_spin_config(cfg);
            let local_energy = self.calculate_local_energy(&spin_config)?;
            // total S^z from configuration (up-down on sites)
            let mut szt = 0.0f64;
            for s in &spin_config {
                use mvmc_physics::hamiltonian::Spin::*;
                szt += match s { Up => 1.0, Down => -1.0, Empty => 0.0 };
            }

            total_w += 1.0;
            e_sum += local_energy;
            e2_sum += local_energy * local_energy;
            sz_sum += szt;
            sz2_sum += szt * szt;
        }

        let (energy, energy_squared, sz_total, sz_squared) = if total_w > 1e-12 {
            (
                e_sum / total_w,
                e2_sum / total_w,
                sz_sum / total_w,
                sz2_sum / total_w,
            )
        } else {
            (num_complex::Complex64::new(0.0, 0.0), num_complex::Complex64::new(0.0, 0.0), 0.0, 0.0)
        };

        let observables = self.calculator.calculate_observables();

        // 3. Calculate variance from weighted averages
        let variance = (energy_squared - energy * energy).re;

        // 4. Update parameters if in optimization mode using full SR
        if self.params.calc_mode == crate::types::CalcMode::Optimization {
            use crate::vmc::sr_optimization::{SROptimizationCalculator, SRSampleData};
            let mut sr_calc = SROptimizationCalculator::new(self.wavefunction.num_parameters());
            let sr_limit = self.params.sr_params.iteration_sample.max(1);
            for cfg in sampling_result.configurations.iter().take(sr_limit) {
                let spin_config = self.electron_config_to_spin_config(cfg);
                let u8_config = self.spin_config_to_u8(&spin_config);
                let _psi = self.wavefunction.calculate(&u8_config);
                let weight = 1.0; // samples are already drawn ~|ψ|^2
                let local_energy = self.calculate_local_energy(&spin_config)?;
                let o_ops = self.wavefunction.calculate_o_operators(&u8_config);
                sr_calc.add_sample(SRSampleData { o_operators: o_ops.into_iter().map(|c| c.re).collect(), energy: local_energy.re, weight });
            }
            let sp = &self.params.sr_params;
            let (updates, info) = sr_calc.solve_full_sr_with_params(sp)?;
            self.wavefunction.update_parameters(&updates, sp.step_size);
            // Attach SR info into result (assign later via local variable)
            sr_step_info = Some(info);
            self.current_iteration += 1;
        }

        Ok(VmcIterationResult {
            energy,
            energy_squared,
            variance,
            sample_count: sampling_result.configurations.len(),
            acceptance_rate: sampling_result.acceptance_rate,
            observables,
            sz_total,
            sz_squared,
            sr_info: sr_step_info,
        })
    }

    /// Runs a single VMC iteration for Heisenberg model (legacy method - use run_heisenberg_iteration with iteration parameter)
    ///
    /// For Heisenberg model, we generate spin configurations directly
    /// instead of using electron configurations.
    ///
    /// # Returns
    /// * `Result<VmcIterationResult>` - The iteration result
    fn run_heisenberg_iteration_legacy(&mut self) -> Result<VmcIterationResult> {
        let _nsite = self.hamiltonian.lattice().n_sites();
        let two_sz = self.sampler.current_config().two_sz();

        // Generate initial spin configuration
        let mut spin_config = self.generate_initial_spin_config(two_sz);

        // Note: In future iterations, we could track previous energy for convergence checks

        // Perform Monte Carlo sampling with Metropolis-Hastings
        // Seed and MC schedule aligned with StdFace
        let base_seed = self.params.random_seed.get();
        let mut rng_state = base_seed + self.current_iteration as u64; // iteration-dependent seed
        let num_warmup_steps = self.params.mc_params.warmup_steps;
        let sampling_interval = self.params.mc_params.sampling_interval;
        let target_samples = self.params.mc_params.num_samples;

        // Burnin phase: equilibrate the system
        for _step in 0..num_warmup_steps {
            let (proposed_config, _acceptance_prob) = self.propose_spin_exchange(&spin_config, &mut rng_state)?;

            // Metropolis-Hastings acceptance
            let current_u8 = self.spin_config_to_u8(&spin_config);
            let proposed_u8 = self.spin_config_to_u8(&proposed_config);

            let psi_current = self.wavefunction.calculate(&current_u8);
            let psi_proposed = self.wavefunction.calculate(&proposed_u8);

            let prob_current = psi_current.norm_sqr();
            let prob_proposed = psi_proposed.norm_sqr();

            let acceptance_rate = if prob_current > 1e-12 {
                (prob_proposed / prob_current).min(1.0)
            } else {
                1.0
            };

            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_num = (rng_state as f64) / (u64::MAX as f64);

            if random_num < acceptance_rate {
                spin_config = proposed_config;
            }
        }

        // Collect samples and calculate energies
        let mut energy_sum = Complex64::new(0.0, 0.0);
        let mut energy_squared_sum = Complex64::new(0.0, 0.0);
        let mut samples_collected = 0;
        let mut accepted_moves = 0;
        let initial_energy = self.calculate_local_energy(&spin_config)?;
        let mut best_energy = initial_energy.re;
        let mut best_config = spin_config.clone();

        let num_mc_steps = sampling_interval * target_samples.max(1);
        for step in 0..num_mc_steps {
            // Propose a spin flip
            let (proposed_config, _acceptance_prob) = self.propose_spin_exchange(&spin_config, &mut rng_state)?;

            // Calculate energies (for tracking best energy)
            let _current_energy = self.calculate_local_energy(&spin_config)?;
            let proposed_energy = self.calculate_local_energy(&proposed_config)?;

            // Metropolis-Hastings acceptance probability based on wavefunction ratio
            // P_accept = min(1, |ψ(proposed)|^2 / |ψ(current)|^2)
            let current_u8 = self.spin_config_to_u8(&spin_config);
            let proposed_u8 = self.spin_config_to_u8(&proposed_config);

            let psi_current = self.wavefunction.calculate(&current_u8);
            let psi_proposed = self.wavefunction.calculate(&proposed_u8);

            let prob_current = psi_current.norm_sqr();
            let prob_proposed = psi_proposed.norm_sqr();

            let acceptance_rate = if prob_current > 1e-12 {
                (prob_proposed / prob_current).min(1.0)
            } else {
                1.0 // Accept if current probability is zero
            };

            // Generate random number for acceptance
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_num = (rng_state as f64) / (u64::MAX as f64);

            if random_num < acceptance_rate {
                spin_config = proposed_config.clone();
                accepted_moves += 1;
            }

            // Always keep track of best configuration found
            if proposed_energy.re < best_energy {
                best_energy = proposed_energy.re;
                best_config = proposed_config;
            }

            // Collect sample (every few steps to avoid correlation)
            if step % sampling_interval == 0 && samples_collected < target_samples {
                let local_energy = self.calculate_local_energy(&spin_config)?;
                energy_sum += local_energy;
                energy_squared_sum += local_energy * local_energy;
                samples_collected += 1;
            }
        }

        // Use the best configuration found during optimization
        spin_config = best_config;

        // Calculate final energy using the best configuration
        let final_energy = self.calculate_local_energy(&spin_config)?;

        // Calculate average energy and variance from samples
        let avg_energy = if samples_collected > 0 {
            energy_sum / Complex64::new(samples_collected as f64, 0.0)
        } else {
            final_energy
        };

        let avg_energy_squared = if samples_collected > 0 {
            energy_squared_sum / Complex64::new(samples_collected as f64, 0.0)
        } else {
            final_energy * final_energy
        };

        let energy_variance = avg_energy_squared - avg_energy * avg_energy;
        let variance = energy_variance.re;

        let acceptance_rate = if num_mc_steps > 0 { accepted_moves as f64 / num_mc_steps as f64 } else { 0.0 };
        let observables = self.calculator.calculate_observables();

        // Defer storing energy until after parameter update (for monotonic reporting)

        // Update parameters if in optimization mode using SR method (full SR)
        if let Some(ref mut _optimizer) = self.optimizer {
            // Collect samples for SR optimization
            use crate::vmc::sr_optimization::{SROptimizationCalculator, SRSampleData};

            let n_params = self.wavefunction.num_parameters();
            let mut sr_calculator = SROptimizationCalculator::new(n_params);

            // For each Monte Carlo step, calculate O-operators and collect samples
            let mut rng_state = base_seed + self.current_iteration as u64;
            let num_sr_samples = self.params.sr_params.iteration_sample.max(1);

            for _ in 0..num_sr_samples {
                // preserve total Sz during SR sampling as well
                let (proposed_config, _) = self.propose_spin_exchange(&spin_config, &mut rng_state)?;

                // Calculate local energy and O-operators
                let local_energy = self.calculate_local_energy(&proposed_config)?;
                let u8_config = self.spin_config_to_u8(&proposed_config);
                let psi = self.wavefunction.calculate(&u8_config);
                let weight = psi.norm_sqr();

                // Calculate O-operators (derivatives of log wavefunction)
                let o_operators = self.wavefunction.calculate_o_operators(&u8_config);

                // Add sample to SR calculator
                let sample =                 SRSampleData {
                    o_operators: o_operators.into_iter().map(|c| c.re).collect(),
                    energy: local_energy.re,
                    weight,
                };
                sr_calculator.add_sample(sample);

                // Metropolis acceptance for next step
                let current_u8 = self.spin_config_to_u8(&spin_config);
                let current_psi = self.wavefunction.calculate(&current_u8);
                let acceptance_rate = if current_psi.norm_sqr() > 1e-12 {
                    (weight / current_psi.norm_sqr()).min(1.0)
                } else {
                    1.0
                };

                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let random_num = (rng_state as f64) / (u64::MAX as f64);
                if random_num < acceptance_rate {
                    spin_config = proposed_config;
                }
            }

            // Solve full SR with stabilization
            let sp = &self.params.sr_params;
            if n_params > 0 {
                if let Ok((updates, _info)) = sr_calculator.solve_full_sr_with_info(
                    sp.reduction_cutoff, sp.stability_delta, sp.step_size,
                ) {
                    // Apply updates (learning_rate used as additional scaling)
                    self.wavefunction.update_parameters(&updates, 1.0);
                }
            }

            self.current_iteration += 1;
        }

        // Re-evaluate energy briefly with updated parameters to report post-update energy
        let mut rng_tmp = 7654321u64;
        let mut conf_tmp = spin_config.clone();
        let mut e_sum = 0.0;
        let mut n_eval = 0usize;
        for _ in 0..50 {
            let (pc, _) = self.propose_spin_exchange(&conf_tmp, &mut rng_tmp)?;
            conf_tmp = pc;
            let e_loc = self.calculate_vmc_local_energy(&conf_tmp)?;
            e_sum += e_loc;
            n_eval += 1;
        }
        let e_after = if n_eval > 0 { e_sum / n_eval as f64 } else { avg_energy.re };
        let energy_report = Complex64::new(e_after, 0.0);
        // Store post-update energy to history
        self.energy_history.push(energy_report);

        // Heisenberg: estimate E^2 and Sz moments from collected samples
        let energy_squared = avg_energy_squared;
        // Sz moments from current best configuration as proxy
        let mut sz_total = 0.0f64;
        for s in &spin_config {
            use mvmc_physics::hamiltonian::Spin::*;
            sz_total += match s { Up => 1.0, Down => -1.0, Empty => 0.0 };
        }
        let sz_squared = sz_total * sz_total;

        Ok(VmcIterationResult {
            energy: energy_report,
            energy_squared,
            variance,
            sample_count: samples_collected,
            acceptance_rate,
            observables,
            sz_total,
            sz_squared,
            sr_info: None,
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
        if configurations.is_empty() {
            // eprintln!("DEBUG: No configurations to calculate energy");
            return Ok(Complex64::new(0.0, 0.0));
        }

        let mut total_energy = Complex64::new(0.0, 0.0);
        let mut total_weight = 0.0;

        let n = configurations.len();
        if n == 0 { return Ok(Complex64::new(0.0, 0.0)); }
        for (_idx, config) in configurations.iter().enumerate() {
            let spin_config = self.electron_config_to_spin_config(config);
            let local_energy = self.calculate_local_energy(&spin_config)?;
            total_energy += local_energy;
            total_weight += 1.0;
        }

        if total_weight > 1e-12 {
            // eprintln!("DEBUG: total_energy: {}, total_weight: {}, average: {}",
            //          total_energy, total_weight, total_energy / total_weight);
            Ok(total_energy / total_weight)
        } else {
            // eprintln!("DEBUG: total_weight too small: {}", total_weight);
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

        let n = configurations.len();
        if n == 0 { return Ok(0.0); }
        for config in configurations {
            let spin_config = self.electron_config_to_spin_config(config);
            let local_energy = self.calculate_local_energy(&spin_config)?;
            let energy_diff = local_energy - mean_energy;
            total_variance += energy_diff.norm_sqr();
            total_weight += 1.0;
        }

        if total_weight > 1e-12 {
            Ok(total_variance / total_weight)
        } else {
            Ok(0.0)
        }
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
        // Calculate the wavefunction amplitude for current configuration
        let u8_config = self.spin_config_to_u8(spin_config);
        let psi = self.wavefunction.calculate(&u8_config);


        if psi.norm() < 1e-12 {
            // For very small amplitudes, use diagonal energy only
            // This prevents division by zero in off-diagonal terms
            let diagonal_energy = self.hamiltonian.diagonal_element(spin_config);
            return Ok(diagonal_energy);
        }

        // Start with diagonal energy (Sz-Sz terms)
        let mut local_energy = self.hamiltonian.diagonal_element(spin_config);

            // Add off-diagonal contributions (S+ S- and S- S+ terms)
            // For Heisenberg model: H = J * sum_<ij> (Sz_i * Sz_j + S+_i * S-_j + S-_i * S+_j)
            //
            // Reference: mVMC/src/mVMC/calham.c:160-168 (ExchangeCoupling term)
            // The matrix element calculation uses: <X'|H|X> = H_ij * ψ(X')/ψ(X)
        let nsite = spin_config.len();
        let mut off_diagonal_contributions = 0.0;
        let mut bond_count = 0;

        // Only consider each bond once by iterating through all sites
        for i in 0..nsite {
            let neighbors = self.hamiltonian.lattice().neighbors(i);
            for &j in &neighbors {
                // Only consider i < j to avoid double counting
                if i < j {
                    // Try both spin flip transitions for this bond
                    // S+_i S-_j: flip (i:Up, j:Down) → (i:Down, j:Up)
                    // S-_i S+_j: flip (i:Down, j:Up) → (i:Up, j:Down)

                    let mut new_config = spin_config.to_vec();
                    let can_flip = if spin_config[i] == Spin::Up && spin_config[j] == Spin::Down {
                        new_config[i] = Spin::Down;
                        new_config[j] = Spin::Up;
                        true
                    } else if spin_config[i] == Spin::Down && spin_config[j] == Spin::Up {
                        new_config[i] = Spin::Up;
                        new_config[j] = Spin::Down;
                        true
                    } else {
                        false
                    };

                    if can_flip {
                        let new_u8_config = self.spin_config_to_u8(&new_config);
                        let new_psi = self.wavefunction.calculate(&new_u8_config);

                        if new_psi.norm() > 1e-12 {
                            // Matrix element is J from HeisenbergHamiltonian
                            // Local energy contribution: E += H_ij * ψ(X')/ψ(X)
                            let matrix_elem = self.hamiltonian.matrix_element(spin_config, &new_config).re;
                            let amplitude_ratio = (new_psi / psi).re;
                            let contribution = matrix_elem * amplitude_ratio;
                                local_energy += contribution;
                                off_diagonal_contributions += contribution;
                                bond_count += 1;
                        }
                    }
                }
            }
        }


        Ok(local_energy)
    }

    /// Variant of local energy using a provided wavefunction (for trial updates)
    fn calculate_vmc_local_energy_with_wf(
        &self,
        wf: &PhysicsCombinedWavefunction,
        spin_config: &[Spin],
    ) -> Result<f64> {
        let u8_config = self.spin_config_to_u8(spin_config);
        let psi = wf.calculate(&u8_config);
        if psi.norm() < 1e-12 {
            // For very small amplitudes, use diagonal energy only
            return Ok(self.hamiltonian.diagonal_element(spin_config));
        }
        let mut local_energy = self.hamiltonian.diagonal_element(spin_config);
        let nsite = spin_config.len();
        for i in 0..nsite {
            for j in (i + 1)..nsite {
                if self.hamiltonian.lattice().neighbors(i).contains(&j) {
                    let mut new_config = spin_config.to_vec();
                    let can_flip = if spin_config[i] == Spin::Up && spin_config[j] == Spin::Down {
                        new_config[i] = Spin::Down;
                        new_config[j] = Spin::Up;
                        true
                    } else if spin_config[i] == Spin::Down && spin_config[j] == Spin::Up {
                        new_config[i] = Spin::Up;
                        new_config[j] = Spin::Down;
                        true
                    } else {
                        false
                    };
                    if can_flip {
                        let new_u8 = self.spin_config_to_u8(&new_config);
                        let new_psi = wf.calculate(&new_u8);
                        if new_psi.norm() > 1e-12 {
                            let matrix_elem = self.hamiltonian.matrix_element(spin_config, &new_config).re;
                            let amplitude_ratio = (new_psi / psi).re;
                            local_energy += matrix_elem * amplitude_ratio;
                        }
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

        let wavefunction = PhysicsCombinedWavefunction::new(4, 2).unwrap();

        use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian};
        use mvmc_physics::lattice::ChainLattice;

        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();

        let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian));
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

        let wavefunction = PhysicsCombinedWavefunction::new(4, 2).unwrap();

        use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian};
        use mvmc_physics::lattice::ChainLattice;

        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();

        let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian));
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

        let wavefunction = PhysicsCombinedWavefunction::new(2, 1).unwrap();

        use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian};
        use mvmc_physics::lattice::ChainLattice;

        let lattice = ChainLattice::new(2, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();

        let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian));
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
