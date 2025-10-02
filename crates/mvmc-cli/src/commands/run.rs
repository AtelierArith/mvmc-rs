//! Run command - Execute a VMC calculation.
//!
//! # Reference
//! Based on mVMC C implementation:
//! - `mVMC/src/mVMC/vmcmain.c` - Main calculation loop

use crate::error::{CliError, CliResult};
use colored::Colorize;
use mvmc_core::{
    vmc::{VmcEngine, VmcResult},
    config::VmcParameters,
    types::{SiteCount, ElectronCount, TwoSz, RandomSeed, CalcMode, LanczosMode},
    monte_carlo::SamplingStatistics,
};
use mvmc_io::{ConfigParser as _, OutputFormat, OutputManager, StdFaceParser, TomlParser, JsonParser};
use mvmc_physics::hamiltonian::{HubbardHamiltonian, Hamiltonian};
use mvmc_physics::lattice::{ChainLattice, SquareLattice};
use mvmc_physics::wavefunction::CombinedWavefunction;
use std::path::PathBuf;
use std::time::Instant;
use std::io::Write;
use num_complex::Complex64;

/// Executes a VMC calculation.
///
/// # Arguments
///
/// * `config` - Path to the configuration file
/// * `output_dir` - Output directory for results
/// * `binary` - Whether to use binary output format
/// * `threads` - Number of threads to use (None = all available)
pub fn execute(
    config: PathBuf,
    output_dir: PathBuf,
    binary: bool,
    threads: Option<usize>,
) -> CliResult<()> {
    let start_time = Instant::now();

    // Check if config file exists
    if !config.exists() {
        return Err(CliError::FileNotFound(config));
    }

    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("{}", "  mVMC - Variational Monte Carlo".cyan().bold());
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!();

    // Detect file format
    let format = mvmc_io::utils::detect_format(
        config.to_str().ok_or_else(|| {
            CliError::InvalidFormat(config.clone(), "Invalid path encoding".to_string())
        })?
    ).ok_or_else(|| {
        CliError::InvalidFormat(config.clone(), "Unknown file extension".to_string())
    })?;

    log::info!("Configuration file: {}", config.display());
    log::info!("Detected format: {}", format);
    log::info!("Output directory: {}", output_dir.display());

    // Parse configuration based on format
    let vmc_params = match format {
        "stdface" => {
            println!("📄 Reading StdFace configuration...");
            let parser = StdFaceParser::new();
            let cfg = parser.parse_file(config.to_str().unwrap())?;
            convert_stdface_to_vmc_params(&cfg)?
        }
        "toml" => {
            println!("📄 Reading TOML configuration...");
            let parser = TomlParser::new();
            let cfg = parser.parse_file(config.to_str().unwrap())?;
            convert_toml_to_vmc_params(&cfg)?
        }
        "json" => {
            println!("📄 Reading JSON configuration...");
            let parser = JsonParser::new();
            let cfg = parser.parse_file(config.to_str().unwrap())?;
            convert_json_to_vmc_params(&cfg)?
        }
        _ => {
            return Err(CliError::InvalidFormat(
                config,
                format!("Unsupported format: {}", format),
            ));
        }
    };

    println!("✓ Configuration loaded successfully");
    println!();

    // Set number of threads
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| CliError::Other(e.into()))?;
        log::info!("Using {} threads", n);
    } else {
        log::info!("Using all available cores");
    }

    // Setup output manager
    let output_format = if binary {
        OutputFormat::Binary
    } else {
        OutputFormat::Text
    };

    let output_manager = OutputManager::new(&output_dir, "zvo", 1);
    output_manager.ensure_output_dir()?;

    println!("⚙️  Initializing VMC calculation...");
    println!("   Model: {:?}", vmc_params.calc_mode);
    println!("   Sites: {}", vmc_params.nsite.get());
    println!("   Electrons: {}", vmc_params.ne.get());
    println!("   Spin: {}", vmc_params.two_sz.as_f64());

    // Initialize VMC engine
    println!("🔧 Creating VMC engine...");
    let wavefunction = CombinedWavefunction::new(vmc_params.nsite.get(), vmc_params.ne.get())
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create wavefunction: {}", e)))?;

    // Create Hamiltonian based on model type
    let hamiltonian = create_hamiltonian(&vmc_params)?;

    let mut engine = VmcEngine::new(vmc_params, wavefunction, hamiltonian)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to initialize VMC engine: {}", e)))?;

    println!("✓ VMC engine initialized successfully");
    println!();

    // Run VMC calculation based on mode
    match engine.params().calc_mode {
        CalcMode::Optimization => {
            println!("🚀 Starting VMC optimization...");
            let result = engine.run::<fn(usize, Complex64, &SamplingStatistics)>(None)
                .map_err(|e| CliError::Other(anyhow::anyhow!("VMC optimization failed: {}", e)))?;

            println!("✓ Optimization completed successfully");
            println!("   Final energy: {:.6}", result.energy.re);
            println!("   Final variance: {:.6}", result.energy_error);
            if let Some(opt) = &result.optimization {
                println!("   Steps: {}", opt.iterations);
                println!("   Converged: {}", opt.converged);
            }

            // Write results
            write_optimization_results(&output_manager, &result, output_format)?;
        }
        CalcMode::Expectation => {
            println!("📊 Starting VMC expectation value calculation...");

            // Create time output file for bin-by-bin data
            let time_file_path = output_manager.time_output_path();
            let mut time_file = std::fs::File::create(&time_file_path)
                .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create time file: {}", e)))?;

            // Define output callback for bin-by-bin data
            let mut sample_count = 0;
            let output_callback = |bin_idx: usize, _local_energy: Complex64, stats: &SamplingStatistics| {
                sample_count += 1;
                let timestamp = chrono::Local::now().format("%a %b %d %H:%M:%S %Y").to_string();
                let acc_hop = stats.acceptance_rate;
                let acc_ex = 0.0; // Placeholder for exchange acceptance rate
                let n_hop = stats.total_steps;
                let n_ex = 0; // Placeholder for exchange trials

                // Write to time file (format: sample_number acc_hop acc_ex n_hop n_ex : timestamp)
                writeln!(&mut time_file, "{:05} {:.6} {:.6} {} {} : {}",
                    bin_idx, acc_hop, acc_ex, n_hop, n_ex, timestamp)
                    .expect("Failed to write to time file");

                // Flush every 10 samples
                if sample_count % 10 == 0 {
                    let _ = time_file.flush();
                }
            };

            let result = engine.run(Some(output_callback))
                .map_err(|e| CliError::Other(anyhow::anyhow!("VMC expectation calculation failed: {}", e)))?;

            println!("✓ Expectation calculation completed successfully");
            println!("   Energy: {:.6}", result.energy.re);
            println!("   Variance: {:.6}", result.energy_error);
            println!("   ✓ Time data written to: {}", time_file_path.display());

            // Write results
            write_expectation_results(&output_manager, &result, output_format)?;
        }
    }

    let elapsed = start_time.elapsed();
    println!();
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("✓ {}", "Run completed successfully".green().bold());
    println!("  Time: {:.2?}", elapsed);
    println!("{}", "═══════════════════════════════════════════".cyan());

    Ok(())
}

/// Creates a Hamiltonian based on VMC parameters
///
/// # Arguments
///
/// * `params` - VMC parameters
///
/// # Returns
///
/// Boxed Hamiltonian trait object
fn create_hamiltonian(params: &VmcParameters) -> CliResult<Box<dyn Hamiltonian>> {
    let nsite = params.nsite.get();

    // For now, create a simple Hubbard model as default
    // In a full implementation, this would be determined by the configuration
    let hamiltonian = if nsite <= 16 {
        // Use 1D chain for small systems
        let lattice = ChainLattice::new(nsite, true)
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create chain lattice: {}", e)))?;
        HubbardHamiltonian::new(
            lattice,
            1.0,  // hopping parameter
            4.0,  // interaction parameter
            0.0   // chemical potential
        ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Hubbard Hamiltonian: {}", e)))?
    } else {
        // Use 2D square lattice for larger systems
        let l = (nsite as f64).sqrt() as usize;
        let lattice = SquareLattice::new(l, l, true)
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create square lattice: {}", e)))?;
        HubbardHamiltonian::new(
            lattice,
            1.0,  // hopping parameter
            4.0,  // interaction parameter
            0.0   // chemical potential
        ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Hubbard Hamiltonian: {}", e)))?
    };

    Ok(Box::new(hamiltonian))
}

/// Converts StdFace configuration to VMC parameters
fn convert_stdface_to_vmc_params(cfg: &mvmc_io::stdface::StdFaceConfig) -> CliResult<VmcParameters> {
    // Validate the configuration first
    cfg.validate()
        .map_err(|e| CliError::Other(anyhow::anyhow!("Invalid StdFace configuration: {}", e)))?;

    // Extract lattice information
    let nsite = SiteCount::new(cfg.total_sites());

    // Extract electron count from configuration (should be set by parser)
    let ne = ElectronCount::new(cfg.calculation.n_particles.unwrap_or(0));

    // Extract spin quantum number
    let two_sz = TwoSz::new(cfg.calculation.total_sz.unwrap_or(0));

    // Determine calculation mode
    let calc_mode = match cfg.monte_carlo.vmc_calculation_mode.unwrap_or(0) {
        0 => CalcMode::Optimization,
        1 => CalcMode::Expectation,
        _ => CalcMode::Optimization, // Default to optimization
    };

    // Extract random seed
    let random_seed = RandomSeed::new(cfg.calculation.random_seed.unwrap_or(123456789));

    // Create SR parameters from StdFace configuration
    let sr_params = mvmc_core::config::SRParameters::new(
        cfg.optimization.sr_steps.unwrap_or(1000),
        cfg.optimization.sr_steps.unwrap_or(1000) / 10, // iteration_sample = 10% of total steps
        1, // fixed_sample_steps
        cfg.optimization.sr_reduction_cutoff.unwrap_or(1e-8),
        cfg.optimization.sr_stabilization_delta.unwrap_or(1e-2),
        cfg.optimization.sr_step_delta.unwrap_or(3e-3),
        1000, // cg_max_iterations
        1e-10, // cg_tolerance
    );

    // Create Monte Carlo parameters from StdFace configuration
    let mc_params = mvmc_core::config::MonteCarloParameters::new(
        100, // warmup_steps
        1, // sampling_interval
        cfg.monte_carlo.vmc_samples.unwrap_or(1000),
        false, // exchange_update
        1, // block_update_size
    );

    // Build VMC parameters
    Ok(VmcParameters::builder()
        .nsite(nsite)
        .ne(ne)
        .two_sz(two_sz)
        .calc_mode(calc_mode)
        .lanczos_mode(LanczosMode::None)
        .random_seed(random_seed)
        .sr_params(sr_params)
        .mc_params(mc_params)
        .build()
        .map_err(|e| CliError::Other(anyhow::anyhow!("Invalid VMC parameters: {}", e)))?)
}

/// Converts TOML configuration to VMC parameters
fn convert_toml_to_vmc_params(cfg: &mvmc_io::toml::TomlConfig) -> CliResult<VmcParameters> {
    let nsite = SiteCount::new(cfg.lattice.dimensions[0]);
    let ne = ElectronCount::new(4); // Default value
    let two_sz = TwoSz::new(0); // Default value
    let calc_mode = CalcMode::Optimization; // Default to optimization
    let lanczos_mode = LanczosMode::None;
    let random_seed = RandomSeed::new(12345); // Default value

    Ok(VmcParameters::builder()
        .nsite(nsite)
        .ne(ne)
        .two_sz(two_sz)
        .calc_mode(calc_mode)
        .lanczos_mode(lanczos_mode)
        .random_seed(random_seed)
        .build()
        .map_err(|e| CliError::Other(anyhow::anyhow!("Invalid VMC parameters: {}", e)))?)
}

/// Converts JSON configuration to VMC parameters
fn convert_json_to_vmc_params(cfg: &mvmc_io::json::JsonConfig) -> CliResult<VmcParameters> {
    let nsite = SiteCount::new(cfg.lattice.dimensions[0]);
    let ne = ElectronCount::new(4); // Default value
    let two_sz = TwoSz::new(0); // Default value
    let calc_mode = CalcMode::Optimization; // Default to optimization
    let lanczos_mode = LanczosMode::None;
    let random_seed = RandomSeed::new(12345); // Default value

    Ok(VmcParameters::builder()
        .nsite(nsite)
        .ne(ne)
        .two_sz(two_sz)
        .calc_mode(calc_mode)
        .lanczos_mode(lanczos_mode)
        .random_seed(random_seed)
        .build()
        .map_err(|e| CliError::Other(anyhow::anyhow!("Invalid VMC parameters: {}", e)))?)
}

/// Writes optimization results to output files
fn write_optimization_results(
    output_manager: &OutputManager,
    result: &VmcResult,
    output_format: OutputFormat,
) -> CliResult<()> {
    use mvmc_io::EnergyData;
    use num_complex::Complex64;

    // Write energy data
    let energy_data = EnergyData::new(
        result.energy,
        Complex64::new(result.energy_error, 0.0),
        Complex64::new(0.0, 0.0), // Sz total
        Complex64::new(0.0, 0.0), // Sz squared
    );

    output_manager.write_energy(&energy_data, output_format)?;
    println!("   ✓ Energy data written to: {}", output_manager.energy_output_path().display());

    // Write variational data if optimization was performed
    if let Some(_opt) = &result.optimization {
        use mvmc_io::VariationalData;
        let variational_data = VariationalData::new(
            energy_data,
            vec![], // TODO: Store actual parameters
        );

        output_manager.write_variational(&variational_data, output_format)?;
        println!("   ✓ Variational data written to: {}", output_manager.variational_output_path().display());
    }

    // Write optimized parameters
    use mvmc_io::OptimizedParameters;
    let optimized_params = OptimizedParameters::new(vec![]); // TODO: Store actual parameters
    output_manager.write_optimized_params(&optimized_params)?;
    println!("   ✓ Optimized parameters written to: {}", output_manager.optimized_params_path().display());

    Ok(())
}

/// Writes expectation results to output files
fn write_expectation_results(
    output_manager: &OutputManager,
    result: &VmcResult,
    output_format: OutputFormat,
) -> CliResult<()> {
    use mvmc_io::EnergyData;
    use num_complex::Complex64;

    // Write energy data
    let energy_data = EnergyData::new(
        result.energy,
        Complex64::new(result.energy_error, 0.0),
        Complex64::new(0.0, 0.0), // Sz total
        Complex64::new(0.0, 0.0), // Sz squared
    );

    output_manager.write_energy(&energy_data, output_format)?;
    println!("   ✓ Energy data written to: {}", output_manager.energy_output_path().display());

    // Write variational data
    use mvmc_io::VariationalData;
    let variational_data = VariationalData::new(
        energy_data,
        vec![], // No parameters for expectation calculation
    );

    output_manager.write_variational(&variational_data, output_format)?;
    println!("   ✓ Variational data written to: {}", output_manager.variational_output_path().display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_execute_with_nonexistent_file() {
        let config = PathBuf::from("nonexistent.def");
        let output = PathBuf::from("output");

        let result = execute(config.clone(), output, false, None);
        assert!(result.is_err());

        if let Err(CliError::FileNotFound(path)) = result {
            assert_eq!(path, config);
        } else {
            panic!("Expected FileNotFound error");
        }
    }

    #[test]
    fn test_execute_with_invalid_extension() {
        // Create a temp file with invalid extension
        let temp_dir = std::env::temp_dir();
        let config = temp_dir.join("test.xyz");
        fs::write(&config, "dummy content").unwrap();

        let output = PathBuf::from("output");
        let result = execute(config.clone(), output, false, None);

        assert!(result.is_err());

        // Clean up
        let _ = fs::remove_file(&config);
    }

    #[test]
    fn test_convert_stdface_to_vmc_params_hubbard() {
        use mvmc_io::stdface::StdFaceConfig;

        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![6];
        config.lattice.sub_dimensions = vec![2];
        config.model.model_type = "Hubbard".to_string();
        config.model.parameters.insert("t".to_string(), 1.0);
        config.model.parameters.insert("U".to_string(), 4.0);
        config.calculation.n_particles = Some(6);
        config.calculation.total_sz = Some(0);
        config.calculation.random_seed = Some(12345);

        let result = convert_stdface_to_vmc_params(&config);
        assert!(result.is_ok());

        let vmc_params = result.unwrap();
        assert_eq!(vmc_params.nsite.get(), 6);
        assert_eq!(vmc_params.ne.get(), 6);
        assert_eq!(vmc_params.two_sz.get(), 0);
    }

    #[test]
    fn test_convert_stdface_to_vmc_params_spin() {
        use mvmc_io::stdface::StdFaceConfig;

        let mut config = StdFaceConfig::new();
        config.lattice.dimensions = vec![8];
        config.lattice.sub_dimensions = vec![2];
        config.model.model_type = "Spin".to_string();
        config.model.parameters.insert("J".to_string(), 1.0);
        config.calculation.n_particles = Some(0);
        config.calculation.total_sz = Some(0);
        config.calculation.random_seed = Some(54321);

        let result = convert_stdface_to_vmc_params(&config);
        assert!(result.is_ok());

        let vmc_params = result.unwrap();
        assert_eq!(vmc_params.nsite.get(), 8);
        assert_eq!(vmc_params.ne.get(), 0);
        assert_eq!(vmc_params.two_sz.get(), 0);
    }
}
