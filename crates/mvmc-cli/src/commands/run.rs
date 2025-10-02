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
    wavefunction::CombinedWavefunction,
};
use mvmc_io::{ConfigParser as _, OutputFormat, OutputManager, StdFaceParser, TomlParser, JsonParser};
use std::path::PathBuf;
use std::time::Instant;

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

    let base_name = mvmc_io::utils::base_name(config.to_str().unwrap());
    let output_manager = OutputManager::new(&output_dir, &base_name);
    output_manager.ensure_output_dir()?;

    println!("⚙️  Initializing VMC calculation...");
    println!("   Model: {:?}", vmc_params.calc_mode);
    println!("   Sites: {}", vmc_params.nsite.get());
    println!("   Electrons: {}", vmc_params.ne.get());
    println!("   Spin: {}", vmc_params.two_sz.as_f64());

    // Initialize VMC engine
    println!("🔧 Creating VMC engine...");
    let wavefunction = CombinedWavefunction::new(vmc_params.nsite, vmc_params.ne);
    let mut engine = VmcEngine::new(vmc_params, wavefunction)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to initialize VMC engine: {}", e)))?;

    println!("✓ VMC engine initialized successfully");
    println!();

    // Run VMC calculation based on mode
    match engine.params().calc_mode {
        CalcMode::Optimization => {
            println!("🚀 Starting VMC optimization...");
            let result = engine.run()
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
            let result = engine.run()
                .map_err(|e| CliError::Other(anyhow::anyhow!("VMC expectation calculation failed: {}", e)))?;

            println!("✓ Expectation calculation completed successfully");
            println!("   Energy: {:.6}", result.energy.re);
            println!("   Variance: {:.6}", result.energy_error);

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

/// Converts StdFace configuration to VMC parameters
fn convert_stdface_to_vmc_params(cfg: &mvmc_io::stdface::StdFaceConfig) -> CliResult<VmcParameters> {
    let nsite = SiteCount::new(cfg.lattice.dimensions[0]);
    let ne = ElectronCount::new(4); // Default value
    let two_sz = TwoSz::new(0); // Default value
    let calc_mode = CalcMode::Optimization; // Default value
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
}
