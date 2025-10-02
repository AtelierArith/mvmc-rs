//! Info command - Display information about a configuration file.

use crate::error::{CliError, CliResult};
use colored::Colorize;
use mvmc_io::{ConfigParser as _, StdFaceParser, TomlParser, JsonParser};
use std::path::PathBuf;

/// Displays information about a configuration file.
///
/// # Arguments
///
/// * `config` - Path to the configuration file
pub fn execute(config: PathBuf) -> CliResult<()> {
    // Check if config file exists
    if !config.exists() {
        return Err(CliError::FileNotFound(config));
    }

    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("{}", "  Configuration Information".cyan().bold());
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!();

    println!("📄 File: {}", config.display());
    println!();

    // Detect file format
    let format = mvmc_io::utils::detect_format(
        config.to_str().ok_or_else(|| {
            CliError::InvalidFormat(config.clone(), "Invalid path encoding".to_string())
        })?
    ).ok_or_else(|| {
        CliError::InvalidFormat(config.clone(), "Unknown file extension".to_string())
    })?;

    println!("Format: {}", format);
    println!();

    // Parse configuration based on format and display details
    match format {
        "stdface" => {
            let parser = StdFaceParser::new();
            let config_data = parser.parse_file(config.to_str().unwrap())?;

            println!("{}", "Model Configuration:".bold());
            println!("  Model type:       {}", config_data.model.model_type);
            println!("  Lattice type:     {}", config_data.lattice.lattice_type);
            println!("  Dimensions:       {:?}", config_data.lattice.dimensions);
            println!();

            if !config_data.model.parameters.is_empty() {
                println!("{}", "Model Parameters:".bold());
                for (key, value) in &config_data.model.parameters {
                    println!("  {}: {}", key, value);
                }
                println!();
            }
        }
        "toml" => {
            let parser = TomlParser::new();
            let config_data = parser.parse_file(config.to_str().unwrap())?;

            println!("{}", "Model Configuration:".bold());
            println!("  Model type:       {}", config_data.model.model_type);
            println!("  Lattice type:     {}", config_data.lattice.lattice_type);
            println!("  Dimensions:       {:?}", config_data.lattice.dimensions);
            println!();
        }
        "json" => {
            let parser = JsonParser::new();
            let config_data = parser.parse_file(config.to_str().unwrap())?;

            println!("{}", "Model Configuration:".bold());
            println!("  Model type:       {}", config_data.model.model_type);
            println!("  Lattice type:     {}", config_data.lattice.lattice_type);
            println!("  Dimensions:       {:?}", config_data.lattice.dimensions);
            println!();
        }
        _ => {
            return Err(CliError::InvalidFormat(
                config,
                format!("Unsupported format: {}", format),
            ));
        }
    }

    println!("{}", "═══════════════════════════════════════════".cyan());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info_with_nonexistent_file() {
        let config = PathBuf::from("nonexistent.def");
        let result = execute(config.clone());

        assert!(result.is_err());
        if let Err(CliError::FileNotFound(path)) = result {
            assert_eq!(path, config);
        } else {
            panic!("Expected FileNotFound error");
        }
    }
}
