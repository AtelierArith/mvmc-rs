//! Validate command - Validate a configuration file.

use crate::error::{CliError, CliResult};
use colored::Colorize;
use mvmc_io::{ConfigParser as _, StdFaceParser, TomlParser, JsonParser};
use std::path::PathBuf;

/// Validates a configuration file.
///
/// # Arguments
///
/// * `config` - Path to the configuration file to validate
pub fn execute(config: PathBuf) -> CliResult<()> {
    // Check if config file exists
    if !config.exists() {
        return Err(CliError::FileNotFound(config));
    }

    println!("🔍 Validating configuration: {}", config.display());
    println!();

    // Detect file format
    let format = mvmc_io::utils::detect_format(
        config.to_str().ok_or_else(|| {
            CliError::InvalidFormat(config.clone(), "Invalid path encoding".to_string())
        })?
    ).ok_or_else(|| {
        CliError::InvalidFormat(config.clone(), "Unknown file extension".to_string())
    })?;

    log::debug!("Detected format: {}", format);

    // Parse configuration based on format
    match format {
        "stdface" => {
            let parser = StdFaceParser::new();
            match parser.parse_file(config.to_str().unwrap()) {
                Ok(config_data) => {
                    println!("✓ {}", "Configuration is valid".green());
                    println!();

                    println!("{}", "Summary:".bold());
                    println!("  Model:    {}", config_data.model.model_type);
                    println!("  Lattice:  {}", config_data.lattice.lattice_type);
                }
                Err(e) => {
                    println!("✗ {}", "Configuration is invalid".red());
                    println!();
                    println!("Error: {}", e);
                    return Err(e.into());
                }
            }
        }
        "toml" => {
            let parser = TomlParser::new();
            match parser.parse_file(config.to_str().unwrap()) {
                Ok(config_data) => {
                    println!("✓ {}", "Configuration is valid".green());
                    println!();
                    println!("{}", "Summary:".bold());
                    println!("  Model:    {}", config_data.model.model_type);
                    println!("  Lattice:  {}", config_data.lattice.lattice_type);
                }
                Err(e) => {
                    println!("✗ {}", "Configuration is invalid".red());
                    println!();
                    println!("Error: {}", e);
                    return Err(e.into());
                }
            }
        }
        "json" => {
            let parser = JsonParser::new();
            match parser.parse_file(config.to_str().unwrap()) {
                Ok(config_data) => {
                    println!("✓ {}", "Configuration is valid".green());
                    println!();
                    println!("{}", "Summary:".bold());
                    println!("  Model:    {}", config_data.model.model_type);
                    println!("  Lattice:  {}", config_data.lattice.lattice_type);
                }
                Err(e) => {
                    println!("✗ {}", "Configuration is invalid".red());
                    println!();
                    println!("Error: {}", e);
                    return Err(e.into());
                }
            }
        }
        _ => {
            return Err(CliError::InvalidFormat(
                config,
                format!("Unsupported format: {}", format),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_validate_with_nonexistent_file() {
        let config = PathBuf::from("nonexistent.def");
        let result = execute(config.clone());

        assert!(result.is_err());
        if let Err(CliError::FileNotFound(path)) = result {
            assert_eq!(path, config);
        } else {
            panic!("Expected FileNotFound error");
        }
    }

    #[test]
    fn test_validate_with_invalid_format() {
        let temp_dir = std::env::temp_dir();
        let config = temp_dir.join("test_invalid.xyz");
        fs::write(&config, "dummy content").unwrap();

        let result = execute(config.clone());
        assert!(result.is_err());

        // Clean up
        let _ = fs::remove_file(&config);
    }
}
