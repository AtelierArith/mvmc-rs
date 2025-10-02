//! mVMC CLI - Command-line interface for many-variable Variational Monte Carlo
//!
//! This is the main executable for running VMC calculations from the command line.
//!
//! # Reference
//! Based on mVMC C implementation:
//! - `mVMC/src/mVMC/vmcmain.c` - Main program structure

mod commands;
mod error;

use clap::{Parser, Subcommand};
use error::{CliError, CliResult};
use std::path::PathBuf;

/// mVMC - Many-variable Variational Monte Carlo method
///
/// A numerical solver for quantum lattice models based on variational Monte Carlo.
#[derive(Parser, Debug)]
#[command(
    name = "mvmc",
    version,
    author,
    about = "Many-variable Variational Monte Carlo method",
    long_about = "A numerical solver package for quantum lattice models\n\
                  based on many-variable Variational Monte Carlo method.\n\n\
                  Supported models:\n  \
                  - Hubbard model\n  \
                  - Heisenberg model\n  \
                  - Kondo lattice model\n  \
                  - Multi-orbital Hubbard model\n\n\
                  Compatible with C implementation:\n  \
                  - Standard mode (-s)\n  \
                  - MultiDef mode (-m)\n  \
                  - OptTrans mode (-o)\n  \
                  - Binary mode (-b)\n  \
                  - Expert mode (-e)"
)]
struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode (suppress output)
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Binary mode (compatible with C implementation)
    #[arg(short = 'b', long, global = true)]
    binary: bool,

    /// MultiDef mode (compatible with C implementation)
    #[arg(short = 'm', long, global = true)]
    multidef: Option<usize>,

    /// OptTrans mode (compatible with C implementation)
    #[arg(long, global = true)]
    opttrans: bool,

    /// File flush interval (compatible with C implementation)
    #[arg(short = 'F', long, global = true)]
    flush_interval: Option<usize>,

    /// Expert mode (compatible with C implementation)
    #[arg(short = 'e', long, global = true)]
    expert: bool,

    /// Standard mode (compatible with C implementation)
    #[arg(short = 's', long, global = true)]
    standard: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a VMC calculation
    Run {
        /// Input configuration file (StdFace .def, TOML, or JSON)
        #[arg(value_name = "CONFIG")]
        config: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Number of threads (default: all available cores)
        #[arg(short = 'j', long)]
        threads: Option<usize>,
    },

    /// Standard mode - Generate mVMC input files from StdFace.def
    Standard {
        /// StdFace configuration file
        #[arg(value_name = "STDFACE_DEF")]
        config: PathBuf,

        /// Output directory for generated files
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },

    /// Show information about the configuration
    Info {
        /// Configuration file to inspect
        #[arg(value_name = "CONFIG")]
        config: PathBuf,
    },

    /// Validate a configuration file
    Validate {
        /// Configuration file to validate
        #[arg(value_name = "CONFIG")]
        config: PathBuf,
    },

    /// Show version information
    Version,
}

fn main() -> CliResult<()> {
    let cli = Cli::parse();

    // Setup logging
    setup_logging(cli.verbose, cli.quiet)?;

    log::info!("mVMC version {}", env!("CARGO_PKG_VERSION"));

    // Handle C implementation compatibility
    if cli.standard {
        println!("CLI: standard mode detected");
        // Standard mode: Generate input files from StdFace.def and run VMC
        match cli.command {
            Commands::Run { config, output, threads } => {
                println!("CLI: About to call standard::execute");
                // First generate input files using Standard mode
                commands::standard::execute(config.clone(), output.clone())?;
                println!("CLI: standard::execute completed");

                // Then run VMC calculation using the original StdFace.def file
                // (not the generated namelist.def)
                println!("CLI: About to call run::execute");
                commands::run::execute(config, output, cli.binary, threads)?;
                println!("CLI: run::execute completed");
            }
            _ => {
                return Err(CliError::Other(anyhow::anyhow!(
                    "Standard mode (-s) requires a Run command"
                )));
            }
        }
    } else {
        // Normal mode: Execute command as specified
        match cli.command {
            Commands::Run {
                config,
                output,
                threads,
            } => {
                commands::run::execute(config, output, cli.binary, threads)?;
            }
            Commands::Standard { config, output } => {
                commands::standard::execute(config, output)?;
            }
            Commands::Info { config } => {
                commands::info::execute(config)?;
            }
            Commands::Validate { config } => {
                commands::validate::execute(config)?;
            }
            Commands::Version => {
                commands::version::execute()?;
            }
        }
    }

    Ok(())
}

/// Sets up logging based on verbosity flags.
fn setup_logging(verbose: bool, quiet: bool) -> CliResult<()> {
    use env_logger::Builder;
    use log::LevelFilter;

    let level = if quiet {
        LevelFilter::Error
    } else if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    Builder::new()
        .filter_level(level)
        .format_timestamp(None)
        .format_module_path(false)
        .init();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test that CLI can be parsed (clap will validate at compile time mostly)
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
