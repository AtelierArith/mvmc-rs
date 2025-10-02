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
use error::CliResult;
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
                  - Multi-orbital Hubbard model"
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

        /// Binary output format
        #[arg(short, long)]
        binary: bool,

        /// Number of threads (default: all available cores)
        #[arg(short = 'j', long)]
        threads: Option<usize>,
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

    match cli.command {
        Commands::Run {
            config,
            output,
            binary,
            threads,
        } => {
            commands::run::execute(config, output, binary, threads)?;
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
