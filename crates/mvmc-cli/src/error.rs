//! Error handling for the CLI.

use std::path::PathBuf;
use thiserror::Error;

/// Result type for CLI operations.
pub type CliResult<T> = Result<T, CliError>;

/// Errors that can occur during CLI operations.
#[derive(Error, Debug)]
pub enum CliError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration file error
    #[error("Configuration error: {0}")]
    Config(String),

    /// File not found
    #[error("File not found: {}", .0.display())]
    FileNotFound(PathBuf),

    /// Invalid file format
    #[error("Invalid file format for {}: {1}", .0.display())]
    InvalidFormat(PathBuf, String),

    /// VMC calculation error
    #[error("VMC calculation error: {0}")]
    VmcError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Any other error
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<mvmc_io::IoError> for CliError {
    fn from(err: mvmc_io::IoError) -> Self {
        CliError::Config(err.to_string())
    }
}

impl From<mvmc_io::StdFaceError> for CliError {
    fn from(err: mvmc_io::StdFaceError) -> Self {
        CliError::Config(err.to_string())
    }
}

impl From<mvmc_io::TomlError> for CliError {
    fn from(err: mvmc_io::TomlError) -> Self {
        CliError::Config(err.to_string())
    }
}

impl From<mvmc_io::JsonError> for CliError {
    fn from(err: mvmc_io::JsonError) -> Self {
        CliError::Config(err.to_string())
    }
}

impl From<mvmc_core::VmcError> for CliError {
    fn from(err: mvmc_core::VmcError) -> Self {
        CliError::VmcError(err.to_string())
    }
}
