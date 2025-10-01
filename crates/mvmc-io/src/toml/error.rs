//! Error types for TOML parsing.

use thiserror::Error;

/// Error types for TOML parsing operations.
#[derive(Debug, Error)]
pub enum TomlError {
    #[error("TOML parse error: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("TOML serialize error: {0}")]
    SerializeError(#[from] toml::ser::Error),

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("File I/O error: {0}")]
    FileError(#[from] std::io::Error),
}

/// Result type for TOML operations.
pub type Result<T> = std::result::Result<T, TomlError>;
