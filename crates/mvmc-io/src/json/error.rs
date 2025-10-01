//! Error types for JSON parsing.

use thiserror::Error;

/// Error types for JSON parsing operations.
#[derive(Debug, Error)]
pub enum JsonError {
    #[error("JSON parse error: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("File I/O error: {0}")]
    FileError(#[from] std::io::Error),
}

/// Result type for JSON operations.
pub type Result<T> = std::result::Result<T, JsonError>;
