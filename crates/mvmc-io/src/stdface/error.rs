//! Error types for StdFace parsing.

use thiserror::Error;

/// Error types for StdFace parsing operations.
#[derive(Debug, Error)]
pub enum StdFaceError {
    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Missing required parameter: {parameter}")]
    MissingParameter { parameter: String },

    #[error("Invalid parameter value: {parameter} = {value}")]
    InvalidParameterValue { parameter: String, value: String },

    #[error("Unsupported model: {model}")]
    UnsupportedModel { model: String },

    #[error("Unsupported lattice: {lattice}")]
    UnsupportedLattice { lattice: String },

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("File I/O error: {0}")]
    FileError(#[from] std::io::Error),
}

/// Result type for StdFace operations.
pub type Result<T> = std::result::Result<T, StdFaceError>;
