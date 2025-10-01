//! Error types for mVMC core library
//!
//! This module defines error types for VMC calculations.

use std::fmt;
use thiserror::Error;

/// Result type for mVMC operations
pub type Result<T> = std::result::Result<T, VmcError>;

/// Errors that can occur during VMC calculations
#[derive(Debug, Error)]
pub enum VmcError {
    /// Invalid parameter value
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Index out of bounds
    #[error("Index out of bounds: {index} >= {limit}")]
    IndexOutOfBounds { index: usize, limit: usize },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Matrix is singular or near-singular
    #[error("Matrix is singular (determinant ≈ 0)")]
    SingularMatrix,

    /// Convergence failure
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Numerical error (overflow, underflow, NaN, etc.)
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

impl VmcError {
    /// Creates an invalid parameter error
    pub fn invalid_param(msg: impl fmt::Display) -> Self {
        VmcError::InvalidParameter(msg.to_string())
    }

    /// Creates an index out of bounds error
    pub fn out_of_bounds(index: usize, limit: usize) -> Self {
        VmcError::IndexOutOfBounds { index, limit }
    }

    /// Creates an invalid configuration error
    pub fn invalid_config(msg: impl fmt::Display) -> Self {
        VmcError::InvalidConfiguration(msg.to_string())
    }

    /// Creates a dimension mismatch error
    pub fn dim_mismatch(expected: usize, actual: usize) -> Self {
        VmcError::DimensionMismatch { expected, actual }
    }

    /// Creates a convergence failure error
    pub fn no_convergence(iterations: usize) -> Self {
        VmcError::ConvergenceFailure { iterations }
    }

    /// Creates a numerical error
    pub fn numerical(msg: impl fmt::Display) -> Self {
        VmcError::NumericalError(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_parameter() {
        let err = VmcError::invalid_param("test parameter");
        assert_eq!(err.to_string(), "Invalid parameter: test parameter");
    }

    #[test]
    fn test_index_out_of_bounds() {
        let err = VmcError::out_of_bounds(10, 5);
        assert_eq!(err.to_string(), "Index out of bounds: 10 >= 5");
    }

    #[test]
    fn test_invalid_configuration() {
        let err = VmcError::invalid_config("bad config");
        assert_eq!(err.to_string(), "Invalid configuration: bad config");
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = VmcError::dim_mismatch(10, 5);
        assert_eq!(err.to_string(), "Dimension mismatch: expected 10, got 5");
    }

    #[test]
    fn test_singular_matrix() {
        let err = VmcError::SingularMatrix;
        assert_eq!(err.to_string(), "Matrix is singular (determinant ≈ 0)");
    }

    #[test]
    fn test_convergence_failure() {
        let err = VmcError::no_convergence(100);
        assert_eq!(err.to_string(), "Failed to converge after 100 iterations");
    }

    #[test]
    fn test_numerical_error() {
        let err = VmcError::numerical("NaN encountered");
        assert_eq!(err.to_string(), "Numerical error: NaN encountered");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VmcError>();
    }

    #[test]
    fn test_result_type() {
        let ok_result: Result<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: Result<i32> = Err(VmcError::invalid_param("test"));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_conversion() {
        // Test that we can create errors using the convenience methods
        let _err1 = VmcError::invalid_param(format!("n={}", 5));
        let _err2 = VmcError::out_of_bounds(10, 20);
        let _err3 = VmcError::invalid_config("test");
        let _err4 = VmcError::dim_mismatch(3, 4);
        let _err5 = VmcError::no_convergence(1000);
        let _err6 = VmcError::numerical("overflow");
    }

    #[test]
    fn test_error_in_function() {
        fn check_index(idx: usize, limit: usize) -> Result<usize> {
            if idx >= limit {
                return Err(VmcError::out_of_bounds(idx, limit));
            }
            Ok(idx)
        }

        assert!(check_index(5, 10).is_ok());
        assert!(check_index(10, 10).is_err());
        assert!(check_index(15, 10).is_err());
    }
}
