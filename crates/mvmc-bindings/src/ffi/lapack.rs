//! LAPACK FFI bindings for numerical linear algebra
//!
//! This module provides raw FFI bindings to LAPACK functions.
//! For safe wrappers, see the `wrappers` module.

use libc::{c_char, c_int, c_double};
use num_complex::Complex64;

/// LAPACK error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapackError {
    /// Invalid input parameter
    InvalidInput(i32),
    /// Memory allocation failed
    MemoryAllocation(i32),
    /// Algorithm failed to converge
    NoConvergence(i32),
    /// Matrix is singular
    Singular(i32),
    /// Other error
    Other(i32),
}

impl LapackError {
    /// Convert LAPACK info code to error
    pub fn from_info(info: i32) -> Option<Self> {
        if info == 0 {
            None
        } else if info < 0 {
            Some(Self::InvalidInput(-info))
        } else {
            Some(Self::Other(info))
        }
    }
}

/// Check LAPACK info code and return error if non-zero
pub fn check_lapack_info(info: i32) -> Result<(), LapackError> {
    if let Some(err) = LapackError::from_info(info) {
        Err(err)
    } else {
        Ok(())
    }
}

// LAPACK function signatures
// Note: These are extern "C" declarations for LAPACK functions

unsafe extern "C" {
    /// ZGESV - Solve system of linear equations using LU decomposition
    ///
    /// # Safety
    /// This function is unsafe because it operates on raw pointers
    pub fn zgesv_(
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut Complex64,
        lda: *const c_int,
        ipiv: *mut c_int,
        b: *mut Complex64,
        ldb: *const c_int,
        info: *mut c_int,
    );

    /// ZGETRF - LU decomposition
    ///
    /// # Safety
    /// This function is unsafe because it operates on raw pointers
    pub fn zgetrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut Complex64,
        lda: *const c_int,
        ipiv: *mut c_int,
        info: *mut c_int,
    );

    /// ZGETRI - Matrix inversion using LU decomposition
    ///
    /// # Safety
    /// This function is unsafe because it operates on raw pointers
    pub fn zgetri_(
        n: *const c_int,
        a: *mut Complex64,
        lda: *const c_int,
        ipiv: *const c_int,
        work: *mut Complex64,
        lwork: *const c_int,
        info: *mut c_int,
    );

    /// ZGEEV - Eigenvalue decomposition
    ///
    /// # Safety
    /// This function is unsafe because it operates on raw pointers
    pub fn zgeev_(
        jobvl: *const c_char,
        jobvr: *const c_char,
        n: *const c_int,
        a: *mut Complex64,
        lda: *const c_int,
        w: *mut Complex64,
        vl: *mut Complex64,
        ldvl: *const c_int,
        vr: *mut Complex64,
        ldvr: *const c_int,
        work: *mut Complex64,
        lwork: *const c_int,
        rwork: *mut c_double,
        info: *mut c_int,
    );

    /// ZGESVD - Singular value decomposition
    ///
    /// # Safety
    /// This function is unsafe because it operates on raw pointers
    pub fn zgesvd_(
        jobu: *const c_char,
        jobvt: *const c_char,
        m: *const c_int,
        n: *const c_int,
        a: *mut Complex64,
        lda: *const c_int,
        s: *mut c_double,
        u: *mut Complex64,
        ldu: *const c_int,
        vt: *mut Complex64,
        ldvt: *const c_int,
        work: *mut Complex64,
        lwork: *const c_int,
        rwork: *mut c_double,
        info: *mut c_int,
    );

    /// DPOSV - Solve real symmetric positive definite system using Cholesky
    ///
    /// # Safety
    /// This function is unsafe because it operates on raw pointers
    pub fn dposv_(
        uplo: *const c_char,
        n: *const c_int,
        nrhs: *const c_int,
        a: *mut c_double,
        lda: *const c_int,
        b: *mut c_double,
        ldb: *const c_int,
        info: *mut c_int,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lapack_error_from_info() {
        assert_eq!(LapackError::from_info(0), None);
        assert_eq!(LapackError::from_info(-1), Some(LapackError::InvalidInput(1)));
        assert_eq!(LapackError::from_info(1), Some(LapackError::Other(1)));
    }

    #[test]
    fn test_check_lapack_info() {
        assert!(check_lapack_info(0).is_ok());
        assert!(check_lapack_info(-1).is_err());
        assert!(check_lapack_info(1).is_err());
    }

    #[test]
    fn test_lapack_error_display() {
        let err = LapackError::InvalidInput(5);
        assert_eq!(format!("{:?}", err), "InvalidInput(5)");
    }
}
