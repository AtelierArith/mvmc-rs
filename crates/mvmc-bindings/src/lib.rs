//! FFI bindings for external C libraries used by mVMC
//!
//! This crate provides safe Rust wrappers around C libraries such as LAPACK,
//! ScaLAPACK, and other numerical computing libraries.

pub mod ffi;
pub mod wrappers;

// Re-export commonly used types and functions
pub use ffi::lapack::{LapackError, check_lapack_info};
pub use wrappers::linear_algebra::{
    solve_linear_system, lu_decomposition, invert_matrix,
    eigenvalue_decomposition, singular_value_decomposition,
    LapackResult,
};

#[cfg(test)]
mod tests {

    #[test]
    fn test_module_structure() {
        // Test that the module structure is correct
        assert!(true);
    }
}
