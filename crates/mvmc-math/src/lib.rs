//! Mathematical utilities for mVMC
//!
//! This crate provides numerical computing primitives for variational Monte Carlo calculations:
//! - Random number generation
//! - Linear algebra operations
//! - Complex number utilities
//! - Matrix decompositions

pub mod complex;
pub mod linear_algebra;
pub mod random;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
