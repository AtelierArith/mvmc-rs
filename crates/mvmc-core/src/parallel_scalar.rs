//! Scalar operations on the Julia-compatible communicator domains.
//!
//! Serial contexts return the input for every domain. MPI group contexts
//! select actual world, sampling-group, or cross-group communicators; they
//! never substitute a world reduction for an unavailable communicator.

use num_complex::Complex64;

/// Julia `comm0`, `comm1`, and `comm2` collective domains.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarCommunicator {
    /// All ranks in the world.
    World,
    /// Ranks in one sampling/QP group.
    Sampling,
    /// Equal local ranks across sampling/QP groups.
    CrossGroup,
}

/// Public scalar counterparts of Julia's scalar sum and signed maximum.
///
/// Every participating rank must call the same operation on the same domain.
/// MPI implementations require the initializing thread. A bare MPI world
/// lacks sampling/cross-group domains and returns an error for those domains;
/// construct a group context instead. There is no NULL-communicator identity.
pub trait ParallelScalarOperations {
    /// Sum one real scalar on the selected communicator.
    fn sum_real(&self, domain: ScalarCommunicator, value: f64) -> Result<f64, String>;
    /// Sum one complex scalar on the selected communicator.
    fn sum_complex(
        &self,
        domain: ScalarCommunicator,
        value: Complex64,
    ) -> Result<Complex64, String>;
    /// Maximum of signed C-compatible integers, including all-negative inputs.
    fn max_integer(&self, domain: ScalarCommunicator, value: i32) -> Result<i32, String>;
}

impl ParallelScalarOperations for crate::reducer::SingleProcessReducer {
    fn sum_real(&self, _: ScalarCommunicator, value: f64) -> Result<f64, String> {
        Ok(value)
    }

    fn sum_complex(&self, _: ScalarCommunicator, value: Complex64) -> Result<Complex64, String> {
        Ok(value)
    }

    fn max_integer(&self, _: ScalarCommunicator, value: i32) -> Result<i32, String> {
        Ok(value)
    }
}
