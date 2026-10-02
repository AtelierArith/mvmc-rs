//! Optional MPI lifecycle and reducer implementation.
//!
//! The feature is intentionally opt-in because linking `rsmpi` requires an
//! MPI installation. The [`MpiContext`] owns the `Universe`, so MPI is
//! finalized when the context is dropped and never through a process-global
//! `MPI_Finalize` call owned by the library.

use num_complex::Complex64;

use ::mpi::traits::*;

use crate::reducer::Reducer;

/// An initialized MPI world and its root-owned lifecycle token.
pub struct MpiContext {
    universe: ::mpi::environment::Universe,
    world: ::mpi::topology::SimpleCommunicator,
}

impl std::fmt::Debug for MpiContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MpiContext")
            .field("rank", &self.rank())
            .field("world_size", &self.world_size())
            .finish()
    }
}

impl MpiContext {
    /// Initialize MPI and retain ownership of its finalization token.
    pub fn initialize() -> Result<Self, String> {
        let universe = ::mpi::initialize()
            .ok_or_else(|| "MPI was already initialized or could not be initialized".to_string())?;
        let world = universe.world();
        Ok(Self { universe, world })
    }

    /// Zero-based rank in `MPI_COMM_WORLD`.
    pub fn rank(&self) -> usize {
        self.world
            .rank()
            .try_into()
            .expect("MPI rank is nonnegative")
    }

    /// Number of ranks in `MPI_COMM_WORLD`.
    pub fn world_size(&self) -> usize {
        self.world
            .size()
            .try_into()
            .expect("MPI world size is positive")
    }

    /// True on the output rank.
    pub fn is_root(&self) -> bool {
        self.rank() == 0
    }

    /// Broadcast a real parameter buffer from `root`.
    pub fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
        let root = i32::try_from(root).map_err(|_| "MPI root rank is too large".to_string())?;
        self.world.process_at_rank(root).broadcast_into(values);
        Ok(())
    }

    /// Broadcast an integer parameter buffer from `root`.
    pub fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        let root = i32::try_from(root).map_err(|_| "MPI root rank is too large".to_string())?;
        self.world.process_at_rank(root).broadcast_into(values);
        Ok(())
    }

    /// Borrow the underlying communicator for grouped execution.
    pub fn world(&self) -> &::mpi::topology::SimpleCommunicator {
        &self.world
    }

    /// Keep the lifecycle token observable to callers that need to enforce
    /// drop order; MPI finalization still happens through `Drop` of Universe.
    pub fn universe(&self) -> &::mpi::environment::Universe {
        &self.universe
    }
}

impl Reducer for MpiContext {
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        use ::mpi::collective::SystemOperation;
        let mut reduced = vec![0.0; values.len()];
        self.world
            .all_reduce_into(values, &mut reduced, SystemOperation::sum());
        values.copy_from_slice(&reduced);
    }

    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        use ::mpi::collective::SystemOperation;
        let real: Vec<_> = values.iter().map(|value| value.re).collect();
        let imag: Vec<_> = values.iter().map(|value| value.im).collect();
        let mut reduced_real = vec![0.0; values.len()];
        let mut reduced_imag = vec![0.0; values.len()];
        self.world
            .all_reduce_into(&real, &mut reduced_real, SystemOperation::sum());
        self.world
            .all_reduce_into(&imag, &mut reduced_imag, SystemOperation::sum());
        for (value, (real, imag)) in values
            .iter_mut()
            .zip(reduced_real.into_iter().zip(reduced_imag))
        {
            *value = Complex64::new(real, imag);
        }
    }

    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        use ::mpi::collective::SystemOperation;
        let mut reduced = vec![0_i64; values.len()];
        self.world
            .all_reduce_into(values, &mut reduced, SystemOperation::sum());
        values.copy_from_slice(&reduced);
    }

    fn world_size(&self) -> usize {
        Self::world_size(self)
    }

    fn rank(&self) -> usize {
        Self::rank(self)
    }
}
