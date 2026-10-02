//! Optional MPI lifecycle and reducer implementation.
//!
//! The feature is intentionally opt-in because linking `rsmpi` requires an
//! MPI installation. The [`MpiContext`] owns the `Universe`, so MPI is
//! finalized when the context is dropped and never through a process-global
//! `MPI_Finalize` call owned by the library.

use num_complex::Complex64;

use ::mpi::topology::{Color, Key};
use ::mpi::traits::*;

use crate::parallel::{assign_group, GroupAssignment, LaunchContext};
use crate::reducer::Reducer;

/// Keep every MPI collective below implementation-specific message-count
/// limits.  Julia reduces its large observable buffers in bounded chunks; the
/// same boundary is required for QQQQ/Green buffers on larger systems.
const MPI_REDUCTION_CHUNK_LEN: usize = 1 << 20;

/// An initialized MPI world and its root-owned lifecycle token.
pub struct MpiContext {
    universe: ::mpi::environment::Universe,
    world: ::mpi::topology::SimpleCommunicator,
}

/// MPI communicator for one Julia-compatible `NSplitSize` group.
///
/// The parent [`MpiContext`] must outlive this value because MPI finalization
/// is owned by the parent `Universe`. All reductions performed by this type
/// are confined to the group communicator (`comm1` in the Julia runner).
pub struct MpiGroupContext {
    communicator: ::mpi::topology::SimpleCommunicator,
    /// Cross-group communicator (`comm2` in Julia/C), connecting equal local
    /// ranks from every QP/sample group.
    cross_communicator: ::mpi::topology::SimpleCommunicator,
    assignment: GroupAssignment,
}

impl std::fmt::Debug for MpiGroupContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MpiGroupContext")
            .field("group", &self.assignment.group)
            .field("rank", &self.rank())
            .field("world_size", &self.world_size())
            .finish()
    }
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

    /// Split `MPI_COMM_WORLD` into `nsplit` contiguous groups.
    pub fn split_groups(&self, nsplit: usize) -> Result<MpiGroupContext, String> {
        let assignment = assign_group(
            LaunchContext {
                rank: self.rank(),
                world_size: self.world_size(),
            },
            nsplit,
        )?;
        let color = Color::with_value(i32::try_from(assignment.group).map_err(|_| {
            format!(
                "MPI group index {} does not fit in an MPI color",
                assignment.group
            )
        })?);
        let key = Key::with_value(i32::try_from(assignment.local_rank).map_err(|_| {
            format!(
                "MPI local rank {} does not fit in an MPI key",
                assignment.local_rank
            )
        })?);
        let communicator = self
            .world
            .split_by_color_with_key(color, key)
            .ok_or_else(|| "MPI communicator split returned MPI_UNDEFINED".to_string())?;
        // Julia's comm2 groups ranks by their local comm1 rank.  The first
        // reduction over comm1 combines sample partitions within each group;
        // the second reduction over comm2 combines the resulting group totals
        // exactly once across groups.
        let cross_color =
            Color::with_value(i32::try_from(assignment.local_rank).map_err(|_| {
                format!(
                    "MPI local rank {} does not fit in an MPI color",
                    assignment.local_rank
                )
            })?);
        let cross_key = Key::with_value(i32::try_from(assignment.group).map_err(|_| {
            format!(
                "MPI group index {} does not fit in an MPI key",
                assignment.group
            )
        })?);
        let cross_communicator = self
            .world
            .split_by_color_with_key(cross_color, cross_key)
            .ok_or_else(|| {
                "MPI cross-group communicator split returned MPI_UNDEFINED".to_string()
            })?;
        Ok(MpiGroupContext {
            communicator,
            cross_communicator,
            assignment,
        })
    }
}

impl Reducer for MpiContext {
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let mut reduced = vec![0.0; chunk.len()];
            self.world
                .all_reduce_into(&*chunk, &mut reduced, SystemOperation::sum());
            chunk.copy_from_slice(&reduced);
        }
    }

    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let real: Vec<_> = chunk.iter().map(|value| value.re).collect();
            let imag: Vec<_> = chunk.iter().map(|value| value.im).collect();
            let mut reduced_real = vec![0.0; chunk.len()];
            let mut reduced_imag = vec![0.0; chunk.len()];
            self.world
                .all_reduce_into(&real, &mut reduced_real, SystemOperation::sum());
            self.world
                .all_reduce_into(&imag, &mut reduced_imag, SystemOperation::sum());
            for (value, (real, imag)) in chunk
                .iter_mut()
                .zip(reduced_real.into_iter().zip(reduced_imag))
            {
                *value = Complex64::new(real, imag);
            }
        }
    }

    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let mut reduced = vec![0_i64; chunk.len()];
            self.world
                .all_reduce_into(&*chunk, &mut reduced, SystemOperation::sum());
            chunk.copy_from_slice(&reduced);
        }
    }

    fn world_size(&self) -> usize {
        Self::world_size(self)
    }

    fn rank(&self) -> usize {
        Self::rank(self)
    }
}

impl MpiGroupContext {
    /// Group index in the world communicator.
    pub fn group(&self) -> usize {
        self.assignment.group
    }

    /// Borrow the group communicator for coordinated failure handling.
    pub fn communicator(&self) -> &::mpi::topology::SimpleCommunicator {
        &self.communicator
    }

    /// Borrow the cross-group communicator used for the second reduction.
    pub fn cross_communicator(&self) -> &::mpi::topology::SimpleCommunicator {
        &self.cross_communicator
    }
}

impl Reducer for MpiGroupContext {
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let mut reduced = vec![0.0; chunk.len()];
            self.communicator
                .all_reduce_into(&*chunk, &mut reduced, SystemOperation::sum());
            self.cross_communicator
                .all_reduce_into(&reduced, chunk, SystemOperation::sum());
        }
    }

    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let real: Vec<_> = chunk.iter().map(|value| value.re).collect();
            let imag: Vec<_> = chunk.iter().map(|value| value.im).collect();
            let mut reduced_real = vec![0.0; chunk.len()];
            let mut reduced_imag = vec![0.0; chunk.len()];
            self.communicator
                .all_reduce_into(&real, &mut reduced_real, SystemOperation::sum());
            self.communicator
                .all_reduce_into(&imag, &mut reduced_imag, SystemOperation::sum());
            let mut global_real = vec![0.0; chunk.len()];
            let mut global_imag = vec![0.0; chunk.len()];
            self.cross_communicator.all_reduce_into(
                &reduced_real,
                &mut global_real,
                SystemOperation::sum(),
            );
            self.cross_communicator.all_reduce_into(
                &reduced_imag,
                &mut global_imag,
                SystemOperation::sum(),
            );
            for (value, (real, imag)) in chunk
                .iter_mut()
                .zip(global_real.into_iter().zip(global_imag))
            {
                *value = Complex64::new(real, imag);
            }
        }
    }

    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let mut reduced = vec![0_i64; chunk.len()];
            self.communicator
                .all_reduce_into(&*chunk, &mut reduced, SystemOperation::sum());
            self.cross_communicator
                .all_reduce_into(&reduced, chunk, SystemOperation::sum());
        }
    }

    fn world_size(&self) -> usize {
        self.communicator
            .size()
            .try_into()
            .expect("MPI group size is positive")
    }

    fn rank(&self) -> usize {
        self.communicator
            .rank()
            .try_into()
            .expect("MPI group rank is nonnegative")
    }

    fn supports_grouped_sampling(&self) -> bool {
        true
    }

    fn seed_offset(&self) -> usize {
        self.assignment.group
    }

    fn is_output_root(&self) -> bool {
        self.assignment.group == 0 && self.assignment.local_rank == 0
    }
}
