//! Optional MPI lifecycle and reducer implementation.
//!
//! The feature is intentionally opt-in because linking `rsmpi` requires an
//! MPI installation. The [`MpiContext`] owns the `Universe`, so MPI is
//! finalized when the context is dropped and never through a process-global
//! `MPI_Finalize` call owned by the library.

use num_complex::Complex64;

use ::mpi::topology::Color;
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
/// is owned by the parent `Universe`. Accumulators use `comm0`, sampling QP
/// work uses `comm1`, and statistical counters use `comm2`.
pub struct MpiGroupContext {
    global_communicator: ::mpi::topology::SimpleCommunicator,
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
        // Inner Rayon workers perform kernels only. Collective call sites
        // stay on the initializing/main thread, requiring MPI_THREAD_FUNNELED.
        let (universe, provided) = ::mpi::initialize_with_threading(
            ::mpi::environment::Threading::Funneled,
        )
        .ok_or_else(|| "MPI was already initialized or could not be initialized".to_string())?;
        if provided == ::mpi::environment::Threading::Single {
            return Err("MPI runtime did not provide MPI_THREAD_FUNNELED".into());
        }
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

    /// Split `MPI_COMM_WORLD` into contiguous groups of width `nsplit`.
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
        let key = i32::try_from(assignment.local_rank).map_err(|_| {
            format!(
                "MPI local rank {} does not fit in an MPI key",
                assignment.local_rank
            )
        })?;
        let communicator = self
            .world
            .split_by_color_with_key(color, key)
            .ok_or_else(|| "MPI communicator split returned MPI_UNDEFINED".to_string())?;
        // C/Julia comm2 connects equal comm1-local ranks across chains. It is
        // for six statistical counters, not a second accumulator reduction.
        let cross_color =
            Color::with_value(i32::try_from(assignment.local_rank).map_err(|_| {
                format!(
                    "MPI local rank {} does not fit in an MPI color",
                    assignment.local_rank
                )
            })?);
        let cross_key = i32::try_from(assignment.group).map_err(|_| {
            format!(
                "MPI group index {} does not fit in an MPI key",
                assignment.group
            )
        })?;
        let cross_communicator = self
            .world
            .split_by_color_with_key(cross_color, cross_key)
            .ok_or_else(|| {
                "MPI cross-group communicator split returned MPI_UNDEFINED".to_string()
            })?;
        Ok(MpiGroupContext {
            global_communicator: self.world.duplicate(),
            communicator,
            cross_communicator,
            assignment,
        })
    }
}

impl crate::parallel_scalar::ParallelScalarOperations for MpiContext {
    fn sum_real(
        &self,
        domain: crate::parallel_scalar::ScalarCommunicator,
        value: f64,
    ) -> Result<f64, String> {
        if domain != crate::parallel_scalar::ScalarCommunicator::World {
            return Err("sampling/cross-group communicator requires split_groups".into());
        }
        let mut result = value;
        self.world.all_reduce_into(
            &value,
            &mut result,
            ::mpi::collective::SystemOperation::sum(),
        );
        Ok(result)
    }

    fn sum_complex(
        &self,
        domain: crate::parallel_scalar::ScalarCommunicator,
        value: Complex64,
    ) -> Result<Complex64, String> {
        Ok(Complex64::new(
            self.sum_real(domain, value.re)?,
            self.sum_real(domain, value.im)?,
        ))
    }

    fn max_integer(
        &self,
        domain: crate::parallel_scalar::ScalarCommunicator,
        value: i32,
    ) -> Result<i32, String> {
        if domain != crate::parallel_scalar::ScalarCommunicator::World {
            return Err("sampling/cross-group communicator requires split_groups".into());
        }
        let mut result = value;
        self.world.all_reduce_into(
            &value,
            &mut result,
            ::mpi::collective::SystemOperation::max(),
        );
        Ok(result)
    }
}

impl crate::parallel_scalar::ParallelScalarOperations for MpiGroupContext {
    fn sum_real(
        &self,
        domain: crate::parallel_scalar::ScalarCommunicator,
        value: f64,
    ) -> Result<f64, String> {
        use crate::parallel_scalar::ScalarCommunicator;
        let comm = match domain {
            ScalarCommunicator::World => &self.global_communicator,
            ScalarCommunicator::Sampling => &self.communicator,
            ScalarCommunicator::CrossGroup => &self.cross_communicator,
        };
        let mut result = value;
        comm.all_reduce_into(
            &value,
            &mut result,
            ::mpi::collective::SystemOperation::sum(),
        );
        Ok(result)
    }

    fn sum_complex(
        &self,
        domain: crate::parallel_scalar::ScalarCommunicator,
        value: Complex64,
    ) -> Result<Complex64, String> {
        Ok(Complex64::new(
            self.sum_real(domain, value.re)?,
            self.sum_real(domain, value.im)?,
        ))
    }

    fn max_integer(
        &self,
        domain: crate::parallel_scalar::ScalarCommunicator,
        value: i32,
    ) -> Result<i32, String> {
        use crate::parallel_scalar::ScalarCommunicator;
        if domain == ScalarCommunicator::Sampling {
            // Dependency: PR291's signed comm1 operation, not a second implementation.
            return self.sampling_max_info(value);
        }
        let comm = match domain {
            ScalarCommunicator::World => &self.global_communicator,
            ScalarCommunicator::CrossGroup => &self.cross_communicator,
            ScalarCommunicator::Sampling => unreachable!(),
        };
        let mut result = value;
        comm.all_reduce_into(
            &value,
            &mut result,
            ::mpi::collective::SystemOperation::max(),
        );
        Ok(result)
    }
}

impl Reducer for MpiContext {
    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        if self.world.size() == 1 {
            return Ok(info);
        }
        let mut result = 0_i32;
        self.world.all_reduce_into(
            &info,
            &mut result,
            ::mpi::collective::SystemOperation::max(),
        );
        Ok(result)
    }

    fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
        Self::broadcast_f64(self, root, values)
    }

    fn barrier(&self) {
        self.world.barrier();
    }

    fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        Self::broadcast_i64(self, root, values)
    }

    fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
        let root = i32::try_from(root).expect("MPI root rank is too large");
        let mut real: Vec<_> = values.iter().map(|value| value.re).collect();
        let mut imag: Vec<_> = values.iter().map(|value| value.im).collect();
        self.world.process_at_rank(root).broadcast_into(&mut real);
        self.world.process_at_rank(root).broadcast_into(&mut imag);
        for (value, (real, imag)) in values.iter_mut().zip(real.into_iter().zip(imag)) {
            *value = Complex64::new(real, imag);
        }
    }

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
    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        if self.communicator.size() == 1 {
            return Ok(info);
        }
        let mut result = 0_i32;
        self.communicator.all_reduce_into(
            &info,
            &mut result,
            ::mpi::collective::SystemOperation::max(),
        );
        Ok(result)
    }

    fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
        if root >= self.global_communicator.size() as usize {
            return Err("MPI broadcast root is outside the global world".into());
        }
        self.global_communicator
            .process_at_rank(i32::try_from(root).map_err(|_| "MPI root rank is too large")?)
            .broadcast_into(values);
        Ok(())
    }

    fn barrier(&self) {
        self.global_communicator.barrier();
    }

    fn sampling_any_failure(&self, failed: bool) -> bool {
        use ::mpi::collective::SystemOperation;
        let flag = i32::from(failed);
        let mut result = 0_i32;
        self.communicator
            .all_reduce_into(&flag, &mut result, SystemOperation::max());
        result != 0
    }

    fn sampling_qp_range(&self, length: usize) -> std::ops::Range<usize> {
        self.assignment.local_range(length)
    }

    fn sampling_sum_f64(&self, values: &mut [f64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let mut reduced = vec![0.0; chunk.len()];
            self.communicator
                .all_reduce_into(&*chunk, &mut reduced, SystemOperation::sum());
            chunk.copy_from_slice(&reduced);
        }
    }

    fn sampling_sum_c64(&self, values: &mut [Complex64]) {
        let mut real: Vec<_> = values.iter().map(|x| x.re).collect();
        let mut imag: Vec<_> = values.iter().map(|x| x.im).collect();
        self.sampling_sum_f64(&mut real);
        self.sampling_sum_f64(&mut imag);
        for (value, (re, im)) in values.iter_mut().zip(real.into_iter().zip(imag)) {
            *value = Complex64::new(re, im);
        }
    }

    fn reduce_counters(&self, counters: &mut [i64]) {
        use ::mpi::collective::SystemOperation;
        let n = counters.len().min(6);
        if n == 0 {
            return;
        }
        let mut reduced = vec![0_i64; n];
        self.cross_communicator.all_reduce_into(
            &counters[..n],
            &mut reduced,
            SystemOperation::sum(),
        );
        if self.cross_communicator.rank() == 0 {
            counters[..n].copy_from_slice(&reduced);
        }
    }

    fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        let root =
            i32::try_from(root).map_err(|_| "MPI seed root rank is too large".to_string())?;
        // A direct comm0 broadcast also reaches shorter final comm1 groups.
        self.global_communicator
            .process_at_rank(root)
            .broadcast_into(values);
        Ok(())
    }

    fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
        let root = i32::try_from(root).expect("MPI global root rank is too large");
        let mut real: Vec<_> = values.iter().map(|value| value.re).collect();
        let mut imag: Vec<_> = values.iter().map(|value| value.im).collect();
        self.global_communicator
            .process_at_rank(root)
            .broadcast_into(&mut real);
        self.global_communicator
            .process_at_rank(root)
            .broadcast_into(&mut imag);
        for (value, (real, imag)) in values.iter_mut().zip(real.into_iter().zip(imag)) {
            *value = Complex64::new(real, imag);
        }
    }

    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        use ::mpi::collective::SystemOperation;
        for chunk in values.chunks_mut(MPI_REDUCTION_CHUNK_LEN) {
            let mut reduced = vec![0.0; chunk.len()];
            self.global_communicator
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
            self.global_communicator.all_reduce_into(
                &real,
                &mut reduced_real,
                SystemOperation::sum(),
            );
            self.global_communicator.all_reduce_into(
                &imag,
                &mut reduced_imag,
                SystemOperation::sum(),
            );
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
            self.global_communicator
                .all_reduce_into(&*chunk, &mut reduced, SystemOperation::sum());
            chunk.copy_from_slice(&reduced);
        }
    }

    fn world_size(&self) -> usize {
        self.communicator
            .size()
            .try_into()
            .expect("MPI group size is positive")
    }

    fn reduction_size(&self) -> usize {
        usize::try_from(self.global_communicator.size()).expect("MPI world size is positive")
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
