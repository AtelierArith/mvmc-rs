//! All-reduce abstraction (Phase 4.1).
//!
//! Centralises the upstream MPI call sites so adding MPI in Phase 7 is
//! a single sibling-impl swap rather than touching every accumulator.
//! Upstream callers that land on this trait once the kernels are
//! ported:
//!
//! * `counter::reduce_counter!` (`counter.jl`).
//! * `parameter_sync::sync_modified_parameter!` (`parameter_sync.jl`).
//! * `weight_average::weight_average_*!` (`weight_average.jl`).
//! * `<O† O>` / `<H O>` accumulators in `vmc_main_cal.jl`.
//!
//! All three numeric overloads keep the same `&mut [T]` shape so the
//! caller does not have to special-case real vs. complex buffers.

use num_complex::Complex64;

/// All-reduce trait. The single-process implementation is a no-op;
/// the MPI implementation (Phase 7) will call `MPI_Allreduce(MPI_SUM)`.
pub trait Reducer {
    /// Broadcast an SR-CG vector with MPI_DOUBLE on the global communicator.
    /// A multi-rank implementation must not silently use the serial no-op.
    fn broadcast_f64(&self, _root: usize, _buf: &mut [f64]) -> Result<(), String> {
        if self.reduction_size() > 1 {
            Err("multi-rank reducer must implement real vector broadcast".into())
        } else {
            Ok(())
        }
    }

    /// Global synchronization before C's sampled SR-CG product reduction.
    /// MPI implementations execute this on the initializing main thread.
    fn barrier(&self) {}

    /// Broadcast a complex parameter buffer from `root`.
    ///
    /// The serial reducer is a no-op; MPI implementations replace the buffer
    /// on non-root ranks and leave the root values unchanged.
    fn broadcast_c64(&self, _root: usize, _buf: &mut [Complex64]) {}

    /// Broadcast an integer buffer from `root`.
    ///
    /// Seed resolution uses this separately from parameter broadcasts so the
    /// root-resolved base seed is shared before each rank adds its group
    /// offset. Default output-directory selection also uses this broadcast.
    /// Grouped implementations must distribute the global root's payload to
    /// every local rank in every group. Only the serial default is a no-op.
    fn broadcast_i64(&self, _root: usize, _buf: &mut [i64]) -> Result<(), String> {
        if self.reduction_size() > 1 {
            Err("multi-rank reducer must implement integer seed broadcast".into())
        } else {
            Ok(())
        }
    }

    /// In-place sum-reduction across all ranks.
    fn allreduce_sum_f64(&self, buf: &mut [f64]);
    /// In-place sum-reduction across all ranks (complex variant).
    fn allreduce_sum_c64(&self, buf: &mut [Complex64]);
    /// In-place sum-reduction across all ranks (integer counters).
    fn allreduce_sum_i64(&self, buf: &mut [i64]);

    /// Sampling IP uses comm1, never the global accumulator communicator.
    /// An ungrouped MPI rank is its own independent chain (NSplitSize=1).
    fn sampling_qp_range(&self, length: usize) -> std::ops::Range<usize> {
        0..length
    }

    /// Sum the QP contributions of this chain only; serial chains are no-ops.
    fn sampling_sum_f64(&self, _buf: &mut [f64]) {}
    /// Complex sampling IP counterpart.
    fn sampling_sum_c64(&self, _buf: &mut [Complex64]) {}

    /// Coordinate sampler initialization status within comm1 only.
    fn sampling_any_failure(&self, failed: bool) -> bool {
        failed
    }

    /// C ReduceCounter: six statistical entries, comm2, root-only writeback.
    /// Logical/configuration fields (including burn status) are not summed.
    fn reduce_counters(&self, counters: &mut [i64]) {
        let n = counters.len().min(6);
        let mut reduced = counters[..n].to_vec();
        self.allreduce_sum_i64(&mut reduced);
        if self.rank() == 0 {
            counters[..n].copy_from_slice(&reduced);
        }
    }

    /// Total number of ranks. The single-process implementation
    /// always returns 1 so `weight_average_*!` can divide safely.
    fn world_size(&self) -> usize {
        1
    }

    /// Number of independent rank contributions represented by a reduction.
    ///
    /// Grouped reducers expose their local communicator as `world_size()` but
    /// reduce accumulator contributions directly on the global communicator. PhysCal uses
    /// this count to average already-normalized per-rank Green accumulators.
    fn reduction_size(&self) -> usize {
        self.world_size()
    }

    /// 0-based rank of this process. The single-process implementation
    /// always returns 0.
    fn rank(&self) -> usize {
        0
    }

    /// Whether this reducer represents one communicator in grouped MPI mode.
    fn supports_grouped_sampling(&self) -> bool {
        false
    }

    /// Seed offset for independent group chains.
    fn seed_offset(&self) -> usize {
        self.rank()
    }

    /// Whether this rank owns the process-wide output files.
    fn is_output_root(&self) -> bool {
        self.rank() == 0
    }

    /// Return true when any rank reports a failure.
    fn any_failure(&self, failed: bool) -> bool {
        let mut flags = [i64::from(failed)];
        self.allreduce_sum_i64(&mut flags);
        flags[0] != 0
    }
}

/// No-op reducer for the v0.1 single-process build.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SingleProcessReducer;

impl Reducer for SingleProcessReducer {
    fn allreduce_sum_f64(&self, _buf: &mut [f64]) {}
    fn allreduce_sum_c64(&self, _buf: &mut [Complex64]) {}
    fn allreduce_sum_i64(&self, _buf: &mut [i64]) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_process_reducer_is_a_no_op() {
        let reducer = SingleProcessReducer;
        let mut floats = [1.0_f64, 2.0, 3.0];
        let mut complex = [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.5)];
        let mut ints = [10_i64, -3, 7];
        let before_floats = floats;
        let before_complex = complex;
        let before_ints = ints;
        reducer.allreduce_sum_f64(&mut floats);
        reducer.allreduce_sum_c64(&mut complex);
        reducer.allreduce_sum_i64(&mut ints);
        assert_eq!(floats, before_floats);
        assert_eq!(complex, before_complex);
        assert_eq!(ints, before_ints);
        assert_eq!(reducer.world_size(), 1);
        assert_eq!(reducer.rank(), 0);
    }
}
