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
    /// In-place sum-reduction across all ranks.
    fn allreduce_sum_f64(&self, buf: &mut [f64]);
    /// In-place sum-reduction across all ranks (complex variant).
    fn allreduce_sum_c64(&self, buf: &mut [Complex64]);
    /// In-place sum-reduction across all ranks (integer counters).
    fn allreduce_sum_i64(&self, buf: &mut [i64]);

    /// Total number of ranks. The single-process implementation
    /// always returns 1 so `weight_average_*!` can divide safely.
    fn world_size(&self) -> usize {
        1
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
