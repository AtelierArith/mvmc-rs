//! Per-step counters (acceptance ratios, RNG-draw counts).
//!
//! Port target: `MVMCOptimizers.jl/src/counter.jl`. The upstream
//! `reduce_counter` is a no-op single-process today, but takes
//! `&dyn Reducer` so MPI can hook in without API churn (Phase 7).

use crate::reducer::Reducer;
use crate::state::VmcOptimizationState;

/// `reduce_counter!(state)` mirror. Single-process implementation is a
/// no-op; the `Reducer` argument is plumbed in for the MPI build.
pub fn reduce_counter<R: Reducer + ?Sized>(state: &mut VmcOptimizationState, reducer: &R) {
    reducer.allreduce_sum_i64(&mut state.electron_config.counter);
}
