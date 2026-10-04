//! Per-step counters (acceptance ratios, RNG-draw counts).
//!
//! C authority: `vmcmake.c::ReduceCounter` reduces only the first six
//! statistical slots across equal-local-rank chains, writing back on that
//! communicator's root. Burn/status slots are local logical state.

use crate::reducer::Reducer;
use crate::state::VmcOptimizationState;

/// Reduce statistical counters without changing logical burn/status slots.
pub fn reduce_counter<R: Reducer + ?Sized>(state: &mut VmcOptimizationState, reducer: &R) {
    reducer.reduce_counters(&mut state.electron_config.counter);
}
