//! Cross-rank variational-parameter sync.
//!
//! Port target: `MVMCOptimizers.jl/src/parameter_sync.jl`. Single-process
//! today; takes `&dyn Reducer` so Phase 7 can drop in an MPI impl without
//! touching the call sites.

use mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter as sync_inner;
use mvmc_expert_parsers::ExpertModeData;

use crate::reducer::Reducer;

/// `sync_modified_parameter!(data)` mirror. The `Reducer` is plumbed
/// through so a future MPI build can broadcast the rescaled Slater
/// values; on the single-process path it is a no-op (the rescaling is
/// fully local).
///
/// Mirrors `MVMCOptimizers.jl/src/parameter_sync.jl`.
pub fn sync_modified_parameter<R: Reducer + ?Sized>(data: &mut ExpertModeData, _reducer: &R) {
    sync_inner(data, true);
}
