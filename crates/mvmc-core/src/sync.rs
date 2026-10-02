//! Cross-rank variational-parameter sync.
//!
//! Port target: `MVMCOptimizers.jl/src/parameter_sync.jl`. Single-process
//! today; takes `&dyn Reducer` so Phase 7 can drop in an MPI impl without
//! touching the call sites.

use mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter as sync_inner;
use mvmc_expert_parsers::ExpertModeData;

use crate::reducer::Reducer;

/// `sync_modified_parameter!(data)` mirror. The `Reducer` is plumbed
/// through so a future MPI build can broadcast parameter values before
/// local correlation shifts and Slater/OptTrans normalization.
///
/// Mirrors `MVMCOptimizers.jl/src/parameter_sync.jl`.
pub fn sync_modified_parameter<R: Reducer + ?Sized>(data: &mut ExpertModeData, _reducer: &R) {
    sync_modified_parameter_local(data, true);
}

/// Apply Julia's local optimizer synchronization, including OptTrans normalization.
/// Correlation shifts can be disabled independently of Slater/OptTrans rescaling.
pub fn sync_modified_parameter_local(data: &mut ExpertModeData, shift_correlations: bool) {
    sync_inner(data, shift_correlations);
    let mut xmax = 0.0_f64;
    for value in &data.opt_trans {
        let amplitude = mvmc_expert_parsers::utils::julia_hypot::hypot(value.re, value.im);
        if amplitude.is_nan() {
            xmax = f64::NAN;
            break;
        }
        xmax = xmax.max(amplitude);
    }
    if xmax > 0.0 {
        let ratio = 1.0 / xmax;
        for value in &mut data.opt_trans {
            *value *= ratio;
        }
    }
}
