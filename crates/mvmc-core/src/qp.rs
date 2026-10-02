//! Quantum-projection weight updates.
//!
//! Port target: `MVMCOptimizers.jl/src/qp_weight_update.jl`. Pairs with
//! `mvmc_expert_parsers::utils::qp_weight::init_qp_weight!`.

use mvmc_expert_parsers::utils::qp_weight::{init_qp_weight as init_inner, update_qp_weight};
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};

/// `init_qp_weight!(data)` mirror, exposed via `mvmc-core` so callers
/// can write `mvmc_core::qp::init_qp_weight(&mut data)` instead of
/// reaching into `mvmc_expert_parsers::utils::qp_weight`.
pub fn init_qp_weight(data: &mut ExpertModeData) {
    init_inner(data);
}

/// `update_qp_weight!(data)` mirror. Refreshes `QPFullWeight` from
/// `QPFixWeight` (no-op for OptTrans-free models).
pub fn update_qp_weight_for(data: &mut ExpertModeData) {
    if let Some(weights) = data.qp_weights.as_mut() {
        update_qp_weight(weights, &data.opt_trans);
    }
}

/// Borrow the cached `QuantumProjectionWeights`. Returns `None` until
/// [`init_qp_weight`] has been called.
pub fn qp_weights(data: &ExpertModeData) -> Option<&QuantumProjectionWeights> {
    data.qp_weights.as_ref()
}
