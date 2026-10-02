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

/// Compose optimized and fixed translations in Julia's site/sign order.
/// Slater-table builders require both complete sector arrays; derivative
/// fallbacks use each available row independently, as in the reference.
pub(crate) fn translated_site(
    data: &ExpertModeData,
    site: usize,
    optidx: usize,
    mpidx: usize,
    require_complete: bool,
) -> (usize, i64) {
    let count = data.n_qp_opt_trans.max(1) as usize;
    let use_opt = !require_complete
        || (data.qp_opt_trans.len() >= count && data.qp_opt_trans_sgn.len() >= count);
    let ori = if use_opt {
        data.qp_opt_trans
            .get(optidx)
            .and_then(|row| row.get(site))
            .copied()
            .unwrap_or(site as i64) as usize
    } else {
        site
    };
    let opt_sign = if use_opt {
        data.qp_opt_trans_sgn
            .get(optidx)
            .and_then(|row| row.get(site))
            .copied()
            .unwrap_or(1)
    } else {
        1
    };
    let trans = data.qp_trans_entries.get(mpidx);
    let mapped = trans
        .and_then(|t| t.site_map.get(ori))
        .copied()
        .unwrap_or(ori as i64) as usize;
    let sign = trans
        .map(|t| t.boundary_sign(ori, data.modpara.nmp_trans < 0))
        .unwrap_or(1);
    (mapped, sign * opt_sign)
}
