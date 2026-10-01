//! Metropolis-Hastings sampler + Pfaffian rank-1/rank-2 updates.
//!
//! Port targets: `MVMCOptimizers.jl/src/{slater_update.jl,vmc_sampling.jl}`
//! (~7.7k LOC combined). The largest single file in the upstream codebase.
//! BIT-PARITY CRITICAL: preserve the upstream BLAS call sequence so
//! floating-point summation order matches Julia.
//!
//! Phase 4.3 is broken into substeps so each lands with its own golden
//! parity test:
//!
//! * **4.3.1 — projection bookkeeping** ([`projection`]): pure-Rust
//!   ports of `init_loc_spn!`, `make_proj_cnt!`, `update_proj_cnt!`,
//!   `log_proj_val`, `log_proj_ratio`, `update_ele_config!`,
//!   `revert_ele_config!`. No RNG, no BLAS — just `Vec<i64>` walks
//!   over `ExpertModeData`.
//! * **4.3.2 — RBM counters + log-cosh ratios** ([`rbm`]): pure-Rust
//!   ports of `make_rbm_cnt`, `update_rbm_cnt_hopping!`,
//!   `log_rbm_ratio`, `log_rbm_val`, `_rbm_log_cosh_stable` using a
//!   synthetic `RbmConfig` until the Phase-3 RBM parsers land.
//! * **4.3.3a — normal-mode candidate generators** ([`candidate`]):
//!   `get_update_type`, `make_candidate_hopping`,
//!   `make_candidate_exchange`; SFMT draw-order parity is covered by
//!   `tests/candidate_vs_julia.rs`. FSZ / local-spin-flip candidates
//!   and Metropolis accept/reject remain pending.
//! * 4.3.4 — Sherman-Morrison Pfaffian updates and the `vmc_make_sample!`
//!   driver. Pending.

pub mod candidate;
pub mod driver;
mod fsz_real;
pub mod initial;
pub mod metropolis;
pub mod one_move;
pub mod projection;
pub mod rbm;
pub mod updates;

pub use candidate::{
    get_update_type, make_candidate_exchange, make_candidate_exchange_fsz, make_candidate_hopping,
    make_candidate_hopping_fsz, make_candidate_local_spin_flip_conduction,
    make_candidate_local_spin_flip_localspin, ExchangeCandidate, FszExchangeCandidate,
    FszHoppingCandidate, HoppingCandidate, LocalSpinFlipCandidate, UpdateType,
};
pub use driver::{vmc_make_sample, vmc_make_sample_fsz, vmc_make_sample_real, SampleStats};
pub use fsz_real::vmc_make_sample_fsz_real;
pub use initial::{
    generate_initial_fsz_configuration, make_initial_sample, make_initial_sample_fsz,
    make_initial_sample_fsz_real, make_initial_sample_init_loc_spn,
};
pub use metropolis::{metropolis_decision, metropolis_weight, MetropolisDecision};
pub use one_move::{
    attempt_exchange_move, attempt_hopping_move, run_generated_hopping_mini_loop,
    run_generated_normal_mini_loop, run_hopping_mini_loop, ExchangeMoveOutcome, ExchangeMoveStatus,
    GeneratedHoppingMiniLoopOutcome, GeneratedHoppingStepInput, GeneratedHoppingStepOutcome,
    GeneratedNormalCandidate, GeneratedNormalMiniLoopOutcome, GeneratedNormalMoveOutcome,
    GeneratedNormalStepInput, GeneratedNormalStepOutcome, HoppingMiniLoopOutcome,
    HoppingMoveOutcome, HoppingMoveStatus, HoppingStepInput,
};
pub use projection::{
    init_loc_spn, log_proj_ratio, log_proj_val, make_proj_cnt, revert_ele_config,
    update_ele_config, update_proj_cnt,
};
pub use rbm::{
    log_cosh_stable, log_rbm_ratio, log_rbm_val, make_rbm_cnt, update_rbm_cnt_hopping, RbmConfig,
    RbmGeneralPhysHiddenTerm, RbmGeneralPhysLayerTerm, RbmHiddenLayerTerm, RbmPhysHiddenTerm,
    RbmPhysLayerTerm,
};
pub use updates::{
    calculate_new_pf_m2_complex_flat, calculate_new_pf_m2_fsz_complex_flat,
    calculate_new_pf_m2_fsz_real_flat, calculate_new_pf_m2_real_flat,
    calculate_new_pf_m_two2_complex_flat, calculate_new_pf_m_two2_real_flat,
    calculate_new_pf_m_two_fsz_complex_flat, calculate_new_pf_m_two_fsz_real_flat,
    update_m_all_complex_flat, update_m_all_fsz_complex_flat, update_m_all_fsz_real_flat,
    update_m_all_real_flat, update_m_all_two_complex_flat, update_m_all_two_fsz_real_flat,
    update_m_all_two_real_flat,
};
