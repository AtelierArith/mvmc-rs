//! Phase 4.3.5b — one normal hopping move scaffold.
//!
//! This is the first thin wiring layer around the kernels from earlier
//! 4.3 substeps. It mirrors one `HOPPING` branch of `vmc_make_sample!`
//! up to the point where the expensive Pfaffian recomputation has
//! already produced `log_ip_new` (and optionally an RBM delta).
//!
//! The function performs exactly this state transition:
//!
//! 1. If the candidate is pre-rejected, do nothing and consume no RNG.
//! 2. `update_ele_config` (post-hop electron buffers).
//! 3. `update_proj_cnt` (with post-hop `ele_num`, matching C call order).
//! 4. Compute `log_proj_ratio`.
//! 5. `metropolis_decision` (draws one SFMT `genrand_real2`).
//! 6. If accepted, commit projection counters; otherwise `revert_ele_config`.
//!
//! Committing `pf_m`, `inv_m`, and `log_ip_old` is deliberately left to
//! the caller because 4.3.4 currently exposes multiple update kernels
//! (real / complex / FSZ) and this scaffold is shared by all of them.

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::sampling::candidate::{
    get_update_type, make_candidate_exchange, make_candidate_hopping, ExchangeCandidate,
    HoppingCandidate, UpdateType,
};
use crate::sampling::metropolis::{metropolis_decision, MetropolisDecision};
use crate::sampling::projection::{
    log_proj_ratio, revert_ele_config, update_ele_config, update_proj_cnt,
};

/// High-level outcome status for [`attempt_hopping_move`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoppingMoveStatus {
    /// Candidate itself was rejected before any state transition.
    CandidateRejected,
    /// Candidate passed and the Metropolis test accepted it.
    Accepted,
    /// Candidate passed but the Metropolis test rejected it; electron
    /// buffers were reverted.
    MetropolisRejected,
}

/// High-level outcome status for [`attempt_exchange_move`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeMoveStatus {
    /// Candidate itself was rejected before any state transition.
    CandidateRejected,
    /// Candidate was structurally invalid (`s == t`). Mirrors the
    /// upstream debug guard that skips such a candidate.
    InvalidCandidate,
    /// Candidate passed and the Metropolis test accepted it.
    Accepted,
    /// Candidate passed but the Metropolis test rejected it; electron
    /// buffers were reverted.
    MetropolisRejected,
}

/// Detailed result of [`attempt_hopping_move`].
#[derive(Debug, Clone, PartialEq)]
pub struct HoppingMoveOutcome {
    /// Final status.
    pub status: HoppingMoveStatus,
    /// Projection log-delta used for the Metropolis exponent. Zero for
    /// candidate-pre-rejection.
    pub log_proj_delta: f64,
    /// Metropolis decision data; `None` when the candidate was rejected
    /// before reaching the Metropolis draw.
    pub decision: Option<MetropolisDecision>,
}

/// Detailed result of [`attempt_exchange_move`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExchangeMoveOutcome {
    /// Final status.
    pub status: ExchangeMoveStatus,
    /// Projection log-delta used for the Metropolis exponent. Zero for
    /// pre-rejected / invalid candidates.
    pub log_proj_delta: f64,
    /// Metropolis decision data; `None` when the candidate was rejected
    /// before reaching the Metropolis draw.
    pub decision: Option<MetropolisDecision>,
}

/// Input to one mini-loop hopping step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoppingStepInput {
    /// Candidate to attempt.
    pub candidate: HoppingCandidate,
    /// Precomputed `log_ip_new` for this candidate.
    pub log_ip_new: Complex64,
    /// Precomputed RBM log-ratio delta; zero when RBM is disabled.
    pub log_rbm_delta: Complex64,
}

/// Summary returned by [`run_hopping_mini_loop`].
#[derive(Debug, Clone, PartialEq)]
pub struct HoppingMiniLoopOutcome {
    /// Per-step outcomes in order.
    pub steps: Vec<HoppingMoveOutcome>,
    /// Number of accepted moves.
    pub accepted_count: usize,
    /// Final `log_ip_old` after committing accepted moves.
    pub log_ip_final: Complex64,
}

/// Input to one RNG-generated hopping mini-loop step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratedHoppingStepInput {
    /// Precomputed `log_ip_new` to use if the generated candidate is
    /// valid (in the full sampler this comes from `calculate_new_pf_m2`).
    pub log_ip_new: Complex64,
    /// Precomputed RBM log-ratio delta; zero when RBM is disabled.
    pub log_rbm_delta: Complex64,
}

/// Outcome for one RNG-generated hopping mini-loop step.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedHoppingStepOutcome {
    /// Candidate generated internally by `make_candidate_hopping`.
    pub candidate: HoppingCandidate,
    /// Result of attempting that candidate.
    pub outcome: HoppingMoveOutcome,
}

/// Summary returned by [`run_generated_hopping_mini_loop`].
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedHoppingMiniLoopOutcome {
    /// Per-step generated candidate + result.
    pub steps: Vec<GeneratedHoppingStepOutcome>,
    /// Number of accepted moves.
    pub accepted_count: usize,
    /// Final `log_ip_old` after committing accepted moves.
    pub log_ip_final: Complex64,
}

/// Per-step input for [`run_generated_normal_mini_loop`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratedNormalStepInput {
    /// Precomputed `log_ip_new` for whichever update type is generated.
    pub log_ip_new: Complex64,
    /// Precomputed RBM log-ratio delta; zero when RBM is disabled.
    pub log_rbm_delta: Complex64,
}

/// Candidate generated in a normal mini-loop step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedNormalCandidate {
    /// Hopping branch.
    Hopping(HoppingCandidate),
    /// Exchange branch.
    Exchange(ExchangeCandidate),
    /// Unsupported branch (e.g. local spin flip in normal-mode helper).
    Unsupported,
}

/// Outcome generated in a normal mini-loop step.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratedNormalMoveOutcome {
    /// Hopping result.
    Hopping(HoppingMoveOutcome),
    /// Exchange result.
    Exchange(ExchangeMoveOutcome),
    /// Unsupported branch result.
    Unsupported,
}

/// One generated-dispatch step snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedNormalStepOutcome {
    /// Update type returned by `get_update_type`.
    pub update_type: UpdateType,
    /// Candidate generated for that branch.
    pub candidate: GeneratedNormalCandidate,
    /// Move result.
    pub outcome: GeneratedNormalMoveOutcome,
}

/// Summary returned by [`run_generated_normal_mini_loop`].
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedNormalMiniLoopOutcome {
    /// Per-step data.
    pub steps: Vec<GeneratedNormalStepOutcome>,
    /// Number of accepted moves across supported branches.
    pub accepted_count: usize,
    /// Final `log_ip_old` after accepted moves.
    pub log_ip_final: Complex64,
}

/// Attempt one normal-mode hopping move.
///
/// `ele_idx`, `ele_cfg`, `ele_num`, and `proj_cnt` are the mutable
/// walker state. `proj_cnt_new` is scratch and may be overwritten. The
/// caller supplies `log_ip_old` and `log_ip_new` (from the Pfaffian / QP
/// pipeline) and `log_rbm_delta` (zero if RBM is disabled).
#[allow(clippy::too_many_arguments)]
pub fn attempt_hopping_move(
    candidate: HoppingCandidate,
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    proj_cnt: &mut [i64],
    proj_cnt_new: &mut [i64],
    data: &ExpertModeData,
    n_site: usize,
    n_elec: usize,
    log_ip_old: Complex64,
    log_ip_new: Complex64,
    log_rbm_delta: Complex64,
    rng: &mut Sfmt19937Rng,
) -> HoppingMoveOutcome {
    if candidate.reject {
        return HoppingMoveOutcome {
            status: HoppingMoveStatus::CandidateRejected,
            log_proj_delta: 0.0,
            decision: None,
        };
    }

    update_ele_config(
        candidate.mi,
        candidate.ri,
        candidate.rj,
        candidate.spin,
        ele_idx,
        ele_cfg,
        ele_num,
        n_site,
        n_elec,
    );

    update_proj_cnt(
        candidate.ri as i64,
        candidate.rj as i64,
        candidate.spin,
        proj_cnt_new,
        proj_cnt,
        ele_num,
        data,
    );

    let log_proj_delta = log_proj_ratio(proj_cnt_new, proj_cnt, data);
    let decision = metropolis_decision(log_proj_delta, log_rbm_delta, log_ip_new, log_ip_old, rng);

    if decision.accepted {
        let n = proj_cnt.len().min(proj_cnt_new.len());
        proj_cnt[..n].copy_from_slice(&proj_cnt_new[..n]);
        HoppingMoveOutcome {
            status: HoppingMoveStatus::Accepted,
            log_proj_delta,
            decision: Some(decision),
        }
    } else {
        revert_ele_config(
            candidate.mi,
            candidate.ri,
            candidate.rj,
            candidate.spin,
            ele_idx,
            ele_cfg,
            ele_num,
            n_site,
            n_elec,
        );
        HoppingMoveOutcome {
            status: HoppingMoveStatus::MetropolisRejected,
            log_proj_delta,
            decision: Some(decision),
        }
    }
}

/// Attempt one normal-mode exchange move.
///
/// This mirrors the `EXCHANGE` branch in `vmc_make_sample!`: first
/// electron update + projection update, then second electron update +
/// projection update, then Metropolis decision; reject reverts the
/// second update first and the first update second.
#[allow(clippy::too_many_arguments)]
pub fn attempt_exchange_move(
    candidate: ExchangeCandidate,
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    proj_cnt: &mut [i64],
    proj_cnt_new: &mut [i64],
    data: &ExpertModeData,
    n_site: usize,
    n_elec: usize,
    log_ip_old: Complex64,
    log_ip_new: Complex64,
    log_rbm_delta: Complex64,
    rng: &mut Sfmt19937Rng,
) -> ExchangeMoveOutcome {
    if candidate.reject {
        return ExchangeMoveOutcome {
            status: ExchangeMoveStatus::CandidateRejected,
            log_proj_delta: 0.0,
            decision: None,
        };
    }
    if candidate.spin == candidate.spin_other {
        return ExchangeMoveOutcome {
            status: ExchangeMoveStatus::InvalidCandidate,
            log_proj_delta: 0.0,
            decision: None,
        };
    }

    update_ele_config(
        candidate.mi,
        candidate.ri,
        candidate.rj,
        candidate.spin,
        ele_idx,
        ele_cfg,
        ele_num,
        n_site,
        n_elec,
    );
    update_proj_cnt(
        candidate.ri as i64,
        candidate.rj as i64,
        candidate.spin,
        proj_cnt_new,
        proj_cnt,
        ele_num,
        data,
    );

    update_ele_config(
        candidate.mj,
        candidate.rj,
        candidate.ri,
        candidate.spin_other,
        ele_idx,
        ele_cfg,
        ele_num,
        n_site,
        n_elec,
    );
    let proj_mid = proj_cnt_new.to_vec();
    update_proj_cnt(
        candidate.rj as i64,
        candidate.ri as i64,
        candidate.spin_other,
        proj_cnt_new,
        &proj_mid,
        ele_num,
        data,
    );

    let log_proj_delta = log_proj_ratio(proj_cnt_new, proj_cnt, data);
    let decision = metropolis_decision(log_proj_delta, log_rbm_delta, log_ip_new, log_ip_old, rng);

    if decision.accepted {
        let n = proj_cnt.len().min(proj_cnt_new.len());
        proj_cnt[..n].copy_from_slice(&proj_cnt_new[..n]);
        ExchangeMoveOutcome {
            status: ExchangeMoveStatus::Accepted,
            log_proj_delta,
            decision: Some(decision),
        }
    } else {
        revert_ele_config(
            candidate.mj,
            candidate.rj,
            candidate.ri,
            candidate.spin_other,
            ele_idx,
            ele_cfg,
            ele_num,
            n_site,
            n_elec,
        );
        revert_ele_config(
            candidate.mi,
            candidate.ri,
            candidate.rj,
            candidate.spin,
            ele_idx,
            ele_cfg,
            ele_num,
            n_site,
            n_elec,
        );
        ExchangeMoveOutcome {
            status: ExchangeMoveStatus::MetropolisRejected,
            log_proj_delta,
            decision: Some(decision),
        }
    }
}

/// Run a deterministic mini loop over normal-mode hopping candidates.
///
/// This is the smallest outer-loop scaffold shared by future
/// `vmc_make_sample!` ports: each step receives a precomputed
/// candidate, `log_ip_new`, and optional RBM delta, then calls
/// [`attempt_hopping_move`]. On accepted steps it commits
/// `log_ip_old = log_ip_new` and increments the accepted counter; on
/// rejected steps the electron buffers have already been reverted by
/// [`attempt_hopping_move`].
#[allow(clippy::too_many_arguments)]
pub fn run_hopping_mini_loop(
    steps: &[HoppingStepInput],
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    proj_cnt: &mut [i64],
    proj_cnt_new: &mut [i64],
    data: &ExpertModeData,
    n_site: usize,
    n_elec: usize,
    mut log_ip_old: Complex64,
    rng: &mut Sfmt19937Rng,
) -> HoppingMiniLoopOutcome {
    let mut outcomes = Vec::with_capacity(steps.len());
    let mut accepted_count = 0usize;

    for step in steps {
        let out = attempt_hopping_move(
            step.candidate,
            ele_idx,
            ele_cfg,
            ele_num,
            proj_cnt,
            proj_cnt_new,
            data,
            n_site,
            n_elec,
            log_ip_old,
            step.log_ip_new,
            step.log_rbm_delta,
            rng,
        );
        if out.status == HoppingMoveStatus::Accepted {
            log_ip_old = step.log_ip_new;
            accepted_count += 1;
        }
        outcomes.push(out);
    }

    HoppingMiniLoopOutcome {
        steps: outcomes,
        accepted_count,
        log_ip_final: log_ip_old,
    }
}

/// Run a deterministic mini loop that generates a normal-mode hopping
/// candidate at each step using [`make_candidate_hopping`].
///
/// This is one layer closer to `vmc_make_sample!` than
/// [`run_hopping_mini_loop`]: RNG is consumed both by candidate
/// generation and by the Metropolis draw. Exchange dispatch and sample
/// saving remain outside this helper.
#[allow(clippy::too_many_arguments)]
pub fn run_generated_hopping_mini_loop(
    steps: &[GeneratedHoppingStepInput],
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    proj_cnt: &mut [i64],
    proj_cnt_new: &mut [i64],
    loc_spn: &[i64],
    data: &ExpertModeData,
    n_site: usize,
    n_elec: usize,
    mut log_ip_old: Complex64,
    rng: &mut Sfmt19937Rng,
) -> GeneratedHoppingMiniLoopOutcome {
    let mut outcomes = Vec::with_capacity(steps.len());
    let mut accepted_count = 0usize;

    for step in steps {
        let candidate = make_candidate_hopping(ele_idx, ele_cfg, n_site, n_elec, loc_spn, rng);
        let outcome = attempt_hopping_move(
            candidate,
            ele_idx,
            ele_cfg,
            ele_num,
            proj_cnt,
            proj_cnt_new,
            data,
            n_site,
            n_elec,
            log_ip_old,
            step.log_ip_new,
            step.log_rbm_delta,
            rng,
        );
        if outcome.status == HoppingMoveStatus::Accepted {
            log_ip_old = step.log_ip_new;
            accepted_count += 1;
        }
        outcomes.push(GeneratedHoppingStepOutcome { candidate, outcome });
    }

    GeneratedHoppingMiniLoopOutcome {
        steps: outcomes,
        accepted_count,
        log_ip_final: log_ip_old,
    }
}

/// Run a deterministic mini loop that dispatches update type each step
/// via [`get_update_type`], then generates and attempts hopping or
/// exchange candidates.
#[allow(clippy::too_many_arguments)]
pub fn run_generated_normal_mini_loop(
    steps: &[GeneratedNormalStepInput],
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    proj_cnt: &mut [i64],
    proj_cnt_new: &mut [i64],
    loc_spn: &[i64],
    data: &ExpertModeData,
    n_site: usize,
    n_elec: usize,
    n_ex_update_path: i64,
    i_flg_orbital_general: i64,
    two_sz: i64,
    mut log_ip_old: Complex64,
    rng: &mut Sfmt19937Rng,
) -> GeneratedNormalMiniLoopOutcome {
    let mut outcomes = Vec::with_capacity(steps.len());
    let mut accepted_count = 0usize;

    for step in steps {
        let update_type = get_update_type(n_ex_update_path, i_flg_orbital_general, two_sz, rng);
        let (candidate, outcome, accepted) = match update_type {
            UpdateType::Hopping => {
                let c = make_candidate_hopping(ele_idx, ele_cfg, n_site, n_elec, loc_spn, rng);
                let out = attempt_hopping_move(
                    c,
                    ele_idx,
                    ele_cfg,
                    ele_num,
                    proj_cnt,
                    proj_cnt_new,
                    data,
                    n_site,
                    n_elec,
                    log_ip_old,
                    step.log_ip_new,
                    step.log_rbm_delta,
                    rng,
                );
                let accepted = out.status == HoppingMoveStatus::Accepted;
                (
                    GeneratedNormalCandidate::Hopping(c),
                    GeneratedNormalMoveOutcome::Hopping(out),
                    accepted,
                )
            }
            UpdateType::Exchange => {
                let c = make_candidate_exchange(ele_idx, ele_cfg, n_site, n_elec, ele_num, rng);
                let out = attempt_exchange_move(
                    c,
                    ele_idx,
                    ele_cfg,
                    ele_num,
                    proj_cnt,
                    proj_cnt_new,
                    data,
                    n_site,
                    n_elec,
                    log_ip_old,
                    step.log_ip_new,
                    step.log_rbm_delta,
                    rng,
                );
                let accepted = out.status == ExchangeMoveStatus::Accepted;
                (
                    GeneratedNormalCandidate::Exchange(c),
                    GeneratedNormalMoveOutcome::Exchange(out),
                    accepted,
                )
            }
            _ => (
                GeneratedNormalCandidate::Unsupported,
                GeneratedNormalMoveOutcome::Unsupported,
                false,
            ),
        };
        if accepted {
            log_ip_old = step.log_ip_new;
            accepted_count += 1;
        }
        outcomes.push(GeneratedNormalStepOutcome {
            update_type,
            candidate,
            outcome,
        });
    }

    GeneratedNormalMiniLoopOutcome {
        steps: outcomes,
        accepted_count,
        log_ip_final: log_ip_old,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm};

    fn data() -> ExpertModeData {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 4;
        data.n_gutzwiller_idx = 1;
        data.gutzwiller_idx = vec![0, 0, 0, 0];
        data.gutzwiller_terms.push(GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.2, 0.0),
            is_complex: false,
        });
        data.n_jastrow_idx = 1;
        data.jastrow_idx = vec![
            vec![-1, 0, 0, 0],
            vec![0, -1, 0, 0],
            vec![0, 0, -1, 0],
            vec![0, 0, 0, -1],
        ];
        data.jastrow_terms.push(JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(-0.1, 0.0),
            is_complex: false,
        });
        data
    }

    #[test]
    fn rejected_candidate_consumes_no_rng_and_mutates_nothing() {
        let data = data();
        let mut ele_idx = vec![0, 1, 2, 3];
        let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
        let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
        let mut proj = vec![0, 0];
        let mut scratch = vec![9, 9];
        let before = (
            ele_idx.clone(),
            ele_cfg.clone(),
            ele_num.clone(),
            proj.clone(),
            scratch.clone(),
        );
        let mut rng = Sfmt19937Rng::new(42);
        let out = attempt_hopping_move(
            HoppingCandidate {
                mi: 0,
                ri: 0,
                rj: 1,
                spin: 0,
                reject: true,
            },
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj,
            &mut scratch,
            &data,
            4,
            2,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            &mut rng,
        );
        assert_eq!(out.status, HoppingMoveStatus::CandidateRejected);
        assert_eq!((ele_idx, ele_cfg, ele_num, proj, scratch), before);
    }

    #[test]
    fn accepted_move_commits_electron_and_proj_buffers() {
        let data = data();
        let mut ele_idx = vec![0, 1, 2, 3];
        let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
        let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
        let mut proj = vec![0, 0];
        let mut scratch = vec![0, 0];
        let mut rng = Sfmt19937Rng::new(42);
        let out = attempt_hopping_move(
            HoppingCandidate {
                mi: 0,
                ri: 0,
                rj: 2,
                spin: 0,
                reject: false,
            },
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj,
            &mut scratch,
            &data,
            4,
            2,
            Complex64::new(0.0, 0.0),
            Complex64::new(10.0, 0.0),
            Complex64::new(0.0, 0.0),
            &mut rng,
        );
        assert_eq!(out.status, HoppingMoveStatus::Accepted);
        assert_eq!(ele_idx[0], 2);
        assert_eq!(ele_cfg[0], -1);
        assert_eq!(ele_cfg[2], 0);
        assert_eq!(proj, scratch);
    }

    #[test]
    fn rejected_metropolis_reverts_electron_buffers() {
        let data = data();
        let mut ele_idx = vec![0, 1, 2, 3];
        let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
        let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
        let mut proj = vec![0, 0];
        let mut scratch = vec![0, 0];
        let before = (
            ele_idx.clone(),
            ele_cfg.clone(),
            ele_num.clone(),
            proj.clone(),
        );
        let mut rng = Sfmt19937Rng::new(42);
        let out = attempt_hopping_move(
            HoppingCandidate {
                mi: 0,
                ri: 0,
                rj: 2,
                spin: 0,
                reject: false,
            },
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj,
            &mut scratch,
            &data,
            4,
            2,
            Complex64::new(0.0, 0.0),
            Complex64::new(-1000.0, 0.0),
            Complex64::new(0.0, 0.0),
            &mut rng,
        );
        assert_eq!(out.status, HoppingMoveStatus::MetropolisRejected);
        assert_eq!((ele_idx, ele_cfg, ele_num, proj), before);
    }

    #[test]
    fn mini_loop_updates_log_ip_only_on_acceptance() {
        let data = data();
        let mut ele_idx = vec![0, 1, 2, 3];
        let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
        let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
        let mut proj = vec![0, 0];
        let mut scratch = vec![0, 0];
        let steps = vec![
            HoppingStepInput {
                candidate: HoppingCandidate {
                    mi: 0,
                    ri: 0,
                    rj: 2,
                    spin: 0,
                    reject: false,
                },
                log_ip_new: Complex64::new(5.0, 0.0),
                log_rbm_delta: Complex64::new(0.0, 0.0),
            },
            HoppingStepInput {
                candidate: HoppingCandidate {
                    mi: 1,
                    ri: 1,
                    rj: 3,
                    spin: 0,
                    reject: false,
                },
                log_ip_new: Complex64::new(-1000.0, 0.0),
                log_rbm_delta: Complex64::new(0.0, 0.0),
            },
        ];
        let mut rng = Sfmt19937Rng::new(42);
        let out = run_hopping_mini_loop(
            &steps,
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj,
            &mut scratch,
            &data,
            4,
            2,
            Complex64::new(0.0, 0.0),
            &mut rng,
        );
        assert_eq!(out.accepted_count, 1);
        assert_eq!(out.steps[0].status, HoppingMoveStatus::Accepted);
        assert_eq!(out.steps[1].status, HoppingMoveStatus::MetropolisRejected);
        assert_eq!(out.log_ip_final, Complex64::new(5.0, 0.0));
    }
}
