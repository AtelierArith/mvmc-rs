//! Block-level sampling engine seam (issue #547).
//!
//! [`crate::sampling::driver::vmc_make_sample_real_engine`] owns the outer structure of the
//! real normal-mode sampler (initialization, burn/restore, the warm-up + measurement block
//! counts, the save points, the per-step counters). The per-block iteration - proposal
//! generation, projection delta, Pfaffian ratio, the acceptance draw, the inverse update and
//! the rebuild window - is a [`SamplingEngine`].
//!
//! [`HostEngine`] runs the production per-proposal body verbatim (`CpuStage` / a device
//! `RealPfStage` as its Pfaffian backend). A device-resident engine can replace it and select
//! the proposals on the accelerator, which the per-proposal [`RealPfStage`] seam cannot
//! express.

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::c_timer::CTimer;
use crate::reducer::Reducer;
use crate::sampling::candidate::{
    get_update_type, make_candidate_exchange, make_candidate_hopping, UpdateType,
};
use crate::sampling::driver::sampling_log_ip_real;
use crate::sampling::metropolis::metropolis_decision;
use crate::sampling::projection::{
    log_proj_ratio, revert_ele_config, update_ele_config, update_proj_cnt,
};
use crate::sampling::rng::SamplingRng;
use crate::sampling::stage::{RealPfStage, StageGeom, StageResult, StageTables};
use crate::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};

/// Mutable working state of one walker shared with the engine for one block.
///
/// The fields are disjoint borrows of the driver's working state, so an engine can mutate the
/// configuration, the projection counters and the tables without copying them.
pub struct BlockCtx<'a, G: SamplingRng, R: Reducer + ?Sized> {
    /// Run data (model, projection tables, dimensions).
    pub data: &'a ExpertModeData,
    /// The chain's reducer (sampling IP reduction; a device engine also rendezvous here).
    pub reducer: &'a R,
    /// The walker's word source (host engine draws here).
    pub rng: &'a mut G,
    /// Shared sizes of this walker (`StageGeom`).
    pub geom: StageGeom,
    /// Slater orbital table `[NQP][2 Nsite][2 Nsite]`.
    pub slater_elm: &'a SlaterElmFlat<f64>,
    /// Host inverse table, mVMC convention.
    pub inv_m: &'a mut InvMColMajor<f64>,
    /// Pfaffians per QP.
    pub pf_m: &'a mut Vec<f64>,
    /// PfaPack workspace of the CPU path.
    pub pool: &'a ThreadedPfaPackWorkspace,
    /// Six statistical counters (`counter[0]` proposes, `[1]` accepts, ...).
    pub counter: &'a mut [i64],
    /// Working electron configuration (`ele_idx`).
    pub ele_idx: &'a mut [i64],
    /// Working occupancy (`ele_cfg`).
    pub ele_cfg: &'a mut [i64],
    /// Working per-site electron numbers (`ele_num`).
    pub ele_num: &'a mut [i64],
    /// Working projection counters.
    pub ele_proj_cnt: &'a mut [i64],
    /// Scratch projection counters of the candidate.
    pub proj_cnt_new: &'a mut [i64],
    /// Scratch Pfaffians of the candidate.
    pub pf_m_new: &'a mut [f64],
    /// Local-spin definition (offsets where ordinary hopping is blocked).
    pub loc_spn: &'a [i64],
    /// `Nsite`.
    pub n_site: usize,
    /// Electrons per spin.
    pub n_elec: usize,
    /// `NExUpdatePath`.
    pub n_ex_path: i64,
    /// `iFlgOrbitalGeneral`.
    pub i_flg_general: i64,
    /// `2Sz`.
    pub two_sz: i64,
    /// Cached `log|inner product|` of the current configuration.
    pub log_ip_old: &'a mut f64,
    /// Accepted moves since the last recomputation.
    pub n_accept_window: &'a mut usize,
    /// Total accepted moves of this series.
    pub accepted_total: &'a mut usize,
}

/// One per-walker sampling engine: the per-block iteration seam of the real sampler.
pub trait SamplingEngine {
    /// Called once after the initial configuration and tables exist (`RealPfStage::begin`).
    fn begin(
        &mut self,
        geom: &StageGeom,
        tables: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<()> {
        let _ = (geom, tables, ele_idx);
        Ok(())
    }

    /// Advance the walker by `n_in_step` proposals. On return the working configuration, its
    /// projection counters, the Pfaffians and `log_ip_old` reflect the block, the counters are
    /// updated, and the walker consumed exactly its own draws.
    fn run_block<const TIMED: bool, G: SamplingRng, R: Reducer + ?Sized>(
        &mut self,
        ctx: &mut BlockCtx<'_, G, R>,
        timer: &mut CTimer<TIMED>,
        n_in_step: usize,
    ) -> StageResult<()>;
}

/// The production per-proposal engine: the unchanged host body around a [`RealPfStage`].
pub struct HostEngine<'a, S: RealPfStage> {
    /// Pfaffian backend (the CPU kernels in production; a device stage under the lock-step runner).
    pub stage: &'a mut S,
}

impl<S: RealPfStage> SamplingEngine for HostEngine<'_, S> {
    fn begin(
        &mut self,
        geom: &StageGeom,
        tables: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<()> {
        self.stage.begin(geom, tables, ele_idx)
    }

    fn run_block<const TIMED: bool, G: SamplingRng, R: Reducer + ?Sized>(
        &mut self,
        ctx: &mut BlockCtx<'_, G, R>,
        timer: &mut CTimer<TIMED>,
        n_in_step: usize,
    ) -> StageResult<()> {
        for _in_step in 0..n_in_step {
            let update_type =
                get_update_type(ctx.n_ex_path, ctx.i_flg_general, ctx.two_sz, ctx.rng);
            match update_type {
                UpdateType::Hopping => {
                    ctx.counter[0] += 1;
                    timer.start(31);
                    let candidate = make_candidate_hopping(
                        ctx.ele_idx,
                        ctx.ele_cfg,
                        ctx.n_site,
                        ctx.n_elec,
                        ctx.loc_spn,
                        ctx.rng,
                    );
                    timer.stop(31);
                    if candidate.reject {
                        continue;
                    }
                    timer.start(32);
                    timer.start(60);
                    update_ele_config(
                        candidate.mi,
                        candidate.ri,
                        candidate.rj,
                        candidate.spin,
                        ctx.ele_idx,
                        ctx.ele_cfg,
                        ctx.ele_num,
                        ctx.n_site,
                        ctx.n_elec,
                    );
                    update_proj_cnt(
                        candidate.ri as i64,
                        candidate.rj as i64,
                        candidate.spin,
                        ctx.proj_cnt_new,
                        ctx.ele_proj_cnt,
                        ctx.ele_num,
                        ctx.data,
                    );
                    timer.stop(60);
                    timer.start(61);
                    self.stage.propose_hop(
                        &ctx.geom,
                        &mut StageTables {
                            slater_elm: ctx.slater_elm,
                            inv_m: &mut *ctx.inv_m,
                            pf_m: &mut *ctx.pf_m,
                            pool: ctx.pool,
                        },
                        ctx.ele_idx,
                        candidate.mi,
                        candidate.spin,
                        ctx.pf_m_new,
                    )?;
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = sampling_log_ip_real(ctx.pf_m_new, ctx.data, ctx.reducer);
                    timer.stop(62);
                    let log_proj_delta =
                        log_proj_ratio(ctx.proj_cnt_new, ctx.ele_proj_cnt, ctx.data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(log_ip_new, 0.0),
                        Complex64::new(*ctx.log_ip_old, 0.0),
                        ctx.rng,
                    );
                    if self.stage.decide(&decision) {
                        timer.start(63);
                        self.stage.accept_hop(
                            &ctx.geom,
                            &mut StageTables {
                                slater_elm: ctx.slater_elm,
                                inv_m: &mut *ctx.inv_m,
                                pf_m: &mut *ctx.pf_m,
                                pool: ctx.pool,
                            },
                            ctx.ele_idx,
                            candidate.mi,
                            candidate.spin,
                            ctx.pf_m_new,
                        )?;
                        timer.stop(63);
                        ctx.ele_proj_cnt.copy_from_slice(ctx.proj_cnt_new);
                        *ctx.log_ip_old = log_ip_new;
                        *ctx.accepted_total += 1;
                        *ctx.n_accept_window += 1;
                        ctx.counter[1] += 1;
                    } else {
                        revert_ele_config(
                            candidate.mi,
                            candidate.ri,
                            candidate.rj,
                            candidate.spin,
                            ctx.ele_idx,
                            ctx.ele_cfg,
                            ctx.ele_num,
                            ctx.n_site,
                            ctx.n_elec,
                        );
                    }
                    timer.stop(32);
                }
                UpdateType::Exchange => {
                    ctx.counter[2] += 1;
                    timer.start(31);
                    let candidate = make_candidate_exchange(
                        ctx.ele_idx,
                        ctx.ele_cfg,
                        ctx.n_site,
                        ctx.n_elec,
                        ctx.ele_num,
                        ctx.rng,
                    );
                    timer.stop(31);
                    if candidate.reject {
                        continue;
                    }
                    timer.start(33);
                    let ri_old = candidate.ri;
                    let rj_old = candidate.rj;
                    timer.start(65);
                    update_ele_config(
                        candidate.mi,
                        ri_old,
                        rj_old,
                        candidate.spin,
                        ctx.ele_idx,
                        ctx.ele_cfg,
                        ctx.ele_num,
                        ctx.n_site,
                        ctx.n_elec,
                    );
                    update_proj_cnt(
                        ri_old as i64,
                        rj_old as i64,
                        candidate.spin,
                        ctx.proj_cnt_new,
                        ctx.ele_proj_cnt,
                        ctx.ele_num,
                        ctx.data,
                    );
                    update_ele_config(
                        candidate.mj,
                        rj_old,
                        ri_old,
                        candidate.spin_other,
                        ctx.ele_idx,
                        ctx.ele_cfg,
                        ctx.ele_num,
                        ctx.n_site,
                        ctx.n_elec,
                    );
                    let mid = ctx.proj_cnt_new.to_vec();
                    update_proj_cnt(
                        rj_old as i64,
                        ri_old as i64,
                        candidate.spin_other,
                        ctx.proj_cnt_new,
                        &mid,
                        ctx.ele_num,
                        ctx.data,
                    );
                    timer.stop(65);
                    timer.start(66);
                    self.stage.propose_exchange(
                        &ctx.geom,
                        &mut StageTables {
                            slater_elm: ctx.slater_elm,
                            inv_m: &mut *ctx.inv_m,
                            pf_m: &mut *ctx.pf_m,
                            pool: ctx.pool,
                        },
                        ctx.ele_idx,
                        [
                            (candidate.mi, candidate.spin),
                            (candidate.mj, candidate.spin_other),
                        ],
                        ctx.pf_m_new,
                    )?;
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = sampling_log_ip_real(ctx.pf_m_new, ctx.data, ctx.reducer);
                    timer.stop(67);
                    let log_proj_delta =
                        log_proj_ratio(ctx.proj_cnt_new, ctx.ele_proj_cnt, ctx.data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(log_ip_new, 0.0),
                        Complex64::new(*ctx.log_ip_old, 0.0),
                        ctx.rng,
                    );
                    if self.stage.decide(&decision) {
                        timer.start(68);
                        self.stage.accept_exchange(
                            &ctx.geom,
                            &mut StageTables {
                                slater_elm: ctx.slater_elm,
                                inv_m: &mut *ctx.inv_m,
                                pf_m: &mut *ctx.pf_m,
                                pool: ctx.pool,
                            },
                            ctx.ele_idx,
                            [
                                (candidate.mi, candidate.spin),
                                (candidate.mj, candidate.spin_other),
                            ],
                            [ri_old, rj_old],
                        )?;
                        timer.stop(68);
                        ctx.ele_proj_cnt.copy_from_slice(ctx.proj_cnt_new);
                        *ctx.log_ip_old = log_ip_new;
                        *ctx.accepted_total += 1;
                        *ctx.n_accept_window += 1;
                        ctx.counter[3] += 1;
                    } else {
                        revert_ele_config(
                            candidate.mj,
                            rj_old,
                            ri_old,
                            candidate.spin_other,
                            ctx.ele_idx,
                            ctx.ele_cfg,
                            ctx.ele_num,
                            ctx.n_site,
                            ctx.n_elec,
                        );
                        revert_ele_config(
                            candidate.mi,
                            ri_old,
                            rj_old,
                            candidate.spin,
                            ctx.ele_idx,
                            ctx.ele_cfg,
                            ctx.ele_num,
                            ctx.n_site,
                            ctx.n_elec,
                        );
                    }
                    timer.stop(33);
                }
                _ => {}
            }

            // Periodic Pfaffian recomputation (mirrors the upstream `n_accept > n_site` guard).
            if *ctx.n_accept_window > ctx.n_site {
                timer.start(34);
                let recompute_failed = self.stage.recompute(
                    &ctx.geom,
                    &mut StageTables {
                        slater_elm: ctx.slater_elm,
                        inv_m: &mut *ctx.inv_m,
                        pf_m: &mut *ctx.pf_m,
                        pool: ctx.pool,
                    },
                    ctx.ele_idx,
                )?;
                if !ctx.reducer.sampling_any_failure(recompute_failed) {
                    *ctx.log_ip_old = sampling_log_ip_real(ctx.pf_m, ctx.data, ctx.reducer);
                }
                *ctx.n_accept_window = 0;
                timer.stop(34);
            }
        }
        Ok(())
    }
}
