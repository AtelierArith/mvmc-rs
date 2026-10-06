//! Phase 4.3.7 — full normal-mode sampler driver.
//!
//! Port target: `vmc_make_sample!` / `vmc_make_sample_real!` in
//! `MVMCOptimizers.jl/src/vmc_sampling.jl` (≈7.7k LOC combined). This
//! module wires the Phase-4.3 sub-step kernels (`make_initial_sample`,
//! `make_candidate_*`, Pfaffian update, projection update, Metropolis)
//! into the full mini-loop that produces `n_vmc_sample` saved walkers.
//!
//! The driver covers the real and complex `i_flg_orbital_general == 0`
//! paths exercised by the upstream `examples/inputs/*` cases. FSZ /
//! BackFlow drivers are deferred to Phase 7.

#![allow(clippy::too_many_arguments)]

#[path = "trace.rs"]
pub mod trace;

use crate::c_timer::CTimer;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::observables::{calculate_ip_complex, calculate_ip_real};
use crate::pfaffian::{calc_m_all_complex_native_info, calc_m_all_real_native_info};
use crate::reducer::{Reducer, SingleProcessReducer};
use crate::sampling::candidate::{
    get_update_type, make_candidate_exchange, make_candidate_exchange_fsz, make_candidate_hopping,
    make_candidate_hopping_fsz, make_candidate_local_spin_flip_conduction,
    make_candidate_local_spin_flip_localspin, FszHoppingCandidate, UpdateType,
};
use crate::sampling::metropolis::metropolis_decision;
use crate::sampling::normal_initial::{
    coordinate_normal_setup, make_initial_sample_normal_with_info, preflight_normal_sampler,
    NormalInitializationError,
};
use crate::sampling::projection::{
    init_loc_spn, log_proj_ratio, revert_ele_config, update_ele_config, update_proj_cnt,
};
use crate::sampling::stage::{CpuStage, RealPfStage, StageGeom, StageTables};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};

/// Output of [`vmc_make_sample_real`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleStats {
    /// Total accepted moves across the mini-loop.
    pub accepted: usize,
    /// Total saved samples (`n_vmc_sample` on success).
    pub saved: usize,
}

/// Real FSZ sampler with group-owned QP work and coordinated initialization.
pub fn vmc_make_sample_fsz_real_with_reducer<R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    reducer: &R,
) -> Result<SampleStats, super::initial::SamplingInitializationError> {
    super::fsz_real::vmc_make_sample_fsz_real_with_reducer(data, state, rng, reducer)
}

/// Range-owned sampling overlap, reduced on comm1 before taking its logarithm.
pub fn sampling_log_ip_real<R: Reducer + ?Sized>(
    pf: &[f64],
    data: &ExpertModeData,
    reducer: &R,
) -> f64 {
    let range = reducer.sampling_qp_range(pf.len());
    let mut ip = [calculate_ip_real(pf, range.start, range.end, data)];
    reducer.sampling_sum_f64(&mut ip);
    // C qp_real.c returns clog(ip) from a double function: retain its real
    // part, including zero -> -inf for the caller's single recovery branch.
    mvmc_expert_parsers::utils::c_math::log(ip[0].abs())
}

/// Complex range-owned sampling overlap; measurement uses its separate full-QP path.
pub fn sampling_log_ip_complex<R: Reducer + ?Sized>(
    pf: &[Complex64],
    data: &ExpertModeData,
    reducer: &R,
) -> Complex64 {
    let range = reducer.sampling_qp_range(pf.len());
    let mut ip = [calculate_ip_complex(pf, range.start, range.end, data)];
    reducer.sampling_sum_c64(&mut ip);
    Complex64::new(ip[0].norm().ln(), ip[0].arg())
}

/// Run one SR-step worth of normal-mode real-arithmetic sampling.
///
/// Side effects:
///  * `state.slater_matrix.slater_elm_real` is repopulated from
///    `slater_elm` (matches upstream `copy_slater_elm_to_real!`).
///  * The flat `inv_m_real` / `pf_m_real` buffers receive the
///    incremental Pfaffian updates.
///  * The walker buffers in `state.electron_config` (`ele_idx`,
///    `ele_cfg`, `ele_num`, `ele_proj_cnt`) hold the final
///    `n_vmc_sample` walkers in row-major layout.
pub fn vmc_make_sample_real(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
) -> Result<SampleStats, NormalInitializationError> {
    vmc_make_sample_real_timed(data, state, rng, &mut CTimer::<false>::new())
}

/// Run the sampler with call-site-specific C timer sections.
pub fn vmc_make_sample_real_timed<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
) -> Result<SampleStats, NormalInitializationError> {
    vmc_make_sample_real_with_reducer_timed(data, state, rng, timer, &SingleProcessReducer)
}

/// C comm1 sampling IP reductions; all ranks retain the full chain length.
pub fn vmc_make_sample_real_with_reducer_timed<const TIMED: bool, R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
    reducer: &R,
) -> Result<SampleStats, NormalInitializationError> {
    vmc_make_sample_real_staged(data, state, rng, timer, reducer, &mut CpuStage)
}

/// [`vmc_make_sample_real_with_reducer_timed`] with a pluggable Pfaffian stage backend
/// ([`RealPfStage`]); `CpuStage` is the production path.
pub fn vmc_make_sample_real_staged<const TIMED: bool, R: Reducer + ?Sized, S: RealPfStage>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
    reducer: &R,
    stage: &mut S,
) -> Result<SampleStats, NormalInitializationError> {
    timer.start(30);
    // This writes LocSpin workspace before typed preflight; the no-mutation
    // contract on rejection covers electron configuration and RNG, not all state.
    // init_loc_spn only writes existing slots; term indices are checked against
    // the slice length before indexing, independently of model dimensions.
    init_loc_spn(&mut state.workspace.loc_spn, data);
    let shape = match preflight_normal_sampler(
        data,
        state,
        &state.workspace.loc_spn,
        |length| reducer.sampling_qp_range(length),
        true,
        |info| reducer.sampling_max_info(info),
    ) {
        Ok(shape) => shape,
        Err(error) => {
            timer.stop(30);
            return Err(error);
        }
    };
    let n_site = shape.n_site;
    let n_elec = shape.n_elec;
    let n_size = shape.n_size;
    let n_qp_full = shape.n_qp_full;
    let qp_start = shape.qp_range.start;
    let qp_end = shape.qp_range.end;
    let n_vmc_sample = shape.n_vmc_sample;
    let n_ex_path = data.modpara.nex_update_path;
    let i_flg_general = data.i_flg_orbital_general;
    let two_sz = data.modpara.two_sz;

    let loc_spn = state.workspace.loc_spn.clone();
    // Working buffers.
    let mut tmp_ele_idx = state.electron_config.tmp_ele_idx.clone();
    let mut tmp_ele_cfg = state.electron_config.tmp_ele_cfg.clone();
    let mut tmp_ele_num = state.electron_config.tmp_ele_num.clone();
    let mut tmp_ele_proj_cnt = state.electron_config.tmp_ele_proj_cnt.clone();

    let mut burn_flag = state.electron_config.counter[9] != 0;
    let pool = ThreadedPfaPackWorkspace::new(state.workspace.n_size, 1);
    let initialization = (|| {
        // Sync the real-mode Slater table from the complex master.
        for qp in 0..state.slater_matrix.slater_elm_real.n_qp_full() {
            for row in 0..state.slater_matrix.slater_elm_real.n_site2() {
                for col in 0..state.slater_matrix.slater_elm_real.n_site2() {
                    state.slater_matrix.slater_elm_real.set(
                        qp,
                        row,
                        col,
                        state.slater_matrix.slater_elm.get(qp, row, col).re,
                    );
                }
            }
        }

        if burn_flag {
            state.electron_config.restore_burn();
            tmp_ele_idx.copy_from_slice(&state.electron_config.tmp_ele_idx);
            tmp_ele_cfg.copy_from_slice(&state.electron_config.tmp_ele_cfg);
            tmp_ele_num.copy_from_slice(&state.electron_config.tmp_ele_num);
            tmp_ele_proj_cnt.copy_from_slice(&state.electron_config.tmp_ele_proj_cnt);
        } else {
            make_initial_sample_normal_with_info(
                &mut tmp_ele_idx,
                &mut tmp_ele_cfg,
                &mut tmp_ele_num,
                &mut tmp_ele_proj_cnt,
                data,
                &loc_spn,
                rng,
                &mut state.slater_matrix,
                qp_start..qp_end,
                &pool,
                |info| reducer.sampling_max_info(info),
            )?;
        }
        // C performs this separate real setup once and ignores numeric INFO.
        // Invalid Rust/input preconditions are not fabricated native INFO.
        let _real_info = coordinate_normal_setup(
            calc_m_all_real_native_info(
                &tmp_ele_idx,
                &state.slater_matrix.slater_elm_real,
                &mut state.slater_matrix.inv_m_real,
                &mut state.slater_matrix.pf_m_real,
                qp_start,
                qp_end,
                n_site,
                n_elec,
                &pool,
            ),
            |info| reducer.sampling_max_info(info),
        )?;
        let mut log_ip = sampling_log_ip_real(&state.slater_matrix.pf_m_real, data, reducer);
        if !log_ip.is_finite() {
            make_initial_sample_normal_with_info(
                &mut tmp_ele_idx,
                &mut tmp_ele_cfg,
                &mut tmp_ele_num,
                &mut tmp_ele_proj_cnt,
                data,
                &loc_spn,
                rng,
                &mut state.slater_matrix,
                qp_start..qp_end,
                &pool,
                |info| reducer.sampling_max_info(info),
            )?;
            let _real_info = coordinate_normal_setup(
                calc_m_all_real_native_info(
                    &tmp_ele_idx,
                    &state.slater_matrix.slater_elm_real,
                    &mut state.slater_matrix.inv_m_real,
                    &mut state.slater_matrix.pf_m_real,
                    qp_start,
                    qp_end,
                    n_site,
                    n_elec,
                    &pool,
                ),
                |info| reducer.sampling_max_info(info),
            )?;
            log_ip = sampling_log_ip_real(&state.slater_matrix.pf_m_real, data, reducer);
            burn_flag = false;
        }
        Ok::<_, NormalInitializationError>(log_ip)
    })();
    let mut log_ip_old = match initialization {
        Ok(log_ip) => log_ip,
        Err(error) => {
            // C aborts with the last actual working configuration, not rollback.
            state.electron_config.tmp_ele_idx = tmp_ele_idx;
            state.electron_config.tmp_ele_cfg = tmp_ele_cfg;
            state.electron_config.tmp_ele_num = tmp_ele_num;
            state.electron_config.tmp_ele_proj_cnt = tmp_ele_proj_cnt;
            timer.stop(30);
            return Err(error);
        }
    };

    let inv_stride = n_size * n_size + 1;
    let geom = StageGeom {
        n_site,
        n_elec,
        qp_start,
        qp_end,
        inv_stride,
    };
    {
        let mut tables = StageTables {
            slater_elm: &state.slater_matrix.slater_elm_real,
            inv_m: &mut state.slater_matrix.inv_m_real,
            pf_m: &mut state.slater_matrix.pf_m_real,
            pool: &pool,
        };
        if let Err(error) = stage.begin(&geom, &mut tables, &tmp_ele_idx) {
            state.electron_config.tmp_ele_idx = tmp_ele_idx;
            state.electron_config.tmp_ele_cfg = tmp_ele_cfg;
            state.electron_config.tmp_ele_num = tmp_ele_num;
            state.electron_config.tmp_ele_proj_cnt = tmp_ele_proj_cnt;
            return Err(NormalInitializationError::Precondition(error));
        }
    }
    let mut pf_m_new = vec![0.0_f64; n_qp_full];
    let mut proj_cnt_new = vec![0_i64; tmp_ele_proj_cnt.len()];
    let n_out_step = if burn_flag {
        shape.n_out_burn
    } else {
        shape.n_out_warm
    };
    let n_in_step = shape.n_in_step;
    let mut accepted_total = 0usize;
    let mut n_accept_window = 0usize;
    let mut saved = 0usize;

    state.electron_config.counter.fill(0);
    timer.stop(30);
    for out_step in 0..n_out_step {
        for _in_step in 0..n_in_step {
            let update_type = get_update_type(n_ex_path, i_flg_general, two_sz, rng);
            match update_type {
                UpdateType::Hopping => {
                    state.electron_config.counter[0] += 1;
                    timer.start(31);
                    let candidate = make_candidate_hopping(
                        &tmp_ele_idx,
                        &tmp_ele_cfg,
                        n_site,
                        n_elec,
                        &loc_spn,
                        rng,
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
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        n_site,
                        n_elec,
                    );
                    update_proj_cnt(
                        candidate.ri as i64,
                        candidate.rj as i64,
                        candidate.spin,
                        &mut proj_cnt_new,
                        &tmp_ele_proj_cnt,
                        &tmp_ele_num,
                        data,
                    );
                    timer.stop(60);
                    timer.start(61);
                    stage
                        .propose_hop(
                            &geom,
                            &mut StageTables {
                                slater_elm: &state.slater_matrix.slater_elm_real,
                                inv_m: &mut state.slater_matrix.inv_m_real,
                                pf_m: &mut state.slater_matrix.pf_m_real,
                                pool: &pool,
                            },
                            &tmp_ele_idx,
                            candidate.mi,
                            candidate.spin,
                            &mut pf_m_new,
                        )
                        .map_err(NormalInitializationError::Precondition)?;
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = sampling_log_ip_real(&pf_m_new, data, reducer);
                    timer.stop(62);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(log_ip_new, 0.0),
                        Complex64::new(log_ip_old, 0.0),
                        rng,
                    );
                    if stage.decide(&decision) {
                        timer.start(63);
                        stage
                            .accept_hop(
                                &geom,
                                &mut StageTables {
                                    slater_elm: &state.slater_matrix.slater_elm_real,
                                    inv_m: &mut state.slater_matrix.inv_m_real,
                                    pf_m: &mut state.slater_matrix.pf_m_real,
                                    pool: &pool,
                                },
                                &tmp_ele_idx,
                                candidate.mi,
                                candidate.spin,
                                &pf_m_new,
                            )
                            .map_err(NormalInitializationError::Precondition)?;
                        timer.stop(63);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
                        state.electron_config.counter[1] += 1;
                    } else {
                        revert_ele_config(
                            candidate.mi,
                            candidate.ri,
                            candidate.rj,
                            candidate.spin,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            n_site,
                            n_elec,
                        );
                    }
                    timer.stop(32);
                }
                UpdateType::Exchange => {
                    state.electron_config.counter[2] += 1;
                    timer.start(31);
                    let candidate = make_candidate_exchange(
                        &tmp_ele_idx,
                        &tmp_ele_cfg,
                        n_site,
                        n_elec,
                        &tmp_ele_num,
                        rng,
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
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        n_site,
                        n_elec,
                    );
                    update_proj_cnt(
                        ri_old as i64,
                        rj_old as i64,
                        candidate.spin,
                        &mut proj_cnt_new,
                        &tmp_ele_proj_cnt,
                        &tmp_ele_num,
                        data,
                    );
                    update_ele_config(
                        candidate.mj,
                        rj_old,
                        ri_old,
                        candidate.spin_other,
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        n_site,
                        n_elec,
                    );
                    let mid = proj_cnt_new.clone();
                    update_proj_cnt(
                        rj_old as i64,
                        ri_old as i64,
                        candidate.spin_other,
                        &mut proj_cnt_new,
                        &mid,
                        &tmp_ele_num,
                        data,
                    );
                    timer.stop(65);
                    timer.start(66);
                    stage
                        .propose_exchange(
                            &geom,
                            &mut StageTables {
                                slater_elm: &state.slater_matrix.slater_elm_real,
                                inv_m: &mut state.slater_matrix.inv_m_real,
                                pf_m: &mut state.slater_matrix.pf_m_real,
                                pool: &pool,
                            },
                            &tmp_ele_idx,
                            [
                                (candidate.mi, candidate.spin),
                                (candidate.mj, candidate.spin_other),
                            ],
                            &mut pf_m_new,
                        )
                        .map_err(NormalInitializationError::Precondition)?;
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = sampling_log_ip_real(&pf_m_new, data, reducer);
                    timer.stop(67);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(log_ip_new, 0.0),
                        Complex64::new(log_ip_old, 0.0),
                        rng,
                    );
                    if stage.decide(&decision) {
                        timer.start(68);
                        stage
                            .accept_exchange(
                                &geom,
                                &mut StageTables {
                                    slater_elm: &state.slater_matrix.slater_elm_real,
                                    inv_m: &mut state.slater_matrix.inv_m_real,
                                    pf_m: &mut state.slater_matrix.pf_m_real,
                                    pool: &pool,
                                },
                                &tmp_ele_idx,
                                [
                                    (candidate.mi, candidate.spin),
                                    (candidate.mj, candidate.spin_other),
                                ],
                                [ri_old, rj_old],
                            )
                            .map_err(NormalInitializationError::Precondition)?;
                        timer.stop(68);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
                        state.electron_config.counter[3] += 1;
                    } else {
                        revert_ele_config(
                            candidate.mj,
                            rj_old,
                            ri_old,
                            candidate.spin_other,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            n_site,
                            n_elec,
                        );
                        revert_ele_config(
                            candidate.mi,
                            ri_old,
                            rj_old,
                            candidate.spin,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            n_site,
                            n_elec,
                        );
                    }
                    timer.stop(33);
                }
                _ => {}
            }

            // Periodic Pfaffian recomputation to keep numerical drift in
            // check (mirrors the upstream `n_accept > n_site` guard).
            if n_accept_window > n_site {
                timer.start(34);
                let recompute_failed = stage
                    .recompute(
                        &geom,
                        &mut StageTables {
                            slater_elm: &state.slater_matrix.slater_elm_real,
                            inv_m: &mut state.slater_matrix.inv_m_real,
                            pf_m: &mut state.slater_matrix.pf_m_real,
                            pool: &pool,
                        },
                        &tmp_ele_idx,
                    )
                    .map_err(NormalInitializationError::Precondition)?;
                if !reducer.sampling_any_failure(recompute_failed) {
                    log_ip_old =
                        sampling_log_ip_real(&state.slater_matrix.pf_m_real, data, reducer);
                }
                n_accept_window = 0;
                timer.stop(34);
            }
        }

        // Save samples after warm-up.
        if out_step + n_vmc_sample >= n_out_step {
            timer.start(35);
            let sample = out_step + n_vmc_sample - n_out_step;
            if sample < n_vmc_sample {
                state
                    .electron_config
                    .ele_idx_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_idx);
                state
                    .electron_config
                    .ele_cfg_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_cfg);
                state
                    .electron_config
                    .ele_num_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_num);
                state
                    .electron_config
                    .ele_proj_cnt_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_proj_cnt);
                saved += 1;
            }
            timer.stop(35);
        }
    }

    state.electron_config.tmp_ele_idx = tmp_ele_idx;
    state.electron_config.tmp_ele_cfg = tmp_ele_cfg;
    state.electron_config.tmp_ele_num = tmp_ele_num;
    state.electron_config.tmp_ele_proj_cnt = tmp_ele_proj_cnt;
    state.electron_config.save_burn();
    state.electron_config.counter[9] = 1;
    trace::checkpoint(state, rng);

    Ok(SampleStats {
        accepted: accepted_total,
        saved,
    })
}

/// Complex-arithmetic counterpart of [`vmc_make_sample_real`]. Mirrors
/// the layout of `vmc_make_sample!` (sz-conserved branch) and only
/// substitutes the complex Pfaffian / inverse / Slater buffers for
/// their real-mode siblings.
pub fn vmc_make_sample(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
) -> Result<SampleStats, NormalInitializationError> {
    vmc_make_sample_timed(data, state, rng, &mut CTimer::<false>::new())
}

/// Run the sampler with call-site-specific C timer sections.
pub fn vmc_make_sample_timed<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
) -> Result<SampleStats, NormalInitializationError> {
    vmc_make_sample_with_reducer_timed(data, state, rng, timer, &SingleProcessReducer)
}

/// Normal complex sampler with group-only QP overlap reductions.
pub fn vmc_make_sample_with_reducer_timed<const TIMED: bool, R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
    reducer: &R,
) -> Result<SampleStats, NormalInitializationError> {
    timer.start(30);
    // This writes LocSpin workspace before typed preflight; the no-mutation
    // contract on rejection covers electron configuration and RNG, not all state.
    // init_loc_spn only writes existing slots; term indices are checked against
    // the slice length before indexing, independently of model dimensions.
    init_loc_spn(&mut state.workspace.loc_spn, data);
    let shape = match preflight_normal_sampler(
        data,
        state,
        &state.workspace.loc_spn,
        |length| reducer.sampling_qp_range(length),
        false,
        |info| reducer.sampling_max_info(info),
    ) {
        Ok(shape) => shape,
        Err(error) => {
            timer.stop(30);
            return Err(error);
        }
    };
    let n_site = shape.n_site;
    let n_elec = shape.n_elec;
    let n_size = shape.n_size;
    let n_qp_full = shape.n_qp_full;
    let qp_start = shape.qp_range.start;
    let qp_end = shape.qp_range.end;
    let n_vmc_sample = shape.n_vmc_sample;
    let n_ex_path = data.modpara.nex_update_path;
    let i_flg_general = data.i_flg_orbital_general;
    let two_sz = data.modpara.two_sz;

    let loc_spn = state.workspace.loc_spn.clone();
    let mut tmp_ele_idx = state.electron_config.tmp_ele_idx.clone();
    let mut tmp_ele_cfg = state.electron_config.tmp_ele_cfg.clone();
    let mut tmp_ele_num = state.electron_config.tmp_ele_num.clone();
    let mut tmp_ele_proj_cnt = state.electron_config.tmp_ele_proj_cnt.clone();

    let mut burn_flag = state.electron_config.counter[9] != 0;
    let pool = ThreadedPfaPackWorkspace::new(state.workspace.n_size, 1);
    let initialization = (|| {
        if burn_flag {
            state.electron_config.restore_burn();
            tmp_ele_idx.copy_from_slice(&state.electron_config.tmp_ele_idx);
            tmp_ele_cfg.copy_from_slice(&state.electron_config.tmp_ele_cfg);
            tmp_ele_num.copy_from_slice(&state.electron_config.tmp_ele_num);
            tmp_ele_proj_cnt.copy_from_slice(&state.electron_config.tmp_ele_proj_cnt);
        } else {
            make_initial_sample_normal_with_info(
                &mut tmp_ele_idx,
                &mut tmp_ele_cfg,
                &mut tmp_ele_num,
                &mut tmp_ele_proj_cnt,
                data,
                &loc_spn,
                rng,
                &mut state.slater_matrix,
                qp_start..qp_end,
                &pool,
                |info| reducer.sampling_max_info(info),
            )?;
        }
        // C's distinct complex setup ignores numeric INFO, not typed errors.
        let _setup_info = coordinate_normal_setup(
            calc_m_all_complex_native_info(
                &tmp_ele_idx,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                qp_start,
                qp_end,
                n_site,
                n_elec,
                &pool,
            ),
            |info| reducer.sampling_max_info(info),
        )?;
        let mut log_ip = sampling_log_ip_complex(&state.slater_matrix.pf_m, data, reducer);
        if !(log_ip.re + log_ip.im).is_finite() {
            make_initial_sample_normal_with_info(
                &mut tmp_ele_idx,
                &mut tmp_ele_cfg,
                &mut tmp_ele_num,
                &mut tmp_ele_proj_cnt,
                data,
                &loc_spn,
                rng,
                &mut state.slater_matrix,
                qp_start..qp_end,
                &pool,
                |info| reducer.sampling_max_info(info),
            )?;
            let _setup_info = coordinate_normal_setup(
                calc_m_all_complex_native_info(
                    &tmp_ele_idx,
                    &state.slater_matrix.slater_elm,
                    &mut state.slater_matrix.inv_m,
                    &mut state.slater_matrix.pf_m,
                    qp_start,
                    qp_end,
                    n_site,
                    n_elec,
                    &pool,
                ),
                |info| reducer.sampling_max_info(info),
            )?;
            log_ip = sampling_log_ip_complex(&state.slater_matrix.pf_m, data, reducer);
            burn_flag = false;
        }
        Ok::<_, NormalInitializationError>(log_ip)
    })();
    let mut log_ip_old = match initialization {
        Ok(log_ip) => log_ip,
        Err(error) => {
            state.electron_config.tmp_ele_idx = tmp_ele_idx;
            state.electron_config.tmp_ele_cfg = tmp_ele_cfg;
            state.electron_config.tmp_ele_num = tmp_ele_num;
            state.electron_config.tmp_ele_proj_cnt = tmp_ele_proj_cnt;
            timer.stop(30);
            return Err(error);
        }
    };
    let rbm_cfg = crate::sampling::rbm::RbmConfig::from(data);
    let use_rbm = data.has_rbm_terms();
    let mut rbm_cnt_old = crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg);
    let mut rbm_cnt_new = vec![Complex64::new(0.0, 0.0); rbm_cnt_old.len()];

    let inv_stride = n_size * n_size + 1;
    let mut pf_m_new = vec![Complex64::new(0.0, 0.0); n_qp_full];
    let mut proj_cnt_new = vec![0_i64; tmp_ele_proj_cnt.len()];
    let n_out_step = if burn_flag {
        shape.n_out_burn
    } else {
        shape.n_out_warm
    };
    let n_in_step = shape.n_in_step;
    let mut accepted_total = 0usize;
    let mut n_accept_window = 0usize;
    let mut saved = 0usize;

    state.electron_config.counter.fill(0);
    timer.stop(30);
    for out_step in 0..n_out_step {
        for _in_step in 0..n_in_step {
            let update_type = get_update_type(n_ex_path, i_flg_general, two_sz, rng);
            match update_type {
                UpdateType::Hopping => {
                    state.electron_config.counter[0] += 1;
                    timer.start(31);
                    let candidate = make_candidate_hopping(
                        &tmp_ele_idx,
                        &tmp_ele_cfg,
                        n_site,
                        n_elec,
                        &loc_spn,
                        rng,
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
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        n_site,
                        n_elec,
                    );
                    update_proj_cnt(
                        candidate.ri as i64,
                        candidate.rj as i64,
                        candidate.spin,
                        &mut proj_cnt_new,
                        &tmp_ele_proj_cnt,
                        &tmp_ele_num,
                        data,
                    );
                    timer.stop(60);
                    timer.start(61);
                    crate::sampling::updates::calculate_new_pf_m2_complex_flat(
                        candidate.mi,
                        candidate.spin,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &state.slater_matrix.slater_elm,
                        state.slater_matrix.inv_m.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = sampling_log_ip_complex(&pf_m_new, data, reducer);
                    timer.stop(62);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let rbm_delta = if use_rbm {
                        crate::sampling::rbm::update_rbm_cnt_hopping(
                            &mut rbm_cnt_new,
                            &rbm_cnt_old,
                            candidate.ri as i64,
                            candidate.rj as i64,
                            candidate.spin,
                            &rbm_cfg,
                        );
                        crate::sampling::rbm::log_rbm_ratio(&rbm_cnt_new, &rbm_cnt_old, &rbm_cfg)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let decision =
                        metropolis_decision(log_proj_delta, rbm_delta, log_ip_new, log_ip_old, rng);
                    if decision.accepted {
                        timer.start(63);
                        crate::sampling::updates::update_m_all_complex_flat(
                            candidate.mi,
                            candidate.spin,
                            &tmp_ele_idx,
                            &state.slater_matrix.slater_elm,
                            state.slater_matrix.inv_m.as_mut_slice(),
                            inv_stride,
                            &mut state.slater_matrix.pf_m,
                            qp_start,
                            qp_end,
                            n_site,
                            n_elec,
                        );
                        timer.stop(63);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        if use_rbm {
                            rbm_cnt_old.copy_from_slice(&rbm_cnt_new);
                        }
                        state.slater_matrix.pf_m.copy_from_slice(&pf_m_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
                        state.electron_config.counter[1] += 1;
                    } else {
                        revert_ele_config(
                            candidate.mi,
                            candidate.ri,
                            candidate.rj,
                            candidate.spin,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            n_site,
                            n_elec,
                        );
                    }
                    timer.stop(32);
                }
                UpdateType::Exchange => {
                    state.electron_config.counter[2] += 1;
                    timer.start(31);
                    let candidate = make_candidate_exchange(
                        &tmp_ele_idx,
                        &tmp_ele_cfg,
                        n_site,
                        n_elec,
                        &tmp_ele_num,
                        rng,
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
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        n_site,
                        n_elec,
                    );
                    update_proj_cnt(
                        ri_old as i64,
                        rj_old as i64,
                        candidate.spin,
                        &mut proj_cnt_new,
                        &tmp_ele_proj_cnt,
                        &tmp_ele_num,
                        data,
                    );
                    update_ele_config(
                        candidate.mj,
                        rj_old,
                        ri_old,
                        candidate.spin_other,
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        n_site,
                        n_elec,
                    );
                    let mid = proj_cnt_new.clone();
                    update_proj_cnt(
                        rj_old as i64,
                        ri_old as i64,
                        candidate.spin_other,
                        &mut proj_cnt_new,
                        &mid,
                        &tmp_ele_num,
                        data,
                    );
                    timer.stop(65);
                    timer.start(66);
                    crate::sampling::updates::calculate_new_pf_m_two2_complex_flat(
                        candidate.mi,
                        candidate.spin,
                        candidate.mj,
                        candidate.spin_other,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &state.slater_matrix.slater_elm,
                        state.slater_matrix.inv_m.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = sampling_log_ip_complex(&pf_m_new, data, reducer);
                    timer.stop(67);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let rbm_delta = if use_rbm {
                        crate::sampling::rbm::update_rbm_cnt_hopping(
                            &mut rbm_cnt_new,
                            &rbm_cnt_old,
                            ri_old as i64,
                            rj_old as i64,
                            candidate.spin,
                            &rbm_cfg,
                        );
                        let mid = rbm_cnt_new.clone();
                        crate::sampling::rbm::update_rbm_cnt_hopping(
                            &mut rbm_cnt_new,
                            &mid,
                            rj_old as i64,
                            ri_old as i64,
                            candidate.spin_other,
                            &rbm_cfg,
                        );
                        crate::sampling::rbm::log_rbm_ratio(&rbm_cnt_new, &rbm_cnt_old, &rbm_cfg)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let decision =
                        metropolis_decision(log_proj_delta, rbm_delta, log_ip_new, log_ip_old, rng);
                    if decision.accepted {
                        timer.start(68);
                        crate::sampling::updates::update_m_all_two_complex_flat(
                            candidate.mi,
                            candidate.spin,
                            candidate.mj,
                            candidate.spin_other,
                            ri_old,
                            rj_old,
                            &tmp_ele_idx,
                            &state.slater_matrix.slater_elm,
                            state.slater_matrix.inv_m.as_mut_slice(),
                            inv_stride,
                            &mut state.slater_matrix.pf_m,
                            qp_start,
                            qp_end,
                            n_site,
                            n_elec,
                        );
                        timer.stop(68);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        if use_rbm {
                            rbm_cnt_old.copy_from_slice(&rbm_cnt_new);
                        }
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
                        state.electron_config.counter[3] += 1;
                    } else {
                        revert_ele_config(
                            candidate.mj,
                            rj_old,
                            ri_old,
                            candidate.spin_other,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            n_site,
                            n_elec,
                        );
                        revert_ele_config(
                            candidate.mi,
                            ri_old,
                            rj_old,
                            candidate.spin,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            n_site,
                            n_elec,
                        );
                    }
                    timer.stop(33);
                }
                _ => {}
            }

            if n_accept_window > n_site {
                timer.start(34);
                // C `CalculateMAll` (ZSKTRF operation order) unless the historical Julia kernel is
                // selected for this thread (#449).
                if !reducer.sampling_any_failure(
                    crate::pfaffian::calc_m_all_complex_production(
                        &tmp_ele_idx,
                        &state.slater_matrix.slater_elm,
                        &mut state.slater_matrix.inv_m,
                        &mut state.slater_matrix.pf_m,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                        &pool,
                    )
                    .is_err(),
                ) {
                    log_ip_old = sampling_log_ip_complex(&state.slater_matrix.pf_m, data, reducer);
                    if use_rbm {
                        rbm_cnt_old = crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg);
                        rbm_cnt_new.resize(rbm_cnt_old.len(), Complex64::new(0.0, 0.0));
                    }
                }
                n_accept_window = 0;
                timer.stop(34);
            }
        }

        if out_step + n_vmc_sample >= n_out_step {
            timer.start(35);
            let sample = out_step + n_vmc_sample - n_out_step;
            if sample < n_vmc_sample {
                state
                    .electron_config
                    .ele_idx_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_idx);
                state
                    .electron_config
                    .ele_cfg_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_cfg);
                state
                    .electron_config
                    .ele_num_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_num);
                state
                    .electron_config
                    .ele_proj_cnt_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_proj_cnt);
                saved += 1;
            }
            timer.stop(35);
        }
    }

    state.electron_config.tmp_ele_idx = tmp_ele_idx;
    state.electron_config.tmp_ele_cfg = tmp_ele_cfg;
    state.electron_config.tmp_ele_num = tmp_ele_num;
    state.electron_config.tmp_ele_proj_cnt = tmp_ele_proj_cnt;
    state.electron_config.save_burn();
    state.electron_config.counter[9] = 1;
    trace::checkpoint(state, rng);

    Ok(SampleStats {
        accepted: accepted_total,
        saved,
    })
}

pub(super) fn update_ele_config_fsz(
    mi: usize,
    org_r: usize,
    dst_r: usize,
    org_spn: u8,
    dst_spn: u8,
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_spn: &mut [i64],
    n_site: usize,
) {
    ele_idx[mi] = dst_r as i64;
    ele_spn[mi] = dst_spn as i64;
    ele_cfg[org_r + org_spn as usize * n_site] = -1;
    ele_cfg[dst_r + dst_spn as usize * n_site] = mi as i64;
    ele_num[org_r + org_spn as usize * n_site] = 0;
    ele_num[dst_r + dst_spn as usize * n_site] = 1;
}

pub(super) fn revert_ele_config_fsz(
    mi: usize,
    org_r: usize,
    dst_r: usize,
    org_spn: u8,
    dst_spn: u8,
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_spn: &mut [i64],
    n_site: usize,
) {
    ele_idx[mi] = org_r as i64;
    ele_spn[mi] = org_spn as i64;
    ele_cfg[org_r + org_spn as usize * n_site] = mi as i64;
    ele_cfg[dst_r + dst_spn as usize * n_site] = -1;
    ele_num[org_r + org_spn as usize * n_site] = 1;
    ele_num[dst_r + dst_spn as usize * n_site] = 0;
}

/// FSZ complex sampler driver (stretch path for `heisenberg_chain_fsz`).
pub fn vmc_make_sample_fsz(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
) -> SampleStats {
    vmc_make_sample_fsz_timed(data, state, rng, &mut CTimer::<false>::new())
}

/// Run the sampler with call-site-specific C timer sections.
pub fn vmc_make_sample_fsz_timed<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
) -> SampleStats {
    vmc_make_sample_fsz_with_reducer_timed(data, state, rng, timer, &SingleProcessReducer)
}

/// Complex FSZ sampler with group-only QP overlap reductions.
pub fn vmc_make_sample_fsz_with_reducer_timed<const TIMED: bool, R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
    reducer: &R,
) -> SampleStats {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let qp_range = reducer.sampling_qp_range(n_qp_full);
    let qp_start = qp_range.start;
    let qp_end = qp_range.end;
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let n_vmc_warmup = data.modpara.nvmc_warmup.max(0) as usize;
    let n_vmc_interval = data.modpara.nvmc_interval.max(0) as usize;
    let n_ex_path = data.modpara.nex_update_path;
    let i_flg_general = data.i_flg_orbital_general;
    let two_sz = data.modpara.two_sz;

    let loc_spn = {
        let ws = &mut state.workspace;
        init_loc_spn(&mut ws.loc_spn, data);
        ws.loc_spn.clone()
    };

    timer.start(30);
    let mut burn_flag = state.electron_config.counter[9] != 0;
    let pool = ThreadedPfaPackWorkspace::new(state.workspace.n_size, 1);
    if burn_flag {
        state.electron_config.restore_burn();
    } else if crate::sampling::initial::make_initial_sample_fsz_with_reducer(
        data, state, rng, qp_start, qp_end, &pool, reducer,
    )
    .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }
    let mut tmp_ele_idx = state.electron_config.tmp_ele_idx.clone();
    let mut tmp_ele_cfg = state.electron_config.tmp_ele_cfg.clone();
    let mut tmp_ele_num = state.electron_config.tmp_ele_num.clone();
    let mut tmp_ele_proj_cnt = state.electron_config.tmp_ele_proj_cnt.clone();
    let mut tmp_ele_spn = state.electron_config.tmp_ele_spn.clone();

    if reducer.sampling_any_failure(
        crate::pfaffian::calc_m_all_fsz_complex(
            &tmp_ele_idx,
            &tmp_ele_spn,
            &state.slater_matrix.slater_elm,
            &mut state.slater_matrix.inv_m,
            &mut state.slater_matrix.pf_m,
            qp_start,
            qp_end,
            n_site,
            n_elec,
            &pool,
        )
        .is_err(),
    ) {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }
    let mut log_ip_old = sampling_log_ip_complex(&state.slater_matrix.pf_m, data, reducer);

    if !log_ip_old.re.is_finite() || !log_ip_old.im.is_finite() {
        if crate::sampling::initial::make_initial_sample_fsz_with_reducer(
            data, state, rng, qp_start, qp_end, &pool, reducer,
        )
        .is_err()
        {
            return SampleStats {
                accepted: 0,
                saved: 0,
            };
        }
        let c = &state.electron_config;
        tmp_ele_idx.copy_from_slice(&c.tmp_ele_idx);
        tmp_ele_cfg.copy_from_slice(&c.tmp_ele_cfg);
        tmp_ele_num.copy_from_slice(&c.tmp_ele_num);
        tmp_ele_proj_cnt.copy_from_slice(&c.tmp_ele_proj_cnt);
        tmp_ele_spn.copy_from_slice(&c.tmp_ele_spn);
        if reducer.sampling_any_failure(
            crate::pfaffian::calc_m_all_fsz_complex(
                &tmp_ele_idx,
                &tmp_ele_spn,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                qp_start,
                qp_end,
                n_site,
                n_elec,
                &pool,
            )
            .is_err(),
        ) {
            return SampleStats {
                accepted: 0,
                saved: 0,
            };
        }
        log_ip_old = sampling_log_ip_complex(&state.slater_matrix.pf_m, data, reducer);
        burn_flag = false;
    }

    let inv_stride = n_size * n_size + 1;
    let mut pf_m_new = vec![Complex64::new(0.0, 0.0); n_qp_full];
    let mut proj_cnt_new = vec![0_i64; tmp_ele_proj_cnt.len()];
    // Issue #403: C's FSZ code has no RBM factor at all (a C defect, see the manual's
    // compatibility chapter); Rust applies it, consistently with the FSZ Hamiltonian and
    // Green-function kernels. Spin-changing moves are not supported by the incremental
    // `update_rbm_cnt_hopping`, so the counters are rebuilt from the updated occupations.
    let rbm_cfg = crate::sampling::rbm::RbmConfig::from(data);
    let use_rbm = data.has_rbm_terms() && !legacy_fsz_sampler_without_rbm();
    let mut rbm_cnt_old = if use_rbm {
        crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg)
    } else {
        Vec::new()
    };
    let mut rbm_cnt_new = rbm_cnt_old.clone();
    let n_out_step = if burn_flag {
        n_vmc_sample + 1
    } else {
        n_vmc_warmup + n_vmc_sample
    };
    let n_in_step = n_vmc_interval * n_site.max(1);
    let mut accepted_total = 0usize;
    let mut n_accept_window = 0usize;
    let mut saved = 0usize;

    timer.stop(30);
    state.electron_config.counter.fill(0);
    for out_step in 0..n_out_step {
        for _ in 0..n_in_step {
            let update_type = get_update_type(n_ex_path, i_flg_general, two_sz, rng);
            match update_type {
                UpdateType::Exchange => {
                    state.electron_config.counter[2] += 1;
                    timer.start(31);
                    let cand = make_candidate_exchange_fsz(
                        &tmp_ele_idx,
                        &tmp_ele_cfg,
                        &tmp_ele_num,
                        &tmp_ele_spn,
                        n_site,
                        n_size,
                        rng,
                    );
                    timer.stop(31);
                    if cand.reject {
                        continue;
                    }
                    timer.start(33);
                    let s = cand.spin;
                    let t = 1 - s;
                    let mi = cand.mi;
                    let ri = cand.ri;
                    let rj = cand.rj;
                    let mj = tmp_ele_cfg[rj + t as usize * n_site] as usize;

                    timer.start(65);
                    update_ele_config_fsz(
                        mi,
                        ri,
                        rj,
                        s,
                        s,
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        &mut tmp_ele_spn,
                        n_site,
                    );
                    update_proj_cnt(
                        ri as i64,
                        rj as i64,
                        s,
                        &mut proj_cnt_new,
                        &tmp_ele_proj_cnt,
                        &tmp_ele_num,
                        data,
                    );
                    update_ele_config_fsz(
                        mj,
                        rj,
                        ri,
                        t,
                        t,
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        &mut tmp_ele_spn,
                        n_site,
                    );
                    let mid = proj_cnt_new.clone();
                    update_proj_cnt(
                        rj as i64,
                        ri as i64,
                        t,
                        &mut proj_cnt_new,
                        &mid,
                        &tmp_ele_num,
                        data,
                    );
                    timer.stop(65);
                    timer.start(66);
                    crate::sampling::updates::calculate_new_pf_m_two_fsz_complex_flat(
                        mi,
                        s,
                        mj,
                        t,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &tmp_ele_spn,
                        &state.slater_matrix.slater_elm,
                        state.slater_matrix.inv_m.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = sampling_log_ip_complex(&pf_m_new, data, reducer);
                    timer.stop(67);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let rbm_delta = if use_rbm {
                        rbm_cnt_new = crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg);
                        crate::sampling::rbm::log_rbm_ratio(&rbm_cnt_new, &rbm_cnt_old, &rbm_cfg)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let decision =
                        metropolis_decision(log_proj_delta, rbm_delta, log_ip_new, log_ip_old, rng);
                    if decision.accepted {
                        timer.start(68);
                        let _ = crate::pfaffian::calc_m_all_fsz_complex(
                            &tmp_ele_idx,
                            &tmp_ele_spn,
                            &state.slater_matrix.slater_elm,
                            &mut state.slater_matrix.inv_m,
                            &mut state.slater_matrix.pf_m,
                            qp_start,
                            qp_end,
                            n_site,
                            n_elec,
                            &pool,
                        );
                        timer.stop(68);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        if use_rbm {
                            rbm_cnt_old.clone_from(&rbm_cnt_new);
                        }
                        state.electron_config.counter[3] += 1;
                        accepted_total += 1;
                        n_accept_window += 1;
                    } else {
                        revert_ele_config_fsz(
                            mj,
                            rj,
                            ri,
                            t,
                            t,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            &mut tmp_ele_spn,
                            n_site,
                        );
                        revert_ele_config_fsz(
                            mi,
                            ri,
                            rj,
                            s,
                            s,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            &mut tmp_ele_spn,
                            n_site,
                        );
                    }
                    timer.stop(33);
                }
                UpdateType::LocalSpinFlip => {
                    state.electron_config.counter[4] += 1;
                    timer.start(31);
                    let cand = make_candidate_local_spin_flip_localspin(
                        &tmp_ele_idx,
                        &tmp_ele_spn,
                        &loc_spn,
                        n_size,
                        rng,
                    );
                    timer.stop(31);
                    if cand.reject {
                        continue;
                    }
                    timer.start(36);
                    timer.start(600);
                    update_ele_config_fsz(
                        cand.mi,
                        cand.ri,
                        cand.rj,
                        cand.spin,
                        cand.spin_to,
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        &mut tmp_ele_spn,
                        n_site,
                    );
                    proj_cnt_new.copy_from_slice(&tmp_ele_proj_cnt);
                    timer.stop(600);
                    timer.start(601);
                    crate::sampling::updates::calculate_new_pf_m2_fsz_complex_flat(
                        cand.mi,
                        cand.spin_to,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &tmp_ele_spn,
                        &state.slater_matrix.slater_elm,
                        state.slater_matrix.inv_m.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    timer.stop(601);
                    timer.start(602);
                    let log_ip_new = sampling_log_ip_complex(&pf_m_new, data, reducer);
                    timer.stop(602);
                    let rbm_delta = if use_rbm {
                        rbm_cnt_new = crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg);
                        crate::sampling::rbm::log_rbm_ratio(&rbm_cnt_new, &rbm_cnt_old, &rbm_cfg)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let decision = metropolis_decision(0.0, rbm_delta, log_ip_new, log_ip_old, rng);
                    if decision.accepted {
                        timer.start(603);
                        crate::sampling::updates::update_m_all_fsz_complex_flat(
                            cand.mi,
                            cand.spin_to,
                            &tmp_ele_idx,
                            &tmp_ele_spn,
                            &state.slater_matrix.slater_elm,
                            state.slater_matrix.inv_m.as_mut_slice(),
                            inv_stride,
                            &mut state.slater_matrix.pf_m,
                            qp_start,
                            qp_end,
                            n_site,
                            n_elec,
                        );
                        timer.stop(603);
                        log_ip_old = log_ip_new;
                        if use_rbm {
                            rbm_cnt_old.clone_from(&rbm_cnt_new);
                        }
                        state.electron_config.counter[5] += 1;
                        accepted_total += 1;
                        n_accept_window += 1;
                    } else {
                        revert_ele_config_fsz(
                            cand.mi,
                            cand.ri,
                            cand.rj,
                            cand.spin,
                            cand.spin_to,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            &mut tmp_ele_spn,
                            n_site,
                        );
                    }
                    timer.stop(36);
                }
                UpdateType::Hopping => {
                    timer.start(31);
                    // C vmcmake_fsz.c: with TwoSz == -1 half of the proposals are
                    // conduction spin flips (Counter[4]/[5]); with TwoSz fixed every
                    // proposal is a hopping with unchanged spin (makeCandidate_hopping_csz,
                    // which `make_candidate_hopping_fsz` reproduces for two_sz != -1).
                    // The short-circuit keeps the draw order.
                    let flag_hop =
                        two_sz != -1 || crate::sampling::driver::trace::draw_real2(rng) < 0.5;
                    let cand = if flag_hop {
                        state.electron_config.counter[0] += 1;
                        make_candidate_hopping_fsz(
                            &tmp_ele_idx,
                            &tmp_ele_cfg,
                            &tmp_ele_spn,
                            &loc_spn,
                            n_site,
                            n_size,
                            two_sz,
                            rng,
                        )
                    } else {
                        state.electron_config.counter[4] += 1;
                        let flip = make_candidate_local_spin_flip_conduction(
                            &tmp_ele_idx,
                            &tmp_ele_cfg,
                            &tmp_ele_spn,
                            &loc_spn,
                            n_site,
                            n_size,
                            rng,
                        );
                        FszHoppingCandidate {
                            mi: flip.mi,
                            ri: flip.ri,
                            rj: flip.rj,
                            spin: flip.spin,
                            spin_to: flip.spin_to,
                            reject: flip.reject,
                        }
                    };
                    timer.stop(31);
                    if cand.reject {
                        continue;
                    }
                    timer.start(32);
                    timer.start(60);
                    update_ele_config_fsz(
                        cand.mi,
                        cand.ri,
                        cand.rj,
                        cand.spin,
                        cand.spin_to,
                        &mut tmp_ele_idx,
                        &mut tmp_ele_cfg,
                        &mut tmp_ele_num,
                        &mut tmp_ele_spn,
                        n_site,
                    );
                    update_proj_cnt(
                        cand.ri as i64,
                        cand.rj as i64,
                        cand.spin_to,
                        &mut proj_cnt_new,
                        &tmp_ele_proj_cnt,
                        &tmp_ele_num,
                        data,
                    );
                    timer.stop(60);
                    timer.start(61);
                    crate::sampling::updates::calculate_new_pf_m2_fsz_complex_flat(
                        cand.mi,
                        cand.spin_to,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &tmp_ele_spn,
                        &state.slater_matrix.slater_elm,
                        state.slater_matrix.inv_m.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = sampling_log_ip_complex(&pf_m_new, data, reducer);
                    timer.stop(62);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let rbm_delta = if use_rbm {
                        rbm_cnt_new = crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg);
                        crate::sampling::rbm::log_rbm_ratio(&rbm_cnt_new, &rbm_cnt_old, &rbm_cfg)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let decision =
                        metropolis_decision(log_proj_delta, rbm_delta, log_ip_new, log_ip_old, rng);
                    if decision.accepted {
                        timer.start(63);
                        crate::sampling::updates::update_m_all_fsz_complex_flat(
                            cand.mi,
                            cand.spin_to,
                            &tmp_ele_idx,
                            &tmp_ele_spn,
                            &state.slater_matrix.slater_elm,
                            state.slater_matrix.inv_m.as_mut_slice(),
                            inv_stride,
                            &mut state.slater_matrix.pf_m,
                            qp_start,
                            qp_end,
                            n_site,
                            n_elec,
                        );
                        timer.stop(63);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        if use_rbm {
                            rbm_cnt_old.clone_from(&rbm_cnt_new);
                        }
                        state.electron_config.counter[if flag_hop { 1 } else { 5 }] += 1;
                        accepted_total += 1;
                        n_accept_window += 1;
                    } else {
                        revert_ele_config_fsz(
                            cand.mi,
                            cand.ri,
                            cand.rj,
                            cand.spin,
                            cand.spin_to,
                            &mut tmp_ele_idx,
                            &mut tmp_ele_cfg,
                            &mut tmp_ele_num,
                            &mut tmp_ele_spn,
                            n_site,
                        );
                    }
                    timer.stop(32);
                }
                _ => {}
            }
            if n_accept_window > n_site {
                timer.start(34);
                let _ = crate::pfaffian::calc_m_all_fsz_complex(
                    &tmp_ele_idx,
                    &tmp_ele_spn,
                    &state.slater_matrix.slater_elm,
                    &mut state.slater_matrix.inv_m,
                    &mut state.slater_matrix.pf_m,
                    qp_start,
                    qp_end,
                    n_site,
                    n_elec,
                    &pool,
                );
                log_ip_old = sampling_log_ip_complex(&state.slater_matrix.pf_m, data, reducer);
                if use_rbm {
                    rbm_cnt_old = crate::sampling::rbm::make_rbm_cnt(&tmp_ele_num, &rbm_cfg);
                }
                n_accept_window = 0;
                timer.stop(34);
            }
        }
        if out_step + n_vmc_sample >= n_out_step {
            timer.start(35);
            let sample = out_step + n_vmc_sample - n_out_step;
            if sample < n_vmc_sample {
                state
                    .electron_config
                    .ele_idx_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_idx);
                state
                    .electron_config
                    .ele_spn_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_spn);
                state
                    .electron_config
                    .ele_cfg_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_cfg);
                state
                    .electron_config
                    .ele_num_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_num);
                state
                    .electron_config
                    .ele_proj_cnt_slice_mut(sample)
                    .copy_from_slice(&tmp_ele_proj_cnt);
                saved += 1;
            }
            timer.stop(35);
        }
    }
    state.electron_config.tmp_ele_idx = tmp_ele_idx;
    state.electron_config.tmp_ele_cfg = tmp_ele_cfg;
    state.electron_config.tmp_ele_num = tmp_ele_num;
    state.electron_config.tmp_ele_proj_cnt = tmp_ele_proj_cnt;
    state.electron_config.tmp_ele_spn = tmp_ele_spn;
    state.electron_config.save_burn();
    state.electron_config.counter[9] = 1;
    trace::checkpoint(state, rng);

    SampleStats {
        accepted: accepted_total,
        saved,
    }
}

// Unit-test hook: the Julia/C reference trajectories of `rbm_fsz` were generated by
// samplers without the RBM factor (a C defect, issue #403). Tests that validate the
// remaining kernels against them can switch the factor off for their own thread.
#[cfg(test)]
thread_local! {
    pub(crate) static LEGACY_FSZ_SAMPLER_WITHOUT_RBM: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn legacy_fsz_sampler_without_rbm() -> bool {
    LEGACY_FSZ_SAMPLER_WITHOUT_RBM.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn legacy_fsz_sampler_without_rbm() -> bool {
    false
}
