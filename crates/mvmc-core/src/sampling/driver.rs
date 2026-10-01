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

use crate::c_timer::CTimer;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::observables::{calculate_log_ip_complex, calculate_log_ip_real};
use crate::pfaffian::{calc_m_all_complex, calc_m_all_real};
use crate::sampling::candidate::{
    get_update_type, make_candidate_exchange, make_candidate_exchange_fsz, make_candidate_hopping,
    make_candidate_hopping_fsz, make_candidate_local_spin_flip_localspin, UpdateType,
};
use crate::sampling::initial::make_initial_sample;
use crate::sampling::metropolis::metropolis_decision;
use crate::sampling::projection::{
    init_loc_spn, log_proj_ratio, revert_ele_config, update_ele_config, update_proj_cnt,
};
use crate::sampling::updates::{
    calculate_new_pf_m2_real_flat, calculate_new_pf_m_two2_real_flat, update_m_all_real_flat,
    update_m_all_two_real_flat,
};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};

/// Output of [`vmc_make_sample_real`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleStats {
    /// Total accepted moves across the mini-loop.
    pub accepted: usize,
    /// Total saved samples (`n_vmc_sample` on success).
    pub saved: usize,
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
) -> SampleStats {
    vmc_make_sample_real_timed(data, state, rng, &mut CTimer::<false>::new())
}

/// Run the sampler with call-site-specific C timer sections.
pub fn vmc_make_sample_real_timed<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
) -> SampleStats {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let n_vmc_warmup = data.modpara.nvmc_warmup.max(0) as usize;
    let n_vmc_interval = data.modpara.nvmc_interval.max(0) as usize;
    let n_ex_path = data.modpara.nex_update_path;
    let i_flg_general = data.i_flg_orbital_general;
    let two_sz = data.modpara.two_sz;
    let n_proj = data.projection_layout().n_proj;

    let loc_spn = {
        let ws = &mut state.workspace;
        init_loc_spn(&mut ws.loc_spn, data);
        ws.loc_spn.clone()
    };

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

    // Working buffers.
    let mut tmp_ele_idx = state.electron_config.tmp_ele_idx.clone();
    let mut tmp_ele_cfg = state.electron_config.tmp_ele_cfg.clone();
    let mut tmp_ele_num = state.electron_config.tmp_ele_num.clone();
    let mut tmp_ele_proj_cnt = state.electron_config.tmp_ele_proj_cnt.clone();

    timer.start(30);
    let burn_flag = state.electron_config.counter[9] != 0;
    if burn_flag {
        tmp_ele_idx.copy_from_slice(&state.electron_config.burn_ele_idx[..n_size]);
        tmp_ele_cfg.copy_from_slice(&state.electron_config.burn_ele_cfg[..n_site2]);
        tmp_ele_num.copy_from_slice(&state.electron_config.burn_ele_num[..n_site2]);
        tmp_ele_proj_cnt.copy_from_slice(&state.electron_config.burn_ele_proj_cnt[..n_proj]);
    } else if make_initial_sample(
        &mut tmp_ele_idx,
        &mut tmp_ele_cfg,
        &mut tmp_ele_num,
        &mut tmp_ele_proj_cnt,
        data,
        &loc_spn,
        rng,
    )
    .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }

    // Initial Pfaffian / inv_m.
    let pool = ThreadedPfaPackWorkspace::new(state.workspace.n_size, 1);
    if calc_m_all_real(
        &tmp_ele_idx,
        &state.slater_matrix.slater_elm_real,
        &mut state.slater_matrix.inv_m_real,
        &mut state.slater_matrix.pf_m_real,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }
    let mut log_ip_old = calculate_log_ip_real(&state.slater_matrix.pf_m_real, 0, n_qp_full, data);

    let inv_stride = n_size * n_size + 1;
    let mut pf_m_new = vec![0.0_f64; n_qp_full];
    let mut proj_cnt_new = vec![0_i64; tmp_ele_proj_cnt.len()];
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
    for out_step in 0..n_out_step {
        for _in_step in 0..n_in_step {
            let update_type = get_update_type(n_ex_path, i_flg_general, two_sz, rng);
            match update_type {
                UpdateType::Hopping => {
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
                    calculate_new_pf_m2_real_flat(
                        candidate.mi,
                        candidate.spin,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &state.slater_matrix.slater_elm_real,
                        state.slater_matrix.inv_m_real.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m_real,
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = calculate_log_ip_real(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(62);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(log_ip_new, 0.0),
                        Complex64::new(log_ip_old, 0.0),
                        rng,
                    );
                    if decision.accepted {
                        timer.start(63);
                        update_m_all_real_flat(
                            candidate.mi,
                            candidate.spin,
                            &tmp_ele_idx,
                            &state.slater_matrix.slater_elm_real,
                            state.slater_matrix.inv_m_real.as_mut_slice(),
                            inv_stride,
                            &mut state.slater_matrix.pf_m_real,
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                        );
                        timer.stop(63);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        state.slater_matrix.pf_m_real.copy_from_slice(&pf_m_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
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
                    calculate_new_pf_m_two2_real_flat(
                        candidate.mi,
                        candidate.spin,
                        candidate.mj,
                        candidate.spin_other,
                        &mut pf_m_new,
                        &tmp_ele_idx,
                        &state.slater_matrix.slater_elm_real,
                        state.slater_matrix.inv_m_real.as_slice(),
                        inv_stride,
                        &state.slater_matrix.pf_m_real,
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = calculate_log_ip_real(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(67);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(log_ip_new, 0.0),
                        Complex64::new(log_ip_old, 0.0),
                        rng,
                    );
                    if decision.accepted {
                        timer.start(68);
                        update_m_all_two_real_flat(
                            candidate.mi,
                            candidate.spin,
                            candidate.mj,
                            candidate.spin_other,
                            ri_old,
                            rj_old,
                            &tmp_ele_idx,
                            &state.slater_matrix.slater_elm_real,
                            state.slater_matrix.inv_m_real.as_mut_slice(),
                            inv_stride,
                            &mut state.slater_matrix.pf_m_real,
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                        );
                        timer.stop(68);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
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
                if calc_m_all_real(
                    &tmp_ele_idx,
                    &state.slater_matrix.slater_elm_real,
                    &mut state.slater_matrix.inv_m_real,
                    &mut state.slater_matrix.pf_m_real,
                    0,
                    n_qp_full,
                    n_site,
                    n_elec,
                    &pool,
                )
                .is_ok()
                {
                    log_ip_old =
                        calculate_log_ip_real(&state.slater_matrix.pf_m_real, 0, n_qp_full, data);
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
    state.electron_config.burn_ele_idx[..n_size]
        .copy_from_slice(&state.electron_config.tmp_ele_idx[..n_size]);
    state.electron_config.burn_ele_cfg[..n_site2]
        .copy_from_slice(&state.electron_config.tmp_ele_cfg[..n_site2]);
    state.electron_config.burn_ele_num[..n_site2]
        .copy_from_slice(&state.electron_config.tmp_ele_num[..n_site2]);
    state.electron_config.burn_ele_proj_cnt[..n_proj]
        .copy_from_slice(&state.electron_config.tmp_ele_proj_cnt[..n_proj]);
    state.electron_config.counter[9] = 1;

    SampleStats {
        accepted: accepted_total,
        saved,
    }
}

/// Complex-arithmetic counterpart of [`vmc_make_sample_real`]. Mirrors
/// the layout of `vmc_make_sample!` (sz-conserved branch) and only
/// substitutes the complex Pfaffian / inverse / Slater buffers for
/// their real-mode siblings.
pub fn vmc_make_sample(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
) -> SampleStats {
    vmc_make_sample_timed(data, state, rng, &mut CTimer::<false>::new())
}

/// Run the sampler with call-site-specific C timer sections.
pub fn vmc_make_sample_timed<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timer: &mut CTimer<TIMED>,
) -> SampleStats {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let n_vmc_warmup = data.modpara.nvmc_warmup.max(0) as usize;
    let n_vmc_interval = data.modpara.nvmc_interval.max(0) as usize;
    let n_ex_path = data.modpara.nex_update_path;
    let i_flg_general = data.i_flg_orbital_general;
    let two_sz = data.modpara.two_sz;
    let n_proj = data.projection_layout().n_proj;

    let loc_spn = {
        let ws = &mut state.workspace;
        init_loc_spn(&mut ws.loc_spn, data);
        ws.loc_spn.clone()
    };

    let mut tmp_ele_idx = state.electron_config.tmp_ele_idx.clone();
    let mut tmp_ele_cfg = state.electron_config.tmp_ele_cfg.clone();
    let mut tmp_ele_num = state.electron_config.tmp_ele_num.clone();
    let mut tmp_ele_proj_cnt = state.electron_config.tmp_ele_proj_cnt.clone();

    timer.start(30);
    let burn_flag = state.electron_config.counter[9] != 0;
    if burn_flag {
        tmp_ele_idx.copy_from_slice(&state.electron_config.burn_ele_idx[..n_size]);
        tmp_ele_cfg.copy_from_slice(&state.electron_config.burn_ele_cfg[..n_site2]);
        tmp_ele_num.copy_from_slice(&state.electron_config.burn_ele_num[..n_site2]);
        tmp_ele_proj_cnt.copy_from_slice(&state.electron_config.burn_ele_proj_cnt[..n_proj]);
    } else if make_initial_sample(
        &mut tmp_ele_idx,
        &mut tmp_ele_cfg,
        &mut tmp_ele_num,
        &mut tmp_ele_proj_cnt,
        data,
        &loc_spn,
        rng,
    )
    .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }

    let pool = ThreadedPfaPackWorkspace::new(state.workspace.n_size, 1);
    if calc_m_all_complex(
        &tmp_ele_idx,
        &state.slater_matrix.slater_elm,
        &mut state.slater_matrix.inv_m,
        &mut state.slater_matrix.pf_m,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }
    let mut log_ip_old = calculate_log_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data);

    let inv_stride = n_size * n_size + 1;
    let mut pf_m_new = vec![Complex64::new(0.0, 0.0); n_qp_full];
    let mut proj_cnt_new = vec![0_i64; tmp_ele_proj_cnt.len()];
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
    for out_step in 0..n_out_step {
        for _in_step in 0..n_in_step {
            let update_type = get_update_type(n_ex_path, i_flg_general, two_sz, rng);
            match update_type {
                UpdateType::Hopping => {
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
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = calculate_log_ip_complex(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(62);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        log_ip_new,
                        log_ip_old,
                        rng,
                    );
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
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                        );
                        timer.stop(63);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        state.slater_matrix.pf_m.copy_from_slice(&pf_m_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
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
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = calculate_log_ip_complex(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(67);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        log_ip_new,
                        log_ip_old,
                        rng,
                    );
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
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                        );
                        timer.stop(68);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        accepted_total += 1;
                        n_accept_window += 1;
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
                if calc_m_all_complex(
                    &tmp_ele_idx,
                    &state.slater_matrix.slater_elm,
                    &mut state.slater_matrix.inv_m,
                    &mut state.slater_matrix.pf_m,
                    0,
                    n_qp_full,
                    n_site,
                    n_elec,
                    &pool,
                )
                .is_ok()
                {
                    log_ip_old =
                        calculate_log_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data);
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
    state.electron_config.burn_ele_idx[..n_size]
        .copy_from_slice(&state.electron_config.tmp_ele_idx[..n_size]);
    state.electron_config.burn_ele_cfg[..n_site2]
        .copy_from_slice(&state.electron_config.tmp_ele_cfg[..n_site2]);
    state.electron_config.burn_ele_num[..n_site2]
        .copy_from_slice(&state.electron_config.tmp_ele_num[..n_site2]);
    state.electron_config.burn_ele_proj_cnt[..n_proj]
        .copy_from_slice(&state.electron_config.tmp_ele_proj_cnt[..n_proj]);
    state.electron_config.counter[9] = 1;

    SampleStats {
        accepted: accepted_total,
        saved,
    }
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
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_qp_full = state.slater_matrix.pf_m.len();
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
    let burn_flag = state.electron_config.counter[9] != 0;
    if burn_flag {
        state.electron_config.restore_burn_fsz();
    }
    let mut tmp_ele_idx = state.electron_config.tmp_ele_idx.clone();
    let mut tmp_ele_cfg = state.electron_config.tmp_ele_cfg.clone();
    let mut tmp_ele_num = state.electron_config.tmp_ele_num.clone();
    let mut tmp_ele_proj_cnt = state.electron_config.tmp_ele_proj_cnt.clone();
    let mut tmp_ele_spn = state.electron_config.tmp_ele_spn.clone();

    if !burn_flag
        && crate::sampling::initial::make_initial_sample_fsz(
            &mut tmp_ele_idx,
            &mut tmp_ele_cfg,
            &mut tmp_ele_num,
            &mut tmp_ele_proj_cnt,
            &mut tmp_ele_spn,
            data,
            &loc_spn,
            rng,
        )
        .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }

    let pool = ThreadedPfaPackWorkspace::new(state.workspace.n_size, 1);
    if crate::pfaffian::calc_m_all_fsz_complex(
        &tmp_ele_idx,
        &tmp_ele_spn,
        &state.slater_matrix.slater_elm,
        &mut state.slater_matrix.inv_m,
        &mut state.slater_matrix.pf_m,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .is_err()
    {
        return SampleStats {
            accepted: 0,
            saved: 0,
        };
    }
    let mut log_ip_old = calculate_log_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data);

    let inv_stride = n_size * n_size + 1;
    let mut pf_m_new = vec![Complex64::new(0.0, 0.0); n_qp_full];
    let mut proj_cnt_new = vec![0_i64; tmp_ele_proj_cnt.len()];
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
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(66);
                    timer.start(67);
                    let log_ip_new = calculate_log_ip_complex(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(67);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        log_ip_new,
                        log_ip_old,
                        rng,
                    );
                    if decision.accepted {
                        timer.start(68);
                        let _ = crate::pfaffian::calc_m_all_fsz_complex(
                            &tmp_ele_idx,
                            &tmp_ele_spn,
                            &state.slater_matrix.slater_elm,
                            &mut state.slater_matrix.inv_m,
                            &mut state.slater_matrix.pf_m,
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                            &pool,
                        );
                        timer.stop(68);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
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
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(601);
                    timer.start(602);
                    let log_ip_new = calculate_log_ip_complex(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(602);
                    let decision = metropolis_decision(
                        0.0,
                        Complex64::new(0.0, 0.0),
                        log_ip_new,
                        log_ip_old,
                        rng,
                    );
                    if decision.accepted {
                        timer.start(603);
                        let _ = crate::pfaffian::calc_m_all_fsz_complex(
                            &tmp_ele_idx,
                            &tmp_ele_spn,
                            &state.slater_matrix.slater_elm,
                            &mut state.slater_matrix.inv_m,
                            &mut state.slater_matrix.pf_m,
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                            &pool,
                        );
                        timer.stop(603);
                        log_ip_old = log_ip_new;
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
                    state.electron_config.counter[0] += 1;
                    timer.start(31);
                    let cand = make_candidate_hopping_fsz(
                        &tmp_ele_idx,
                        &tmp_ele_cfg,
                        &tmp_ele_spn,
                        &loc_spn,
                        n_site,
                        n_size,
                        two_sz,
                        rng,
                    );
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
                        0,
                        n_qp_full,
                        n_site,
                        n_elec,
                    );
                    timer.stop(61);
                    timer.start(62);
                    let log_ip_new = calculate_log_ip_complex(&pf_m_new, 0, n_qp_full, data);
                    timer.stop(62);
                    let log_proj_delta = log_proj_ratio(&proj_cnt_new, &tmp_ele_proj_cnt, data);
                    let decision = metropolis_decision(
                        log_proj_delta,
                        Complex64::new(0.0, 0.0),
                        log_ip_new,
                        log_ip_old,
                        rng,
                    );
                    if decision.accepted {
                        timer.start(63);
                        let _ = crate::pfaffian::calc_m_all_fsz_complex(
                            &tmp_ele_idx,
                            &tmp_ele_spn,
                            &state.slater_matrix.slater_elm,
                            &mut state.slater_matrix.inv_m,
                            &mut state.slater_matrix.pf_m,
                            0,
                            n_qp_full,
                            n_site,
                            n_elec,
                            &pool,
                        );
                        timer.stop(63);
                        tmp_ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                        log_ip_old = log_ip_new;
                        state.electron_config.counter[1] += 1;
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
                    0,
                    n_qp_full,
                    n_site,
                    n_elec,
                    &pool,
                );
                log_ip_old =
                    calculate_log_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data);
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
    state.electron_config.save_burn_fsz();
    state.electron_config.counter[9] = 1;

    SampleStats {
        accepted: accepted_total,
        saved,
    }
}
