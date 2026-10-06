//! Real FSZ sampling with explicit electron spins, mirroring Julia's driver.
use mvmc_expert_parsers::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use super::candidate::{
    get_update_type, make_candidate_exchange_fsz, make_candidate_hopping_fsz,
    make_candidate_local_spin_flip_conduction, make_candidate_local_spin_flip_localspin,
    UpdateType,
};
use super::driver::sampling_log_ip_real;
use super::driver::{revert_ele_config_fsz, update_ele_config_fsz, SampleStats};
use super::initial::{
    coordinate_initialization_result, make_initial_sample_fsz_real_with_reducer,
    SamplingInitializationError,
};
use super::projection::{init_loc_spn, log_proj_ratio, update_proj_cnt};
use super::updates::{
    calculate_new_pf_m2_fsz_real_flat, calculate_new_pf_m_two_fsz_real_flat,
    update_m_all_fsz_real_flat, update_m_all_two_fsz_real_flat,
};
use crate::pfaffian::{calc_m_all_fsz_real, CalcMAllError};
use crate::reducer::{Reducer, SingleProcessReducer};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};

/// Run one real FSZ sampling sweep, including burn restoration, real move
/// updates, Julia's proposal/Metropolis draws and six attempt/accept counters.
/// Initial Pfaffian failures propagate before saved samples are published.
pub fn vmc_make_sample_fsz_real(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
) -> Result<SampleStats, CalcMAllError> {
    vmc_make_sample_fsz_real_with_reducer(data, state, rng, &SingleProcessReducer).map_err(
        |error| match error {
            SamplingInitializationError::Local(error) => error,
            SamplingInitializationError::PeerFailure => unreachable!("serial reducer has no peers"),
        },
    )
}

/// Real FSZ chain with comm1-only projected-overlap reductions.
pub fn vmc_make_sample_fsz_real_with_reducer<R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    reducer: &R,
) -> Result<SampleStats, SamplingInitializationError> {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_size = 2 * n_elec;
    let n_qp_full = state.slater_matrix.pf_m_real.len();
    let qp_range = reducer.sampling_qp_range(n_qp_full);
    let qp_start = qp_range.start;
    let qp_end = qp_range.end;
    let n_proj = data.projection_layout().n_proj;
    let n_sample = data.modpara.nvmc_sample as usize;
    let pool = ThreadedPfaPackWorkspace::new(n_size, 1);
    init_loc_spn(&mut state.workspace.loc_spn, data);
    let loc_spn = state.workspace.loc_spn.clone();
    for (dst, src) in state
        .slater_matrix
        .slater_elm_real
        .as_mut_slice()
        .iter_mut()
        .zip(state.slater_matrix.slater_elm.as_slice())
    {
        *dst = src.re;
    }
    // Rust compresses Julia's reserved counter slot 10; slot 9 is the
    // Julia counter[11] burn marker, shared with the existing Rust drivers.
    let mut burn_flag = state.electron_config.counter[9] != 0;
    if !burn_flag {
        make_initial_sample_fsz_real_with_reducer(
            data, state, rng, qp_start, qp_end, &pool, reducer,
        )?;
    }
    let config = &mut state.electron_config;
    if burn_flag {
        config.restore_burn();
    }
    coordinate_initialization_result(
        calc_m_all_fsz_real(
            &config.tmp_ele_idx,
            &config.tmp_ele_spn,
            &mut state.slater_matrix,
            qp_start,
            qp_end,
            n_site,
            n_elec,
            &pool,
        ),
        reducer,
    )?;
    let mut log_ip_old = sampling_log_ip_real(&state.slater_matrix.pf_m_real, data, reducer);
    if !log_ip_old.is_finite() {
        make_initial_sample_fsz_real_with_reducer(
            data, state, rng, qp_start, qp_end, &pool, reducer,
        )?;
        let config = &state.electron_config;
        coordinate_initialization_result(
            calc_m_all_fsz_real(
                &config.tmp_ele_idx,
                &config.tmp_ele_spn,
                &mut state.slater_matrix,
                qp_start,
                qp_end,
                n_site,
                n_elec,
                &pool,
            ),
            reducer,
        )?;
        log_ip_old = sampling_log_ip_real(&state.slater_matrix.pf_m_real, data, reducer);
        burn_flag = false;
    }
    let config = &mut state.electron_config;
    config.counter.fill(0);
    let mut ele_idx = config.tmp_ele_idx.clone();
    let mut ele_cfg = config.tmp_ele_cfg.clone();
    let mut ele_num = config.tmp_ele_num.clone();
    let mut ele_proj_cnt = config.tmp_ele_proj_cnt.clone();
    let mut ele_spn = config.tmp_ele_spn.clone();
    let mut proj_cnt_new = vec![0; n_proj];
    let mut pf_m_new = vec![0.0; n_qp_full];
    let stride = n_size * n_size + 1;
    let n_out_step = if burn_flag {
        n_sample + 1
    } else {
        data.modpara.nvmc_warmup as usize + n_sample
    };
    let n_in_step = data.modpara.nvmc_interval as usize * n_site;
    let mut window_accepted = 0;
    let mut accepted = 0;
    let mut saved = 0;
    for out_step in 0..n_out_step {
        for _ in 0..n_in_step {
            let kind = get_update_type(
                data.modpara.nex_update_path,
                data.i_flg_orbital_general,
                data.modpara.two_sz,
                rng,
            );
            if kind == UpdateType::Exchange {
                config.counter[2] += 1;
                let cand = make_candidate_exchange_fsz(
                    &ele_idx, &ele_cfg, &ele_num, &ele_spn, n_site, n_size, rng,
                );
                if cand.reject {
                    continue;
                }
                let (mi, ri, rj, s) = (cand.mi, cand.ri, cand.rj, cand.spin);
                let t = 1 - s;
                let mj = ele_cfg[rj + t as usize * n_site] as usize;
                update_ele_config_fsz(
                    mi,
                    ri,
                    rj,
                    s,
                    s,
                    &mut ele_idx,
                    &mut ele_cfg,
                    &mut ele_num,
                    &mut ele_spn,
                    n_site,
                );
                update_proj_cnt(
                    ri as i64,
                    rj as i64,
                    s,
                    &mut proj_cnt_new,
                    &ele_proj_cnt,
                    &ele_num,
                    data,
                );
                update_ele_config_fsz(
                    mj,
                    rj,
                    ri,
                    t,
                    t,
                    &mut ele_idx,
                    &mut ele_cfg,
                    &mut ele_num,
                    &mut ele_spn,
                    n_site,
                );
                let mid = proj_cnt_new.clone();
                update_proj_cnt(
                    rj as i64,
                    ri as i64,
                    t,
                    &mut proj_cnt_new,
                    &mid,
                    &ele_num,
                    data,
                );
                let matrix = &mut state.slater_matrix;
                calculate_new_pf_m_two_fsz_real_flat(
                    mi,
                    s,
                    mj,
                    t,
                    &mut pf_m_new,
                    &ele_idx,
                    &ele_spn,
                    &matrix.slater_elm_real,
                    matrix.inv_m_real.as_slice(),
                    stride,
                    &matrix.pf_m_real,
                    qp_start,
                    qp_end,
                    n_site,
                    n_elec,
                );
                let log_ip_new = sampling_log_ip_real(&pf_m_new, data, reducer);
                if accept(
                    log_proj_ratio(&proj_cnt_new, &ele_proj_cnt, data),
                    log_ip_new,
                    log_ip_old,
                    rng,
                ) {
                    update_m_all_two_fsz_real_flat(
                        mi,
                        s,
                        mj,
                        t,
                        ri,
                        rj,
                        &ele_idx,
                        &ele_spn,
                        &matrix.slater_elm_real,
                        matrix.inv_m_real.as_mut_slice(),
                        stride,
                        &mut matrix.pf_m_real,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                    log_ip_old = log_ip_new;
                    accepted += 1;
                    window_accepted += 1;
                    config.counter[3] += 1;
                } else {
                    revert_ele_config_fsz(
                        mj,
                        rj,
                        ri,
                        t,
                        t,
                        &mut ele_idx,
                        &mut ele_cfg,
                        &mut ele_num,
                        &mut ele_spn,
                        n_site,
                    );
                    revert_ele_config_fsz(
                        mi,
                        ri,
                        rj,
                        s,
                        s,
                        &mut ele_idx,
                        &mut ele_cfg,
                        &mut ele_num,
                        &mut ele_spn,
                        n_site,
                    );
                }
            } else if matches!(kind, UpdateType::Hopping | UpdateType::LocalSpinFlip) {
                let (mi, ri, rj, s, t, reject, hop) = if kind == UpdateType::LocalSpinFlip {
                    config.counter[4] += 1;
                    let cand = make_candidate_local_spin_flip_localspin(
                        &ele_idx, &ele_spn, &loc_spn, n_size, rng,
                    );
                    (
                        cand.mi,
                        cand.ri,
                        cand.rj,
                        cand.spin,
                        cand.spin_to,
                        cand.reject,
                        false,
                    )
                } else if data.modpara.two_sz != -1
                    || crate::sampling::driver::trace::draw_real2(rng) < 0.5
                {
                    config.counter[0] += 1;
                    let cand = make_candidate_hopping_fsz(
                        &ele_idx,
                        &ele_cfg,
                        &ele_spn,
                        &loc_spn,
                        n_site,
                        n_size,
                        data.modpara.two_sz,
                        rng,
                    );
                    (
                        cand.mi,
                        cand.ri,
                        cand.rj,
                        cand.spin,
                        cand.spin_to,
                        cand.reject,
                        true,
                    )
                } else {
                    config.counter[4] += 1;
                    let cand = make_candidate_local_spin_flip_conduction(
                        &ele_idx, &ele_cfg, &ele_spn, &loc_spn, n_site, n_size, rng,
                    );
                    (
                        cand.mi,
                        cand.ri,
                        cand.rj,
                        cand.spin,
                        cand.spin_to,
                        cand.reject,
                        false,
                    )
                };
                if reject {
                    continue;
                }
                update_ele_config_fsz(
                    mi,
                    ri,
                    rj,
                    s,
                    t,
                    &mut ele_idx,
                    &mut ele_cfg,
                    &mut ele_num,
                    &mut ele_spn,
                    n_site,
                );
                // Julia's FSZ projection update uses the same post-move
                // occupations/count arithmetic as update_proj_cnt; s/t are
                // not read. A spin flip at ri == rj leaves counts unchanged.
                update_proj_cnt(
                    ri as i64,
                    rj as i64,
                    s,
                    &mut proj_cnt_new,
                    &ele_proj_cnt,
                    &ele_num,
                    data,
                );
                let matrix = &mut state.slater_matrix;
                calculate_new_pf_m2_fsz_real_flat(
                    mi,
                    t,
                    &mut pf_m_new,
                    &ele_idx,
                    &ele_spn,
                    &matrix.slater_elm_real,
                    matrix.inv_m_real.as_slice(),
                    stride,
                    &matrix.pf_m_real,
                    qp_start,
                    qp_end,
                    n_site,
                    n_elec,
                );
                let log_ip_new = sampling_log_ip_real(&pf_m_new, data, reducer);
                if accept(
                    log_proj_ratio(&proj_cnt_new, &ele_proj_cnt, data),
                    log_ip_new,
                    log_ip_old,
                    rng,
                ) {
                    update_m_all_fsz_real_flat(
                        mi,
                        t,
                        &ele_idx,
                        &ele_spn,
                        &matrix.slater_elm_real,
                        matrix.inv_m_real.as_mut_slice(),
                        stride,
                        &mut matrix.pf_m_real,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                    );
                    ele_proj_cnt.copy_from_slice(&proj_cnt_new);
                    log_ip_old = log_ip_new;
                    accepted += 1;
                    window_accepted += 1;
                    config.counter[if hop { 1 } else { 5 }] += 1;
                } else {
                    revert_ele_config_fsz(
                        mi,
                        ri,
                        rj,
                        s,
                        t,
                        &mut ele_idx,
                        &mut ele_cfg,
                        &mut ele_num,
                        &mut ele_spn,
                        n_site,
                    );
                }
            }
            if window_accepted > n_site {
                if !reducer.sampling_any_failure(
                    calc_m_all_fsz_real(
                        &ele_idx,
                        &ele_spn,
                        &mut state.slater_matrix,
                        qp_start,
                        qp_end,
                        n_site,
                        n_elec,
                        &pool,
                    )
                    .is_err(),
                ) {
                    log_ip_old =
                        sampling_log_ip_real(&state.slater_matrix.pf_m_real, data, reducer);
                }
                window_accepted = 0;
            }
        }
        if out_step >= n_out_step - n_sample {
            let sample = out_step - (n_out_step - n_sample);
            config.ele_idx_slice_mut(sample).copy_from_slice(&ele_idx);
            config.ele_cfg_slice_mut(sample).copy_from_slice(&ele_cfg);
            config.ele_num_slice_mut(sample).copy_from_slice(&ele_num);
            config
                .ele_proj_cnt_slice_mut(sample)
                .copy_from_slice(&ele_proj_cnt);
            config.ele_spn_slice_mut(sample).copy_from_slice(&ele_spn);
            saved += 1;
        }
    }
    config.tmp_ele_idx = ele_idx;
    config.tmp_ele_cfg = ele_cfg;
    config.tmp_ele_num = ele_num;
    config.tmp_ele_proj_cnt = ele_proj_cnt;
    config.tmp_ele_spn = ele_spn;
    config.save_burn();
    config.counter[9] = 1;
    super::driver::trace::checkpoint(state, rng);
    Ok(SampleStats { accepted, saved })
}

fn accept(delta: f64, log_ip_new: f64, log_ip_old: f64, rng: &mut Sfmt19937Rng) -> bool {
    let w = mvmc_expert_parsers::utils::c_math::exp(2.0 * (delta + (log_ip_new - log_ip_old)));
    let w = if w.is_finite() { w } else { -1.0 };
    let draw = crate::sampling::driver::trace::draw_real2(rng);
    super::driver::trace::record_decision(w, draw);
    super::driver::trace::record(7, &[i64::from(w > draw), (draw * 4294967296.0) as i64]);
    w > draw
}
