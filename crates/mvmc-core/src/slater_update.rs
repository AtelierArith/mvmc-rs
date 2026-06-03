//! Phase 4 — Slater-element table updater.
//!
//! Port target: `update_slater_elm_fcmp!` in
//! `MVMCOptimizers.jl/src/slater_update.jl`. The Phase-4 cut covers
//! the `i_flg_orbital_general == 0` branch with `OrbitalAntiParallel`
//! tables (the Heisenberg / Hubbard examples). Full FSZ and
//! `OrbitalParallel` translations land alongside the full FSZ port.

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::state::VmcOptimizationState;

/// `update_slater_elm_fcmp!(data, state)` mirror for the
/// `i_flg_orbital_general == 0` path. The Slater table is rebuilt for
/// every QP plane using the cached orbital-idx matrix and the parsed
/// `qptransidx.def` translation maps. `QPOptTrans` is identity in the
/// v0.1 fixtures (`n_qp_opt_trans == 1`).
pub fn update_slater_elm(data: &mut ExpertModeData, state: &mut VmcOptimizationState) {
    data.ensure_orbital_idx_matrix();
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_site2 = 2 * n_site;
    let weights = match data.qp_weights.as_ref() {
        Some(w) => w,
        None => return,
    };
    let n_sp_gauss_leg = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp_trans = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_qp_opt_trans = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_full = n_sp_gauss_leg * n_mp_trans * n_qp_opt_trans;

    let orb_idx = match data.orbital_idx_matrix.as_ref() {
        Some(m) => m,
        None => return,
    };
    let orb_sgn = data
        .orbital_sgn_matrix
        .as_ref()
        .expect("orbital_sgn_matrix populated alongside orbital_idx_matrix");
    let n_orb = data.modpara.n_orbital_idx.max(0) as usize;
    if n_orb == 0 {
        return;
    }

    let mut slater = vec![Complex64::new(0.0, 0.0); n_orb];
    for term in &data.orbital_terms {
        let idx = term.idx;
        if idx >= 0 && (idx as usize) < n_orb {
            slater[idx as usize] = term.value;
        }
    }

    for qp in 0..n_qp_full {
        let n_qp_fix = n_sp_gauss_leg * n_mp_trans;
        let rem = qp % n_qp_fix;
        let mpidx = rem / n_sp_gauss_leg;
        let spidx = qp % n_sp_gauss_leg;
        let cs = weights.spgl_cos_sin[spidx];
        let cc = weights.spgl_cos_cos[spidx];
        let ss = weights.spgl_sin_sin[spidx];
        let trans = data.qp_trans_entries.get(mpidx);

        let qp_offset = qp * n_site2 * n_site2;
        for ri in 0..n_site {
            let ori = ri;
            let tri = trans
                .and_then(|t| t.site_map.get(ori))
                .copied()
                .unwrap_or(ori as i64) as usize;
            let sgni = trans
                .and_then(|t| t.site_sign.get(ori))
                .copied()
                .unwrap_or(1);
            if tri >= n_site {
                continue;
            }
            for rj in 0..n_site {
                let orj = rj;
                let trj = trans
                    .and_then(|t| t.site_map.get(orj))
                    .copied()
                    .unwrap_or(orj as i64) as usize;
                let sgnj = trans
                    .and_then(|t| t.site_sign.get(orj))
                    .copied()
                    .unwrap_or(1);
                if trj >= n_site {
                    continue;
                }
                let qpsgn = sgni * sgnj;
                let idx_ij = orb_idx[tri][trj];
                let idx_ji = orb_idx[trj][tri];
                if idx_ij < 0 || idx_ji < 0 {
                    continue;
                }
                let sgn_ij = orb_sgn[tri][trj] * qpsgn;
                let sgn_ji = orb_sgn[trj][tri] * qpsgn;
                let slt_ij = slater[idx_ij as usize] * Complex64::new(sgn_ij as f64, 0.0);
                let slt_ji = slater[idx_ji as usize] * Complex64::new(sgn_ji as f64, 0.0);

                let rsi0 = ri;
                let rsi1 = ri + n_site;
                let rsj0 = rj;
                let rsj1 = rj + n_site;

                let row0 = qp_offset + rsi0 * n_site2;
                let row1 = qp_offset + rsi1 * n_site2;

                let flat = state.slater_matrix.slater_elm.as_mut_slice();
                flat[row0 + rsj0] = -(slt_ij - slt_ji) * cs;
                flat[row0 + rsj1] = slt_ij * cc + slt_ji * ss;
                flat[row1 + rsj0] = -slt_ij * ss - slt_ji * cc;
                flat[row1 + rsj1] = (slt_ij - slt_ji) * cs;
            }
        }
    }

    // Real shadow buffer: refresh the per-QP real plane from the
    // freshly-updated complex master so the real-mode sampler sees a
    // consistent view.
    let n_real = state
        .slater_matrix
        .slater_elm
        .len()
        .min(state.slater_matrix.slater_elm_real.len());
    for i in 0..n_real {
        state.slater_matrix.slater_elm_real.as_mut_slice()[i] =
            state.slater_matrix.slater_elm.as_slice()[i].re;
    }
}

/// `update_slater_elm_fsz!(data, state)` mirror for the AP+P FSZ fixtures.
///
/// FSZ uses an explicit `2*n_site × 2*n_site` orbital matrix and does not
/// apply spin projection (`NSPGaussLeg` is treated as 1). QP translation is
/// applied to the site index and the spin offset is kept explicit.
pub fn update_slater_elm_fsz(data: &mut ExpertModeData, state: &mut VmcOptimizationState) {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_site2 = 2 * n_site;
    if n_site == 0 {
        return;
    }
    let n_sp_gauss_leg = 1usize;
    let n_mp_trans = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_qp_opt_trans = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_fix = n_sp_gauss_leg * n_mp_trans;
    let n_qp_full = n_qp_fix * n_qp_opt_trans;

    let (orbital_idx, orbital_sgn, slater) = build_orbital_idx_sgn_matrices_fsz(data, n_site);
    for qp in 0..n_qp_full {
        let rem = qp % n_qp_fix;
        let mpidx = rem / n_sp_gauss_leg;
        let trans = data.qp_trans_entries.get(mpidx);
        let qp_offset = qp * n_site2 * n_site2;
        for ri in 0..n_site {
            let ori = ri;
            let tri = trans
                .and_then(|t| t.site_map.get(ori))
                .copied()
                .unwrap_or(ori as i64) as usize;
            let sgni = trans
                .and_then(|t| t.site_sign.get(ori))
                .copied()
                .unwrap_or(1);
            if tri >= n_site {
                continue;
            }
            for rj in 0..n_site {
                let orj = rj;
                let trj = trans
                    .and_then(|t| t.site_map.get(orj))
                    .copied()
                    .unwrap_or(orj as i64) as usize;
                let sgnj = trans
                    .and_then(|t| t.site_sign.get(orj))
                    .copied()
                    .unwrap_or(1);
                if trj >= n_site {
                    continue;
                }
                let qpsgn = sgni * sgnj;
                for si in 0..2 {
                    for sj in 0..2 {
                        let rsi = ri + si * n_site;
                        let rsj = rj + sj * n_site;
                        let tri_s = tri + si * n_site;
                        let trj_s = trj + sj * n_site;
                        let idx_ij = orbital_idx[tri_s][trj_s];
                        let idx_ji = orbital_idx[trj_s][tri_s];
                        let slt_ij = if idx_ij >= 0 && (idx_ij as usize) < slater.len() {
                            slater[idx_ij as usize]
                                * Complex64::new((orbital_sgn[tri_s][trj_s] * qpsgn) as f64, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        let slt_ji = if idx_ji >= 0 && (idx_ji as usize) < slater.len() {
                            slater[idx_ji as usize]
                                * Complex64::new((orbital_sgn[trj_s][tri_s] * qpsgn) as f64, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        state.slater_matrix.slater_elm.as_mut_slice()
                            [qp_offset + rsi * n_site2 + rsj] = slt_ij - slt_ji;
                    }
                }
            }
        }
    }
}

pub(crate) fn build_orbital_idx_sgn_matrices_fsz(
    data: &ExpertModeData,
    n_site: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Complex64>) {
    let n_site2 = 2 * n_site;
    let mut orbital_idx = vec![vec![-1_i64; n_site2]; n_site2];
    let mut orbital_sgn = vec![vec![0_i64; n_site2]; n_site2];
    let max_idx = data.orbital_terms.iter().map(|t| t.idx).max().unwrap_or(0);
    let mut slater = vec![Complex64::new(0.0, 0.0); (max_idx + 1).max(1) as usize];

    for term in &data.orbital_terms {
        if term.idx >= 0 && (term.idx as usize) < slater.len() {
            slater[term.idx as usize] = term.value;
        }
    }

    let n_anti_parallel =
        if data.i_flg_orbital_anti_parallel == 1 && data.i_flg_orbital_parallel == 1 {
            let mut pair_min = i64::MAX;
            for (i, lhs) in data.orbital_terms.iter().enumerate() {
                for rhs in data.orbital_terms.iter().skip(i + 1) {
                    if lhs.site1 == rhs.site1
                        && lhs.site2 == rhs.site2
                        && lhs.sign == rhs.sign
                        && (lhs.idx - rhs.idx).abs() == 1
                    {
                        pair_min = pair_min.min(lhs.idx.min(rhs.idx));
                    }
                }
            }
            if pair_min == i64::MAX {
                (n_site * n_site) as i64
            } else {
                pair_min
            }
        } else {
            0
        };

    for term in &data.orbital_terms {
        if term.site1 < 0 || term.site2 < 0 {
            continue;
        }
        let site1 = term.site1 as usize;
        let site2 = term.site2 as usize;
        if site1 >= n_site || site2 >= n_site {
            continue;
        }
        if data.i_flg_orbital_anti_parallel == 1 && term.idx < n_anti_parallel {
            let all_i = site1;
            let all_j = site2 + n_site;
            orbital_idx[all_i][all_j] = term.idx;
            orbital_sgn[all_i][all_j] = term.sign;
            orbital_idx[all_j][all_i] = term.idx;
            orbital_sgn[all_j][all_i] = -term.sign;
        } else {
            let rel_idx = term.idx - n_anti_parallel;
            let is_down_down = rel_idx % 2 == 1;
            let (all_i, all_j) = if is_down_down {
                (site1 + n_site, site2 + n_site)
            } else {
                (site1, site2)
            };
            orbital_idx[all_i][all_j] = term.idx;
            orbital_sgn[all_i][all_j] = term.sign;
            if all_i != all_j {
                orbital_idx[all_j][all_i] = term.idx;
                orbital_sgn[all_j][all_i] = -term.sign;
            }
        }
    }
    (orbital_idx, orbital_sgn, slater)
}
