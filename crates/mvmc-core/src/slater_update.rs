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
    data.normalize_projection_count();
    data.ensure_orbital_idx_matrix();
    let n_site = data.modpara.nsite.max(0) as usize;
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

        for ri in 0..n_site {
            let ori = ri;
            let tri = trans
                .and_then(|t| t.site_map.get(ori))
                .copied()
                .unwrap_or(ori as i64) as usize;
            let sgni = trans
                .map(|t| t.boundary_sign(ori, data.modpara.nmp_trans < 0))
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
                    .map(|t| t.boundary_sign(orj, data.modpara.nmp_trans < 0))
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
                state
                    .slater_matrix
                    .slater_elm
                    .set(qp, rsi0, rsj0, -(slt_ij - slt_ji) * cs);
                state
                    .slater_matrix
                    .slater_elm
                    .set(qp, rsi0, rsj1, slt_ij * cc + slt_ji * ss);
                state
                    .slater_matrix
                    .slater_elm
                    .set(qp, rsi1, rsj0, -slt_ij * ss - slt_ji * cc);
                state
                    .slater_matrix
                    .slater_elm
                    .set(qp, rsi1, rsj1, (slt_ij - slt_ji) * cs);
            }
        }
    }

    // Real shadow buffer: refresh the per-QP real plane from the
    // freshly-updated complex master so the real-mode sampler sees a
    // consistent view.
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
}

/// `update_slater_elm_fsz!(data, state)` mirror for the AP+P FSZ fixtures.
///
/// FSZ uses an explicit `2*n_site × 2*n_site` orbital matrix and does not
/// apply spin projection (`NSPGaussLeg` is treated as 1). QP translation is
/// applied to the site index and the spin offset is kept explicit.
pub fn update_slater_elm_fsz(data: &mut ExpertModeData, state: &mut VmcOptimizationState) {
    data.normalize_projection_count();
    let n_site = data.modpara.nsite.max(0) as usize;
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
        for ri in 0..n_site {
            let ori = ri;
            let tri = trans
                .and_then(|t| t.site_map.get(ori))
                .copied()
                .unwrap_or(ori as i64) as usize;
            let sgni = trans
                .map(|t| t.boundary_sign(ori, data.modpara.nmp_trans < 0))
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
                    .map(|t| t.boundary_sign(orj, data.modpara.nmp_trans < 0))
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
                        state
                            .slater_matrix
                            .slater_elm
                            .set(qp, rsi, rsj, slt_ij - slt_ji);
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
    debug_assert_eq!(n_site, data.modpara.nsite.max(0) as usize);
    let (orbital_idx, orbital_sgn) = data.build_orbital_matrices();
    let n_slater = mvmc_expert_parsers::utils::parameter_init::n_slater(data);
    let mut slater = vec![Complex64::new(0.0, 0.0); n_slater];
    for term in &data.orbital_terms {
        if term.idx >= 0 && (term.idx as usize) < slater.len() {
            slater[term.idx as usize] = term.value;
        }
    }
    (orbital_idx, orbital_sgn, slater)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvmc_expert_parsers::OrbitalTerm;

    #[test]
    fn pure_general_spin_site_indices_and_sparse_signs_match_julia() {
        for nmp in [1, -1] {
            let mut data = ExpertModeData::new();
            data.modpara.nsite = 2;
            data.modpara.nmp_trans = nmp;
            data.i_flg_orbital_general = 1;
            data.modpara.n_orbital_idx = 2;
            data.orbital_terms = vec![OrbitalTerm {
                site1: 2,
                site2: 3,
                idx: 1,
                value: Complex64::new(0.3, 0.2),
                is_complex: true,
                sign: -1,
            }];
            let (indices, signs, values) = build_orbital_idx_sgn_matrices_fsz(&data, 2);
            assert_eq!(indices[2][3], 1);
            assert_eq!(indices[3][2], 1);
            assert_eq!(indices[0][3], 0);
            assert_eq!(values[1], Complex64::new(0.3, 0.2));
            if nmp > 0 {
                for (i, row) in signs.iter().enumerate() {
                    for (j, &sign) in row.iter().enumerate() {
                        assert_eq!(sign, (j as i64 - i as i64).signum());
                    }
                }
            } else {
                assert_eq!(signs[2][3], -1);
                assert_eq!(signs[3][2], 1);
                assert_eq!(signs[0][3], 0);
            }
            data.modpara.nsp_gauss_leg = 1;
            data.para_qp_trans = vec![Complex64::new(1.0, 0.0)];
            mvmc_expert_parsers::utils::qp_weight::init_qp_weight(&mut data);
            let mut state = VmcOptimizationState::zeros(2, 1, 0, 2, 1, 1, true, true);
            update_slater_elm_fsz(&mut data, &mut state);
            // Live Julia v0.5.0: F(2,3)=+/- (0.6+0.4im),
            // F(3,2)=-F(2,3); every unmapped cell is zero here.
            let expected = if nmp > 0 {
                Complex64::new(0.6, 0.4)
            } else {
                Complex64::new(-0.6, -0.4)
            };
            assert_eq!(state.slater_matrix.slater_elm.get(0, 2, 3), expected);
            assert_eq!(state.slater_matrix.slater_elm.get(0, 3, 2), -expected);
            assert_eq!(
                state.slater_matrix.slater_elm.get(0, 0, 3),
                Complex64::new(0.0, 0.0)
            );
        }
    }

    #[test]
    fn translation_signs_apply_only_with_antiperiodic_boundaries() {
        use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
        use mvmc_expert_parsers::QPTransEntry;
        for nmp in [1, -1] {
            let mut data = ExpertModeData::new();
            data.modpara.nsite = 2;
            data.modpara.nmp_trans = nmp;
            data.modpara.nsp_gauss_leg = 1;
            data.modpara.n_orbital_idx = 1;
            for i in 0..2 {
                for j in 0..2 {
                    data.orbital_terms.push(OrbitalTerm {
                        site1: i,
                        site2: j,
                        idx: 0,
                        value: Complex64::new(2.0, 0.0),
                        is_complex: false,
                        sign: 1,
                    });
                }
            }
            data.qp_trans_entries.push(QPTransEntry {
                weight: Complex64::new(1.0, 0.0),
                site_map: vec![0, 1],
                site_sign: vec![1, -1],
            });
            data.para_qp_trans = vec![Complex64::new(1.0, 0.0)];
            init_qp_weight(&mut data);
            let mut state = VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, false, false);
            update_slater_elm(&mut data, &mut state);
            assert_eq!(
                state.slater_matrix.slater_elm.get(0, 0, 3),
                Complex64::new(if nmp > 0 { 2.0 } else { -2.0 }, 0.0)
            );
        }
    }

    #[test]
    fn fsz_uses_explicit_ap_boundary_even_with_adjacent_duplicate_mappings() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.n_orbital_anti_parallel = 7;
        data.modpara.n_orbital_idx = 13;
        data.i_flg_orbital_anti_parallel = 1;
        data.i_flg_orbital_parallel = 1;
        data.i_flg_orbital_general = 1;
        data.orbital_terms = [0, 1, 7, 8]
            .into_iter()
            .map(|idx| OrbitalTerm {
                site1: 0,
                site2: 1,
                idx,
                value: Complex64::new(idx as f64, 0.0),
                is_complex: true,
                sign: 1,
            })
            .collect();
        let (indices, signs, _) = build_orbital_idx_sgn_matrices_fsz(&data, 2);
        assert_eq!(indices[0][3], 1);
        assert_eq!(indices[3][0], 1);
        assert_eq!(signs[3][0], -1);
        assert_eq!(indices[0][1], 7);
        assert_eq!(indices[2][3], 8);
    }
}
