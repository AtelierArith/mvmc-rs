//! Phase 4.3.1 — projection bookkeeping helpers.
//!
//! Port targets in `MVMCOptimizers.jl/src/vmc_sampling.jl`:
//! `init_loc_spn!`, `make_proj_cnt!`, `update_proj_cnt!`,
//! `log_proj_val`, `log_proj_ratio`, `update_ele_config!`,
//! `revert_ele_config!`.
//!
//! These are pure combinatorial helpers — no RNG, no Pfaffian, no
//! BLAS. They keep the per-walker projection counters in sync with the
//! electron configuration so the sampler's Metropolis ratio can fall
//! back on cheap incremental updates instead of recomputing
//! `MakeProjCnt` after every move.

#![allow(clippy::too_many_arguments)]

use mvmc_expert_parsers::{ExpertModeData, ProjectionLayout};

/// Source DH tails are rebuilt after the incremental Gutzwiller/Jastrow work.
/// Repeated neighbors contribute repeatedly, and self-neighbors are retained.
fn recompute_dh_counts(
    proj_cnt: &mut [i64],
    ele_num: &[i64],
    data: &ExpertModeData,
    layout: ProjectionLayout,
) {
    if layout.n_dh2 == 0 && layout.n_dh4 == 0 {
        return;
    }
    let stop = layout.n_proj.min(proj_cnt.len());
    if layout.dh2_offset < stop {
        proj_cnt[layout.dh2_offset..stop].fill(0);
    }
    let n_site = data.modpara.nsite as usize;
    for (definition, table) in data.doublon_holon_2site_indices.iter().enumerate() {
        for site in 0..n_site {
            let occupation = ele_num[site] + ele_num[n_site + site];
            if occupation == 1 {
                continue;
            }
            let class = occupation / 2;
            let wanted = if class == 0 { 1 } else { 0 };
            let opposite = table.neighbors[site]
                .iter()
                .filter(|&&neighbor| {
                    let neighbor = neighbor as usize;
                    ele_num[neighbor] == wanted && ele_num[n_site + neighbor] == wanted
                })
                .count();
            let index =
                layout.dh2_offset + definition + (class as usize + 2 * opposite) * layout.n_dh2;
            if index < proj_cnt.len() {
                proj_cnt[index] += 1;
            }
        }
    }
    for (definition, table) in data.doublon_holon_4site_indices.iter().enumerate() {
        for site in 0..n_site {
            let occupation = ele_num[site] + ele_num[n_site + site];
            if occupation == 1 {
                continue;
            }
            let class = occupation / 2;
            let wanted = if class == 0 { 1 } else { 0 };
            let opposite = table.neighbors[site]
                .iter()
                .filter(|&&neighbor| {
                    let neighbor = neighbor as usize;
                    ele_num[neighbor] == wanted && ele_num[n_site + neighbor] == wanted
                })
                .count();
            let index =
                layout.dh4_offset + definition + (class as usize + 2 * opposite) * layout.n_dh4;
            if index < proj_cnt.len() {
                proj_cnt[index] += 1;
            }
        }
    }
}

/// Initialise `loc_spn[ri] = 1` for sites flagged as local-spin in
/// `data.locspin_terms` (mirrors `init_loc_spn!`).
///
/// Raw integer definitions are not restricted to binary values. Only exactly
/// `spin_value == 1` selects a local spin, matching C's placement predicate;
/// the resulting internal flags are binary even for nonbinary public input.
pub fn init_loc_spn(loc_spn: &mut [i64], data: &ExpertModeData) {
    for slot in loc_spn.iter_mut() {
        *slot = 0;
    }
    for term in &data.locspin_terms {
        if term.spin_value == 1 && term.site >= 0 && (term.site as usize) < loc_spn.len() {
            loc_spn[term.site as usize] = 1;
        }
    }
}

/// Recompute the projection counters from scratch given the current
/// electron-number array (`ele_num`).
///
/// Layout (mirrors upstream):
/// * `proj_cnt` is indexed by the projection-parameter index, with the
///   Gutzwiller block first (`n_gutzwiller_idx` slots) followed by the
///   Jastrow block (`n_jastrow_idx` slots), then six components per DH2 table and ten per DH4 table.
/// * `ele_num` has length `2 * n_site`: `[ele_num[0..n_site]]` is the
///   up-spin occupancy and `[ele_num[n_site..2*n_site]]` is down.
pub fn make_proj_cnt(proj_cnt: &mut [i64], ele_num: &[i64], data: &ExpertModeData) {
    let n_site = data.modpara.nsite as usize;
    let n_proj = proj_cnt.len();
    let n_gutz = data.projection_layout().n_gutzwiller;
    let n_jast = data.projection_layout().n_jastrow;
    let gutz_idx = data.gutzwiller_idx.as_slice();
    let jastrow_idx = &data.jastrow_idx;

    for slot in proj_cnt.iter_mut() {
        *slot = 0;
    }
    if n_site == 0 || ele_num.len() < 2 * n_site {
        return;
    }
    let (n0, n1) = ele_num.split_at(n_site);
    let n1 = &n1[..n_site];

    // Gutzwiller block.
    if n_gutz > 0 {
        if !gutz_idx.is_empty() {
            for ri in 0..n_site {
                if ri >= gutz_idx.len() {
                    continue;
                }
                let idx = gutz_idx[ri];
                if idx >= 0 && (idx as usize) < n_proj {
                    proj_cnt[idx as usize] += n0[ri] * n1[ri];
                }
            }
        } else {
            for ri in 0..n_site {
                proj_cnt[0] += n0[ri] * n1[ri];
            }
        }
    }

    // Jastrow block. `JastrowIdx[ri, rj]` is the 0-based parameter
    // slot (matches `jastrow_idx_matrix` in upstream).
    let offset = n_gutz;
    if n_jast > 0 && !jastrow_idx.is_empty() {
        for ri in 0..n_site {
            let xi = n0[ri] + n1[ri] - 1;
            if xi == 0 {
                continue;
            }
            for rj in (ri + 1)..n_site {
                if ri >= jastrow_idx.len() || rj >= jastrow_idx[ri].len() {
                    continue;
                }
                let idx = jastrow_idx[ri][rj];
                if idx < 0 {
                    continue;
                }
                let xj = n0[rj] + n1[rj] - 1;
                let proj_idx = offset + idx as usize;
                if proj_idx < n_proj {
                    proj_cnt[proj_idx] += xi * xj;
                }
            }
        }
    } else if !data.jastrow_terms.is_empty() {
        let proj_idx = offset;
        if proj_idx < n_proj {
            for term in &data.jastrow_terms {
                let ri = term.site1;
                let rj = term.site2;
                if ri < 0 || rj < 0 {
                    continue;
                }
                let ri = ri as usize;
                let rj = rj as usize;
                if ri >= n_site || rj >= n_site {
                    continue;
                }
                let xi = n0[ri] + n1[ri] - 1;
                if xi == 0 {
                    continue;
                }
                let xj = n0[rj] + n1[rj] - 1;
                proj_cnt[proj_idx] += xi * xj;
            }
        }
    }
    recompute_dh_counts(proj_cnt, ele_num, data, data.projection_layout());
}

/// Incrementally update the projection counters after a single-spin
/// hop from site `ri` to site `rj`. `proj_cnt_new` and `proj_cnt_old`
/// may alias; when they do not, the old counter block is copied across
/// first.
///
/// **Call ordering**: this function expects the **post-hop**
/// occupancy in `ele_num`, matching the upstream C call order
/// (`vmcmake.c:164-165`):
///
/// ```text
/// updateEleConfig(mi, ri, rj, s, ele_idx, ele_cfg, ele_num);  // updates ele_num
/// UpdateProjCnt(ri, rj, s, proj_cnt_new, proj_cnt_old, ele_num);
/// ```
pub fn update_proj_cnt(
    ri: i64,
    rj: i64,
    _spin: u8,
    proj_cnt_new: &mut [i64],
    proj_cnt_old: &[i64],
    ele_num: &[i64],
    data: &ExpertModeData,
) {
    let n_site = data.modpara.nsite as usize;
    let n_proj = proj_cnt_new.len();
    let n_gutz = data.projection_layout().n_gutzwiller;
    let gutz_idx = data.gutzwiller_idx.as_slice();
    let jastrow_idx = &data.jastrow_idx;

    if !std::ptr::eq(proj_cnt_new.as_ptr(), proj_cnt_old.as_ptr()) {
        let n = proj_cnt_new.len().min(proj_cnt_old.len());
        proj_cnt_new[..n].copy_from_slice(&proj_cnt_old[..n]);
    }
    if ri == rj {
        return;
    }
    if n_site == 0 || ele_num.len() < 2 * n_site {
        return;
    }
    let (n0, n1) = ele_num.split_at(n_site);
    let n1 = &n1[..n_site];

    let ri = ri as usize;
    let rj = rj as usize;
    if ri >= n_site || rj >= n_site {
        return;
    }

    // Gutzwiller block.
    if n_gutz > 0 {
        if !gutz_idx.is_empty() && ri < gutz_idx.len() && rj < gutz_idx.len() {
            let idx_ri = gutz_idx[ri];
            let idx_rj = gutz_idx[rj];
            if idx_ri >= 0 && (idx_ri as usize) < n_proj {
                proj_cnt_new[idx_ri as usize] -= n0[ri] + n1[ri];
            }
            if idx_rj >= 0 && (idx_rj as usize) < n_proj {
                proj_cnt_new[idx_rj as usize] += n0[rj] * n1[rj];
            }
        } else {
            proj_cnt_new[0] -= n0[ri] + n1[ri];
            proj_cnt_new[0] += n0[rj] * n1[rj];
        }
    }

    // C keeps directional mappings; projection kernels always look up the
    // upper-triangle entry, with the smaller site index first.
    let offset = n_gutz;
    if !jastrow_idx.is_empty() {
        let get_idx = |ra: usize, rb: usize| -> i64 {
            let (a, b) = if ra < rb { (ra, rb) } else { (rb, ra) };
            if a >= jastrow_idx.len() || b >= jastrow_idx[a].len() {
                -1
            } else {
                jastrow_idx[a][b]
            }
        };

        let xi_ri = n0[ri] + n1[ri];
        let xi_rj = n0[rj] + n1[rj];

        let idx = get_idx(ri, rj);
        if idx >= 0 {
            let proj_idx = offset + idx as usize;
            if proj_idx < n_proj {
                proj_cnt_new[proj_idx] += xi_ri - xi_rj + 1;
            }
        }
        for rk in 0..n_site {
            if rk == ri || rk == rj {
                continue;
            }
            let xi_rk_minus_1 = n0[rk] + n1[rk] - 1;
            let idx_ri_rk = get_idx(ri, rk);
            if idx_ri_rk >= 0 {
                let proj_idx = offset + idx_ri_rk as usize;
                if proj_idx < n_proj {
                    proj_cnt_new[proj_idx] -= xi_rk_minus_1;
                }
            }
            let idx_rj_rk = get_idx(rj, rk);
            if idx_rj_rk >= 0 {
                let proj_idx = offset + idx_rj_rk as usize;
                if proj_idx < n_proj {
                    proj_cnt_new[proj_idx] += xi_rk_minus_1;
                }
            }
        }
    } else if !data.jastrow_terms.is_empty() {
        let proj_idx = offset;
        if proj_idx < n_proj {
            let xi_ri = n0[ri] + n1[ri];
            let xi_rj = n0[rj] + n1[rj];
            proj_cnt_new[proj_idx] += xi_ri - xi_rj + 1;
            for rk in 0..n_site {
                if rk == ri || rk == rj {
                    continue;
                }
                let xi_rk_minus_1 = n0[rk] + n1[rk] - 1;
                proj_cnt_new[proj_idx] -= xi_rk_minus_1;
                proj_cnt_new[proj_idx] += xi_rk_minus_1;
            }
        }
    }
    recompute_dh_counts(proj_cnt_new, ele_num, data, data.projection_layout());
}

/// `log_proj_val(proj_cnt, data)` -- sum of `Re(Proj[idx]) * proj_cnt[idx]`
/// over all projection blocks. DH2 imaginary parts do not enter this value.
pub fn log_proj_val(proj_cnt: &[i64], data: &ExpertModeData) -> f64 {
    let mut z = 0.0_f64;
    for (value, &count) in data.projection_parameters().iter().zip(proj_cnt) {
        z += value.re * count as f64;
    }
    z
}

/// `log_proj_ratio(new, old, data)` -- sum of
/// `Re(Proj[idx]) * (new[idx] - old[idx])`.
pub fn log_proj_ratio(proj_cnt_new: &[i64], proj_cnt_old: &[i64], data: &ExpertModeData) -> f64 {
    let mut z = 0.0_f64;
    for ((value, &new), &old) in data
        .projection_parameters()
        .iter()
        .zip(proj_cnt_new)
        .zip(proj_cnt_old)
    {
        z += value.re * (new - old) as f64;
    }
    z
}

/// `update_ele_config!`: apply a `mi/ri/rj/spin` hop to the per-walker
/// electron-configuration buffers.
pub fn update_ele_config(
    mi: usize,
    ri: usize,
    rj: usize,
    spin: u8,
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    n_site: usize,
    n_elec: usize,
) {
    let s = spin as usize;
    let msa = mi + s * n_elec;
    let rsa_old = ri + s * n_site;
    let rsa_new = rj + s * n_site;
    ele_idx[msa] = rj as i64;
    ele_cfg[rsa_old] = -1;
    ele_cfg[rsa_new] = mi as i64;
    ele_num[rsa_old] = 0;
    ele_num[rsa_new] = 1;
}

/// `revert_ele_config!`: undo `update_ele_config`.
pub fn revert_ele_config(
    mi: usize,
    ri: usize,
    rj: usize,
    spin: u8,
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    n_site: usize,
    n_elec: usize,
) {
    let s = spin as usize;
    let msa = mi + s * n_elec;
    let rsa_old = ri + s * n_site;
    let rsa_new = rj + s * n_site;
    ele_idx[msa] = ri as i64;
    ele_cfg[rsa_old] = mi as i64;
    ele_cfg[rsa_new] = -1;
    ele_num[rsa_old] = 1;
    ele_num[rsa_new] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvmc_expert_parsers::{ExpertModeData, JastrowTerm, LocSpinTerm};
    use num_complex::Complex64;

    fn small_data() -> ExpertModeData {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 4;
        data.locspin_terms = vec![
            LocSpinTerm {
                site: 0,
                spin_value: 1,
            },
            LocSpinTerm {
                site: 2,
                spin_value: 1,
            },
        ];
        data.n_gutzwiller_idx = 1;
        data.gutzwiller_idx = vec![0, 0, 0, 0];
        data.gutzwiller_terms = vec![mvmc_expert_parsers::GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.5, 0.0),
            is_complex: false,
        }];
        data.n_jastrow_idx = 1;
        data.jastrow_idx = vec![
            vec![-1, 0, 0, 0],
            vec![0, -1, 0, 0],
            vec![0, 0, -1, 0],
            vec![0, 0, 0, -1],
        ];
        data.jastrow_terms = vec![JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(-0.25, 0.0),
            is_complex: false,
        }];
        data
    }

    #[test]
    fn init_loc_spn_sets_flagged_sites() {
        let data = small_data();
        let mut loc = vec![99_i64; 4];
        init_loc_spn(&mut loc, &data);
        assert_eq!(loc, vec![1, 0, 1, 0]);
    }

    #[test]
    fn init_loc_spn_accepts_nonbinary_definition_values() {
        let mut data = small_data();
        data.locspin_terms = vec![
            LocSpinTerm {
                site: 0,
                spin_value: 2,
            },
            LocSpinTerm {
                site: 1,
                spin_value: -1,
            },
            LocSpinTerm {
                site: 2,
                spin_value: 0,
            },
            LocSpinTerm {
                site: 3,
                spin_value: 1,
            },
        ];
        let mut loc = [99; 4];
        init_loc_spn(&mut loc, &data);
        assert_eq!(loc, [0, 0, 0, 1]);
        // Projection must not rewrite or reject the original definitions.
        assert_eq!(
            data.locspin_terms
                .iter()
                .map(|term| term.spin_value)
                .collect::<Vec<_>>(),
            [2, -1, 0, 1]
        );
    }

    #[test]
    fn make_proj_cnt_handles_gutzwiller_and_jastrow() {
        let data = small_data();
        // 2-up / 2-down occupancy (half filling).
        let ele_num = vec![1, 0, 1, 0, /*down*/ 0, 1, 0, 1];
        let mut proj_cnt = vec![0_i64; 1 + 1];
        make_proj_cnt(&mut proj_cnt, &ele_num, &data);
        // Gutzwiller: sum n0_i * n1_i = 0 (no doublons).
        assert_eq!(proj_cnt[0], 0);
        // Jastrow: pairs (xi, xj) with xi = n0+n1-1, only pairs where xi != 0
        // contribute. For our fully-occupied configuration, every xi = 0, so 0.
        assert_eq!(proj_cnt[1], 0);
    }

    #[test]
    fn make_proj_cnt_doublon_counted_in_gutzwiller_block() {
        let data = small_data();
        // Put two electrons on site 0 (one of each spin) -> doublon.
        let ele_num = vec![1, 0, 0, 0, /*down*/ 1, 0, 0, 0];
        let mut proj_cnt = vec![0_i64; 2];
        make_proj_cnt(&mut proj_cnt, &ele_num, &data);
        assert_eq!(proj_cnt[0], 1);
    }

    #[test]
    fn update_proj_cnt_matches_make_proj_cnt_after_hop() {
        let data = small_data();
        // Start with up-spin doublon-ish layout.
        let mut ele_num = vec![1, 0, 0, 0, 1, 0, 0, 0];
        let mut proj_cnt_old = vec![0_i64; 2];
        make_proj_cnt(&mut proj_cnt_old, &ele_num, &data);

        // Apply the hop (up electron 0 -> 1) before calling update_proj_cnt,
        // matching the upstream C convention (vmcmake.c:164-165).
        ele_num[0] = 0;
        ele_num[1] = 1;

        let mut proj_cnt_new = proj_cnt_old.clone();
        update_proj_cnt(0, 1, 0, &mut proj_cnt_new, &proj_cnt_old, &ele_num, &data);

        let mut proj_cnt_fresh = vec![0_i64; 2];
        make_proj_cnt(&mut proj_cnt_fresh, &ele_num, &data);
        assert_eq!(proj_cnt_new, proj_cnt_fresh);
    }

    #[test]
    fn log_proj_val_and_ratio_align() {
        let data = small_data();
        let proj_cnt = vec![3_i64, -2_i64];
        let val = log_proj_val(&proj_cnt, &data);
        // Re(g) * 3 + Re(j) * -2 = 0.5*3 + (-0.25)*(-2) = 1.5 + 0.5 = 2.0.
        assert!((val - 2.0).abs() < 1e-15);

        let proj_cnt_old = vec![0_i64; 2];
        let ratio = log_proj_ratio(&proj_cnt, &proj_cnt_old, &data);
        assert!((ratio - val).abs() < 1e-15);
    }

    #[test]
    fn sparse_projection_values_use_declared_offsets() {
        let mut data = small_data();
        data.n_gutzwiller_idx = 3;
        data.n_jastrow_idx = 4;
        data.gutzwiller_terms[0].value.re = 1.0;
        data.jastrow_terms[0].value.re = 2.0;
        let counts = [1, 9, 9, 2, 9, 9, 9];
        assert_eq!(log_proj_val(&counts, &data), 5.0);
        assert_eq!(log_proj_ratio(&counts, &[0; 7], &data), 5.0);
    }

    #[test]
    fn update_then_revert_ele_config_is_identity() {
        let n_site = 4;
        let n_elec = 2; // per spin
        let mut ele_idx = vec![0, 1, 2, 3]; // up: (0, 1), down: (2, 3)
        let mut ele_cfg = vec![0, 1, -1, -1, /*down*/ -1, -1, 0, 1];
        let mut ele_num = vec![1, 1, 0, 0, /*down*/ 0, 0, 1, 1];
        let snapshot = (ele_idx.clone(), ele_cfg.clone(), ele_num.clone());
        update_ele_config(
            0,
            0,
            3,
            0,
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            n_site,
            n_elec,
        );
        // Hop happened: up electron 0 moved 0 -> 3.
        assert_eq!(ele_idx[0], 3);
        assert_eq!(ele_cfg[0], -1);
        assert_eq!(ele_cfg[3], 0);
        revert_ele_config(
            0,
            0,
            3,
            0,
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            n_site,
            n_elec,
        );
        assert_eq!((ele_idx, ele_cfg, ele_num), snapshot);
    }
}
