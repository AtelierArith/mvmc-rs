//! Normal and FSZ Slater-element table updates from Julia's cached orbital
//! matrices in `MVMCOptimizers.jl/src/slater_update.jl`.

use std::borrow::Cow;

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::state::VmcOptimizationState;

/// `update_slater_elm_fcmp!(data, state)` mirror for the
/// `i_flg_orbital_general == 0` path. The Slater table is rebuilt for
/// every QP plane using the cached orbital-idx matrix and the parsed
/// `qptransidx.def` maps composed after optimized translations and their signs.
pub fn update_slater_elm(data: &mut ExpertModeData, state: &mut VmcOptimizationState) {
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
    let all_complex = crate::run::get_all_complex_flag(data);
    if n_orb == 0 {
        return;
    }

    let mut slater = vec![Complex64::new(0.0, 0.0); n_orb];
    for (idx, &value) in data.slater_params.iter().take(n_orb).enumerate() {
        // C's UpdateSlaterElm_fcmp reads the declared coefficient directly;
        // it does not apply Julia's historical 1e-14 amplitude cutoff.
        slater[idx] = value;
    }

    for qp in 0..n_qp_full {
        let n_qp_fix = n_sp_gauss_leg * n_mp_trans;
        let rem = qp % n_qp_fix;
        let mpidx = rem / n_sp_gauss_leg;
        let spidx = qp % n_sp_gauss_leg;
        let cs = weights.spgl_cos_sin[spidx];
        let cc = weights.spgl_cos_cos[spidx];
        let ss = weights.spgl_sin_sin[spidx];
        let optidx = qp / n_qp_fix;

        for ri in 0..n_site {
            let (tri, sgni) = crate::qp::translated_site(data, ri, optidx, mpidx, true);
            if tri >= n_site {
                continue;
            }
            for rj in 0..n_site {
                let (trj, sgnj) = crate::qp::translated_site(data, rj, optidx, mpidx, true);
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
                let slt_ij = slater[idx_ij as usize] * sgn_ij as f64;
                let slt_ji = slater[idx_ji as usize] * sgn_ji as f64;

                let rsi0 = ri;
                let rsi1 = ri + n_site;
                let rsj0 = rj;
                let rsj1 = rj + n_site;
                if all_complex {
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
                } else {
                    let cs = cs.re;
                    let cc = cc.re;
                    let ss = ss.re;
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

/// `update_slater_elm_fsz!(data, state)` for pure General and AP+P orbitals.
///
/// FSZ uses an explicit `2*n_site × 2*n_site` orbital matrix and does not
/// apply spin projection (`NSPGaussLeg` is treated as 1). QP translation is
/// applied to the site index and the spin offset is kept explicit.
pub fn update_slater_elm_fsz(data: &mut ExpertModeData, state: &mut VmcOptimizationState) {
    data.ensure_orbital_idx_matrix();
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
        let optidx = qp / n_qp_fix;
        for ri in 0..n_site {
            let (tri, sgni) = crate::qp::translated_site(data, ri, optidx, mpidx, true);
            if tri >= n_site {
                continue;
            }
            for rj in 0..n_site {
                let (trj, sgnj) = crate::qp::translated_site(data, rj, optidx, mpidx, true);
                if trj >= n_site {
                    continue;
                }
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
                                * orbital_sgn[tri_s][trj_s] as f64
                                * sgni as f64
                                * sgnj as f64
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        let slt_ji = if idx_ji >= 0 && (idx_ji as usize) < slater.len() {
                            slater[idx_ji as usize]
                                * orbital_sgn[trj_s][tri_s] as f64
                                * sgni as f64
                                * sgnj as f64
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

type OrbitalMatrix<'a> = Cow<'a, [Vec<i64>]>;

pub(crate) fn build_orbital_idx_sgn_matrices_fsz(
    data: &ExpertModeData,
    n_site: usize,
) -> (OrbitalMatrix<'_>, OrbitalMatrix<'_>, Vec<Complex64>) {
    debug_assert_eq!(n_site, data.modpara.nsite.max(0) as usize);
    let (orbital_idx, orbital_sgn) = match (&data.orbital_idx_matrix, &data.orbital_sgn_matrix) {
        (Some(idx), Some(sgn)) => (Cow::Borrowed(idx.as_slice()), Cow::Borrowed(sgn.as_slice())),
        _ => {
            let (idx, sgn) = data.build_orbital_matrices();
            (Cow::Owned(idx), Cow::Owned(sgn))
        }
    };
    let slater = data.slater_params.clone();
    (orbital_idx, orbital_sgn, slater)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvmc_expert_parsers::OrbitalTerm;

    #[test]
    fn real_normal_slater_uses_c_real_spgl_weights() {
        use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;

        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.modpara.complex_flag = 0;
        data.modpara.nmp_trans = 1;
        data.modpara.nsp_gauss_leg = 1;
        data.modpara.n_orbital_idx = 1;
        data.slater_params = vec![Complex64::new(0.25, 0.0)];
        data.orbital_terms = vec![OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: 0,
            is_complex: false,
            sign: 1,
        }];
        data.ensure_orbital_idx_matrix();
        init_qp_weight(&mut data);
        let weights = data.qp_weights.as_mut().unwrap();
        weights.spgl_cos_sin = vec![Complex64::new(0.25, 9.0)];
        weights.spgl_cos_cos = vec![Complex64::new(0.75, -7.0)];
        weights.spgl_sin_sin = vec![Complex64::new(0.5, 5.0)];
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, false, false);
        update_slater_elm(&mut data, &mut state);
        assert!(state
            .slater_matrix
            .slater_elm
            .as_slice()
            .iter()
            .all(|value| value.im == 0.0));
    }

    #[test]
    fn general_slater_and_derivatives_match_julia_with_sparse_and_cached_layouts() {
        use crate::historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
        use crate::slater_derivative::{slater_elm_diff_fsz_with_scratch, SlaterDerivativeScratch};
        use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
        use mvmc_expert_parsers::QPTransEntry;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/orbital_general");
        let golden = std::fs::read_to_string(root.join("matrices.txt")).unwrap();
        let mut lines = golden.lines().filter(|line| !line.starts_with('#'));
        let ints = |line: &str| -> Vec<i64> {
            line.split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect()
        };
        let complexes = |line: &str| -> Vec<Complex64> {
            let vals: Vec<_> = line
                .split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect();
            vals.chunks_exact(2)
                .map(|z| Complex64::new(z[0], z[1]))
                .collect()
        };
        let bits = |vals: &[Complex64]| -> Vec<u64> {
            vals.iter()
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect()
        };
        while let Some(header) = lines.next() {
            let fields: Vec<_> = header.split_whitespace().collect();
            let kind = fields[0];
            let boundary: i64 = fields[1].parse().unwrap();
            let n: usize = fields[2].parse().unwrap();
            let input = if kind == "cached" { "general" } else { kind };
            let mut data =
                parse_expert_mode_files(root.join(format!("namelist_{input}.def"))).unwrap();
            assert_eq!(data.modpara.nelec, 2);
            data.modpara.nmp_trans = 2 * boundary;
            data.orbital_idx_matrix = None;
            data.orbital_sgn_matrix = None;
            data.ensure_orbital_idx_matrix();
            if kind == "cached" {
                let idx = data.orbital_idx_matrix.as_mut().unwrap();
                idx[0][1] = 4;
                idx[1][0] = 4;
                let sgn = data.orbital_sgn_matrix.as_mut().unwrap();
                sgn[0][1] = -1;
                sgn[1][0] = 1;
            }
            assert_eq!(
                data.orbital_idx_matrix
                    .as_ref()
                    .unwrap()
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                ints(lines.next().unwrap()),
                "{header} indices"
            );
            assert_eq!(
                data.orbital_sgn_matrix
                    .as_ref()
                    .unwrap()
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                ints(lines.next().unwrap()),
                "{header} signs"
            );
            data.slater_params = (0..data.modpara.n_orbital_idx)
                .map(|idx| Complex64::new((idx + 1) as f64 / 7.0, (idx % 3 - 1) as f64 / 5.0))
                .collect();
            data.n_qp_trans = 2;
            data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-0.375, 0.0)];
            data.qp_trans_entries = [[0, 1, 2], [1, 2, 0]]
                .into_iter()
                .enumerate()
                .map(|(qp, map)| QPTransEntry {
                    weight: data.para_qp_trans[qp],
                    site_map: map.to_vec(),
                    site_sign: if boundary > 0 || qp == 0 {
                        vec![1, 1, 1]
                    } else {
                        vec![-1, 1, -1]
                    },
                })
                .collect();
            init_qp_weight(&mut data);
            let mut state = VmcOptimizationState::zeros(3, 2, 0, n, 2, 1, true, true);
            state
                .slater_matrix
                .pf_m
                .copy_from_slice(&complexes(lines.next().unwrap()));
            let inverse = complexes(lines.next().unwrap());
            for qp in 0..2 {
                state
                    .slater_matrix
                    .inv_m
                    .qp_matrix_slice_mut(qp)
                    .copy_from_slice(&inverse[qp * 16..(qp + 1) * 16]);
            }
            let ip = complexes(lines.next().unwrap())[0];
            let expected_slater = complexes(lines.next().unwrap());
            let expected_o = complexes(lines.next().unwrap());
            update_slater_elm_fsz(&mut data, &mut state);
            assert_eq!(
                bits(state.slater_matrix.slater_elm.as_slice()),
                bits(&expected_slater),
                "{header} Slater"
            );
            let mut o = vec![Complex64::new(0.0, 0.0); 2 * n];
            slater_elm_diff_fsz_with_scratch(
                &mut o,
                ip,
                &[0, 1, 1, 2],
                &[0, 0, 1, 1],
                &data,
                &state.slater_matrix,
                &mut SlaterDerivativeScratch::new(),
            );
            assert_eq!(bits(&o), bits(&expected_o), "{header} derivative");
        }
    }

    #[test]
    fn shared_coefficient_matrix_and_unmapped_normalization_maximum_match_c() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nmp_trans = -1;
        data.modpara.nsp_gauss_leg = 1;
        data.modpara.n_orbital_idx = 13;
        data.slater_params = vec![Complex64::new(0.0, 0.0); 13];
        data.slater_params[0] = Complex64::new(0.3, 0.2);
        data.slater_params[12] = Complex64::new(8.0, 0.0);
        data.orbital_terms = (0..2)
            .flat_map(|i| {
                (0..2).map(move |j| OrbitalTerm {
                    site1: i,
                    site2: j,
                    idx: 0,
                    sign: if i == j { 1 } else { -1 },
                    is_complex: true,
                })
            })
            .collect();
        data.ensure_orbital_idx_matrix();
        mvmc_expert_parsers::utils::qp_weight::init_qp_weight(&mut data);
        let weights = data.qp_weights.as_mut().unwrap();
        weights.spgl_cos_sin = vec![Complex64::new(0.25, 0.0)];
        weights.spgl_cos_cos = vec![Complex64::new(0.75, 0.0)];
        weights.spgl_sin_sin = vec![Complex64::new(0.5, 0.0)];
        mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter(&mut data, false);
        let rows: Vec<_> =
            include_str!("../../../tests/fixtures/orbital_general/c_shared_matrix.txt")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect();
        assert_eq!(rows.len(), 2);
        let expected = |row: &str| {
            row.split_whitespace()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect::<Vec<_>>()
        };
        let bits = |values: &[Complex64]| {
            values
                .iter()
                .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&data.slater_params), expected(rows[0]));
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 13, 1, 1, true, false);
        update_slater_elm(&mut data, &mut state);
        let matrix = expected(rows[1])
            .chunks_exact(2)
            .map(|pair| Complex64::new(f64::from_bits(pair[0]), f64::from_bits(pair[1])))
            .collect::<Vec<_>>();
        // Exact numeric equality checks coefficient sharing. C multiplies
        // real SPGL weights; Rust's complex-weight signed zeros remain #42.
        assert_eq!(state.slater_matrix.slater_elm.as_slice(), matrix);
    }

    #[test]
    fn pure_general_spin_site_indices_and_sparse_signs_match_julia() {
        for nmp in [1, -1] {
            let mut data = ExpertModeData::new();
            data.modpara.nsite = 2;
            data.modpara.nmp_trans = nmp;
            data.i_flg_orbital_general = 1;
            data.modpara.n_orbital_idx = 2;
            data.slater_params = vec![Complex64::new(0.0, 0.0), Complex64::new(0.3, 0.2)];
            data.orbital_terms = vec![OrbitalTerm {
                site1: 2,
                site2: 3,
                idx: 1,
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
            data.slater_params = vec![Complex64::new(2.0, 0.0)];
            for i in 0..2 {
                for j in 0..2 {
                    data.orbital_terms.push(OrbitalTerm {
                        site1: i,
                        site2: j,
                        idx: 0,
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
        data.slater_params = (0..13).map(|idx| Complex64::new(idx as f64, 0.0)).collect();
        data.i_flg_orbital_anti_parallel = 1;
        data.i_flg_orbital_parallel = 1;
        data.i_flg_orbital_general = 1;
        data.orbital_terms = [0, 1, 7, 8]
            .into_iter()
            .map(|idx| OrbitalTerm {
                site1: 0,
                site2: 1,
                idx,
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
