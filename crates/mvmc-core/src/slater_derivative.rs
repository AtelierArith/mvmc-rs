//! Slater-parameter derivative scratch and QP-weighted reductions.

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use tenferro_tensor::TypedTensor;

use crate::c_timer::CTimer;
use crate::state::SlaterMatrixData;

type ComplexTensor = TypedTensor<Complex64>;

pub(crate) struct SlaterDerivativeScratch {
    qp_orbital: Option<ComplexTensor>,
    weighted_orbital: Option<ComplexTensor>,
    n_slater: usize,
    n_qp_full: usize,
}

impl SlaterDerivativeScratch {
    pub(crate) fn new() -> Self {
        Self {
            qp_orbital: None,
            weighted_orbital: None,
            n_slater: 0,
            n_qp_full: 0,
        }
    }

    pub(crate) fn ensure_shape(&mut self, n_slater: usize, n_qp_full: usize) {
        if self.qp_orbital.is_some() && self.n_slater == n_slater && self.n_qp_full == n_qp_full {
            return;
        }

        self.qp_orbital = Some(
            tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
                vec![n_slater, n_qp_full],
                vec![Complex64::new(0.0, 0.0); n_slater * n_qp_full],
            )
            .expect("SlaterDerivativeScratch shape and data length must match"),
        );
        self.weighted_orbital = Some(
            tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
                vec![n_slater],
                vec![Complex64::new(0.0, 0.0); n_slater],
            )
            .expect("SlaterDerivativeScratch weighted shape and data length must match"),
        );
        self.n_slater = n_slater;
        self.n_qp_full = n_qp_full;
    }

    pub(crate) fn zero_qp_orbital(&mut self) {
        let data = self
            .qp_orbital
            .as_mut()
            .expect("SlaterDerivativeScratch::ensure_shape must be called first")
            .host_data_mut()
            .expect("SlaterDerivativeScratch requires host-backed storage");
        data.fill(Complex64::new(0.0, 0.0));
    }

    pub(crate) fn add_qp_orbital(&mut self, qpidx: usize, orbidx: usize, value: Complex64) {
        debug_assert!(qpidx < self.n_qp_full);
        debug_assert!(orbidx < self.n_slater);
        let offset = qpidx * self.n_slater + orbidx;
        let data = self
            .qp_orbital
            .as_mut()
            .expect("SlaterDerivativeScratch::ensure_shape must be called first")
            .host_data_mut()
            .expect("SlaterDerivativeScratch requires host-backed storage");
        data[offset] += value;
    }

    #[cfg(test)]
    fn qp_orbital_host_data(&self) -> &[Complex64] {
        self.qp_orbital
            .as_ref()
            .expect("SlaterDerivativeScratch::ensure_shape must be called first")
            .host_data()
            .expect("SlaterDerivativeScratch requires host-backed storage")
    }

    pub(crate) fn reduce_qp_weighted_julia_into(
        &mut self,
        weights: &[Complex64],
        ip: Complex64,
        sr_opt_o: &mut [Complex64],
    ) {
        if self.n_slater == 0 || self.n_qp_full == 0 || ip.norm() == 0.0 {
            return;
        }
        assert!(
            sr_opt_o.len() >= 2 * self.n_slater,
            "Slater derivative output must have real/imag slots for every orbital"
        );
        assert_eq!(weights.len(), self.n_qp_full);
        let buffer = self.qp_orbital.as_ref().unwrap().host_data().unwrap();
        let weighted = self
            .weighted_orbital
            .as_mut()
            .unwrap()
            .host_data_mut()
            .unwrap();
        // Julia _store_slater_sr_opt_o_fast! folds QP terms in order. A GEMM
        // contraction can reassociate/fuse this sum; a single ulp then changes
        // the truncated CG solution. Keep tensor storage and fold explicitly.
        for (orbidx, value) in weighted.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (qpidx, &weight) in weights.iter().enumerate() {
                acc += weight * buffer[qpidx * self.n_slater + orbidx];
            }
            *value = acc;
        }
        let weighted_data = self
            .weighted_orbital
            .as_ref()
            .expect("SlaterDerivativeScratch::ensure_shape must be called first")
            .host_data()
            .expect("Slater derivative weighted result must be host-readable");
        let inv_ip = crate::julia_complex::reciprocal(ip);
        for orbidx in 0..self.n_slater {
            let acc = weighted_data[orbidx] * inv_ip;
            sr_opt_o[2 * orbidx] = acc;
            sr_opt_o[2 * orbidx + 1] = acc * Complex64::new(0.0, 1.0);
        }
    }
}

impl Default for SlaterDerivativeScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn slater_elm_diff_with_scratch_timed<const TIMED: bool>(
    sr_opt_o: &mut [Complex64],
    ip: Complex64,
    ele_idx: &[i64],
    data: &ExpertModeData,
    slater_matrix: &SlaterMatrixData,
    scratch: &mut SlaterDerivativeScratch,
    timer: &mut CTimer<TIMED>,
) {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    if n_site == 0 || n_elec == 0 || ip.norm() == 0.0 {
        return;
    }
    let weights = match data.qp_weights.as_ref() {
        Some(weights) => weights,
        None => return,
    };
    let n_qp_full = weights.qp_full_weight.len().min(slater_matrix.pf_m.len());
    let n_sp_gauss_leg = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp_trans = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_slater = if data.modpara.n_orbital_idx > 0 {
        data.modpara.n_orbital_idx as usize
    } else if let Some(max_idx) = data.orbital_terms.iter().map(|t| t.idx).max() {
        (max_idx + 1).max(0) as usize
    } else {
        0
    };
    if n_qp_full == 0 || n_slater == 0 || sr_opt_o.len() < 2 * n_slater {
        return;
    }

    let diag = timer.diagnostics.slater;
    timer.start_diag(932, diag);
    let (orbital_idx, orbital_sgn) = data.build_orbital_matrices();

    let mut trans_orb_idx = vec![-1_i64; n_mp_trans * n_size * n_size];
    let mut trans_orb_sgn = vec![1_i64; n_mp_trans * n_size * n_size];
    for mpidx in 0..n_mp_trans {
        let trans = data.qp_trans_entries.get(mpidx);
        for msi in 0..n_size {
            let ri = ele_idx.get(msi).copied().unwrap_or(-1);
            if ri < 0 || ri as usize >= n_site {
                continue;
            }
            let ori = ri as usize;
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
            for msj in 0..n_size {
                let rj = ele_idx.get(msj).copied().unwrap_or(-1);
                if rj < 0 || rj as usize >= n_site {
                    continue;
                }
                let orj = rj as usize;
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
                let idx = mpidx * n_size * n_size + msi * n_size + msj;
                trans_orb_idx[idx] = orbital_idx[tri][trj];
                trans_orb_sgn[idx] = sgni * sgnj * orbital_sgn[tri][trj];
            }
        }
    }

    timer.stop_diag(932, diag);
    timer.start_diag(931, diag);
    scratch.ensure_shape(n_slater, n_qp_full);
    scratch.zero_qp_orbital();
    timer.stop_diag(931, diag);
    timer.start_diag(933, diag);
    for qpidx in 0..n_qp_full {
        let mpidx = (qpidx / n_sp_gauss_leg).min(n_mp_trans.saturating_sub(1));
        let spidx = qpidx % n_sp_gauss_leg;
        if spidx >= weights.spgl_cos_sin.len()
            || spidx >= weights.spgl_cos_cos.len()
            || spidx >= weights.spgl_sin_sin.len()
        {
            continue;
        }
        let pf = slater_matrix.pf_m[qpidx];
        let cs = pf * weights.spgl_cos_sin[spidx];
        let cc = pf * weights.spgl_cos_cos[spidx];
        let ss = pf * weights.spgl_sin_sin[spidx];
        let tbase = mpidx * n_size * n_size;
        let inv_plane = slater_matrix.inv_m.qp_matrix_slice(qpidx);

        for msi in 0..n_elec {
            for msj in 0..n_elec {
                accumulate_slater_diff(
                    scratch,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    inv_plane[msj + msi * n_size] * cs,
                );
            }
            for msj in n_elec..n_size {
                accumulate_slater_diff(
                    scratch,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    -inv_plane[msj + msi * n_size] * cc,
                );
            }
        }
        for msi in n_elec..n_size {
            for msj in 0..n_elec {
                accumulate_slater_diff(
                    scratch,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    inv_plane[msj + msi * n_size] * ss,
                );
            }
            for msj in n_elec..n_size {
                accumulate_slater_diff(
                    scratch,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    -inv_plane[msj + msi * n_size] * cs,
                );
            }
        }
    }

    timer.stop_diag(933, diag);
    timer.start_diag(934, diag);
    scratch.reduce_qp_weighted_julia_into(&weights.qp_full_weight[..n_qp_full], ip, sr_opt_o);
    timer.stop_diag(934, diag);
}

fn accumulate_slater_diff(
    scratch: &mut SlaterDerivativeScratch,
    qpidx: usize,
    n_slater: usize,
    orbidx: i64,
    sign: i64,
    value: Complex64,
) {
    if orbidx >= 0 && (orbidx as usize) < n_slater {
        scratch.add_qp_orbital(
            qpidx,
            orbidx as usize,
            value * Complex64::new(sign as f64, 0.0),
        );
    }
}

pub(crate) fn slater_elm_diff_fsz_with_scratch(
    sr_opt_o: &mut [Complex64],
    ip: Complex64,
    ele_idx: &[i64],
    ele_spn: &[i64],
    data: &ExpertModeData,
    slater_matrix: &SlaterMatrixData,
    scratch: &mut SlaterDerivativeScratch,
) {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    if n_site == 0 || n_elec == 0 || ip.norm() == 0.0 {
        return;
    }
    let weights = match data.qp_weights.as_ref() {
        Some(weights) => weights,
        None => return,
    };
    let n_qp_full = weights.qp_full_weight.len().min(slater_matrix.pf_m.len());
    let n_mp_trans = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_slater = if data.modpara.n_orbital_idx > 0 {
        data.modpara.n_orbital_idx as usize
    } else if let Some(max_idx) = data.orbital_terms.iter().map(|t| t.idx).max() {
        (max_idx + 1).max(0) as usize
    } else {
        0
    };
    if n_qp_full == 0 || n_slater == 0 || sr_opt_o.len() < 2 * n_slater {
        return;
    }
    let (orbital_idx, orbital_sgn, _) =
        crate::slater_update::build_orbital_idx_sgn_matrices_fsz(data, n_site);
    let n_trans = n_mp_trans * data.n_qp_opt_trans.max(1) as usize;
    let mut trans_orb_idx = vec![-1_i64; n_trans * n_size * n_size];
    let mut trans_orb_sgn = vec![1_i64; n_trans * n_size * n_size];

    for trans_idx in 0..n_trans {
        let mpidx = trans_idx % n_mp_trans.max(1);
        let trans = data.qp_trans_entries.get(mpidx);
        for msi in 0..n_size {
            let ri = ele_idx[msi];
            let si = ele_spn[msi];
            if ri < 0 || si < 0 || ri as usize >= n_site || si > 1 {
                continue;
            }
            let ori = ri as usize;
            let tri_site = trans
                .and_then(|t| t.site_map.get(ori))
                .copied()
                .unwrap_or(ori as i64) as usize;
            let sgni = trans
                .map(|t| t.boundary_sign(ori, data.modpara.nmp_trans < 0))
                .unwrap_or(1);
            if tri_site >= n_site {
                continue;
            }
            let tri = tri_site + si as usize * n_site;
            for msj in 0..n_size {
                let rj = ele_idx[msj];
                let sj = ele_spn[msj];
                if rj < 0 || sj < 0 || rj as usize >= n_site || sj > 1 {
                    continue;
                }
                let orj = rj as usize;
                let trj_site = trans
                    .and_then(|t| t.site_map.get(orj))
                    .copied()
                    .unwrap_or(orj as i64) as usize;
                let sgnj = trans
                    .map(|t| t.boundary_sign(orj, data.modpara.nmp_trans < 0))
                    .unwrap_or(1);
                if trj_site >= n_site {
                    continue;
                }
                let trj = trj_site + sj as usize * n_site;
                let idx = trans_idx * n_size * n_size + msi * n_size + msj;
                trans_orb_idx[idx] = orbital_idx[tri][trj];
                trans_orb_sgn[idx] = sgni * sgnj * orbital_sgn[tri][trj];
            }
        }
    }

    scratch.ensure_shape(n_slater, n_qp_full);
    scratch.zero_qp_orbital();
    for qpidx in 0..n_qp_full {
        let mpidx = qpidx.min(n_mp_trans.saturating_sub(1));
        let pf = slater_matrix.pf_m[qpidx];
        let inv_plane = slater_matrix.inv_m.qp_matrix_slice(qpidx);
        let tbase = mpidx * n_size * n_size;
        for msi in 0..n_size {
            for msj in 0..n_size {
                let orbidx = trans_orb_idx[tbase + msi * n_size + msj];
                if orbidx >= 0 && (orbidx as usize) < n_slater {
                    let sign = trans_orb_sgn[tbase + msi * n_size + msj];
                    let value =
                        -inv_plane[msj + msi * n_size] * pf * Complex64::new(sign as f64, 0.0);
                    scratch.add_qp_orbital(qpidx, orbidx as usize, value);
                }
            }
        }
    }
    // FSZ accumulates both component columns before multiplying by inv(ip).
    // Rotating the normalized column later changes signed zeros.
    let inv_ip = crate::julia_complex::reciprocal(ip);
    for orb in 0..n_slater {
        let mut real = Complex64::new(0.0, 0.0);
        let mut imag = Complex64::new(0.0, 0.0);
        for qp in 0..n_qp_full {
            let tmp = weights.qp_full_weight[qp]
                * scratch.qp_orbital.as_ref().unwrap().host_data().unwrap()[qp * n_slater + orb];
            real += tmp;
            imag += Complex64::new(-tmp.im, tmp.re);
        }
        sr_opt_o[2 * orb] = real * inv_ip;
        sr_opt_o[2 * orb + 1] = imag * inv_ip;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_slater_derivative_matches_julia_numerical_bits() {
        let mut lines = include_str!("../../../tests/fixtures/pfaffian_cg/slater_derivative.txt")
            .lines()
            .filter(|l| !l.starts_with('#'));
        let shape: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let parse = |l: &str| -> Vec<Complex64> {
            let doubles: Vec<_> = l
                .split_whitespace()
                .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                .collect();
            doubles
                .chunks_exact(2)
                .map(|z| Complex64::new(z[0], z[1]))
                .collect()
        };
        let buffer = parse(lines.next().unwrap());
        let weights = parse(lines.next().unwrap());
        let ip = parse(lines.next().unwrap())[0];
        let expected = parse(lines.next().unwrap());
        let mut scratch = SlaterDerivativeScratch::new();
        scratch.ensure_shape(shape[0], shape[1]);
        scratch
            .qp_orbital
            .as_mut()
            .unwrap()
            .host_data_mut()
            .unwrap()
            .copy_from_slice(&buffer);
        let mut actual = vec![Complex64::new(0.0, 0.0); expected.len()];
        scratch.reduce_qp_weighted_julia_into(&weights, ip, &mut actual);
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                (a.re.to_bits(), a.im.to_bits()),
                (b.re.to_bits(), b.im.to_bits()),
                "Slater component {i}: {a} != {b}"
            );
        }
    }

    #[test]
    fn scratch_weighted_reduction_preserves_tensor_layout() {
        let mut scratch = SlaterDerivativeScratch::new();
        scratch.ensure_shape(2, 3);
        scratch.zero_qp_orbital();
        scratch.add_qp_orbital(0, 0, Complex64::new(1.0, 0.0));
        scratch.add_qp_orbital(0, 1, Complex64::new(2.0, 1.0));
        scratch.add_qp_orbital(1, 0, Complex64::new(-1.0, 0.5));
        scratch.add_qp_orbital(1, 1, Complex64::new(0.0, -2.0));
        scratch.add_qp_orbital(2, 0, Complex64::new(3.0, 1.5));
        scratch.add_qp_orbital(2, 1, Complex64::new(-4.0, 0.25));

        let expected_legacy_flat = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.0, -2.0),
            Complex64::new(3.0, 1.5),
            Complex64::new(-4.0, 0.25),
        ];
        assert_eq!(
            scratch.qp_orbital_host_data(),
            expected_legacy_flat.as_slice()
        );

        let weights = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(-1.0, 1.0),
            Complex64::new(2.0, -0.5),
        ];
        let ip = Complex64::new(2.0, 0.0);
        let mut sr_opt_o = vec![Complex64::new(0.0, 0.0); 4];

        scratch.reduce_qp_weighted_julia_into(&weights, ip, &mut sr_opt_o);

        let mut expected = vec![Complex64::new(0.0, 0.0); 4];
        for o in 0..2 {
            let mut acc = Complex64::new(0.0, 0.0);
            for q in 0..3 {
                acc += weights[q] * expected_legacy_flat[q * 2 + o];
            }
            acc /= ip;
            expected[2 * o] = acc;
            expected[2 * o + 1] = acc * Complex64::new(0.0, 1.0);
        }

        assert_eq!(sr_opt_o, expected);
    }
}
