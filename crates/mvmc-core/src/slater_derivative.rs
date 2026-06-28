//! Slater-parameter derivative scratch and QP-weighted reductions.

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::state::SlaterMatrixData;

fn qp_weighted_orbital_sum_einsum(
    backend: &mut tenferro_cpu::CpuBackend,
    weights: &tenferro_tensor::TypedTensor<Complex64>,
    buffer: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<Complex64>> {
    use tenferro_einsum::TypedTensorEinsumExt;

    [buffer, weights].einsum("oq,q->o", backend)
}

pub(crate) struct SlaterDerivativeScratch {
    qp_orbital: Option<tenferro_tensor::TypedTensor<Complex64>>,
    qp_weights: Option<tenferro_tensor::TypedTensor<Complex64>>,
    backend: tenferro_cpu::CpuBackend,
    n_slater: usize,
    n_qp_full: usize,
}

impl SlaterDerivativeScratch {
    pub(crate) fn new() -> Self {
        Self {
            qp_orbital: None,
            qp_weights: None,
            backend: tenferro_cpu::CpuBackend::new(),
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
        self.qp_weights = None;
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

    fn ensure_weights(&mut self, weights: &[Complex64]) {
        assert_eq!(
            weights.len(),
            self.n_qp_full,
            "QP weight length must match Slater derivative scratch shape"
        );
        match self.qp_weights.as_mut() {
            Some(qp_weights) if qp_weights.shape() == [self.n_qp_full] => {
                qp_weights
                    .host_data_mut()
                    .expect("SlaterDerivativeScratch weights require host-backed storage")
                    .copy_from_slice(weights);
            }
            _ => {
                self.qp_weights = Some(
                    tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
                        vec![self.n_qp_full],
                        weights.to_vec(),
                    )
                    .expect("SlaterDerivativeScratch weight shape and data length must match"),
                );
            }
        }
    }

    pub(crate) fn reduce_qp_weighted_einsum_into(
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
        self.ensure_weights(weights);
        let weighted = qp_weighted_orbital_sum_einsum(
            &mut self.backend,
            self.qp_weights
                .as_ref()
                .expect("SlaterDerivativeScratch weights must be initialized"),
            self.qp_orbital
                .as_ref()
                .expect("SlaterDerivativeScratch::ensure_shape must be called first"),
        )
        .expect("Slater derivative QP weighted einsum must run");
        let weighted_data = weighted
            .host_data()
            .expect("Slater derivative weighted result must be host-readable");
        let inv_ip = Complex64::new(1.0, 0.0) / ip;
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

pub(crate) fn slater_elm_diff_with_scratch(
    sr_opt_o: &mut [Complex64],
    ip: Complex64,
    ele_idx: &[i64],
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

    let mut orbital_idx = vec![vec![-1_i64; n_site]; n_site];
    let mut orbital_sgn = vec![vec![1_i64; n_site]; n_site];
    for term in &data.orbital_terms {
        if term.site1 < 0 || term.site2 < 0 {
            continue;
        }
        let ri = term.site1 as usize;
        let rj = term.site2 as usize;
        if ri >= n_site || rj >= n_site {
            continue;
        }
        let sign = if term.sign == 0 { 1 } else { term.sign };
        orbital_idx[ri][rj] = term.idx;
        orbital_sgn[ri][rj] = sign;
    }

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
                .and_then(|t| t.site_sign.get(ori))
                .copied()
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
                    .and_then(|t| t.site_sign.get(orj))
                    .copied()
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

    scratch.ensure_shape(n_slater, n_qp_full);
    scratch.zero_qp_orbital();
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

    scratch.reduce_qp_weighted_einsum_into(&weights.qp_full_weight[..n_qp_full], ip, sr_opt_o);
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
                .and_then(|t| t.site_sign.get(ori))
                .copied()
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
                    .and_then(|t| t.site_sign.get(orj))
                    .copied()
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
    scratch.reduce_qp_weighted_einsum_into(&weights.qp_full_weight[..n_qp_full], ip, sr_opt_o);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_einsum_matches_legacy_reduction_layout() {
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

        scratch.reduce_qp_weighted_einsum_into(&weights, ip, &mut sr_opt_o);

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
