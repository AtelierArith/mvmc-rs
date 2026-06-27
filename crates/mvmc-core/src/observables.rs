//! `<H>`, `<H^2>`, `<Sz>`, `<Sz^2>` + 1- and 2-body Green-function accumulators.
//!
//! Port targets: `MVMCOptimizers.jl/src/{vmc_main_cal.jl,green_func_calc.jl}`
//! (~3.2k LOC).
//!
//! Phase 4.4 ships the Heisenberg / Hubbard model path needed by the
//! upstream `examples/inputs/*` cases: diagonal CoulombIntra,
//! CoulombInter, Hund (cheap density-density), Exchange (uses the 2-body
//! Green function), plus the `(set_projection_diff, calculate_oo_real,
//! calculate_oo, finalize_oo_store)` accumulators that feed the SR step.
//! Transfer / PairHopping / InterAll terms are still pending and will
//! land alongside the QP-trans-aware `slater_elm_diff` port.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::items_after_test_module
)]

use num_complex::Complex64;

use mvmc_expert_parsers::{ExpertModeData, Spin};

use crate::pfaffian::calc_m_all_complex;
use crate::sampling::projection::update_proj_cnt;
use crate::sampling::updates::calculate_new_pf_m_two2_real_flat;
use crate::state::VmcOptimizationState;

/// Reset the accumulators that `vmc_main_cal!` clears at the top of each
/// SR step (`clear_phys_quantity!` in upstream).
pub fn clear_phys_quantity(state: &mut VmcOptimizationState) {
    state.energy.wc = Complex64::new(0.0, 0.0);
    state.energy.etot = Complex64::new(0.0, 0.0);
    state.energy.etot2 = Complex64::new(0.0, 0.0);
    state.energy.sztot = Complex64::new(0.0, 0.0);
    state.energy.sztot2 = Complex64::new(0.0, 0.0);
    for x in state.sr_opt.sr_opt_oo.iter_mut() {
        *x = Complex64::new(0.0, 0.0);
    }
    for x in state.sr_opt.sr_opt_ho.iter_mut() {
        *x = Complex64::new(0.0, 0.0);
    }
    for x in state.sr_opt.sr_opt_o.iter_mut() {
        *x = Complex64::new(0.0, 0.0);
    }
    for x in state.sr_opt.sr_opt_oo_real.iter_mut() {
        *x = 0.0;
    }
    for x in state.sr_opt.sr_opt_ho_real.iter_mut() {
        *x = 0.0;
    }
    for x in state.sr_opt.sr_opt_o_real.iter_mut() {
        *x = 0.0;
    }
}

/// `calculate_ip_real(pf_m_real, qp_start, qp_end, data)` mirror.
pub fn calculate_ip_real(
    pf_m_real: &[f64],
    qp_start: usize,
    qp_end: usize,
    data: &ExpertModeData,
) -> f64 {
    let weights = match data.qp_weights.as_ref() {
        Some(w) => w,
        None => return 0.0,
    };
    let mut ip = 0.0;
    for qpidx in qp_start..qp_end {
        if qpidx < weights.qp_full_weight.len() && qpidx < pf_m_real.len() {
            ip += weights.qp_full_weight[qpidx].re * pf_m_real[qpidx];
        }
    }
    ip
}

/// Complex `calculate_ip_fcmp` mirror.
pub fn calculate_ip_complex(
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    data: &ExpertModeData,
) -> Complex64 {
    let weights = match data.qp_weights.as_ref() {
        Some(w) => w,
        None => return Complex64::new(0.0, 0.0),
    };
    let mut ip = Complex64::new(0.0, 0.0);
    for qpidx in qp_start..qp_end {
        if qpidx < weights.qp_full_weight.len() && qpidx < pf_m.len() {
            ip += weights.qp_full_weight[qpidx] * pf_m[qpidx];
        }
    }
    ip
}

/// `log(|ip| + 1e-100)` mirror used by the sampler’s acceptance test.
pub fn calculate_log_ip_real(
    pf_m_real: &[f64],
    qp_start: usize,
    qp_end: usize,
    data: &ExpertModeData,
) -> f64 {
    let ip = calculate_ip_real(pf_m_real, qp_start, qp_end, data);
    (ip.abs() + 1.0e-100).ln()
}

/// Complex `calculate_log_ip_fcmp` mirror.
pub fn calculate_log_ip_complex(
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    data: &ExpertModeData,
) -> Complex64 {
    let ip = calculate_ip_complex(pf_m, qp_start, qp_end, data);
    let mag = ip.norm() + 1.0e-100;
    Complex64::new(mag.ln(), ip.arg())
}

/// `calculate_sz_fsz(ele_num, n_site)` mirror.
pub fn calculate_sz(ele_num: &[i64], n_site: usize) -> f64 {
    let mut sz = 0i64;
    for ri in 0..n_site {
        sz += ele_num[ri] - ele_num[ri + n_site];
    }
    sz as f64
}

/// `set_projection_diff!(sr_opt_o, ele_proj_cnt, n_proj)` mirror.
pub fn set_projection_diff(sr_opt_o: &mut [Complex64], ele_proj_cnt: &[i64], n_proj: usize) {
    if sr_opt_o.len() < 2 {
        return;
    }
    sr_opt_o[0] = Complex64::new(1.0, 0.0);
    sr_opt_o[1] = Complex64::new(0.0, 0.0);
    for i in 0..n_proj {
        let cnt = ele_proj_cnt.get(i).copied().unwrap_or(0);
        let r = (i + 1) * 2;
        if r + 1 >= sr_opt_o.len() {
            break;
        }
        sr_opt_o[r] = Complex64::new(cnt as f64, 0.0);
        sr_opt_o[r + 1] = Complex64::new(0.0, 0.0);
    }
}

/// Real-mode `calculate_oo_real!(sr_opt_oo_real, sr_opt_ho_real, o, w, e, sr_opt_size)`.
pub fn calculate_oo_real(
    sr_opt_oo: &mut [f64],
    sr_opt_ho: &mut [f64],
    sr_opt_o: &[f64],
    w: f64,
    e: f64,
    sr_opt_size: usize,
) {
    let we = w * e;
    let lda = sr_opt_size;
    for j in 0..sr_opt_size {
        let oj = sr_opt_o[j];
        for i in 0..sr_opt_size {
            let idx = i + lda * j;
            sr_opt_oo[idx] += w * sr_opt_o[i] * oj;
        }
    }
    for i in 0..sr_opt_size {
        sr_opt_ho[i] += we * sr_opt_o[i];
    }
}

/// Complex `calculate_oo!` (single-sample, no store).
pub fn calculate_oo(
    sr_opt_oo: &mut [Complex64],
    sr_opt_ho: &mut [Complex64],
    sr_opt_o: &[Complex64],
    w: f64,
    e: Complex64,
    sr_opt_size: usize,
) {
    let size_2 = 2 * sr_opt_size;
    for j in 0..size_2 {
        let tmp = sr_opt_o[j] * w;
        sr_opt_oo[j] += tmp;
        sr_opt_ho[j] += e * tmp;
    }
    for i in 2..size_2 {
        for j in 0..size_2 {
            sr_opt_oo[i * size_2 + j] += sr_opt_o[j] * sr_opt_o[i].conj() * w;
        }
    }
}

/// Complex `calculate_oo_store!` mirror (stores `sqrt(w) * O`).
pub fn calculate_oo_store(
    sr_opt_ho: &mut [Complex64],
    sr_opt_o_store: &mut [Complex64],
    sr_opt_o: &[Complex64],
    w: f64,
    e: Complex64,
    sample: usize,
    sr_opt_size: usize,
) {
    let we = e * w;
    let sqrtw = w.sqrt();
    let size_2 = 2 * sr_opt_size;
    let store = &mut sr_opt_o_store[sample * size_2..(sample + 1) * size_2];
    for i in 0..size_2 {
        store[i] = sr_opt_o[i] * sqrtw;
        sr_opt_ho[i] += sr_opt_o[i] * we;
    }
}

/// Complex `finalize_oo_store!` mirror — sums `O * O^H` from the sample
/// store back into `sr_opt_oo`.
pub fn finalize_oo_store(
    sr_opt_oo: &mut [Complex64],
    sr_opt_o_store: &[Complex64],
    sr_opt_size: usize,
    sample_size: usize,
) {
    let size_2 = 2 * sr_opt_size;
    if size_2 == 0 {
        return;
    }
    if sample_size == 0 {
        for i in 0..size_2 {
            for j in 0..size_2 {
                sr_opt_oo[i * size_2 + j] = Complex64::new(0.0, 0.0);
            }
        }
        return;
    }

    let mut backend = tenferro_cpu::CpuBackend::new();
    let store_tensor = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
        vec![size_2, sample_size],
        sr_opt_o_store.to_vec(),
    )
    .expect("SR store shape and data length must match");
    let gram =
        sr_store_gram_einsum(&mut backend, &store_tensor).expect("SR store Gram einsum must run");
    let gram = gram
        .host_data()
        .expect("SR store Gram tensor must be host-backed");
    for j in 0..size_2 {
        for i in 0..size_2 {
            sr_opt_oo[i * size_2 + j] = gram[i + j * size_2];
        }
    }
}

fn sr_store_gram_einsum(
    backend: &mut tenferro_cpu::CpuBackend,
    store: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<Complex64>> {
    use tenferro_einsum::TypedTensorEinsumExt;

    let shape = store.shape();
    assert_eq!(shape.len(), 2, "SR store must be [component, sample]");
    let raw = store.host_data()?;
    let conj = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
        shape.to_vec(),
        raw.iter().map(|z| z.conj()).collect(),
    )?;
    [store, &conj].einsum("is,js->ij", backend)
}

// The current Slater derivative buffers are still Vec-backed and live in a
// per-sample hot path. Keep the tensor einsum contract tested here, but do not
// pay a Vec -> tensor conversion for every sample until the store is tensor-backed.
#[allow(dead_code)]
fn qp_weighted_orbital_sum_einsum(
    backend: &mut tenferro_cpu::CpuBackend,
    weights: &tenferro_tensor::TypedTensor<Complex64>,
    buffer: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<Complex64>> {
    use tenferro_einsum::TypedTensorEinsumExt;

    [buffer, weights].einsum("oq,q->o", backend)
}

/// Diagonal-only Hamiltonian terms (no off-diagonal Green-function calls).
/// Mirrors the CoulombIntra / CoulombInter / Hund sums in
/// `calculate_hamiltonian`.
fn calculate_hamiltonian_diagonal(ele_num: &[i64], data: &ExpertModeData) -> Complex64 {
    let n_site = data.modpara.nsite.max(0) as usize;
    if n_site == 0 || ele_num.len() < 2 * n_site {
        return Complex64::new(0.0, 0.0);
    }
    let (n0, n1) = ele_num.split_at(n_site);
    let n1 = &n1[..n_site];
    let mut e = Complex64::new(0.0, 0.0);
    for term in &data.coulomb_intra_terms {
        if term.site >= 0 && (term.site as usize) < n_site {
            let ri = term.site as usize;
            e += Complex64::new(term.value * (n0[ri] * n1[ri]) as f64, 0.0);
        }
    }
    for term in &data.coulomb_inter_terms {
        if term.site1 >= 0
            && term.site2 >= 0
            && (term.site1 as usize) < n_site
            && (term.site2 as usize) < n_site
        {
            let ri = term.site1 as usize;
            let rj = term.site2 as usize;
            let occ_i = (n0[ri] + n1[ri]) as f64;
            let occ_j = (n0[rj] + n1[rj]) as f64;
            e += Complex64::new(term.value * occ_i * occ_j, 0.0);
        }
    }
    for term in &data.hund_terms {
        if term.site1 >= 0
            && term.site2 >= 0
            && (term.site1 as usize) < n_site
            && (term.site2 as usize) < n_site
        {
            let ri = term.site1 as usize;
            let rj = term.site2 as usize;
            let s_up = (n0[ri] * n0[rj]) as f64;
            let s_down = (n1[ri] * n1[rj]) as f64;
            e -= Complex64::new(term.value * (s_up + s_down), 0.0);
        }
    }
    e
}

/// `green_func2(ri, rj, rj, ri, s, t, ...)` for the Heisenberg exchange
/// term. This is the only 2-body Green function the Phase-4 gate touches
/// because the exchange contribution requires it; for other 2-body
/// operators (PairHop, InterAll) the call sites can land in a follow-up.
#[allow(clippy::too_many_arguments)]
pub fn green_func_exchange_real(
    ri: usize,
    rj: usize,
    spin: u8,
    spin_other: u8,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
) -> Complex64 {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    if ri == rj || n_elec == 0 || n_site == 0 {
        return Complex64::new(0.0, 0.0);
    }
    // Pre-conditions matching `green_func2`'s early returns for the
    // exchange tile.
    if ele_num[ri + spin as usize * n_site] != 0
        || ele_num[rj + spin as usize * n_site] != 1
        || ele_num[rj + spin_other as usize * n_site] != 0
        || ele_num[ri + spin_other as usize * n_site] != 1
    {
        return Complex64::new(0.0, 0.0);
    }

    let n_proj = ele_proj_cnt.len();
    let mut my_ele_idx = ele_idx.to_vec();
    let mut my_ele_num = ele_num.to_vec();
    let mut proj_mid = vec![0_i64; n_proj];
    let mut proj_final = vec![0_i64; n_proj];

    // 1st hop: electron at rj with spin `spin` -> ri.
    let mj = ele_cfg[rj + spin as usize * n_site];
    if mj < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mj = mj as usize;
    let msj = mj + spin as usize * n_elec;
    my_ele_idx[msj] = ri as i64;
    my_ele_num[ri + spin as usize * n_site] = 1;
    my_ele_num[rj + spin as usize * n_site] = 0;
    update_proj_cnt(
        rj as i64,
        ri as i64,
        spin,
        &mut proj_mid,
        ele_proj_cnt,
        &my_ele_num,
        data,
    );

    // 2nd hop: electron at ri with spin_other -> rj.
    let mi = ele_cfg[ri + spin_other as usize * n_site];
    if mi < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mi = mi as usize;
    let msi = mi + spin_other as usize * n_elec;
    my_ele_idx[msi] = rj as i64;
    my_ele_num[rj + spin_other as usize * n_site] = 1;
    my_ele_num[ri + spin_other as usize * n_site] = 0;
    update_proj_cnt(
        ri as i64,
        rj as i64,
        spin_other,
        &mut proj_final,
        &proj_mid,
        &my_ele_num,
        data,
    );

    let log_proj_delta =
        crate::sampling::projection::log_proj_ratio(&proj_final, ele_proj_cnt, data);
    let proj_ratio = log_proj_delta.exp();

    let _ = my_ele_num;

    // Fast path: rank-2 Woodbury Pfaffian update for real mode.
    //
    // Port of `calculate_new_pf_m_two2_real!` from upstream
    // `MVMCOptimizers.jl/src/vmc_sampling.jl`. Uses the cached
    // `inv_m_real` and `slater_elm_real` to compute the new Pfaffian in
    // O(N²) instead of the O(N³) full recomputation below.
    //
    // `mj` (spin) hops ri→rj was the 1st hop and `mi` (spin_other)
    // hops ri→rj was the 2nd hop. After both hops, `my_ele_idx` holds
    // the updated electron positions, from which `rsa` and `rsb` are
    // derived inside `calculate_new_pf_m_two2_real_flat`.
    if !state.slater_matrix.pf_m_real.is_empty() {
        let n_size = 2 * n_elec;
        let inv_stride = n_size * n_size + 1;
        let mut pf_m_new_real = vec![0.0_f64; n_qp_full];
        calculate_new_pf_m_two2_real_flat(
            mj,
            spin,
            mi,
            spin_other,
            &mut pf_m_new_real,
            &my_ele_idx,
            &state.slater_matrix.slater_elm_real,
            state.slater_matrix.inv_m_real.as_slice(),
            inv_stride,
            &state.slater_matrix.pf_m_real,
            0,
            n_qp_full,
            n_site,
            n_elec,
        );
        let new_ip_real = calculate_ip_real(&pf_m_new_real, 0, n_qp_full, data);
        // In real mode all quantities are real, so conj is a no-op.
        // Return as Complex64 to match the function signature.
        return Complex64::new(proj_ratio * new_ip_real, 0.0);
    }

    // Fallback: full O(N³) complex recomputation for all-complex mode.
    //
    // `green_func2` upstream returns `conj(z / ip)` where `z` is
    // the projection ratio times the new inner product. We emit the
    // `conj(z)` factor here so the caller multiplies by `conj(inv_ip)`.
    let n_site_l = n_site;
    let n_elec_l = n_elec;
    let ws = &mut state.workspace;
    let pool = &ws.pfapack;
    let mut new_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
    let scratch_slater = state.slater_matrix.slater_elm.clone();
    let mut scratch_inv = state.slater_matrix.inv_m.clone();
    if calc_m_all_complex(
        &my_ele_idx,
        &scratch_slater,
        &mut scratch_inv,
        &mut new_pf,
        0,
        n_qp_full,
        n_site_l,
        n_elec_l,
        pool,
    )
    .is_err()
    {
        return Complex64::new(0.0, 0.0);
    }
    let new_ip = calculate_ip_complex(&new_pf, 0, n_qp_full, data);
    let _ = scratch_inv;
    (Complex64::new(proj_ratio, 0.0) * new_ip).conj()
}

/// Non-FSZ Slater-parameter derivative block (`SlaterElmDiff_fcmp!`).
///
/// `sr_opt_o` is the view starting at the first Slater parameter slot,
/// i.e. index `2 * (1 + n_proj)` of the full `SROptO` array. For every
/// unique orbital parameter `k`, this writes `O_re[k]` and `O_im[k]` as
/// adjacent complex slots, matching the upstream real/imag packed layout.
pub fn slater_elm_diff(
    sr_opt_o: &mut [Complex64],
    ip: Complex64,
    ele_idx: &[i64],
    data: &ExpertModeData,
    state: &VmcOptimizationState,
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
    let n_qp_full = weights
        .qp_full_weight
        .len()
        .min(state.slater_matrix.pf_m.len());
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
            let ri = ri as usize;
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
            for msj in 0..n_size {
                let rj = ele_idx.get(msj).copied().unwrap_or(-1);
                if rj < 0 || rj as usize >= n_site {
                    continue;
                }
                let rj = rj as usize;
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
                let idx = mpidx * n_size * n_size + msi * n_size + msj;
                trans_orb_idx[idx] = orbital_idx[tri][trj];
                trans_orb_sgn[idx] = sgni * sgnj * orbital_sgn[tri][trj];
            }
        }
    }

    let mut buffer = vec![Complex64::new(0.0, 0.0); n_qp_full * n_slater];
    for qpidx in 0..n_qp_full {
        let mpidx = (qpidx / n_sp_gauss_leg).min(n_mp_trans.saturating_sub(1));
        let spidx = qpidx % n_sp_gauss_leg;
        if spidx >= weights.spgl_cos_sin.len()
            || spidx >= weights.spgl_cos_cos.len()
            || spidx >= weights.spgl_sin_sin.len()
        {
            continue;
        }
        let pf = state.slater_matrix.pf_m[qpidx];
        let cs = pf * weights.spgl_cos_sin[spidx];
        let cc = pf * weights.spgl_cos_cos[spidx];
        let ss = pf * weights.spgl_sin_sin[spidx];
        let tbase = mpidx * n_size * n_size;
        let inv_plane = state.slater_matrix.inv_m.qp_matrix_slice(qpidx);
        // Upstream `SlaterElmDiff` reads the inverse plane through the
        // transposed flat convention `msi*n_size + msj`; keep that contract
        // explicit while borrowing only the matrix portion of the QP plane.

        for msi in 0..n_elec {
            for msj in 0..n_elec {
                accumulate_slater_diff(
                    &mut buffer,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    inv_plane[msj + msi * n_size] * cs,
                );
            }
            for msj in n_elec..n_size {
                accumulate_slater_diff(
                    &mut buffer,
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
                    &mut buffer,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    inv_plane[msj + msi * n_size] * ss,
                );
            }
            for msj in n_elec..n_size {
                accumulate_slater_diff(
                    &mut buffer,
                    qpidx,
                    n_slater,
                    trans_orb_idx[tbase + msi * n_size + msj],
                    trans_orb_sgn[tbase + msi * n_size + msj],
                    -inv_plane[msj + msi * n_size] * cs,
                );
            }
        }
    }

    let inv_ip = Complex64::new(1.0, 0.0) / ip;
    for orbidx in 0..n_slater {
        let mut acc = Complex64::new(0.0, 0.0);
        for qpidx in 0..n_qp_full {
            acc += weights.qp_full_weight[qpidx] * buffer[qpidx * n_slater + orbidx];
        }
        acc *= inv_ip;
        sr_opt_o[2 * orbidx] = acc;
        sr_opt_o[2 * orbidx + 1] = acc * Complex64::new(0.0, 1.0);
    }
}

fn accumulate_slater_diff(
    buffer: &mut [Complex64],
    qpidx: usize,
    n_slater: usize,
    orbidx: i64,
    sign: i64,
    value: Complex64,
) {
    if orbidx >= 0 && (orbidx as usize) < n_slater {
        buffer[qpidx * n_slater + orbidx as usize] += value * Complex64::new(sign as f64, 0.0);
    }
}

fn spin_code(spin: Spin) -> u8 {
    match spin {
        Spin::Up => 0,
        Spin::Down => 1,
    }
}

/// FSZ one-body Green function for `<c†_{ri,s} c_{rj,t}> / <x>`.
#[allow(clippy::too_many_arguments)]
pub fn green_func1_fsz(
    ri: usize,
    rj: usize,
    spin_create: u8,
    spin_annihilate: u8,
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    ele_spn: &[i64],
) -> Complex64 {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    if ip.norm() == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    let dst = ri + spin_create as usize * n_site;
    let src = rj + spin_annihilate as usize * n_site;
    if spin_create == spin_annihilate && ri == rj {
        return Complex64::new(ele_num[dst] as f64, 0.0);
    }
    if ele_num[dst] == 1 || ele_num[src] == 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mj = ele_cfg[src];
    if mj < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mj = mj as usize;
    let mut my_ele_idx = ele_idx.to_vec();
    let mut my_ele_spn = ele_spn.to_vec();
    let mut my_ele_num = ele_num.to_vec();
    let mut proj_new = vec![0_i64; ele_proj_cnt.len()];
    my_ele_idx[mj] = ri as i64;
    my_ele_spn[mj] = spin_create as i64;
    my_ele_num[src] = 0;
    my_ele_num[dst] = 1;

    let proj_ratio = if ri == rj {
        1.0
    } else {
        update_proj_cnt(
            rj as i64,
            ri as i64,
            spin_create,
            &mut proj_new,
            ele_proj_cnt,
            &my_ele_num,
            data,
        );
        crate::sampling::projection::log_proj_ratio(&proj_new, ele_proj_cnt, data).exp()
    };

    let pool = crate::state::ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
    let mut scratch_inv = state.slater_matrix.inv_m.clone();
    let mut new_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
    if crate::pfaffian::calc_m_all_fsz_complex(
        &my_ele_idx,
        &my_ele_spn,
        &state.slater_matrix.slater_elm,
        &mut scratch_inv,
        &mut new_pf,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .is_err()
    {
        return Complex64::new(0.0, 0.0);
    }
    ((Complex64::new(proj_ratio, 0.0) * calculate_ip_complex(&new_pf, 0, n_qp_full, data)) / ip)
        .conj()
}

/// FSZ exchange Green-function numerator for `green_func2_fsz`'s
/// Heisenberg exchange tile. Returns `conj(proj_ratio * ip_new)` so the
/// caller can multiply by `conj(1/ip_old)`.
#[allow(clippy::too_many_arguments)]
pub fn green_func_exchange_fsz(
    ri: usize,
    rj: usize,
    spin: u8,
    spin_other: u8,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    ele_spn: &[i64],
) -> Complex64 {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    if ri == rj || n_elec == 0 || n_site == 0 {
        return Complex64::new(0.0, 0.0);
    }
    if ele_num[ri + spin as usize * n_site] != 0
        || ele_num[rj + spin as usize * n_site] != 1
        || ele_num[rj + spin_other as usize * n_site] != 0
        || ele_num[ri + spin_other as usize * n_site] != 1
    {
        return Complex64::new(0.0, 0.0);
    }
    let n_proj = ele_proj_cnt.len();
    let mut my_ele_idx = ele_idx.to_vec();
    let mut my_ele_spn = ele_spn.to_vec();
    let mut my_ele_num = ele_num.to_vec();
    let mut proj_mid = vec![0_i64; n_proj];
    let mut proj_final = vec![0_i64; n_proj];

    let mj = ele_cfg[rj + spin as usize * n_site];
    if mj < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mj = mj as usize;
    my_ele_idx[mj] = ri as i64;
    my_ele_spn[mj] = spin as i64;
    my_ele_num[ri + spin as usize * n_site] = 1;
    my_ele_num[rj + spin as usize * n_site] = 0;
    update_proj_cnt(
        rj as i64,
        ri as i64,
        spin,
        &mut proj_mid,
        ele_proj_cnt,
        &my_ele_num,
        data,
    );

    let mi = ele_cfg[ri + spin_other as usize * n_site];
    if mi < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mi = mi as usize;
    my_ele_idx[mi] = rj as i64;
    my_ele_spn[mi] = spin_other as i64;
    my_ele_num[rj + spin_other as usize * n_site] = 1;
    my_ele_num[ri + spin_other as usize * n_site] = 0;
    update_proj_cnt(
        ri as i64,
        rj as i64,
        spin_other,
        &mut proj_final,
        &proj_mid,
        &my_ele_num,
        data,
    );

    let proj_ratio =
        crate::sampling::projection::log_proj_ratio(&proj_final, ele_proj_cnt, data).exp();
    let pool = crate::state::ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
    let mut scratch_inv = state.slater_matrix.inv_m.clone();
    let mut new_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
    if crate::pfaffian::calc_m_all_fsz_complex(
        &my_ele_idx,
        &my_ele_spn,
        &state.slater_matrix.slater_elm,
        &mut scratch_inv,
        &mut new_pf,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .is_err()
    {
        return Complex64::new(0.0, 0.0);
    }
    (Complex64::new(proj_ratio, 0.0) * calculate_ip_complex(&new_pf, 0, n_qp_full, data)).conj()
}

/// FSZ local Hamiltonian for the Heisenberg/Hubbard fixture path.
pub fn calculate_local_energy_fsz(
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    ele_spn: &[i64],
) -> Complex64 {
    let mut e = calculate_hamiltonian_diagonal(ele_num, data);
    for term in &data.transfer_terms {
        if term.site1 >= 0 && term.site2 >= 0 {
            let ri = term.site1 as usize;
            let rj = term.site2 as usize;
            if ri < data.modpara.nsite as usize && rj < data.modpara.nsite as usize {
                e += -term.value
                    * green_func1_fsz(
                        ri,
                        rj,
                        spin_code(term.spin1),
                        spin_code(term.spin2),
                        ip,
                        data,
                        state,
                        ele_idx,
                        ele_cfg,
                        ele_num,
                        ele_proj_cnt,
                        ele_spn,
                    );
            }
        }
    }
    if !data.exchange_terms.is_empty() && ip.norm() > 0.0 {
        let n_site = data.modpara.nsite as usize;
        let inv_ip = (Complex64::new(1.0, 0.0) / ip).conj();
        for term in &data.exchange_terms {
            let ri = term.site1;
            let rj = term.site2;
            if ri < 0 || rj < 0 || ri == rj || (ri as usize) >= n_site || (rj as usize) >= n_site {
                continue;
            }
            let g01 = green_func_exchange_fsz(
                ri as usize,
                rj as usize,
                0,
                1,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
                ele_spn,
            );
            let g10 = green_func_exchange_fsz(
                ri as usize,
                rj as usize,
                1,
                0,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
                ele_spn,
            );
            e += Complex64::new(term.value, 0.0) * (g01 + g10) * inv_ip;
        }
    }
    e
}

/// Non-FSZ 1-body Green function `<c†_{ri,spin} c_{rj,spin}> / <Ψ|x>`.
///
/// Mirrors `green_func1` from `MVMCOptimizers.jl/src/green_func_calc.jl`
/// for the non-FSZ (i_flg_orbital_general == 0) path. The calculation
/// applies to same-spin hops (spin_create == spin_annihilate); for
/// different-spin Transfer terms (spin-flip hops) in a non-FSZ basis the
/// electron-block layout is incompatible, so this function returns 0 —
/// only the FSZ path handles those.
///
/// Return value: `conj((proj_ratio * ip_new) / ip_old)`, matching the
/// `green_func1` upstream convention used by the Transfer accumulator.
#[allow(clippy::too_many_arguments)]
pub fn green_func1(
    ri: usize,
    rj: usize,
    spin_create: u8,
    spin_annihilate: u8,
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
) -> Complex64 {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();

    if ip.norm() == 0.0 {
        return Complex64::new(0.0, 0.0);
    }

    let dst = ri + spin_create as usize * n_site;
    let src = rj + spin_annihilate as usize * n_site;

    // Diagonal: <n_{ri,spin}>
    if spin_create == spin_annihilate && ri == rj {
        return Complex64::new(ele_num[src] as f64, 0.0);
    }

    // Spin-flip hops are not representable in the non-FSZ block layout.
    if spin_create != spin_annihilate {
        return Complex64::new(0.0, 0.0);
    }

    // Destination must be empty; source must be occupied.
    if ele_num[dst] == 1 || ele_num[src] == 0 {
        return Complex64::new(0.0, 0.0);
    }

    // Find which electron (index within spin block) sits at rj.
    let mj_raw = ele_cfg[src];
    if mj_raw < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mj = mj_raw as usize;
    let msj = mj + spin_annihilate as usize * n_elec; // full index in ele_idx

    // Build proposed electron configuration: move electron msj from rj → ri.
    let mut my_ele_idx = ele_idx.to_vec();
    let mut my_ele_num = ele_num.to_vec();
    my_ele_idx[msj] = ri as i64;
    my_ele_num[dst] = 1;
    my_ele_num[src] = 0;

    // Update projection counts for the hop rj → ri (same spin).
    let n_proj = ele_proj_cnt.len();
    let mut proj_new = vec![0_i64; n_proj];
    update_proj_cnt(
        rj as i64,
        ri as i64,
        spin_create,
        &mut proj_new,
        ele_proj_cnt,
        &my_ele_num,
        data,
    );

    let proj_ratio = if n_proj > 0 {
        crate::sampling::projection::log_proj_ratio(&proj_new, ele_proj_cnt, data).exp()
    } else {
        1.0
    };

    // Recompute Pfaffian for proposed configuration.
    let pool = crate::state::ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
    let mut scratch_inv = state.slater_matrix.inv_m.clone();
    let mut new_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
    if calc_m_all_complex(
        &my_ele_idx,
        &state.slater_matrix.slater_elm,
        &mut scratch_inv,
        &mut new_pf,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .is_err()
    {
        return Complex64::new(0.0, 0.0);
    }

    let new_ip = calculate_ip_complex(&new_pf, 0, n_qp_full, data);
    ((Complex64::new(proj_ratio, 0.0) * new_ip) / ip).conj()
}

/// Compute the local energy for a given sample. Non-FSZ path:
/// diagonal contributions (CoulombIntra / CoulombInter / Hund),
/// 1-body Transfer (kinetic hopping via `green_func1`),
/// and 2-body Exchange (via `green_func_exchange_real`).
pub fn calculate_local_energy(
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
) -> Complex64 {
    let mut e = calculate_hamiltonian_diagonal(ele_num, data);
    let n_site = data.modpara.nsite as usize;

    // Transfer (kinetic hopping) terms: H = Σ T_ij c†_{i,s1} c_{j,s2}.
    // The convention in upstream Julia is `e_local += -T * G1` where G1 is
    // the 1-body Green function ratio. Same-spin hops only for non-FSZ.
    if !data.transfer_terms.is_empty() && ip.norm() > 0.0 {
        for term in &data.transfer_terms {
            if term.site1 < 0 || term.site2 < 0 {
                continue;
            }
            let ri = term.site1 as usize;
            let rj = term.site2 as usize;
            if ri >= n_site || rj >= n_site {
                continue;
            }
            let spin_create = spin_code(term.spin1);
            let spin_annihilate = spin_code(term.spin2);
            let g1 = green_func1(
                ri,
                rj,
                spin_create,
                spin_annihilate,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
            e += -term.value * g1;
        }
    }

    // Exchange terms (2-body).
    if !data.exchange_terms.is_empty() && ip.norm() > 0.0 {
        let inv_ip = Complex64::new(1.0, 0.0) / ip;
        for term in &data.exchange_terms {
            let ri = term.site1;
            let rj = term.site2;
            if ri < 0 || rj < 0 || ri == rj || (ri as usize) >= n_site || (rj as usize) >= n_site {
                continue;
            }
            let g01 = green_func_exchange_real(
                ri as usize,
                rj as usize,
                0,
                1,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
            let g10 = green_func_exchange_real(
                ri as usize,
                rj as usize,
                1,
                0,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
            e += Complex64::new(term.value, 0.0) * (g01 + g10) * inv_ip.conj();
        }
    }

    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_hamiltonian_doublon_count() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.coulomb_intra_terms = vec![
            mvmc_expert_parsers::CoulombIntraTerm {
                site: 0,
                value: 4.0,
            },
            mvmc_expert_parsers::CoulombIntraTerm {
                site: 1,
                value: 2.0,
            },
        ];
        // Up = [1, 0], Down = [1, 1] -> site 0 doublon, site 1 single.
        let ele_num = vec![1, 0, 1, 1];
        let e = calculate_hamiltonian_diagonal(&ele_num, &data);
        assert!((e.re - 4.0).abs() < 1e-15);
        assert!(e.im.abs() < 1e-15);
    }

    #[test]
    fn set_projection_diff_writes_real_block() {
        let mut buf = vec![Complex64::new(0.0, 0.0); 8];
        let proj = vec![3_i64, -2_i64, 5_i64];
        set_projection_diff(&mut buf, &proj, 3);
        assert_eq!(buf[0], Complex64::new(1.0, 0.0));
        assert_eq!(buf[1], Complex64::new(0.0, 0.0));
        assert_eq!(buf[2], Complex64::new(3.0, 0.0));
        assert_eq!(buf[4], Complex64::new(-2.0, 0.0));
        assert_eq!(buf[6], Complex64::new(5.0, 0.0));
    }

    #[test]
    fn finalize_oo_store_preserves_legacy_flat_order() {
        let mut sr_opt_oo = vec![Complex64::new(0.0, 0.0); 4];
        let sr_opt_o_store = vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];

        finalize_oo_store(&mut sr_opt_oo, &sr_opt_o_store, 1, 1);

        assert_eq!(sr_opt_oo[0], Complex64::new(5.0, 0.0));
        assert_eq!(sr_opt_oo[1], Complex64::new(5.0, -10.0));
        assert_eq!(sr_opt_oo[2], Complex64::new(5.0, 10.0));
        assert_eq!(sr_opt_oo[3], Complex64::new(25.0, 0.0));
    }

    #[test]
    fn sr_store_gram_einsum_matches_manual_complex_reference() {
        let mut backend = tenferro_cpu::CpuBackend::new();
        let store = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
            vec![2, 3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-3.0, 0.5),
                Complex64::new(4.0, 2.0),
                Complex64::new(-1.5, 3.0),
                Complex64::new(0.25, -0.75),
            ],
        )
        .expect("typed tensor");

        let gram = sr_store_gram_einsum(&mut backend, &store).expect("einsum result");
        let gram_data = gram.host_data().expect("host data");

        let mut expected = vec![Complex64::new(0.0, 0.0); 4];
        let store_data = store.host_data().expect("host data");
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = Complex64::new(0.0, 0.0);
                for s in 0..3 {
                    let a = store_data[i + s * 2];
                    let b = store_data[j + s * 2];
                    acc += a * b.conj();
                }
                expected[i + j * 2] = acc;
            }
        }

        assert_eq!(gram_data, expected.as_slice());
    }

    #[test]
    fn qp_weighted_orbital_sum_einsum_matches_manual_complex_reference() {
        let mut backend = tenferro_cpu::CpuBackend::new();
        let buffer = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
            vec![2, 3],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.0, -2.0),
                Complex64::new(3.0, 1.5),
                Complex64::new(-4.0, 0.25),
            ],
        )
        .expect("typed tensor");
        let weights = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(-1.0, 1.0),
                Complex64::new(2.0, -0.5),
            ],
        )
        .expect("typed tensor");

        let weighted =
            qp_weighted_orbital_sum_einsum(&mut backend, &weights, &buffer).expect("einsum result");
        let weighted_data = weighted.host_data().expect("host data");

        let buffer_data = buffer.host_data().expect("host data");
        let weights_data = weights.host_data().expect("host data");
        let mut expected = vec![Complex64::new(0.0, 0.0); 2];
        for o in 0..2 {
            let mut acc = Complex64::new(0.0, 0.0);
            for q in 0..3 {
                acc += buffer_data[o + q * 2] * weights_data[q];
            }
            expected[o] = acc;
        }

        assert_eq!(weighted_data, expected.as_slice());
    }
}

/// FSZ Slater-parameter derivative block (`SlaterElmDiff_fsz!`).
pub fn slater_elm_diff_fsz(
    sr_opt_o: &mut [Complex64],
    ip: Complex64,
    ele_idx: &[i64],
    ele_spn: &[i64],
    data: &ExpertModeData,
    state: &VmcOptimizationState,
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
    let n_qp_full = weights
        .qp_full_weight
        .len()
        .min(state.slater_matrix.pf_m.len());
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

    let mut buffer = vec![Complex64::new(0.0, 0.0); n_qp_full * n_slater];
    for qpidx in 0..n_qp_full {
        let mpidx = qpidx.min(n_mp_trans.saturating_sub(1));
        let pf = state.slater_matrix.pf_m[qpidx];
        let inv_plane = state.slater_matrix.inv_m.qp_matrix_slice(qpidx);
        // See `slater_elm_diff`: this upstream path uses the same transposed
        // flat inverse convention.
        let tbase = mpidx * n_size * n_size;
        for msi in 0..n_size {
            for msj in 0..n_size {
                let orbidx = trans_orb_idx[tbase + msi * n_size + msj];
                if orbidx >= 0 && (orbidx as usize) < n_slater {
                    let sign = trans_orb_sgn[tbase + msi * n_size + msj];
                    let value =
                        -inv_plane[msj + msi * n_size] * pf * Complex64::new(sign as f64, 0.0);
                    buffer[qpidx * n_slater + orbidx as usize] += value;
                }
            }
        }
    }
    let inv_ip = Complex64::new(1.0, 0.0) / ip;
    for orbidx in 0..n_slater {
        let mut acc = Complex64::new(0.0, 0.0);
        for qpidx in 0..n_qp_full {
            acc += weights.qp_full_weight[qpidx] * buffer[qpidx * n_slater + orbidx];
        }
        acc *= inv_ip;
        sr_opt_o[2 * orbidx] = acc;
        sr_opt_o[2 * orbidx + 1] = acc * Complex64::new(0.0, 1.0);
    }
}
