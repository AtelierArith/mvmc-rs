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
//! Transfer and general fixed-Sz/FSZ Green ratios are implemented. FSZ local
//! energy includes InterAll; its runner support remains gated by validation.
//! Fixed-Sz InterAll production contributions are pending.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::items_after_test_module
)]

use num_complex::Complex64;

use mvmc_expert_parsers::{ExpertModeData, Spin};

use crate::c_timer::CTimer;
use crate::sampling::projection::update_proj_cnt;
use crate::sampling::updates::{
    calculate_new_pf_m2_complex_flat, calculate_new_pf_m2_fsz_complex_flat,
    calculate_new_pf_m2_real_flat, calculate_new_pf_m_two2_complex_flat,
    calculate_new_pf_m_two2_real_flat,
};
use crate::state::{TransferGreenScratch as GreenScratch, VmcOptimizationState};
use mvmc_expert_parsers::utils::julia_exp::exp as julia_exp;

mod fsz_green;
pub use fsz_green::green_func2_fsz;

/// Complete Julia's projection ratio with the RBM ratio for an operator move.
/// Counters are rebuilt from occupations and current parameters, including saved walkers.
pub(crate) fn with_rbm_ratio(
    projection_ratio: f64,
    new_num: &[i64],
    old_num: &[i64],
    data: &ExpertModeData,
) -> Complex64 {
    let mut ratio = Complex64::new(projection_ratio, 0.0);
    if data.has_rbm_terms() {
        let cfg = crate::sampling::rbm::RbmConfig::from(data);
        let new = crate::sampling::rbm::make_rbm_cnt(new_num, &cfg);
        let old = crate::sampling::rbm::make_rbm_cnt(old_num, &cfg);
        ratio *=
            crate::sampling::rbm_math::exp(crate::sampling::rbm::log_rbm_ratio(&new, &old, &cfg));
    }
    ratio
}

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
    if let Some(phys) = state.phys_quantities.as_mut() {
        phys.phys_lanczos_qqqq.fill(Complex64::new(0.0, 0.0));
        phys.phys_lanczos_qcisajsq.fill(Complex64::new(0.0, 0.0));
        phys.phys_lanczos_qcisajscktaltq
            .fill(Complex64::new(0.0, 0.0));
        phys.phys_lanczos_qcisajscktaltq_dc
            .fill(Complex64::new(0.0, 0.0));
        phys.local_cis_ajs.fill(Complex64::new(0.0, 0.0));
        phys.phys_cis_ajs.fill(Complex64::new(0.0, 0.0));
        phys.phys_cis_ajs_ckt_alt.fill(Complex64::new(0.0, 0.0));
        phys.local_cis_ajs_ckt_alt_dc.fill(Complex64::new(0.0, 0.0));
        phys.phys_cis_ajs_ckt_alt_dc.fill(Complex64::new(0.0, 0.0));
    }
}

/// Accumulate one sampled TwoBodyGEx product in canonical term order.
///
/// Julia forms the product from the same sample and conjugates the second
/// one-body factor before adding it to the weighted accumulator.
pub fn accumulate_two_body_gex_sample(
    accumulator: &mut [Complex64],
    one_body_values: &[Complex64],
    canonical_indices: &[(usize, usize)],
    weight: Complex64,
) {
    for (slot, &(first, second)) in canonical_indices.iter().enumerate() {
        if slot >= accumulator.len()
            || first >= one_body_values.len()
            || second >= one_body_values.len()
        {
            continue;
        }
        accumulator[slot] += weight * one_body_values[first] * one_body_values[second].conj();
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
    mvmc_expert_parsers::utils::julia_log::log(ip.abs() + 1.0e-100)
}

/// Complex `calculate_log_ip_fcmp` mirror.
pub fn calculate_log_ip_complex(
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    data: &ExpertModeData,
) -> Complex64 {
    let ip = calculate_ip_complex(pf_m, qp_start, qp_end, data);
    // Julia's complex path takes log(ip) directly, including log(0).
    // A nonfinite initial overlap triggers the sampler's original remake.
    let mag = ip.norm();
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

/// Active sample window and SR-CG finalization mode.
#[derive(Debug, Default, Clone, Copy)]
pub struct StoreFinalization {
    /// Zero-based first stored sample, used by grouped/MPI sample ranges.
    pub sample_start: usize,
    /// Materialize only the mean and diagonal blocks needed by SR-CG.
    pub diagonal_only: bool,
}

/// Store sqrt(weight)*O and accumulate weight*energy*O in real mode.
pub fn calculate_oo_store_real(
    sr_opt_ho: &mut [f64],
    sr_opt_o_store: &mut [f64],
    sr_opt_o: &[f64],
    w: f64,
    e: f64,
    sample: usize,
    sr_opt_size: usize,
) {
    let we = w * e;
    let sqrtw = w.sqrt();
    let store = &mut sr_opt_o_store[sample * sr_opt_size..(sample + 1) * sr_opt_size];
    for i in 0..sr_opt_size {
        store[i] = sqrtw * sr_opt_o[i];
        sr_opt_ho[i] += we * sr_opt_o[i];
    }
}

/// Finalize real O*O^T with Julia's BLAS GEMM path, preserving extra buffer slots.
/// SR-CG writes just the first two blocks, in sample order as in Julia.
pub fn finalize_oo_store_real(
    sr_opt_oo: &mut [f64],
    sr_opt_o_store: &[f64],
    sr_opt_size: usize,
    sample_size: usize,
    options: StoreFinalization,
) {
    let n = sr_opt_size;
    if options.diagonal_only {
        for i in 0..n {
            let mut mean = 0.0;
            let mut diagonal = 0.0;
            for sample in options.sample_start..options.sample_start + sample_size {
                let o = sr_opt_o_store[i + sample * n];
                mean += o;
                diagonal += o * o;
            }
            sr_opt_oo[i] = mean;
            sr_opt_oo[i + n] = diagonal;
        }
        return;
    }
    if n == 0 {
        return;
    }
    if sample_size == 0 {
        sr_opt_oo[..n * n].fill(0.0);
        return;
    }
    let active =
        &sr_opt_o_store[options.sample_start * n..(options.sample_start + sample_size) * n];
    let dim = i32::try_from(n).expect("SR Gram dimension must fit BLAS LP64");
    let samples = i32::try_from(sample_size).expect("sample count must fit BLAS LP64");
    // Julia mul!(C, O, transpose(O)) recognizes the shared operand and
    // selects SYRK. Below its max(n,samples)>=4 cutoff, generic_syrk!
    // accumulates with muladd instead. Both paths copy the upper triangle.
    if n.max(sample_size) < 4 {
        sr_opt_oo[..n * n].fill(0.0);
        for sample in 0..sample_size {
            for j in 0..n {
                let oj = active[j + sample * n];
                for i in 0..=j {
                    let index = i + j * n;
                    sr_opt_oo[index] = active[i + sample * n].mul_add(oj, sr_opt_oo[index]);
                }
            }
        }
    } else {
        // SAFETY: O is [n,samples], leading n; the writable output has n*n
        // entries. SYRK reads O and overwrites the output's upper triangle.
        unsafe {
            blas::dsyrk(
                b'U',
                b'N',
                dim,
                samples,
                1.0,
                active,
                dim,
                0.0,
                &mut sr_opt_oo[..n * n],
                dim,
            );
        }
    }
    for j in 0..n {
        for i in j + 1..n {
            sr_opt_oo[i + j * n] = sr_opt_oo[j + i * n];
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
    let we = w * e;
    let sqrtw = w.sqrt();
    let size_2 = 2 * sr_opt_size;
    let store = &mut sr_opt_o_store[sample * size_2..(sample + 1) * size_2];
    for i in 0..size_2 {
        store[i] = sqrtw * sr_opt_o[i];
        sr_opt_ho[i] += we * sr_opt_o[i];
    }
}

/// Complex `finalize_oo_store!` mirror — sums `O * O^H` from the sample
/// store back into `sr_opt_oo`.
pub fn finalize_oo_store(
    sr_opt_oo: &mut [Complex64],
    sr_opt_o_store: &[Complex64],
    sr_opt_size: usize,
    sample_size: usize,
    options: StoreFinalization,
) {
    let size_2 = 2 * sr_opt_size;
    if options.diagonal_only {
        for i in 0..size_2 {
            let mut mean = Complex64::new(0.0, 0.0);
            let mut diagonal = 0.0;
            for sample in options.sample_start..options.sample_start + sample_size {
                let o = sr_opt_o_store[i + sample * size_2];
                mean += o;
                diagonal += o.norm_sqr();
            }
            sr_opt_oo[i] = mean;
            sr_opt_oo[i + size_2] = Complex64::new(diagonal, 0.0);
        }
        return;
    }
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

    let store_tensor = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
        vec![size_2, sample_size],
        sr_opt_o_store
            [options.sample_start * size_2..(options.sample_start + sample_size) * size_2]
            .to_vec(),
    )
    .expect("SR store shape and data length must match");
    let gram = sr_store_gram_julia(&store_tensor).expect("SR store Gram must run");
    let gram = gram
        .host_data()
        .expect("SR store Gram tensor must be host-backed");
    for j in 0..size_2 {
        for i in 0..size_2 {
            sr_opt_oo[i * size_2 + j] = gram[i + j * size_2];
        }
    }
}

// Preserve the authoritative complex finalizer's sequential sample sum.
// A general einsum backend can change rounding and signed zeros, which changes
// the direct SR input even when the sampling/RNG trajectory is identical.
fn sr_store_gram_julia(
    store: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<Complex64>> {
    let shape = store.shape();
    assert_eq!(shape.len(), 2, "SR store must be [component, sample]");
    let (n, samples) = (shape[0], shape[1]);
    let raw = store.host_data()?;
    let mut gram = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for sample in 0..samples {
                sum += raw[i + sample * n] * raw[j + sample * n].conj();
            }
            gram[i + j * n] = sum;
        }
    }
    tenferro_tensor::TypedTensor::from_vec_col_major(vec![n, n], gram)
}

// The current Slater derivative buffers are still Vec-backed and live in a
// per-sample hot path. Keep the tensor einsum contract tested here, but do not
// pay a Vec -> tensor conversion for every sample until the store is tensor-backed.
#[allow(dead_code)]
fn qp_weighted_orbital_sum_einsum(
    backend: &mut tenferro_cpu::CpuBackend,
    weights: &tenferro_tensor::TypedTensor<Complex64>,
    buffer: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_einsum::Result<tenferro_tensor::TypedTensor<Complex64>> {
    use tenferro_einsum::TypedTensorEinsumExt;
    use tenferro_tensor::BackendSessionHost;

    backend.with_backend_session(|session| [buffer, weights].einsum("oq,q->o", session))
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

/// Fixed-Sz ratio for `c†(ri,spin) c(rj,spin)
/// c†(rk,spin_other) c(rl,spin_other)`, matching Julia's `green_func2`.
/// Coincident indices reduce to densities or one-body ratios before hopping.
#[allow(clippy::too_many_arguments)]
pub fn green_func2(
    ri: usize,
    rj: usize,
    rk: usize,
    rl: usize,
    spin: u8,
    spin_other: u8,
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
    if n_elec == 0 || n_site == 0 {
        return Complex64::new(0.0, 0.0);
    }
    let one = |ri, rj, s, state: &mut VmcOptimizationState| {
        green_func1(
            ri,
            rj,
            s,
            s,
            ip,
            data,
            state,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
        )
    };
    let zero = Complex64::new(0.0, 0.0);
    let rsi = ri + spin as usize * n_site;
    let rsj = rj + spin as usize * n_site;
    let rtk = rk + spin_other as usize * n_site;
    if spin == spin_other {
        if rk == rl {
            return if ele_num[rtk] == 0 {
                zero
            } else {
                one(ri, rj, spin, state)
            };
        } else if rj == rl {
            return zero;
        } else if ri == rl {
            return if ele_num[rsi] == 0 {
                zero
            } else if rj == rk {
                Complex64::new((1 - ele_num[rsj]) as f64, 0.0)
            } else {
                -one(rk, rj, spin, state)
            };
        } else if rj == rk {
            return if ele_num[rsj] == 1 {
                zero
            } else {
                one(ri, rl, spin, state)
            };
        } else if ri == rk {
            return zero;
        } else if ri == rj {
            return if ele_num[rsi] == 0 {
                zero
            } else {
                one(rk, rl, spin, state)
            };
        }
    } else if rk == rl {
        return if ele_num[rtk] == 0 {
            zero
        } else if ri == rj {
            Complex64::new(ele_num[rsi] as f64, 0.0)
        } else {
            one(ri, rj, spin, state)
        };
    } else if ri == rj {
        return if ele_num[rsi] == 0 {
            zero
        } else {
            one(rk, rl, spin_other, state)
        };
    }
    // General case: apply the rightmost hop before the leftmost hop.
    if ele_num[ri + spin as usize * n_site] != 0
        || ele_num[rj + spin as usize * n_site] != 1
        || ele_num[rk + spin_other as usize * n_site] != 0
        || ele_num[rl + spin_other as usize * n_site] != 1
    {
        return Complex64::new(0.0, 0.0);
    }

    let n_proj = ele_proj_cnt.len();
    let mut my_ele_idx = ele_idx.to_vec();
    let mut my_ele_num = ele_num.to_vec();
    let mut proj_mid = vec![0_i64; n_proj];
    let mut proj_final = vec![0_i64; n_proj];

    let mj = ele_cfg[rj + spin as usize * n_site];
    let mi = ele_cfg[rl + spin_other as usize * n_site];
    if mj < 0 || mi < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let (mj, mi) = (mj as usize, mi as usize);
    let msj = mj + spin as usize * n_elec;
    let msi = mi + spin_other as usize * n_elec;

    // Preserve Julia's hop order in projection and Pfaffian updates.
    my_ele_idx[msi] = rk as i64;
    my_ele_num[rl + spin_other as usize * n_site] = 0;
    my_ele_num[rk + spin_other as usize * n_site] = 1;
    update_proj_cnt(
        rl as i64,
        rk as i64,
        spin_other,
        &mut proj_mid,
        ele_proj_cnt,
        &my_ele_num,
        data,
    );
    my_ele_idx[msj] = ri as i64;
    my_ele_num[rj + spin as usize * n_site] = 0;
    my_ele_num[ri + spin as usize * n_site] = 1;
    update_proj_cnt(
        rj as i64,
        ri as i64,
        spin,
        &mut proj_final,
        &proj_mid,
        &my_ele_num,
        data,
    );

    let log_proj_delta =
        crate::sampling::projection::log_proj_ratio(&proj_final, ele_proj_cnt, data);
    let proj_ratio = with_rbm_ratio(julia_exp(log_proj_delta), &my_ele_num, ele_num, data);

    // Fast path: rank-2 Woodbury Pfaffian update for real mode.
    //
    // Port of `calculate_new_pf_m_two2_real!` from upstream
    // `MVMCOptimizers.jl/src/vmc_sampling.jl`. Uses the cached
    // `inv_m_real` and `slater_elm_real` to compute the new Pfaffian in
    // O(N²), using the same two-electron proposal kernels as sampling.
    //
    // `mi` (spin_other) hops rl→rk first, then `mj` (spin) hops rj→ri.
    // After both hops, `my_ele_idx` holds
    // the updated electron positions, from which `rsa` and `rsb` are
    // derived inside `calculate_new_pf_m_two2_real_flat`.
    if !state.slater_matrix.pf_m_real.is_empty() {
        let n_size = 2 * n_elec;
        let inv_stride = n_size * n_size + 1;
        let mut pf_m_new_real = vec![0.0_f64; n_qp_full];
        calculate_new_pf_m_two2_real_flat(
            mi,
            spin_other,
            mj,
            spin,
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
        return crate::julia_complex::divide(proj_ratio * Complex64::new(new_ip_real, 0.0), ip)
            .conj();
    }

    let n_size = 2 * n_elec;
    let mut new_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
    calculate_new_pf_m_two2_complex_flat(
        mi,
        spin_other,
        mj,
        spin,
        &mut new_pf,
        &my_ele_idx,
        &state.slater_matrix.slater_elm,
        state.slater_matrix.inv_m.as_slice(),
        n_size * n_size + 1,
        &state.slater_matrix.pf_m,
        0,
        n_qp_full,
        n_site,
        n_elec,
    );
    let new_ip = calculate_ip_complex(&new_pf, 0, n_qp_full, data);
    crate::julia_complex::divide(proj_ratio * new_ip, ip).conj()
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
    let mut scratch = crate::slater_derivative::SlaterDerivativeScratch::new();
    let mut timer = crate::c_timer::CTimer::<false>::new();
    crate::slater_derivative::slater_elm_diff_with_scratch_timed::<false>(
        sr_opt_o,
        ip,
        ele_idx,
        data,
        &state.slater_matrix,
        &mut scratch,
        &mut timer,
    );
}

pub(crate) fn spin_code(spin: Spin) -> u8 {
    match spin {
        Spin::Up => 0,
        Spin::Down => 1,
    }
}

/// OptTrans derivative components in sector order, using fixed QP weights.
/// Each pair is `sum(QPFixWeight * PfM) / ip` and its imaginary component.
/// This follows Julia's separate OptTrans block, including bounded output views.
pub fn opt_trans_diff(
    sr_opt_o: &mut [Complex64],
    ip: Complex64,
    data: &ExpertModeData,
    pf_m: &[Complex64],
) {
    let Some(weights) = data.qp_weights.as_ref() else {
        return;
    };
    let n_fix = weights.qp_fix_weight.len();
    if n_fix == 0 {
        return;
    }
    for sector in 0..data.count_opt_trans_parameters() {
        let mut acc = Complex64::new(0.0, 0.0);
        for (j, &weight) in weights.qp_fix_weight.iter().enumerate() {
            if let Some(&pf) = pf_m.get(sector * n_fix + j) {
                acc += weight * pf;
            }
        }
        let value = crate::julia_complex::divide(acc, ip);
        let real = 2 * sector;
        if real + 1 < sr_opt_o.len() {
            sr_opt_o[real] = value;
            sr_opt_o[real + 1] = Complex64::new(0.0, 1.0) * value;
        }
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
        julia_exp(crate::sampling::projection::log_proj_ratio(
            &proj_new,
            ele_proj_cnt,
            data,
        ))
    };

    let proj_ratio = with_rbm_ratio(proj_ratio, &my_ele_num, ele_num, data);
    let n_size = 2 * n_elec;
    let mut new_pf = vec![Complex64::new(0.0, 0.0); n_qp_full];
    calculate_new_pf_m2_fsz_complex_flat(
        mj,
        spin_create,
        &mut new_pf,
        &my_ele_idx,
        &my_ele_spn,
        &state.slater_matrix.slater_elm,
        state.slater_matrix.inv_m.as_slice(),
        n_size * n_size + 1,
        &state.slater_matrix.pf_m,
        0,
        n_qp_full,
        n_site,
        n_elec,
    );
    let numerator = proj_ratio * calculate_ip_complex(&new_pf, 0, n_qp_full, data);
    crate::julia_complex::divide(numerator, ip).conj()
}

/// FSZ local Hamiltonian, including general four-spin InterAll contributions.
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
    calculate_local_energy_fsz_timed(
        ip,
        data,
        state,
        ele_idx,
        ele_cfg,
        ele_num,
        ele_proj_cnt,
        ele_spn,
        &mut CTimer::<false>::new(),
    )
}

/// Evaluate this kernel with call-site-specific section and diagnostic timers.
pub fn calculate_local_energy_fsz_timed<const TIMED: bool>(
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    ele_spn: &[i64],
    timer: &mut CTimer<TIMED>,
) -> Complex64 {
    timer.start(70);
    let mut e = calculate_hamiltonian_diagonal(ele_num, data);
    timer.stop(70);
    timer.start(71);
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
    timer.stop(71);
    timer.start(72);
    for term in &data.pair_hop_terms {
        let n_site = data.modpara.nsite;
        if !(0..n_site).contains(&term.site1) || !(0..n_site).contains(&term.site2) {
            continue;
        }
        let ri = term.site1 as usize;
        let rj = term.site2 as usize;
        e += term.value
            * green_func2_fsz(
                ri,
                rj,
                ri,
                rj,
                0,
                0,
                1,
                1,
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
    if !data.exchange_terms.is_empty() && ip.norm() > 0.0 {
        let n_site = data.modpara.nsite as usize;
        for term in &data.exchange_terms {
            let ri = term.site1;
            let rj = term.site2;
            if ri < 0 || rj < 0 || (ri as usize) >= n_site || (rj as usize) >= n_site {
                continue;
            }
            let g01 = green_func2_fsz(
                ri as usize,
                rj as usize,
                rj as usize,
                ri as usize,
                0,
                0,
                1,
                1,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
                ele_spn,
            );
            let g10 = green_func2_fsz(
                ri as usize,
                rj as usize,
                rj as usize,
                ri as usize,
                1,
                1,
                0,
                0,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
                ele_spn,
            );
            e += term.value * (g01 + g10);
        }
    }
    for term in &data.inter_all_terms {
        let sites = [term.site0, term.site1, term.site2, term.site3];
        if sites
            .iter()
            .any(|&site| site < 0 || site >= data.modpara.nsite)
        {
            continue;
        }
        let spins = [term.spin0, term.spin1, term.spin2, term.spin3];
        assert!(
            spins.iter().all(|spin| (0..=1).contains(spin)),
            "invalid InterAll spin index"
        );
        e += term.value
            * green_func2_fsz(
                sites[0] as usize,
                sites[1] as usize,
                sites[2] as usize,
                sites[3] as usize,
                spins[0] as u8,
                spins[1] as u8,
                spins[2] as u8,
                spins[3] as u8,
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
    timer.stop(72);
    e
}

/// Non-FSZ 1-body Green function `<c†_{ri,spin} c_{rj,spin}> / <Ψ|x>`.
///
/// Mirrors `green_func1` from `MVMCOptimizers.jl/src/vmc_main_cal.jl`
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
    let mut scratch = GreenScratch::default();
    green_func1_impl::<false, false>(
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
        &mut scratch,
        &mut CTimer::<false>::new(),
    )
}

fn transfer_cache_signature(data: &ExpertModeData) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    mix(data.modpara.nsite as u64);
    mix(data.modpara.nelec as u64);
    mix(u64::from(crate::run::get_all_complex_flag(data)));
    for term in &data.transfer_terms {
        mix(term.site1 as u64);
        mix(term.site2 as u64);
        mix(term.spin1.as_code() as u64);
        mix(term.spin2.as_code() as u64);
        mix(term.value.re.to_bits());
        mix(term.value.im.to_bits());
    }
    for term in &data.gutzwiller_terms {
        mix(term.value.re.to_bits());
        mix(term.value.im.to_bits());
    }
    for term in &data.jastrow_terms {
        mix(term.value.re.to_bits());
        mix(term.value.im.to_bits());
    }
    mix(data.rbm_params.len() as u64);
    for value in &data.rbm_params {
        mix(value.re.to_bits());
        mix(value.im.to_bits());
    }
    mix(u64::from(data.has_rbm_terms()));
    mix(data.gutzwiller_idx.len() as u64);
    mix(data.n_gutzwiller_idx as u64);
    for &index in &data.gutzwiller_idx {
        mix(index as u64);
    }
    mix(data.jastrow_idx.len() as u64);
    mix(data.n_jastrow_idx as u64);
    for row in &data.jastrow_idx {
        mix(row.len() as u64);
        for &index in row {
            mix(index as u64);
        }
    }
    mix(data.doublon_holon_2site_indices.len() as u64);
    for table in &data.doublon_holon_2site_indices {
        mix(table.neighbors.len() as u64);
        for row in &table.neighbors {
            for &site in row {
                mix(site as u64);
            }
        }
    }
    mix(data.doublon_holon_2site_params.len() as u64);
    mix(data.doublon_holon_4site_indices.len() as u64);
    for table in &data.doublon_holon_4site_indices {
        mix(table.neighbors.len() as u64);
        for row in &table.neighbors {
            for &site in row {
                mix(site as u64);
            }
        }
    }
    mix(data.doublon_holon_4site_params.len() as u64);
    hash
}

fn refresh_transfer_cache(data: &ExpertModeData, state: &mut VmcOptimizationState) {
    let signature = transfer_cache_signature(data);
    if state.transfer_cache.signature == signature {
        return;
    }
    // The Julia real Transfer shortcut sums the section separately and is
    // disabled for RBM. The generic RBM/C complex path adds each contribution
    // directly to the diagonal energy; grouping that sum changes SR's HO.
    let all_real = !data.has_rbm_terms()
        && !crate::run::get_all_complex_flag(data)
        && data.transfer_terms.iter().all(|term| term.value.im == 0.0);
    let n_site = data.modpara.nsite.max(0) as usize;
    let direct_projection_eligible = !data.has_rbm_terms()
        && data.doublon_holon_2site_indices.is_empty()
        && data.doublon_holon_2site_params.is_empty()
        && data.doublon_holon_4site_indices.is_empty()
        && data.doublon_holon_4site_params.is_empty()
        && (data.n_gutzwiller_idx == 0 || data.gutzwiller_idx.len() >= n_site)
        && (data.n_jastrow_idx == 0
            || (data.jastrow_idx.len() >= n_site
                && data
                    .jastrow_idx
                    .iter()
                    .take(n_site)
                    .all(|row| row.len() >= n_site)));
    state.transfer_cache.terms = data
        .transfer_terms
        .iter()
        .filter_map(|term| {
            if term.site1 < 0
                || term.site2 < 0
                || term.site1 as usize >= n_site
                || term.site2 as usize >= n_site
            {
                return None;
            }
            Some(crate::state::TransferTermMetadata {
                site1: term.site1 as usize,
                site2: term.site2 as usize,
                spin1: term.spin1.as_code(),
                spin2: term.spin2.as_code(),
                value: term.value,
            })
        })
        .collect();
    state.transfer_cache.signature = signature;
    state.transfer_cache.all_real = all_real;
    state.transfer_cache.direct_projection_eligible = direct_projection_eligible;
}

// The cached real Transfer path evaluates the moved configuration directly.
// Its Jastrow subtraction must precede multiplication by the site charge.
fn calh1_direct_projection_ratio(
    source: usize,
    dest: usize,
    ele_num: &[i64],
    data: &ExpertModeData,
) -> Option<f64> {
    let n = data.modpara.nsite as usize;
    let ng = data.n_gutzwiller_idx.max(0) as usize;
    let nj = data.n_jastrow_idx.max(0) as usize;
    if data.has_rbm_terms()
        || !data.doublon_holon_2site_indices.is_empty()
        || !data.doublon_holon_2site_params.is_empty()
        || !data.doublon_holon_4site_indices.is_empty()
        || !data.doublon_holon_4site_params.is_empty()
        || (ng == 0 && !data.gutzwiller_terms.is_empty())
        || (nj == 0 && !data.jastrow_terms.is_empty())
        || ng > data.gutzwiller_terms.len()
        || nj > data.jastrow_terms.len()
        || (ng > 0 && data.gutzwiller_idx.len() < n)
        || (nj > 0
            && (data.jastrow_idx.len() < n
                || data.jastrow_idx.iter().take(n).any(|row| row.len() < n)))
    {
        return None;
    }
    let gutz = |site: usize| {
        if ng == 0 {
            return 0.0;
        }
        let index = data.gutzwiller_idx[site];
        if index < 0 || index as usize >= ng {
            0.0
        } else {
            data.gutzwiller_terms[index as usize].value.re
        }
    };
    let jastrow = |a: usize, b: usize| {
        if nj == 0 || a == b {
            return 0.0;
        }
        let index = data.jastrow_idx[a.min(b)][a.max(b)];
        if index < 0 || index as usize >= nj {
            0.0
        } else {
            data.jastrow_terms[index as usize].value.re
        }
    };
    let charge = |site: usize| ele_num[site] + ele_num[n + site] - 1;
    let mut z = 0.0;
    z -= gutz(source) * (ele_num[source] + ele_num[n + source]) as f64;
    z += gutz(dest) * (ele_num[dest] * ele_num[n + dest]) as f64;
    z += jastrow(source, dest) * (charge(source) - charge(dest) + 1) as f64;
    for site in 0..n {
        if site != source && site != dest {
            z += (jastrow(dest, site) - jastrow(source, site)) * charge(site) as f64;
        }
    }
    Some(julia_exp(z))
}

/// Evaluate Transfer's main-calculation one-body kernel with section timers.
/// Real mode follows its direct-projection and real-quotient arithmetic.
pub fn green_func1_timed<const TIMED: bool>(
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
    timer: &mut CTimer<TIMED>,
) -> Complex64 {
    let mut scratch = GreenScratch::default();
    green_func1_timed_with_scratch(
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
        &mut scratch,
        timer,
    )
}

pub(crate) fn green_func1_timed_with_scratch<const TIMED: bool>(
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
    scratch: &mut GreenScratch,
    timer: &mut CTimer<TIMED>,
) -> Complex64 {
    green_func1_impl::<TIMED, true>(
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
        scratch,
        timer,
    )
}

// Transfer's real main-calculation path uses direct projection arithmetic and
// a real quotient. General Green operators use Julia's projection-count ratio
// and complex quotient, including one-body reductions of two-body operators.
fn green_func1_impl<const TIMED: bool, const TRANSFER: bool>(
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
    scratch: &mut GreenScratch,
    timer: &mut CTimer<TIMED>,
) -> Complex64 {
    let diag = timer.diagnostics.calham1 && !crate::run::get_all_complex_flag(data);
    timer.start_diag(927, diag);
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();

    if ip.norm() == 0.0 {
        timer.stop_diag(927, diag);
        return Complex64::new(0.0, 0.0);
    }

    let dst = ri + spin_create as usize * n_site;
    let src = rj + spin_annihilate as usize * n_site;

    // Diagonal: <n_{ri,spin}>
    if spin_create == spin_annihilate && ri == rj {
        timer.stop_diag(927, diag);
        return Complex64::new(ele_num[src] as f64, 0.0);
    }

    // Spin-flip hops are not representable in the non-FSZ block layout.
    if spin_create != spin_annihilate {
        timer.stop_diag(927, diag);
        return Complex64::new(0.0, 0.0);
    }

    // Destination must be empty; source must be occupied.
    if ele_num[dst] == 1 || ele_num[src] == 0 {
        timer.stop_diag(927, diag);
        return Complex64::new(0.0, 0.0);
    }

    // Find which electron (index within spin block) sits at rj.
    let mj_raw = ele_cfg[src];
    if mj_raw < 0 {
        timer.stop_diag(927, diag);
        return Complex64::new(0.0, 0.0);
    }
    let mj = mj_raw as usize;
    let msj = mj + spin_annihilate as usize * n_elec; // full index in ele_idx

    // Build proposed electron configuration: move electron msj from rj → ri.
    scratch.ele_idx.clear();
    scratch.ele_idx.extend_from_slice(ele_idx);
    scratch.ele_num.clear();
    scratch.ele_num.extend_from_slice(ele_num);
    scratch.ele_idx[msj] = ri as i64;
    scratch.ele_num[dst] = 1;
    scratch.ele_num[src] = 0;

    timer.stop_diag(927, diag);
    timer.start_diag(921, diag);
    // Update projection counts for the hop rj → ri (same spin).
    let n_proj = ele_proj_cnt.len();
    scratch.proj_new.resize(n_proj, 0);
    scratch.proj_new.fill(0);
    update_proj_cnt(
        rj as i64,
        ri as i64,
        spin_create,
        &mut scratch.proj_new,
        ele_proj_cnt,
        &scratch.ele_num,
        data,
    );

    timer.stop_diag(921, diag);
    timer.start_diag(922, diag);
    let direct_ratio = if TRANSFER
        && !crate::run::get_all_complex_flag(data)
        && state.transfer_cache.direct_projection_eligible
    {
        calh1_direct_projection_ratio(rj, ri, &scratch.ele_num, data)
    } else {
        None
    };
    let proj_ratio = if let Some(ratio) = direct_ratio {
        ratio
    } else if n_proj > 0 {
        julia_exp(crate::sampling::projection::log_proj_ratio(
            &scratch.proj_new,
            ele_proj_cnt,
            data,
        ))
    } else {
        1.0
    };

    let proj_ratio = with_rbm_ratio(proj_ratio, &scratch.ele_num, ele_num, data);
    timer.stop_diag(922, diag);
    // The main-calculation state keeps one pad slot per QP; Julia's
    // wrapper compacts the same inverse planes before its Green helper.
    let inv_stride = (2 * n_elec).pow(2) + 1;
    if !state.slater_matrix.pf_m_real.is_empty() {
        scratch.new_pf_real.resize(n_qp_full, 0.0);
        scratch.new_pf_real.fill(0.0);
        timer.start_diag(923, diag);
        calculate_new_pf_m2_real_flat(
            mj,
            spin_annihilate,
            &mut scratch.new_pf_real,
            &scratch.ele_idx,
            &state.slater_matrix.slater_elm_real,
            state.slater_matrix.inv_m_real.as_slice(),
            inv_stride,
            &state.slater_matrix.pf_m_real,
            0,
            n_qp_full,
            n_site,
            n_elec,
        );
        timer.stop_diag(923, diag);
        timer.start_diag(924, diag);
        let new_ip = calculate_ip_real(&scratch.new_pf_real, 0, n_qp_full, data);
        timer.stop_diag(924, diag);
        return if TRANSFER && !data.has_rbm_terms() {
            Complex64::new(proj_ratio.re * new_ip / ip.re, 0.0)
        } else {
            crate::julia_complex::divide(proj_ratio * Complex64::new(new_ip, 0.0), ip).conj()
        };
    }
    scratch
        .new_pf_complex
        .resize(n_qp_full, Complex64::new(0.0, 0.0));
    scratch.new_pf_complex.fill(Complex64::new(0.0, 0.0));
    calculate_new_pf_m2_complex_flat(
        mj,
        spin_annihilate,
        &mut scratch.new_pf_complex,
        &scratch.ele_idx,
        &state.slater_matrix.slater_elm,
        state.slater_matrix.inv_m.as_slice(),
        inv_stride,
        &state.slater_matrix.pf_m,
        0,
        n_qp_full,
        n_site,
        n_elec,
    );
    let new_ip = calculate_ip_complex(&scratch.new_pf_complex, 0, n_qp_full, data);
    crate::julia_complex::divide(proj_ratio * new_ip, ip).conj()
}

/// Compute the local energy for a given sample. Non-FSZ path:
/// diagonal contributions (CoulombIntra / CoulombInter / Hund),
/// 1-body Transfer (kinetic hopping via `green_func1`),
/// and 2-body Exchange (via `green_func_exchange`).
pub fn calculate_local_energy(
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
) -> Complex64 {
    calculate_local_energy_timed(
        ip,
        data,
        state,
        ele_idx,
        ele_cfg,
        ele_num,
        ele_proj_cnt,
        &mut CTimer::<false>::new(),
    )
}

/// Calculate the second Lanczos Hamiltonian moment for diagonal and transfer
/// terms. The moved configuration is evaluated with the same local-energy
/// kernel as the original sample, preserving the Julia operator order.
/// PairHop and Exchange use the sequential two-body moves from Julia;
/// InterAll and FSZ callers remain outside this helper.
type LanczosMovedConfig = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>);

type LanczosMovedTerm = (Complex64, LanczosMovedConfig);

fn lanczos_apply_one_body(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    create_site: usize,
    annihilate_site: usize,
    spin: u8,
    data: &ExpertModeData,
) -> Option<LanczosMovedConfig> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    if create_site >= n_site
        || annihilate_site >= n_site
        || spin > 1
        || ele_num.len() < 2 * n_site
        || ele_cfg.len() < 2 * n_site
        || ele_idx.len() < 2 * n_elec
    {
        return None;
    }
    let dst = create_site + spin as usize * n_site;
    let src = annihilate_site + spin as usize * n_site;
    if create_site == annihilate_site {
        return (ele_num[dst] == 1).then(|| {
            (
                ele_idx.to_vec(),
                ele_cfg.to_vec(),
                ele_num.to_vec(),
                ele_proj_cnt.to_vec(),
            )
        });
    }
    if ele_num[dst] == 1 || ele_num[src] == 0 {
        return None;
    }
    let electron = usize::try_from(ele_cfg[src]).ok()?;
    if electron >= n_elec {
        return None;
    }
    let mut moved_idx = ele_idx.to_vec();
    let mut moved_cfg = ele_cfg.to_vec();
    let mut moved_num = ele_num.to_vec();
    let mut moved_proj = ele_proj_cnt.to_vec();
    moved_idx[electron + spin as usize * n_elec] = create_site as i64;
    moved_cfg[src] = -1;
    moved_cfg[dst] = electron as i64;
    moved_num[src] = 0;
    moved_num[dst] = 1;
    crate::sampling::projection::make_proj_cnt(&mut moved_proj, &moved_num, data);
    Some((moved_idx, moved_cfg, moved_num, moved_proj))
}

fn lanczos_apply_one_body_terms(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    create_site: usize,
    annihilate_site: usize,
    spin: u8,
    data: &ExpertModeData,
) -> Vec<LanczosMovedTerm> {
    lanczos_apply_one_body(
        ele_idx,
        ele_cfg,
        ele_num,
        ele_proj_cnt,
        create_site,
        annihilate_site,
        spin,
        data,
    )
    .into_iter()
    .map(|moved| (Complex64::new(1.0, 0.0), moved))
    .collect()
}

fn lanczos_apply_pair_hop(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    destination: usize,
    source: usize,
    data: &ExpertModeData,
) -> Option<LanczosMovedConfig> {
    lanczos_apply_two_body(
        ele_idx,
        ele_cfg,
        ele_num,
        ele_proj_cnt,
        destination,
        source,
        destination,
        source,
        0,
        1,
        data,
    )
}

fn lanczos_apply_two_body(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    first_create: usize,
    first_annihilate: usize,
    second_create: usize,
    second_annihilate: usize,
    first_spin: u8,
    second_spin: u8,
    data: &ExpertModeData,
) -> Option<LanczosMovedConfig> {
    lanczos_apply_two_body_terms(
        ele_idx,
        ele_cfg,
        ele_num,
        ele_proj_cnt,
        first_create,
        first_annihilate,
        second_create,
        second_annihilate,
        first_spin,
        second_spin,
        data,
    )
    .into_iter()
    .next()
    .map(|(_, moved)| moved)
}

/// Apply `c†_{ri,s} c_{rj,s} c†_{rk,t} c_{rl,t}` in Julia's operator order.
/// The special coincident-index branches preserve the fermionic signs and
/// diagonal coefficients used by `_lanczos_apply_two_body`.
fn lanczos_apply_two_body_terms(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    first_create: usize,
    first_annihilate: usize,
    second_create: usize,
    second_annihilate: usize,
    first_spin: u8,
    second_spin: u8,
    data: &ExpertModeData,
) -> Vec<LanczosMovedTerm> {
    let n_site = data.modpara.nsite.max(0) as usize;
    if first_spin > 1
        || second_spin > 1
        || [
            first_create,
            first_annihilate,
            second_create,
            second_annihilate,
        ]
        .iter()
        .any(|&site| site >= n_site)
    {
        return Vec::new();
    }
    let first_dst = first_create + first_spin as usize * n_site;
    let first_src = first_annihilate + first_spin as usize * n_site;
    let second_dst = second_create + second_spin as usize * n_site;
    let second_src = second_annihilate + second_spin as usize * n_site;
    let occupied = |site: usize| ele_num.get(site).copied().unwrap_or(0) == 1;
    let empty = |site: usize| ele_num.get(site).copied().unwrap_or(0) == 0;
    let same = || {
        vec![(
            Complex64::new(1.0, 0.0),
            (
                ele_idx.to_vec(),
                ele_cfg.to_vec(),
                ele_num.to_vec(),
                ele_proj_cnt.to_vec(),
            ),
        )]
    };
    let one = |create: usize,
               annihilate: usize,
               spin: u8,
               idx: &[i64],
               cfg: &[i64],
               num: &[i64],
               proj: &[i64]| {
        lanczos_apply_one_body_terms(idx, cfg, num, proj, create, annihilate, spin, data)
    };
    if first_spin == second_spin {
        if second_create == second_annihilate {
            if empty(second_dst) {
                return Vec::new();
            }
            if first_create == first_annihilate {
                return if occupied(first_dst) {
                    same()
                } else {
                    Vec::new()
                };
            }
            return one(
                first_create,
                first_annihilate,
                first_spin,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
        }
        if first_create == first_annihilate {
            if empty(first_dst) {
                return Vec::new();
            }
            return one(
                second_create,
                second_annihilate,
                second_spin,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
        }
        if first_annihilate == second_annihilate
            || first_create == second_annihilate
            || first_create == second_create
        {
            return Vec::new();
        }
        if occupied(first_dst) || empty(first_src) || occupied(second_dst) || empty(second_src) {
            return Vec::new();
        }
        let mut result = Vec::new();
        for (coef2, (idx, cfg, num, proj)) in one(
            second_create,
            second_annihilate,
            second_spin,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
        ) {
            for (coef1, moved) in one(
                first_create,
                first_annihilate,
                first_spin,
                &idx,
                &cfg,
                &num,
                &proj,
            ) {
                result.push((coef1 * coef2, moved));
            }
        }
        return result;
    }

    if second_create == second_annihilate {
        if empty(second_dst) {
            return Vec::new();
        }
        if first_create == first_annihilate {
            return if occupied(first_dst) {
                same()
            } else {
                Vec::new()
            };
        }
        return one(
            first_create,
            first_annihilate,
            first_spin,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
        );
    }
    if first_create == first_annihilate {
        if empty(first_dst) {
            return Vec::new();
        }
        return one(
            second_create,
            second_annihilate,
            second_spin,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
        );
    }
    if occupied(first_dst) || empty(first_src) || occupied(second_dst) || empty(second_src) {
        return Vec::new();
    }
    let mut result = Vec::new();
    for (coef2, (idx, cfg, num, proj)) in one(
        second_create,
        second_annihilate,
        second_spin,
        ele_idx,
        ele_cfg,
        ele_num,
        ele_proj_cnt,
    ) {
        for (coef1, moved) in one(
            first_create,
            first_annihilate,
            first_spin,
            &idx,
            &cfg,
            &num,
            &proj,
        ) {
            result.push((coef1 * coef2, moved));
        }
    }
    result
}

fn lanczos_evaluate_moved(
    moved_idx: &[i64],
    moved_cfg: &[i64],
    moved_num: &[i64],
    moved_proj: &[i64],
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    original_slater: &crate::state::SlaterMatrixData,
    all_complex: bool,
    n_site: usize,
    n_elec: usize,
    n_qp_full: usize,
    pool: &crate::state::ThreadedPfaPackWorkspace,
) -> Option<Complex64> {
    let calculation = if all_complex {
        crate::pfaffian::calc_m_all_complex_c_compat(
            moved_idx,
            &state.slater_matrix.slater_elm,
            &mut state.slater_matrix.inv_m,
            &mut state.slater_matrix.pf_m,
            0,
            n_qp_full,
            n_site,
            n_elec,
            pool,
        )
    } else {
        crate::pfaffian::calc_m_all_real(
            moved_idx,
            &state.slater_matrix.slater_elm_real,
            &mut state.slater_matrix.inv_m_real,
            &mut state.slater_matrix.pf_m_real,
            0,
            n_qp_full,
            n_site,
            n_elec,
            pool,
        )
    };
    if calculation.is_err() {
        state.slater_matrix = original_slater.clone();
        return None;
    }
    let moved_ip = if all_complex {
        calculate_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data)
    } else {
        Complex64::new(
            calculate_ip_real(&state.slater_matrix.pf_m_real, 0, n_qp_full, data),
            0.0,
        )
    };
    let value = if moved_ip.norm() > 0.0 {
        Some(calculate_local_energy(
            moved_ip, data, state, moved_idx, moved_cfg, moved_num, moved_proj,
        ))
    } else {
        None
    };
    state.slater_matrix = original_slater.clone();
    value
}

/// QPhysQ accumulators produced by Lanczos mode 2 for one sample.
#[derive(Debug, Default)]
pub(crate) struct LanczosGreenValues {
    pub(crate) one_body: Vec<Complex64>,
    pub(crate) factored_two_body: Vec<Complex64>,
    pub(crate) direct_two_body: Vec<Complex64>,
}

fn lanczos_local_value(value: Complex64, all_complex: bool) -> Complex64 {
    if all_complex {
        value
    } else {
        Complex64::new(value.re, 0.0)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn calculate_lanczos_green(
    h1: Complex64,
    _ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    local_one_body: &[Complex64],
    local_direct_two_body: &[Complex64],
    all_complex: bool,
) -> LanczosGreenValues {
    let n_one = data.green_one_terms.len();
    let n_factored = data.green_two_ex_indices.len();
    let n_direct = data.green_two_terms.len();
    let mut result = LanczosGreenValues {
        one_body: vec![Complex64::new(0.0, 0.0); 4 * n_one],
        factored_two_body: vec![Complex64::new(0.0, 0.0); 4 * n_factored],
        direct_two_body: vec![Complex64::new(0.0, 0.0); 4 * n_direct],
    };
    if data.modpara.lanczos_mode < 2 {
        return result;
    }

    let original_slater = state.slater_matrix.clone();
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let pool = crate::state::ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
    let mut lslca = vec![Complex64::new(0.0, 0.0); 2 * n_one];
    for (index, value) in local_one_body.iter().copied().enumerate().take(n_one) {
        lslca[index] = lanczos_local_value(value, all_complex);
    }
    for (index, term) in data.green_one_terms.iter().enumerate() {
        let Some((create, annihilate, spin)) = lanczos_one_body_indices(term, n_site) else {
            continue;
        };
        let mut value = Complex64::new(0.0, 0.0);
        for (_coef, (moved_idx, moved_cfg, moved_num, moved_proj)) in lanczos_apply_one_body_terms(
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            create,
            annihilate,
            spin,
            data,
        ) {
            if let Some(moved_h) = lanczos_evaluate_moved(
                &moved_idx,
                &moved_cfg,
                &moved_num,
                &moved_proj,
                data,
                state,
                &original_slater,
                all_complex,
                n_site,
                n_elec,
                n_qp_full,
                &pool,
            ) {
                // `local_one_body` is Julia's overlap ratio for this
                // operator; multiplying it by the moved configuration's
                // local Hamiltonian gives HCA.
                value += local_one_body.get(index).copied().unwrap_or_default() * moved_h;
            }
        }
        lslca[n_one + index] = lanczos_local_value(value, all_complex);
    }

    for rq in 0..2 {
        for rp in 0..2 {
            let block_one = n_one * (rp + 2 * rq);
            let right_q = if rp == 0 {
                Complex64::new(1.0, 0.0)
            } else {
                lanczos_local_value(h1, all_complex)
            };
            for index in 0..n_one {
                let left = lslca[rq * n_one + index];
                result.one_body[block_one + index] +=
                    lanczos_local_value(left, all_complex).conj() * right_q;
                if !all_complex {
                    result.one_body[block_one + index] =
                        Complex64::new(result.one_body[block_one + index].re, 0.0);
                }
            }
            for (index, &(idx0, idx1)) in data.green_two_ex_indices.iter().enumerate() {
                if idx0 >= n_one || idx1 >= n_one {
                    continue;
                }
                let left = lslca[rq * n_one + idx0];
                let right = lslca[rp * n_one + idx1];
                let left = if all_complex { left.conj() } else { left };
                let block_factored = n_factored * (rp + 2 * rq);
                result.factored_two_body[block_factored + index] += left * right;
            }
        }
    }

    let mut hca_direct = vec![Complex64::new(0.0, 0.0); n_direct];
    for (index, term) in data.green_two_terms.iter().enumerate() {
        let first_spin = term.spin1.as_code();
        let second_spin = term.spin3.as_code();
        let Some(first_create) = usize::try_from(term.site1).ok() else {
            continue;
        };
        let Some(first_annihilate) = usize::try_from(term.site2).ok() else {
            continue;
        };
        let Some(second_create) = usize::try_from(term.site3).ok() else {
            continue;
        };
        let Some(second_annihilate) = usize::try_from(term.site4).ok() else {
            continue;
        };
        let mut value = Complex64::new(0.0, 0.0);
        for (_coef, (moved_idx, moved_cfg, moved_num, moved_proj)) in lanczos_apply_two_body_terms(
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            first_create,
            first_annihilate,
            second_create,
            second_annihilate,
            first_spin,
            second_spin,
            data,
        ) {
            if let Some(moved_h) = lanczos_evaluate_moved(
                &moved_idx,
                &moved_cfg,
                &moved_num,
                &moved_proj,
                data,
                state,
                &original_slater,
                all_complex,
                n_site,
                n_elec,
                n_qp_full,
                &pool,
            ) {
                // As above, the direct Green value supplies the operator
                // overlap/sign while `moved_h` supplies the Hamiltonian.
                value += local_direct_two_body
                    .get(index)
                    .copied()
                    .unwrap_or_default()
                    * moved_h;
            }
        }
        hca_direct[index] = lanczos_local_value(value, all_complex);
    }
    for rq in 0..2 {
        for rp in 0..2 {
            let block = n_direct * (rp + 2 * rq);
            let right_q = if rp == 0 {
                Complex64::new(1.0, 0.0)
            } else {
                lanczos_local_value(h1, all_complex)
            };
            for index in 0..n_direct {
                let value = if rq == 0 {
                    local_direct_two_body
                        .get(index)
                        .copied()
                        .unwrap_or_default()
                } else {
                    hca_direct[index]
                };
                result.direct_two_body[block + index] +=
                    lanczos_local_value(value, all_complex) * right_q;
            }
        }
    }
    state.slater_matrix = original_slater;
    result
}

fn lanczos_one_body_indices(
    term: &mvmc_expert_parsers::GreenOneTerm,
    n_site: usize,
) -> Option<(usize, usize, u8)> {
    let create = usize::try_from(term.site1).ok()?;
    let annihilate = usize::try_from(term.site2).ok()?;
    if create >= n_site || annihilate >= n_site || term.spin1 != term.spin2 {
        return None;
    }
    Some((create, annihilate, term.spin1.as_code()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn calculate_lanczos_h2_transfer(
    h1: Complex64,
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    all_complex: bool,
) -> Complex64 {
    let mut h2 = h1 * calculate_hamiltonian_diagonal(ele_num, data);
    let original_slater = state.slater_matrix.clone();
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let pool = crate::state::ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
    let mut scratch = GreenScratch::default();

    for term in &data.transfer_terms {
        let ri = term.site1;
        let rj = term.site2;
        let spin_create = term.spin1.as_code();
        let spin_annihilate = term.spin2.as_code();
        if ri < 0
            || rj < 0
            || ri as usize >= n_site
            || rj as usize >= n_site
            || spin_create != spin_annihilate
        {
            continue;
        }
        let green = green_func1_impl::<false, true>(
            ri as usize,
            rj as usize,
            spin_create,
            spin_annihilate,
            ip,
            data,
            state,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            &mut scratch,
            &mut CTimer::<false>::new(),
        );
        if green.norm() == 0.0 {
            continue;
        }
        let src = rj as usize + spin_annihilate as usize * n_site;
        let dst = ri as usize + spin_create as usize * n_site;
        let moved_idx = scratch.ele_idx.clone();
        let moved_num = scratch.ele_num.clone();
        let moved_proj = scratch.proj_new.clone();
        let mut moved_cfg = ele_cfg.to_vec();
        let Some(electron) = ele_cfg[src].checked_abs().map(|v| v as usize) else {
            continue;
        };
        moved_cfg[src] = -1;
        moved_cfg[dst] = electron as i64;

        if all_complex {
            let _ = crate::pfaffian::calc_m_all_complex_c_compat(
                &moved_idx,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            );
        } else {
            let _ = crate::pfaffian::calc_m_all_real(
                &moved_idx,
                &state.slater_matrix.slater_elm_real,
                &mut state.slater_matrix.inv_m_real,
                &mut state.slater_matrix.pf_m_real,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            );
        }
        let moved_ip = if all_complex {
            calculate_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data)
        } else {
            Complex64::new(
                calculate_ip_real(&state.slater_matrix.pf_m_real, 0, n_qp_full, data),
                0.0,
            )
        };
        if moved_ip.norm() > 0.0 {
            let moved_h = calculate_local_energy(
                moved_ip,
                data,
                state,
                &moved_idx,
                &moved_cfg,
                &moved_num,
                &moved_proj,
            );
            // Match C's calHCA: form the local H overlap first, then apply
            // the transfer coefficient.  The grouping is observable for the
            // ill-conditioned Full Lanczos alpha equation.
            let hca = moved_h * green;
            h2 += (-term.value) * hca;
        }
        state.slater_matrix = original_slater.clone();
    }

    for term in &data.pair_hop_terms {
        let destination = usize::try_from(term.site1).ok();
        let source = usize::try_from(term.site2).ok();
        let (Some(destination), Some(source)) = (destination, source) else {
            continue;
        };
        let Some((moved_idx, moved_cfg, moved_num, moved_proj)) = lanczos_apply_pair_hop(
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            destination,
            source,
            data,
        ) else {
            continue;
        };
        let green = green_func2(
            destination,
            source,
            destination,
            source,
            0,
            1,
            ip,
            data,
            state,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
        );
        if green.norm() == 0.0 {
            continue;
        }
        if all_complex {
            let _ = crate::pfaffian::calc_m_all_complex_c_compat(
                &moved_idx,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            );
        } else {
            let _ = crate::pfaffian::calc_m_all_real(
                &moved_idx,
                &state.slater_matrix.slater_elm_real,
                &mut state.slater_matrix.inv_m_real,
                &mut state.slater_matrix.pf_m_real,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            );
        }
        let moved_ip = if all_complex {
            calculate_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data)
        } else {
            Complex64::new(
                calculate_ip_real(&state.slater_matrix.pf_m_real, 0, n_qp_full, data),
                0.0,
            )
        };
        if moved_ip.norm() > 0.0 {
            let moved_h = calculate_local_energy(
                moved_ip,
                data,
                state,
                &moved_idx,
                &moved_cfg,
                &moved_num,
                &moved_proj,
            );
            let hcaca = moved_h * green;
            h2 += term.value * hcaca;
        }
        state.slater_matrix = original_slater.clone();
    }

    for term in &data.exchange_terms {
        let ri = usize::try_from(term.site1).ok();
        let rj = usize::try_from(term.site2).ok();
        let (Some(ri), Some(rj)) = (ri, rj) else {
            continue;
        };
        // C's calculateHW accumulates both exchange spin channels into a
        // temporary before applying ParaExchange. Preserve that grouping.
        let mut exchange = Complex64::new(0.0, 0.0);
        for (first_spin, second_spin) in [(0_u8, 1_u8), (1_u8, 0_u8)] {
            let Some((moved_idx, moved_cfg, moved_num, moved_proj)) = lanczos_apply_two_body(
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
                ri,
                rj,
                rj,
                ri,
                first_spin,
                second_spin,
                data,
            ) else {
                continue;
            };
            let green = green_func2(
                ri,
                rj,
                rj,
                ri,
                first_spin,
                second_spin,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
            if green.norm() == 0.0 {
                continue;
            }
            if let Some(moved_h) = lanczos_evaluate_moved(
                &moved_idx,
                &moved_cfg,
                &moved_num,
                &moved_proj,
                data,
                state,
                &original_slater,
                all_complex,
                n_site,
                n_elec,
                n_qp_full,
                &pool,
            ) {
                let hcaca = moved_h * green;
                exchange += hcaca;
            }
        }
        h2 += term.value * exchange;
    }
    state.slater_matrix = original_slater;
    h2
}

/// Evaluate this kernel with call-site-specific section and diagnostic timers.
pub fn calculate_local_energy_timed<const TIMED: bool>(
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    timer: &mut CTimer<TIMED>,
) -> Complex64 {
    timer.start(70);
    let mut e = calculate_hamiltonian_diagonal(ele_num, data);
    timer.stop(70);
    timer.start(71);
    let n_site = data.modpara.nsite as usize;

    // Transfer (kinetic hopping) terms: H = Σ T_ij c†_{i,s1} c_{j,s2}.
    // The convention in upstream Julia is `e_local += -T * G1` where G1 is
    // the 1-body Green function ratio. Same-spin hops only for non-FSZ.
    if !data.transfer_terms.is_empty() && ip.norm() > 0.0 {
        refresh_transfer_cache(data, state);
        let real_transfer = state.transfer_cache.all_real;
        let mut transfer_energy = 0.0;
        // Keep the Julia-style reusable workspace in the optimization state.
        // Taking it out temporarily avoids aliasing the mutable state passed
        // to the Green kernel while preserving its capacities for the next
        // local-energy call.
        let mut green_scratch = std::mem::take(&mut state.transfer_scratch);
        for index in 0..state.transfer_cache.terms.len() {
            let term = state.transfer_cache.terms[index];
            let ri = term.site1;
            let rj = term.site2;
            let spin_create = term.spin1;
            let spin_annihilate = term.spin2;
            let diag = timer.diagnostics.calham1 && !crate::run::get_all_complex_flag(data);
            timer.start_diag(920, diag);
            let g1 = green_func1_timed_with_scratch(
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
                &mut green_scratch,
                timer,
            );
            timer.stop_diag(920, diag);
            if real_transfer {
                transfer_energy -= term.value.re * g1.re;
            } else {
                e += -term.value * g1;
            }
        }
        // Julia's real Transfer path sums this section before adding it to
        // the diagonal energy. Combining the two sums changes SR gradients.
        if real_transfer {
            e += Complex64::new(transfer_energy, 0.0);
        }
        state.transfer_scratch = green_scratch;
    }

    // Julia accumulates PairHop before Exchange in the two-body section.
    timer.stop(71);
    timer.start(72);
    for term in &data.pair_hop_terms {
        if !(0..data.modpara.nsite).contains(&term.site1)
            || !(0..data.modpara.nsite).contains(&term.site2)
        {
            continue;
        }
        let ri = term.site1 as usize;
        let rj = term.site2 as usize;
        e += term.value
            * green_func2(
                ri,
                rj,
                ri,
                rj,
                0,
                1,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
    }
    if !data.exchange_terms.is_empty() && ip.norm() > 0.0 {
        for term in &data.exchange_terms {
            let ri = term.site1;
            let rj = term.site2;
            if ri < 0 || rj < 0 || (ri as usize) >= n_site || (rj as usize) >= n_site {
                continue;
            }
            let g01 = green_func2(
                ri as usize,
                rj as usize,
                rj as usize,
                ri as usize,
                0,
                1,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
            let g10 = green_func2(
                ri as usize,
                rj as usize,
                rj as usize,
                ri as usize,
                1,
                0,
                ip,
                data,
                state,
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
            );
            e += term.value * (g01 + g10);
        }
    }

    timer.stop(72);
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
    fn transfer_cache_rebuilds_when_coefficients_change() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            value: Complex64::new(1.0, 0.0),
        });
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, false);
        refresh_transfer_cache(&data, &mut state);
        let first = state.transfer_cache.signature;
        assert_eq!(state.transfer_cache.terms.len(), 1);
        assert!(state.transfer_cache.all_real);
        data.transfer_terms[0].value.re = 2.0;
        refresh_transfer_cache(&data, &mut state);
        assert_ne!(state.transfer_cache.signature, first);
        assert_eq!(state.transfer_cache.terms[0].value.re, 2.0);
    }

    #[test]
    fn real_transfer_cache_eligibility_matches_julia_fast_path_requirements() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            value: Complex64::new(1.0, 0.0),
        });
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, false);
        refresh_transfer_cache(&data, &mut state);
        assert!(state.transfer_cache.all_real);
        assert!(state.transfer_cache.direct_projection_eligible);

        data.charge_rbm_phys_layer_terms
            .push(mvmc_expert_parsers::ChargeRBMPhysLayerTerm {
                site: 0,
                idx: 0,
                value: Complex64::new(0.25, 0.0),
                is_complex: false,
            });
        refresh_transfer_cache(&data, &mut state);
        assert!(!state.transfer_cache.direct_projection_eligible);
        data.charge_rbm_phys_layer_terms.clear();

        let before_mapping = state.transfer_cache.signature;
        data.gutzwiller_idx = vec![0, 1];
        refresh_transfer_cache(&data, &mut state);
        assert_ne!(state.transfer_cache.signature, before_mapping);
        let before_width = state.transfer_cache.signature;
        data.n_gutzwiller_idx = 1;
        refresh_transfer_cache(&data, &mut state);
        assert_ne!(state.transfer_cache.signature, before_width);

        data.doublon_holon_2site_indices
            .push(mvmc_expert_parsers::DoublonHolon2SiteIndex {
                neighbors: vec![[1, 1], [0, 0]],
            });
        refresh_transfer_cache(&data, &mut state);
        assert!(!state.transfer_cache.direct_projection_eligible);
        data.doublon_holon_2site_indices.clear();

        data.complex_flags = vec![1];
        refresh_transfer_cache(&data, &mut state);
        assert!(!state.transfer_cache.all_real);
    }

    #[test]
    fn transfer_local_energy_reuses_workspace_capacity() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
            site1: 1,
            spin1: Spin::Up,
            site2: 0,
            spin2: Spin::Up,
            value: Complex64::new(1.0, 0.0),
        });
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, false);
        state.slater_matrix.pf_m_real[0] = 1.0;
        let idx = [0_i64, 0];
        let cfg = [0_i64, -1, -1, -1];
        let num = [1_i64, 0, 0, 0];
        let counts: [i64; 0] = [];

        let _ = calculate_local_energy(
            Complex64::new(1.0, 0.0),
            &data,
            &mut state,
            &idx,
            &cfg,
            &num,
            &counts,
        );
        let first_capacity = (
            state.transfer_scratch.ele_idx.capacity(),
            state.transfer_scratch.ele_num.capacity(),
            state.transfer_scratch.new_pf_real.capacity(),
        );
        assert!(first_capacity.0 >= idx.len());
        assert!(first_capacity.1 >= num.len());
        assert!(first_capacity.2 >= 1);

        let _ = calculate_local_energy(
            Complex64::new(1.0, 0.0),
            &data,
            &mut state,
            &idx,
            &cfg,
            &num,
            &counts,
        );
        assert_eq!(
            first_capacity,
            (
                state.transfer_scratch.ele_idx.capacity(),
                state.transfer_scratch.ele_num.capacity(),
                state.transfer_scratch.new_pf_real.capacity(),
            )
        );
    }

    #[test]
    fn lanczos_pair_hop_applies_down_then_up_like_julia() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        let idx = [0_i64, 0];
        let cfg = [0_i64, -1, 0, -1];
        let num = [1_i64, 0, 1, 0];
        let proj: [i64; 0] = [];
        let (moved_idx, moved_cfg, moved_num, moved_proj) =
            lanczos_apply_pair_hop(&idx, &cfg, &num, &proj, 1, 0, &data)
                .expect("occupied source pair can hop to an empty destination");
        assert_eq!(moved_idx, [1, 1]);
        assert_eq!(moved_cfg, [-1, 0, -1, 0]);
        assert_eq!(moved_num, [0, 1, 0, 1]);
        assert!(moved_proj.is_empty());
    }

    #[test]
    fn lanczos_exchange_applies_each_spin_channel_in_julia_order() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        // Up at site 1 and down at site 0 exchange positions.
        let idx = [1_i64, 0];
        let cfg = [-1_i64, 0, 0, -1];
        let num = [0_i64, 1, 1, 0];
        let proj: [i64; 0] = [];
        let moved = lanczos_apply_two_body(&idx, &cfg, &num, &proj, 0, 1, 1, 0, 0, 1, &data)
            .expect("exchange move should be valid");
        assert_eq!(moved.0, [0, 1]);
        assert_eq!(moved.1, [0, -1, -1, 0]);
        assert_eq!(moved.2, [1, 0, 0, 1]);
        assert!(moved.3.is_empty());
    }

    #[test]
    fn real_transfer_fast_green_matches_generic_and_local_energy() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
            site1: 1,
            spin1: Spin::Up,
            site2: 0,
            spin2: Spin::Up,
            value: Complex64::new(0.75, 0.0),
        });
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, false);
        state.slater_matrix.pf_m_real[0] = 1.0;
        // The one-electron real quotient is nonzero and deterministic:
        // inv row 0 contracts with the moved Slater row to give -1.
        state.slater_matrix.inv_m_real.as_mut_slice()[0] = 1.0;
        state.slater_matrix.slater_elm_real.set(0, 1, 0, 1.0);
        let idx = [0_i64, 0];
        let cfg = [0_i64, -1, -1, -1];
        let num = [1_i64, 0, 0, 0];
        let counts: [i64; 0] = [];
        let ip = Complex64::new(1.0, 0.0);

        let generic = green_func1(1, 0, 0, 0, ip, &data, &mut state, &idx, &cfg, &num, &counts);
        refresh_transfer_cache(&data, &mut state);
        let mut timer = CTimer::<false>::new();
        let fast = green_func1_timed(
            1, 0, 0, 0, ip, &data, &mut state, &idx, &cfg, &num, &counts, &mut timer,
        );
        assert_eq!(fast.re.to_bits(), generic.re.to_bits());
        assert_eq!(fast.im, generic.im);

        let energy = calculate_local_energy(ip, &data, &mut state, &idx, &cfg, &num, &counts);
        assert_eq!(energy.re, -data.transfer_terms[0].value.re * generic.re);
        assert_eq!(energy.im, 0.0);
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

        finalize_oo_store(
            &mut sr_opt_oo,
            &sr_opt_o_store,
            1,
            1,
            StoreFinalization::default(),
        );

        assert_eq!(sr_opt_oo[0], Complex64::new(5.0, 0.0));
        assert_eq!(sr_opt_oo[1], Complex64::new(5.0, -10.0));
        assert_eq!(sr_opt_oo[2], Complex64::new(5.0, 10.0));
        assert_eq!(sr_opt_oo[3], Complex64::new(25.0, 0.0));
    }

    #[test]
    fn sr_store_gram_julia_matches_manual_complex_reference() {
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

        let gram = sr_store_gram_julia(&store).expect("source-ordered Gram");
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
    fn stored_direct_sr_gram_matches_sampled_julia_bits() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/sr_direct");
        for case in [
            "real",
            "cmp",
            "fsz",
            "hubbard",
            "pairhop_real",
            "pairhop_fsz",
            "dh2_real",
            "dh2_cmp",
            "dh2_fsz",
            "dh4_real",
            "dh4_cmp",
            "dh4_fsz",
            "dh24_real",
            "dh24_cmp",
            "dh24_fsz",
            "rbm_real",
            "rbm_cmp",
            "rbm_general_cmp",
            "rbm_dh24_cmp",
            "rbm_fsz",
            "rbm_reference_cmp",
        ] {
            let fixture =
                std::fs::read_to_string(root.join(format!("{case}_store_runner/gram.txt")))
                    .unwrap();
            let mut lines = fixture.lines();
            let dims: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let parse = |line: &str| -> Vec<f64> {
                line.split_whitespace()
                    .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                    .collect()
            };
            let store = parse(lines.next().unwrap());
            let expected = parse(lines.next().unwrap());
            let actual = if matches!(
                case,
                "cmp"
                    | "fsz"
                    | "pairhop_fsz"
                    | "dh2_cmp"
                    | "dh2_fsz"
                    | "dh4_cmp"
                    | "dh4_fsz"
                    | "dh24_cmp"
                    | "dh24_fsz"
                    | "rbm_cmp"
                    | "rbm_general_cmp"
                    | "rbm_dh24_cmp"
                    | "rbm_fsz"
                    | "rbm_reference_cmp"
            ) {
                let store: Vec<Complex64> = store
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|z| Complex64::new(z[0], z[1]))
                    .collect();
                let mut oo = vec![Complex64::new(0.0, 0.0); expected.len() / 2];
                finalize_oo_store(
                    &mut oo,
                    &store,
                    dims[0],
                    dims[1],
                    StoreFinalization::default(),
                );
                oo.into_iter()
                    .flat_map(|z| [z.re, z.im])
                    .collect::<Vec<_>>()
            } else {
                let mut oo = vec![0.0; expected.len()];
                finalize_oo_store_real(
                    &mut oo,
                    &store,
                    dims[0],
                    dims[1],
                    StoreFinalization::default(),
                );
                oo
            };
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{case} Gram entry {i}: {a:?} versus {b:?}"
                );
            }
        }
    }

    #[test]
    fn real_gram_matches_julia_generic_and_syrk_dispatch_boundary() {
        let fixture = include_str!("../../../tests/fixtures/sr_direct/small_gram.txt");
        let mut lines = fixture.lines().filter(|s| !s.starts_with('#'));
        while let Some(dimensions) = lines.next() {
            let dims: Vec<usize> = dimensions
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let parse = |s: &str| {
                s.split_whitespace()
                    .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                    .collect::<Vec<_>>()
            };
            let store = parse(lines.next().unwrap());
            let expected = parse(lines.next().unwrap());
            let mut oo = vec![777.0; expected.len()];
            finalize_oo_store_real(
                &mut oo,
                &store,
                dims[0],
                dims[1],
                StoreFinalization::default(),
            );
            for (i, (a, b)) in oo.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "n={}, samples={}, entry {i}",
                    dims[0],
                    dims[1]
                );
            }
        }
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

    #[test]
    fn two_body_gex_accumulates_same_sample_conjugated_products() {
        let values = [
            Complex64::new(1.0, 2.0),
            Complex64::new(-0.5, 0.25),
            Complex64::new(2.0, -1.0),
        ];
        let mut accumulator = vec![Complex64::new(0.0, 0.0); 2];
        accumulate_two_body_gex_sample(
            &mut accumulator,
            &values,
            &[(0, 1), (2, 0)],
            Complex64::new(0.5, -0.25),
        );
        assert_eq!(
            accumulator,
            vec![Complex64::new(-0.3125, -0.625), Complex64::new(-1.25, -2.5),]
        );
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
    let mut scratch = crate::slater_derivative::SlaterDerivativeScratch::new();
    crate::slater_derivative::slater_elm_diff_fsz_with_scratch(
        sr_opt_o,
        ip,
        ele_idx,
        ele_spn,
        data,
        &state.slater_matrix,
        &mut scratch,
    );
}
