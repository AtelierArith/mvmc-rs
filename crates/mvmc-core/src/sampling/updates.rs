//! Phase 4.3.4a — one-electron Pfaffian update ratios.
//!
//! Port targets from `MVMCOptimizers.jl/src/vmc_sampling.jl`:
//! `calculate_new_pf_m2!`, `calculate_new_pf_m2_fsz!`,
//! `calculate_new_pf_m2_real!`, `calculate_new_pf_m2_fsz_real!`,
//! `calculate_new_pf_m_two2!`, `calculate_new_pf_m_two_fsz!`, and
//! the corresponding real variants.
//!
//! These helpers intentionally operate on **flat raw `inv_m` slices**
//! with an explicit `inv_stride` so we can mirror the active upstream
//! path exactly. The Julia state wrapper copies `inv_m_temp[:,:,qp]`
//! into `state.slater_matrix.inv_m` using `n_size*n_size` stride even
//! though the allocation reserves one pad slot per QP. Later update
//! helpers read that same no-pad row-major flat layout.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use num_complex::Complex64;

use crate::state::SlaterElmFlat;

/// Complex normal-mode `calculate_new_pf_m2!`.
///
/// `n_elec` is per spin (`Ne`); `n_size = 2*n_elec`.
/// `qp_start..qp_end` is 0-based and half-open.
pub fn calculate_new_pf_m2_complex_flat(
    ma: usize,
    spin: u8,
    pf_m_new: &mut [Complex64],
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &[Complex64],
    inv_stride: usize,
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride + msa * n_size;
        let mut ratio = Complex64::new(0.0, 0.0);
        for msj in 0..n_size {
            let rsj = if msj < n_elec {
                ele_idx[msj] as usize
            } else {
                ele_idx[msj] as usize + n_site
            };
            let inv_val = inv_m_flat[inv_base + msj];
            let slt_val = slater_elm.get(qp, rsa, rsj);
            ratio += inv_val * slt_val;
        }
        pf_m_new[qp] = -ratio * pf_m[qp];
    }
}

/// Real normal-mode `calculate_new_pf_m2_real!`.
pub fn calculate_new_pf_m2_real_flat(
    ma: usize,
    spin: u8,
    pf_m_new: &mut [f64],
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &[f64],
    inv_stride: usize,
    pf_m: &[f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride + msa * n_size;
        let mut ratio = 0.0;
        for msj in 0..n_size {
            let rsj = if msj < n_elec {
                ele_idx[msj] as usize
            } else {
                ele_idx[msj] as usize + n_site
            };
            ratio += inv_m_flat[inv_base + msj] * slater_elm.get(qp, rsa, rsj);
        }
        pf_m_new[qp] = -ratio * pf_m[qp];
    }
}

/// Complex FSZ `calculate_new_pf_m2_fsz!`.
pub fn calculate_new_pf_m2_fsz_complex_flat(
    ma: usize,
    spin: u8,
    pf_m_new: &mut [Complex64],
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &[Complex64],
    inv_stride: usize,
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride + msa * n_size;
        let mut ratio = Complex64::new(0.0, 0.0);
        for msj in 0..n_size {
            let rsj = ele_idx[msj] as usize + (ele_spn[msj] as usize) * n_site;
            ratio += inv_m_flat[inv_base + msj] * slater_elm.get(qp, rsa, rsj);
        }
        pf_m_new[qp] = -ratio * pf_m[qp];
    }
}

/// Real FSZ `calculate_new_pf_m2_fsz_real!`.
pub fn calculate_new_pf_m2_fsz_real_flat(
    ma: usize,
    spin: u8,
    pf_m_new: &mut [f64],
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &[f64],
    inv_stride: usize,
    pf_m: &[f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride + msa * n_size;
        let mut ratio = 0.0;
        for msj in 0..n_size {
            let rsj = ele_idx[msj] as usize + (ele_spn[msj] as usize) * n_site;
            ratio += inv_m_flat[inv_base + msj] * slater_elm.get(qp, rsa, rsj);
        }
        pf_m_new[qp] = -ratio * pf_m[qp];
    }
}

/// Complex normal-mode `calculate_new_pf_m_two2!`.
pub fn calculate_new_pf_m_two2_complex_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    pf_m_new: &mut [Complex64],
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &[Complex64],
    inv_stride: usize,
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let msb = mb + (spin_other as usize) * n_elec;
    if msa == msb {
        calculate_new_pf_m2_complex_flat(
            mb, spin_other, pf_m_new, ele_idx, slater_elm, inv_m_flat, inv_stride, pf_m, qp_start,
            qp_end, n_site, n_elec,
        );
        return;
    }

    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let mut vec_a = vec![Complex64::new(0.0, 0.0); n_size];
    let mut vec_b = vec![Complex64::new(0.0, 0.0); n_size];

    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride;
        fill_vecs_normal_complex(
            qp, rsa, rsb, ele_idx, slater_elm, &mut vec_a, &mut vec_b, n_site, n_elec,
        );
        let ratio = two_ratio_complex(msa, msb, inv_m_flat, inv_base, n_size, &vec_a, &vec_b);
        pf_m_new[qp] = ratio * pf_m[qp];
    }
}

/// Real normal-mode `calculate_new_pf_m_two2_real!`.
pub fn calculate_new_pf_m_two2_real_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    pf_m_new: &mut [f64],
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &[f64],
    inv_stride: usize,
    pf_m: &[f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let msb = mb + (spin_other as usize) * n_elec;
    if msa == msb {
        calculate_new_pf_m2_real_flat(
            mb, spin_other, pf_m_new, ele_idx, slater_elm, inv_m_flat, inv_stride, pf_m, qp_start,
            qp_end, n_site, n_elec,
        );
        return;
    }

    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let mut vec_a = vec![0.0; n_size];
    let mut vec_b = vec![0.0; n_size];

    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride;
        fill_vecs_normal_real(
            qp, rsa, rsb, ele_idx, slater_elm, &mut vec_a, &mut vec_b, n_site, n_elec,
        );
        let ratio = two_ratio_real(msa, msb, inv_m_flat, inv_base, n_size, &vec_a, &vec_b);
        pf_m_new[qp] = ratio * pf_m[qp];
    }
}

/// Complex FSZ `calculate_new_pf_m_two_fsz!`.
pub fn calculate_new_pf_m_two_fsz_complex_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    pf_m_new: &mut [Complex64],
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &[Complex64],
    inv_stride: usize,
    pf_m: &[Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let msb = mb;
    if msa == msb {
        calculate_new_pf_m2_fsz_complex_flat(
            mb, spin_other, pf_m_new, ele_idx, ele_spn, slater_elm, inv_m_flat, inv_stride, pf_m,
            qp_start, qp_end, n_site, n_elec,
        );
        return;
    }

    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let mut vec_a = vec![Complex64::new(0.0, 0.0); n_size];
    let mut vec_b = vec![Complex64::new(0.0, 0.0); n_size];

    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride;
        fill_vecs_fsz_complex(
            qp, rsa, rsb, ele_idx, ele_spn, slater_elm, &mut vec_a, &mut vec_b, n_site,
        );
        let ratio = two_ratio_complex(msa, msb, inv_m_flat, inv_base, n_size, &vec_a, &vec_b);
        pf_m_new[qp] = ratio * pf_m[qp];
    }
}

/// Real FSZ `calculate_new_pf_m_two2_fsz_real!`.
pub fn calculate_new_pf_m_two_fsz_real_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    pf_m_new: &mut [f64],
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &[f64],
    inv_stride: usize,
    pf_m: &[f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let msb = mb;
    if msa == msb {
        calculate_new_pf_m2_fsz_real_flat(
            mb, spin_other, pf_m_new, ele_idx, ele_spn, slater_elm, inv_m_flat, inv_stride, pf_m,
            qp_start, qp_end, n_site, n_elec,
        );
        return;
    }

    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let mut vec_a = vec![0.0; n_size];
    let mut vec_b = vec![0.0; n_size];

    for qp in qp_start..qp_end {
        if qp >= pf_m.len() || qp >= pf_m_new.len() {
            continue;
        }
        let inv_base = qp * inv_stride;
        fill_vecs_fsz_real(
            qp, rsa, rsb, ele_idx, ele_spn, slater_elm, &mut vec_a, &mut vec_b, n_site,
        );
        let ratio = two_ratio_real(msa, msb, inv_m_flat, inv_base, n_size, &vec_a, &vec_b);
        pf_m_new[qp] = ratio * pf_m[qp];
    }
}

/// Complex normal-mode `update_m_all!`.
pub fn update_m_all_complex_flat(
    ma: usize,
    spin: u8,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &mut [Complex64],
    inv_stride: usize,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let mut slt_vec = vec![Complex64::new(0.0, 0.0); n_size];
    let mut vec1 = vec![Complex64::new(0.0, 0.0); n_size];
    let mut vec2 = vec![Complex64::new(0.0, 0.0); n_size];

    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        fill_slt_vec_normal_complex(qp, rsa, ele_idx, slater_elm, &mut slt_vec, n_site, n_elec);
        let base = qp * inv_stride;
        update_one_complex(
            base,
            msa,
            inv_m_flat,
            &slt_vec,
            &mut vec1,
            &mut vec2,
            n_size,
            &mut pf_m[qp],
        );
    }
}

/// Real normal-mode `update_m_all_real!`.
pub fn update_m_all_real_flat(
    ma: usize,
    spin: u8,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &mut [f64],
    inv_stride: usize,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let mut slt_vec = vec![0.0; n_size];
    let mut vec1 = vec![0.0; n_size];
    let mut vec2 = vec![0.0; n_size];

    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        fill_slt_vec_normal_real(qp, rsa, ele_idx, slater_elm, &mut slt_vec, n_site, n_elec);
        let base = qp * inv_stride;
        update_one_real(
            base,
            msa,
            inv_m_flat,
            &slt_vec,
            &mut vec1,
            &mut vec2,
            n_size,
            &mut pf_m[qp],
        );
    }
}

/// Complex FSZ `update_m_all_fsz!`.
pub fn update_m_all_fsz_complex_flat(
    ma: usize,
    spin: u8,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &mut [Complex64],
    inv_stride: usize,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let mut slt_vec = vec![Complex64::new(0.0, 0.0); n_size];
    let mut vec1 = vec![Complex64::new(0.0, 0.0); n_size];
    let mut vec2 = vec![Complex64::new(0.0, 0.0); n_size];
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        fill_slt_vec_fsz_complex(qp, rsa, ele_idx, ele_spn, slater_elm, &mut slt_vec, n_site);
        let base = qp * inv_stride;
        update_one_complex(
            base,
            msa,
            inv_m_flat,
            &slt_vec,
            &mut vec1,
            &mut vec2,
            n_size,
            &mut pf_m[qp],
        );
    }
}

/// Real FSZ `update_m_all_fsz_real!`.
pub fn update_m_all_fsz_real_flat(
    ma: usize,
    spin: u8,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &mut [f64],
    inv_stride: usize,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let mut slt_vec = vec![0.0; n_size];
    let mut vec1 = vec![0.0; n_size];
    let mut vec2 = vec![0.0; n_size];
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        fill_slt_vec_fsz_real(qp, rsa, ele_idx, ele_spn, slater_elm, &mut slt_vec, n_site);
        let base = qp * inv_stride;
        update_one_real(
            base,
            msa,
            inv_m_flat,
            &slt_vec,
            &mut vec1,
            &mut vec2,
            n_size,
            &mut pf_m[qp],
        );
    }
}

/// Complex normal-mode `update_m_all_two!`.
pub fn update_m_all_two_complex_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    ra_old: usize,
    rb_old: usize,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m_flat: &mut [Complex64],
    inv_stride: usize,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let msb = mb + (spin_other as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let rsa_old = ra_old + (spin as usize) * n_site;
    let rsb_old = rb_old + (spin_other as usize) * n_site;
    let mut work = TwoUpdateWorkComplex::new(n_size);
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        let base = qp * inv_stride;
        let m_old_ab = slater_elm.get(qp, rsa_old, rsb_old);
        fill_vecs_normal_complex(
            qp,
            rsa,
            rsb,
            ele_idx,
            slater_elm,
            &mut work.vec_s,
            &mut work.vec_t,
            n_site,
            n_elec,
        );
        work.vec_s[msb] = m_old_ab;
        update_two_complex(base, msa, msb, inv_m_flat, n_size, &mut work, &mut pf_m[qp]);
    }
}

/// Real normal-mode `update_m_all_two_real!`.
///
/// This intentionally preserves the upstream real-path quirk:
/// `rsb_old = ra_old + t*n_site` (not `rb_old + t*n_site`).
pub fn update_m_all_two_real_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    ra_old: usize,
    _rb_old: usize,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &mut [f64],
    inv_stride: usize,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let msb = mb + (spin_other as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let rsa_old = ra_old + (spin as usize) * n_site;
    let rsb_old = ra_old + (spin_other as usize) * n_site;
    let mut work = TwoUpdateWorkReal::new(n_size);
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        let base = qp * inv_stride;
        let m_old_ab = slater_elm.get(qp, rsa_old, rsb_old);
        fill_vecs_normal_real(
            qp,
            rsa,
            rsb,
            ele_idx,
            slater_elm,
            &mut work.vec_s,
            &mut work.vec_t,
            n_site,
            n_elec,
        );
        work.vec_s[msb] = m_old_ab;
        update_two_real(base, msa, msb, inv_m_flat, n_size, &mut work, &mut pf_m[qp]);
    }
}

/// Real FSZ `update_m_all_two_fsz_real!`.
///
/// This intentionally preserves the upstream real-FSZ quirk:
/// `rsb_old = ra_old + t*n_site` (not `rb_old + t*n_site`).
pub fn update_m_all_two_fsz_real_flat(
    ma: usize,
    spin: u8,
    mb: usize,
    spin_other: u8,
    ra_old: usize,
    _rb_old: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &mut [f64],
    inv_stride: usize,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    let msa = ma;
    let msb = mb;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let rsb = ele_idx[msb] as usize + (spin_other as usize) * n_site;
    let rsa_old = ra_old + (spin as usize) * n_site;
    let rsb_old = ra_old + (spin_other as usize) * n_site;
    let mut work = TwoUpdateWorkReal::new(n_size);
    for qp in qp_start..qp_end {
        if qp >= pf_m.len() {
            continue;
        }
        let base = qp * inv_stride;
        let m_old_ab = slater_elm.get(qp, rsa_old, rsb_old);
        fill_vecs_fsz_real(
            qp,
            rsa,
            rsb,
            ele_idx,
            ele_spn,
            slater_elm,
            &mut work.vec_s,
            &mut work.vec_t,
            n_site,
        );
        work.vec_s[msb] = m_old_ab;
        update_two_real(base, msa, msb, inv_m_flat, n_size, &mut work, &mut pf_m[qp]);
    }
}

fn fill_vecs_normal_complex(
    qp: usize,
    rsa: usize,
    rsb: usize,
    ele_idx: &[i64],
    slater: &SlaterElmFlat<Complex64>,
    vec_a: &mut [Complex64],
    vec_b: &mut [Complex64],
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    for msi in 0..n_size {
        let rsi = if msi < n_elec {
            ele_idx[msi] as usize
        } else {
            ele_idx[msi] as usize + n_site
        };
        vec_a[msi] = slater.get(qp, rsa, rsi);
        vec_b[msi] = slater.get(qp, rsb, rsi);
    }
}

fn fill_vecs_normal_real(
    qp: usize,
    rsa: usize,
    rsb: usize,
    ele_idx: &[i64],
    slater: &SlaterElmFlat<f64>,
    vec_a: &mut [f64],
    vec_b: &mut [f64],
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    for msi in 0..n_size {
        let rsi = if msi < n_elec {
            ele_idx[msi] as usize
        } else {
            ele_idx[msi] as usize + n_site
        };
        vec_a[msi] = slater.get(qp, rsa, rsi);
        vec_b[msi] = slater.get(qp, rsb, rsi);
    }
}

fn fill_vecs_fsz_complex(
    qp: usize,
    rsa: usize,
    rsb: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater: &SlaterElmFlat<Complex64>,
    vec_a: &mut [Complex64],
    vec_b: &mut [Complex64],
    n_site: usize,
) {
    for msi in 0..vec_a.len() {
        let rsi = ele_idx[msi] as usize + (ele_spn[msi] as usize) * n_site;
        vec_a[msi] = slater.get(qp, rsa, rsi);
        vec_b[msi] = slater.get(qp, rsb, rsi);
    }
}

fn fill_vecs_fsz_real(
    qp: usize,
    rsa: usize,
    rsb: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater: &SlaterElmFlat<f64>,
    vec_a: &mut [f64],
    vec_b: &mut [f64],
    n_site: usize,
) {
    for msi in 0..vec_a.len() {
        let rsi = ele_idx[msi] as usize + (ele_spn[msi] as usize) * n_site;
        vec_a[msi] = slater.get(qp, rsa, rsi);
        vec_b[msi] = slater.get(qp, rsb, rsi);
    }
}

fn fill_slt_vec_normal_complex(
    qp: usize,
    rsa: usize,
    ele_idx: &[i64],
    slater: &SlaterElmFlat<Complex64>,
    slt_vec: &mut [Complex64],
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    for msj in 0..n_size {
        let rsj = if msj < n_elec {
            ele_idx[msj] as usize
        } else {
            ele_idx[msj] as usize + n_site
        };
        slt_vec[msj] = slater.get(qp, rsa, rsj);
    }
}

fn fill_slt_vec_normal_real(
    qp: usize,
    rsa: usize,
    ele_idx: &[i64],
    slater: &SlaterElmFlat<f64>,
    slt_vec: &mut [f64],
    n_site: usize,
    n_elec: usize,
) {
    let n_size = 2 * n_elec;
    for msj in 0..n_size {
        let rsj = if msj < n_elec {
            ele_idx[msj] as usize
        } else {
            ele_idx[msj] as usize + n_site
        };
        slt_vec[msj] = slater.get(qp, rsa, rsj);
    }
}

fn fill_slt_vec_fsz_complex(
    qp: usize,
    rsa: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater: &SlaterElmFlat<Complex64>,
    slt_vec: &mut [Complex64],
    n_site: usize,
) {
    for msj in 0..slt_vec.len() {
        let rsj = ele_idx[msj] as usize + (ele_spn[msj] as usize) * n_site;
        slt_vec[msj] = slater.get(qp, rsa, rsj);
    }
}

fn fill_slt_vec_fsz_real(
    qp: usize,
    rsa: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater: &SlaterElmFlat<f64>,
    slt_vec: &mut [f64],
    n_site: usize,
) {
    for msj in 0..slt_vec.len() {
        let rsj = ele_idx[msj] as usize + (ele_spn[msj] as usize) * n_site;
        slt_vec[msj] = slater.get(qp, rsa, rsj);
    }
}

fn update_one_complex(
    base: usize,
    msa: usize,
    inv: &mut [Complex64],
    slt_vec: &[Complex64],
    vec1: &mut [Complex64],
    vec2: &mut [Complex64],
    n_size: usize,
    pf: &mut Complex64,
) {
    for i in 0..n_size {
        vec1[i] = Complex64::new(0.0, 0.0);
    }
    for msj in 0..n_size {
        let slt = slt_vec[msj];
        for msi in 0..n_size {
            vec1[msi] += -inv[base + msj * n_size + msi] * slt;
        }
    }
    let tmp = vec1[msa];
    *pf *= -tmp;
    let inv_vec1_a = -Complex64::new(1.0, 0.0) / tmp;
    for msi in 0..n_size {
        vec2[msi] = inv[base + msa * n_size + msi] * inv_vec1_a;
    }
    for msi in 0..n_size {
        let vec1_i = vec1[msi];
        let vec2_i = vec2[msi];
        for msj in 0..n_size {
            inv[base + msi * n_size + msj] += vec1_i * vec2[msj] - vec1[msj] * vec2_i;
        }
        inv[base + msi * n_size + msa] -= vec2_i;
    }
    for msj in 0..n_size {
        inv[base + msa * n_size + msj] += vec2[msj];
    }
}

fn update_one_real(
    base: usize,
    msa: usize,
    inv: &mut [f64],
    slt_vec: &[f64],
    vec1: &mut [f64],
    vec2: &mut [f64],
    n_size: usize,
    pf: &mut f64,
) {
    for i in 0..n_size {
        vec1[i] = 0.0;
    }
    for msj in 0..n_size {
        let slt = slt_vec[msj];
        for msi in 0..n_size {
            vec1[msi] += -inv[base + msj * n_size + msi] * slt;
        }
    }
    let tmp = vec1[msa];
    *pf *= -tmp;
    let inv_vec1_a = -1.0 / tmp;
    for msi in 0..n_size {
        vec2[msi] = inv[base + msa * n_size + msi] * inv_vec1_a;
    }
    for msi in 0..n_size {
        let vec1_i = vec1[msi];
        let vec2_i = vec2[msi];
        for msj in 0..n_size {
            inv[base + msi * n_size + msj] += vec1_i * vec2[msj] - vec1[msj] * vec2_i;
        }
        inv[base + msi * n_size + msa] -= vec2_i;
    }
    for msj in 0..n_size {
        inv[base + msa * n_size + msj] += vec2[msj];
    }
}

struct TwoUpdateWorkComplex {
    vec_p: Vec<Complex64>,
    vec_q: Vec<Complex64>,
    vec_s: Vec<Complex64>,
    vec_t: Vec<Complex64>,
}

impl TwoUpdateWorkComplex {
    fn new(n_size: usize) -> Self {
        Self {
            vec_p: vec![Complex64::new(0.0, 0.0); n_size],
            vec_q: vec![Complex64::new(0.0, 0.0); n_size],
            vec_s: vec![Complex64::new(0.0, 0.0); n_size],
            vec_t: vec![Complex64::new(0.0, 0.0); n_size],
        }
    }
}

struct TwoUpdateWorkReal {
    vec_p: Vec<f64>,
    vec_q: Vec<f64>,
    vec_s: Vec<f64>,
    vec_t: Vec<f64>,
}

impl TwoUpdateWorkReal {
    fn new(n_size: usize) -> Self {
        Self {
            vec_p: vec![0.0; n_size],
            vec_q: vec![0.0; n_size],
            vec_s: vec![0.0; n_size],
            vec_t: vec![0.0; n_size],
        }
    }
}

fn update_two_complex(
    base: usize,
    msa: usize,
    msb: usize,
    inv: &mut [Complex64],
    n_size: usize,
    work: &mut TwoUpdateWorkComplex,
    pf: &mut Complex64,
) {
    for i in 0..n_size {
        work.vec_p[i] = Complex64::new(0.0, 0.0);
        work.vec_q[i] = Complex64::new(0.0, 0.0);
    }
    for i in 0..n_size {
        for j in 0..n_size {
            let inv_ij = inv[base + i * n_size + j];
            work.vec_p[i] += inv_ij * work.vec_s[j];
            work.vec_q[i] += inv_ij * work.vec_t[j];
        }
    }

    let mut bma = Complex64::new(0.0, 0.0);
    for i in 0..n_size {
        bma += work.vec_t[i] * work.vec_p[i];
    }
    let inv_ab = inv[base + msa * n_size + msb];
    let ratio = inv_ab * work.vec_t[msa] + inv_ab * bma + work.vec_p[msa] * work.vec_q[msb]
        - work.vec_p[msb] * work.vec_q[msa];
    *pf *= ratio;

    let a = -work.vec_p[msa];
    let b = work.vec_p[msb];
    let c = work.vec_q[msa];
    let d = -work.vec_q[msb];
    let e = -bma - work.vec_t[msa];
    let f = inv_ab;
    let det = a * d - b * c - e * f;
    let inv_det = Complex64::new(1.0, 0.0) / det;

    for i in 0..n_size {
        work.vec_s[i] = inv_det * inv[base + msa * n_size + i];
        work.vec_t[i] = inv_det * inv[base + msb * n_size + i];
    }

    for i in 0..n_size {
        let p_i = work.vec_p[i];
        let q_i = work.vec_q[i];
        let s_i = work.vec_s[i];
        let t_i = work.vec_t[i];
        for j in 0..n_size {
            let p_j = work.vec_p[j];
            let q_j = work.vec_q[j];
            let s_j = work.vec_s[j];
            let t_j = work.vec_t[j];
            inv[base + i * n_size + j] += a * (q_i * t_j - q_j * t_i)
                + b * (q_i * s_j - q_j * s_i)
                + c * (p_i * t_j - p_j * t_i)
                + d * (p_i * s_j - p_j * s_i)
                + e * det * (s_i * t_j - s_j * t_i)
                + f * inv_det * (p_i * q_j - q_i * p_j);
        }
        inv[base + i * n_size + msa] += -c * t_i - d * s_i - f * inv_det * q_i;
        inv[base + i * n_size + msb] += -a * t_i - b * s_i + f * inv_det * p_i;
    }
    for j in 0..n_size {
        let p_j = work.vec_p[j];
        let q_j = work.vec_q[j];
        let s_j = work.vec_s[j];
        let t_j = work.vec_t[j];
        inv[base + msa * n_size + j] += c * t_j + d * s_j + f * inv_det * q_j;
        inv[base + msb * n_size + j] += a * t_j + b * s_j - f * inv_det * p_j;
    }
    inv[base + msa * n_size + msb] += f * inv_det;
    inv[base + msb * n_size + msa] -= f * inv_det;
}

fn update_two_real(
    base: usize,
    msa: usize,
    msb: usize,
    inv: &mut [f64],
    n_size: usize,
    work: &mut TwoUpdateWorkReal,
    pf: &mut f64,
) {
    for i in 0..n_size {
        work.vec_p[i] = 0.0;
        work.vec_q[i] = 0.0;
    }
    for i in 0..n_size {
        for j in 0..n_size {
            let inv_ij = inv[base + i * n_size + j];
            work.vec_p[i] += inv_ij * work.vec_s[j];
            work.vec_q[i] += inv_ij * work.vec_t[j];
        }
    }

    let mut bma = 0.0;
    for i in 0..n_size {
        bma += work.vec_t[i] * work.vec_p[i];
    }
    let inv_ab = inv[base + msa * n_size + msb];
    let ratio = inv_ab * work.vec_t[msa] + inv_ab * bma + work.vec_p[msa] * work.vec_q[msb]
        - work.vec_p[msb] * work.vec_q[msa];
    *pf *= ratio;

    let a = -work.vec_p[msa];
    let b = work.vec_p[msb];
    let c = work.vec_q[msa];
    let d = -work.vec_q[msb];
    let e = -bma - work.vec_t[msa];
    let f = inv_ab;
    let det = a * d - b * c - e * f;
    let inv_det = 1.0 / det;

    for i in 0..n_size {
        work.vec_s[i] = inv_det * inv[base + msa * n_size + i];
        work.vec_t[i] = inv_det * inv[base + msb * n_size + i];
    }

    for i in 0..n_size {
        let p_i = work.vec_p[i];
        let q_i = work.vec_q[i];
        let s_i = work.vec_s[i];
        let t_i = work.vec_t[i];
        for j in 0..n_size {
            let p_j = work.vec_p[j];
            let q_j = work.vec_q[j];
            let s_j = work.vec_s[j];
            let t_j = work.vec_t[j];
            inv[base + i * n_size + j] += a * (q_i * t_j - q_j * t_i)
                + b * (q_i * s_j - q_j * s_i)
                + c * (p_i * t_j - p_j * t_i)
                + d * (p_i * s_j - p_j * s_i)
                + e * det * (s_i * t_j - s_j * t_i)
                + f * inv_det * (p_i * q_j - q_i * p_j);
        }
        inv[base + i * n_size + msa] += -c * t_i - d * s_i - f * inv_det * q_i;
        inv[base + i * n_size + msb] += -a * t_i - b * s_i + f * inv_det * p_i;
    }
    for j in 0..n_size {
        let p_j = work.vec_p[j];
        let q_j = work.vec_q[j];
        let s_j = work.vec_s[j];
        let t_j = work.vec_t[j];
        inv[base + msa * n_size + j] += c * t_j + d * s_j + f * inv_det * q_j;
        inv[base + msb * n_size + j] += a * t_j + b * s_j - f * inv_det * p_j;
    }
    inv[base + msa * n_size + msb] += f * inv_det;
    inv[base + msb * n_size + msa] -= f * inv_det;
}

fn two_ratio_complex(
    msa: usize,
    msb: usize,
    inv: &[Complex64],
    inv_base: usize,
    n_size: usize,
    vec_a: &[Complex64],
    vec_b: &[Complex64],
) -> Complex64 {
    let vec_ba = vec_b[msa];
    let mut p_a = Complex64::new(0.0, 0.0);
    let mut p_b = Complex64::new(0.0, 0.0);
    let mut q_a = Complex64::new(0.0, 0.0);
    let mut q_b = Complex64::new(0.0, 0.0);
    for msi in 0..n_size {
        let inv_a = inv[inv_base + msa * n_size + msi];
        let inv_b = inv[inv_base + msb * n_size + msi];
        p_a += inv_a * vec_a[msi];
        p_b += inv_b * vec_a[msi];
        q_a += inv_a * vec_b[msi];
        q_b += inv_b * vec_b[msi];
    }
    let inv_ab = inv[inv_base + msa * n_size + msb];
    let mut bma = Complex64::new(0.0, 0.0);
    for msi in 0..n_size {
        let mut tmp = Complex64::new(0.0, 0.0);
        for msj in 0..n_size {
            tmp += inv[inv_base + msi * n_size + msj] * vec_a[msj];
        }
        bma += vec_b[msi] * tmp;
    }
    inv_ab * vec_ba + inv_ab * bma + p_a * q_b - p_b * q_a
}

fn two_ratio_real(
    msa: usize,
    msb: usize,
    inv: &[f64],
    inv_base: usize,
    n_size: usize,
    vec_a: &[f64],
    vec_b: &[f64],
) -> f64 {
    let vec_ba = vec_b[msa];
    let mut p_a = 0.0;
    let mut p_b = 0.0;
    let mut q_a = 0.0;
    let mut q_b = 0.0;
    for msi in 0..n_size {
        let inv_a = inv[inv_base + msa * n_size + msi];
        let inv_b = inv[inv_base + msb * n_size + msi];
        p_a += inv_a * vec_a[msi];
        p_b += inv_b * vec_a[msi];
        q_a += inv_a * vec_b[msi];
        q_b += inv_b * vec_b[msi];
    }
    let inv_ab = inv[inv_base + msa * n_size + msb];
    let mut bma = 0.0;
    for msi in 0..n_size {
        let mut tmp = 0.0;
        for msj in 0..n_size {
            tmp += inv[inv_base + msi * n_size + msj] * vec_a[msj];
        }
        bma += vec_b[msi] * tmp;
    }
    inv_ab * vec_ba + inv_ab * bma + p_a * q_b - p_b * q_a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_normal_smoke() {
        let n_site = 2;
        let n_elec = 1;
        let n_qp = 1;
        let n_size = 2;
        let ele_idx = vec![0, 1];
        let mut slater = SlaterElmFlat::<f64>::zeros(n_qp, n_site);
        slater.set(0, 0, 0, 1.0);
        slater.set(0, 0, 3, 2.0);
        let inv = vec![3.0, 5.0, 7.0, 11.0];
        let pf = vec![13.0];
        let mut out = vec![0.0];
        calculate_new_pf_m2_real_flat(
            0,
            0,
            &mut out,
            &ele_idx,
            &slater,
            &inv,
            n_size * n_size,
            &pf,
            0,
            1,
            n_site,
            n_elec,
        );
        // msa=0, rsa=0, rsj=(0,3): ratio=3*1 + 5*2 = 13; out=-13*13
        assert_eq!(out[0], -169.0);
    }
}
