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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 8 * n_size),
        || (),
        |_, qp| {
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
            -ratio * pf_m[qp]
        },
    );
}

/// Real normal-mode `calculate_new_pf_m2_real!`.
///
/// The `n_size`-term sum of each QP is a serial floating-point chain (its order is part
/// of the C/Julia contract), so four QPs are advanced together to overlap the chains'
/// add latency; every chain still adds the same products in the same order.
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
    const BLOCK: usize = 4;
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let rsa = ele_idx[msa] as usize + (spin as usize) * n_site;
    let end = qp_end.min(pf_m.len());
    // Column of the Slater row for every electron slot (same for all QPs).
    let mut rsj_stack = [0usize; 128];
    let rsj_heap;
    let rsj: &[usize] = if n_size <= rsj_stack.len() {
        for (msj, slot) in rsj_stack[..n_size].iter_mut().enumerate() {
            *slot = ele_idx[msj] as usize + if msj < n_elec { 0 } else { n_site };
        }
        &rsj_stack[..n_size]
    } else {
        rsj_heap = (0..n_size)
            .map(|msj| ele_idx[msj] as usize + if msj < n_elec { 0 } else { n_site })
            .collect::<Vec<_>>();
        &rsj_heap
    };
    let n_site2 = slater_elm.n_site2();
    let slater = slater_elm.as_slice();
    let row_of = |qp: usize| {
        let start = (qp * n_site2 + rsa) * n_site2;
        &slater[start..start + n_site2]
    };
    let inv_of = |qp: usize| {
        let start = qp * inv_stride + msa * n_size;
        &inv_m_flat[start..start + n_size]
    };
    crate::threading::qp_fill_blocks(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 4 * n_size),
        BLOCK,
        |first, out| {
            if out.len() == BLOCK {
                let rows = [
                    row_of(first),
                    row_of(first + 1),
                    row_of(first + 2),
                    row_of(first + 3),
                ];
                let invs = [
                    inv_of(first),
                    inv_of(first + 1),
                    inv_of(first + 2),
                    inv_of(first + 3),
                ];
                let mut ratio = [0.0_f64; BLOCK];
                for (msj, &col) in rsj.iter().enumerate() {
                    for k in 0..BLOCK {
                        ratio[k] += invs[k][msj] * rows[k][col];
                    }
                }
                for k in 0..BLOCK {
                    out[k] = -ratio[k] * pf_m[first + k];
                }
            } else {
                for (k, slot) in out.iter_mut().enumerate() {
                    let qp = first + k;
                    let (row, inv) = (row_of(qp), inv_of(qp));
                    let mut ratio = 0.0;
                    for (msj, &col) in rsj.iter().enumerate() {
                        ratio += inv[msj] * row[col];
                    }
                    *slot = -ratio * pf_m[qp];
                }
            }
        },
    );
}

/// Fused serial `calculate_new_pf_m2_real!` + `calculate_ip_real!` for one moved electron
/// (the CalHamiltonian1 transfer terms): returns `sum_qp w_qp * new_pf_qp`, adding the
/// terms in QP order exactly as the separate kernels do, without the `new_pf` buffer.
///
/// `moved = (msj, site)` is the electron slot whose site becomes `site` (the unmoved
/// `ele_idx` is not copied); `ma`/`spin` select the same slot as in
/// [`calculate_new_pf_m2_real_flat`]. QPs beyond `qp_weights` contribute nothing, as in
/// `calculate_ip_real`.
pub fn calculate_new_pf_m2_ip_real_flat(
    ma: usize,
    spin: u8,
    moved: (usize, i64),
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m_flat: &[f64],
    inv_stride: usize,
    pf_m: &[f64],
    qp_weights: &[Complex64],
    n_site: usize,
    n_elec: usize,
) -> f64 {
    const BLOCK: usize = 4;
    let n_size = 2 * n_elec;
    let msa = ma + (spin as usize) * n_elec;
    let site_of = |msj: usize| {
        if msj == moved.0 {
            moved.1
        } else {
            ele_idx[msj]
        }
    };
    let rsa = site_of(msa) as usize + (spin as usize) * n_site;
    let mut rsj_stack = [0usize; 128];
    let rsj_heap;
    let rsj: &[usize] = if n_size <= rsj_stack.len() {
        for (msj, slot) in rsj_stack[..n_size].iter_mut().enumerate() {
            *slot = site_of(msj) as usize + if msj < n_elec { 0 } else { n_site };
        }
        &rsj_stack[..n_size]
    } else {
        rsj_heap = (0..n_size)
            .map(|msj| site_of(msj) as usize + if msj < n_elec { 0 } else { n_site })
            .collect::<Vec<_>>();
        &rsj_heap
    };
    let n_site2 = slater_elm.n_site2();
    let slater = slater_elm.as_slice();
    let row_of = |qp: usize| {
        let start = (qp * n_site2 + rsa) * n_site2;
        &slater[start..start + n_site2]
    };
    let inv_of = |qp: usize| {
        let start = qp * inv_stride + msa * n_size;
        &inv_m_flat[start..start + n_size]
    };
    let n_qp = pf_m.len();
    let mut ip = 0.0;
    let mut first = 0;
    while first < n_qp {
        let mut new_pf = [0.0_f64; BLOCK];
        let count = (n_qp - first).min(BLOCK);
        if count == BLOCK {
            let rows = [
                row_of(first),
                row_of(first + 1),
                row_of(first + 2),
                row_of(first + 3),
            ];
            let invs = [
                inv_of(first),
                inv_of(first + 1),
                inv_of(first + 2),
                inv_of(first + 3),
            ];
            let mut ratio = [0.0_f64; BLOCK];
            for (msj, &col) in rsj.iter().enumerate() {
                for k in 0..BLOCK {
                    ratio[k] += invs[k][msj] * rows[k][col];
                }
            }
            for k in 0..BLOCK {
                new_pf[k] = -ratio[k] * pf_m[first + k];
            }
        } else {
            for (k, slot) in new_pf[..count].iter_mut().enumerate() {
                let qp = first + k;
                let (row, inv) = (row_of(qp), inv_of(qp));
                let mut ratio = 0.0;
                for (msj, &col) in rsj.iter().enumerate() {
                    ratio += inv[msj] * row[col];
                }
                *slot = -ratio * pf_m[qp];
            }
        }
        for (k, &value) in new_pf[..count].iter().enumerate() {
            if let Some(weight) = qp_weights.get(first + k) {
                ip += weight.re * value;
            }
        }
        first += count;
    }
    ip
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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 8 * n_size),
        || (),
        |_, qp| {
            let inv_base = qp * inv_stride + msa * n_size;
            let mut ratio = Complex64::new(0.0, 0.0);
            for msj in 0..n_size {
                let rsj = ele_idx[msj] as usize + (ele_spn[msj] as usize) * n_site;
                ratio += inv_m_flat[inv_base + msj] * slater_elm.get(qp, rsa, rsj);
            }
            -ratio * pf_m[qp]
        },
    );
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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 4 * n_size),
        || (),
        |_, qp| {
            let inv_base = qp * inv_stride + msa * n_size;
            let mut ratio = 0.0;
            for msj in 0..n_size {
                let rsj = ele_idx[msj] as usize + (ele_spn[msj] as usize) * n_site;
                ratio += inv_m_flat[inv_base + msj] * slater_elm.get(qp, rsa, rsj);
            }
            -ratio * pf_m[qp]
        },
    );
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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 16 * n_size),
        || {
            (
                vec![Complex64::new(0.0, 0.0); n_size],
                vec![Complex64::new(0.0, 0.0); n_size],
            )
        },
        |(vec_a, vec_b), qp| {
            let inv_base = qp * inv_stride;
            fill_vecs_normal_complex(
                qp, rsa, rsb, ele_idx, slater_elm, vec_a, vec_b, n_site, n_elec,
            );
            let ratio = two_ratio_complex(msa, msb, inv_m_flat, inv_base, n_size, vec_a, vec_b);
            ratio * pf_m[qp]
        },
    );
}

/// Real normal-mode two-electron Pfaffian update.
/// `SCALAR` selects C's nested sum; otherwise preserve Julia's vector reduction.
pub fn calculate_new_pf_m_two2_real_flat<const SCALAR: bool>(
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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 8 * n_size),
        || (vec![0.0; n_size], vec![0.0; n_size]),
        |(vec_a, vec_b), qp| {
            let inv_base = qp * inv_stride;
            fill_vecs_normal_real(
                qp, rsa, rsb, ele_idx, slater_elm, vec_a, vec_b, n_site, n_elec,
            );
            let ratio =
                two_ratio_real::<SCALAR>(msa, msb, inv_m_flat, inv_base, n_size, vec_a, vec_b);
            ratio * pf_m[qp]
        },
    );
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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 16 * n_size),
        || {
            (
                vec![Complex64::new(0.0, 0.0); n_size],
                vec![Complex64::new(0.0, 0.0); n_size],
            )
        },
        |(vec_a, vec_b), qp| {
            let inv_base = qp * inv_stride;
            fill_vecs_fsz_complex(
                qp, rsa, rsb, ele_idx, ele_spn, slater_elm, vec_a, vec_b, n_site,
            );
            let ratio = two_ratio_complex(msa, msb, inv_m_flat, inv_base, n_size, vec_a, vec_b);
            ratio * pf_m[qp]
        },
    );
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
    let end = qp_end.min(pf_m.len());
    crate::threading::qp_fill(
        pf_m_new,
        qp_start,
        end,
        crate::threading::scaled_cost_ns(n_size, 8 * n_size),
        || (vec![0.0; n_size], vec![0.0; n_size]),
        |(vec_a, vec_b), qp| {
            let inv_base = qp * inv_stride;
            fill_vecs_fsz_real(
                qp, rsa, rsb, ele_idx, ele_spn, slater_elm, vec_a, vec_b, n_site,
            );
            let ratio =
                two_ratio_real::<true>(msa, msb, inv_m_flat, inv_base, n_size, vec_a, vec_b);
            ratio * pf_m[qp]
        },
    );
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, 2 * (n_size * n_size + 16 * n_size)),
        || {
            (
                vec![Complex64::new(0.0, 0.0); n_size],
                vec![Complex64::new(0.0, 0.0); n_size],
                vec![Complex64::new(0.0, 0.0); n_size],
            )
        },
        |(slt_vec, vec1, vec2), qp, pf, window| {
            fill_slt_vec_normal_complex(qp, rsa, ele_idx, slater_elm, slt_vec, n_site, n_elec);
            let base = 0;
            update_one_complex(base, msa, window, slt_vec, vec1, vec2, n_size, pf);
        },
    );
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, n_size * n_size + 16 * n_size),
        || (vec![0.0; n_size], vec![0.0; n_size], vec![0.0; n_size]),
        |(slt_vec, vec1, vec2), qp, pf, window| {
            fill_slt_vec_normal_real(qp, rsa, ele_idx, slater_elm, slt_vec, n_site, n_elec);
            let base = 0;
            update_one_real(base, msa, window, slt_vec, vec1, vec2, n_size, pf);
        },
    );
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, 2 * (n_size * n_size + 16 * n_size)),
        || {
            (
                vec![Complex64::new(0.0, 0.0); n_size],
                vec![Complex64::new(0.0, 0.0); n_size],
                vec![Complex64::new(0.0, 0.0); n_size],
            )
        },
        |(slt_vec, vec1, vec2), qp, pf, window| {
            fill_slt_vec_fsz_complex(qp, rsa, ele_idx, ele_spn, slater_elm, slt_vec, n_site);
            let base = 0;
            update_one_complex(base, msa, window, slt_vec, vec1, vec2, n_size, pf);
        },
    );
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, n_size * n_size + 16 * n_size),
        || (vec![0.0; n_size], vec![0.0; n_size], vec![0.0; n_size]),
        |(slt_vec, vec1, vec2), qp, pf, window| {
            fill_slt_vec_fsz_real(qp, rsa, ele_idx, ele_spn, slater_elm, slt_vec, n_site);
            let base = 0;
            update_one_real(base, msa, window, slt_vec, vec1, vec2, n_size, pf);
        },
    );
}

/// Complex normal-mode `update_m_all_two!`.
///
/// C (`pfupdate_two_fcmp.c:227`) defines `rsbOld = raOld + t*Nsite`, whereas this function uses
/// `rb_old + t*n_site`. The old `(a,b)` element only enters through `vec_s[msb]`, and the
/// updated Pfaffian and inverse do not depend on it beyond roundoff, so the difference is
/// unobservable; see `tests/two_electron_rsbold.rs` for the equivalence check (#365).
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, 4 * (n_size * n_size + 16 * n_size)),
        || TwoUpdateWorkComplex::new(n_size),
        |work, qp, pf, window| {
            let base = 0;
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
            update_two_complex(base, msa, msb, window, n_size, work, pf);
        },
    );
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, 2 * (n_size * n_size + 16 * n_size)),
        || TwoUpdateWorkReal::new(n_size),
        |work, qp, pf, window| {
            let base = 0;
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
            update_two_real(base, msa, msb, window, n_size, work, pf);
        },
    );
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
    crate::threading::qp_update(
        pf_m,
        inv_m_flat,
        inv_stride,
        n_size * n_size,
        qp_start,
        qp_end,
        crate::threading::scaled_cost_ns(n_size, 2 * (n_size * n_size + 16 * n_size)),
        || TwoUpdateWorkReal::new(n_size),
        |work, qp, pf, window| {
            let base = 0;
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
            update_two_real(base, msa, msb, window, n_size, work, pf);
        },
    );
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
    let n_site2 = slater.n_site2();
    let start = (qp * n_site2 + rsa) * n_site2;
    let row = &slater.as_slice()[start..start + n_site2];
    for (msj, slot) in slt_vec[..n_size].iter_mut().enumerate() {
        let rsj = if msj < n_elec {
            ele_idx[msj] as usize
        } else {
            ele_idx[msj] as usize + n_site
        };
        *slot = row[rsj];
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
    let inv_vec1_a = crate::julia_complex::divide(-Complex64::new(1.0, 0.0), tmp);
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
    // Slice-based loops (same operations and order as the indexed C loops) let the
    // compiler drop bounds checks and vectorize the independent elements.
    let inv = &mut inv[base..base + n_size * n_size];
    let (vec1, vec2) = (&mut vec1[..n_size], &mut vec2[..n_size]);
    vec1.fill(0.0);
    for (column, &slt) in inv.chunks_exact(n_size).zip(slt_vec) {
        for (v, &x) in vec1.iter_mut().zip(column) {
            *v += -x * slt;
        }
    }
    let tmp = vec1[msa];
    *pf *= -tmp;
    let inv_vec1_a = -1.0 / tmp;
    for (v2, &x) in vec2.iter_mut().zip(&inv[msa * n_size..(msa + 1) * n_size]) {
        *v2 = x * inv_vec1_a;
    }
    for (msi, row) in inv.chunks_exact_mut(n_size).enumerate() {
        let vec1_i = vec1[msi];
        let vec2_i = vec2[msi];
        for ((x, &vec1_j), &vec2_j) in row.iter_mut().zip(vec1.iter()).zip(vec2.iter()) {
            *x += vec1_i * vec2_j - vec1_j * vec2_i;
        }
        row[msa] -= vec2_i;
    }
    for (x, &v2) in inv[msa * n_size..(msa + 1) * n_size]
        .iter_mut()
        .zip(vec2.iter())
    {
        *x += v2;
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

fn two_ratio_real<const FSZ: bool>(
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
    let bma = if FSZ {
        two_hop_bilinear_fsz_real(inv, inv_base, n_size, vec_a, vec_b)
    } else {
        two_hop_bilinear_real(inv, inv_base, n_size, vec_a, vec_b)
    };
    inv_ab * vec_ba + inv_ab * bma + p_a * q_b - p_b * q_a
}

/// Two-hop bilinear form `b^T M a` in the operation order of the authoritative C kernel
/// `calculateNewPfMTwo_child_real` (`extern/mVMC-1.3.0/src/mVMC/pfupdate_two_real.c:167-177`):
///
/// ```text
/// for i: tmp = 0; for j: tmp += M[i][j] * a[j];   bMa += b[i] * tmp;
/// ```
///
/// Every row is a sequential left-to-right sum of separately rounded products and the rows are
/// accumulated in order: no fused multiply-add and no lane-split partial sums (C built for the
/// default x86-64 target contracts nothing; only an FMA-enabled C build such as
/// `-march=native` would differ, see `c_toolbox/two_hop_bilinear_449/README.md`). Rust never
/// contracts `a * b + c`, so this is bit-identical to the C order on every platform.
///
/// Speed: the inner sum of one row is a serial dependency chain, so four rows are evaluated
/// side by side (four independent chains, each still sequential in `j`), and `bMa` is then
/// accumulated over the rows in their original order. The association of every result is
/// unchanged, hence the result is bit-identical to the plain nested loop (tested).
///
/// Julia's `@turbo` version used four inner lanes and six outer accumulators with FMA
/// (issue #449; the archived Julia values remain a historical comparison with a reordering
/// bound only).
fn two_hop_bilinear_c_order(inv: &[f64], base: usize, n: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut b_ma = 0.0;
    let mut i = 0;
    while i + 4 <= n {
        let r0 = &inv[base + i * n..base + i * n + n];
        let r1 = &inv[base + (i + 1) * n..base + (i + 1) * n + n];
        let r2 = &inv[base + (i + 2) * n..base + (i + 2) * n + n];
        let r3 = &inv[base + (i + 3) * n..base + (i + 3) * n + n];
        let (mut t0, mut t1, mut t2, mut t3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for (j, &aj) in a[..n].iter().enumerate() {
            t0 += r0[j] * aj;
            t1 += r1[j] * aj;
            t2 += r2[j] * aj;
            t3 += r3[j] * aj;
        }
        b_ma += b[i] * t0;
        b_ma += b[i + 1] * t1;
        b_ma += b[i + 2] * t2;
        b_ma += b[i + 3] * t3;
        i += 4;
    }
    while i < n {
        let row = &inv[base + i * n..base + i * n + n];
        let mut tmp = 0.0;
        for (j, &aj) in a[..n].iter().enumerate() {
            tmp += row[j] * aj;
        }
        b_ma += b[i] * tmp;
        i += 1;
    }
    b_ma
}

fn two_hop_bilinear_real(inv: &[f64], base: usize, n: usize, a: &[f64], b: &[f64]) -> f64 {
    two_hop_bilinear_c_order(inv, base, n, a, b)
}

fn two_hop_bilinear_fsz_real(inv: &[f64], base: usize, n: usize, a: &[f64], b: &[f64]) -> f64 {
    two_hop_bilinear_c_order(inv, base, n, a, b)
}

#[cfg(test)]
mod tests {
    /// Straight indexed transcription of the pre-#207 kernels (the C loop order); the
    /// slice-based and QP-blocked versions must reproduce them bit for bit.
    fn reference_update_one_real(
        msa: usize,
        inv: &mut [f64],
        slt_vec: &[f64],
        n_size: usize,
        pf: &mut f64,
    ) {
        let mut vec1 = vec![0.0; n_size];
        let mut vec2 = vec![0.0; n_size];
        for msj in 0..n_size {
            let slt = slt_vec[msj];
            for msi in 0..n_size {
                vec1[msi] += -inv[msj * n_size + msi] * slt;
            }
        }
        let tmp = vec1[msa];
        *pf *= -tmp;
        let inv_vec1_a = -1.0 / tmp;
        for msi in 0..n_size {
            vec2[msi] = inv[msa * n_size + msi] * inv_vec1_a;
        }
        for msi in 0..n_size {
            let vec1_i = vec1[msi];
            let vec2_i = vec2[msi];
            for msj in 0..n_size {
                inv[msi * n_size + msj] += vec1_i * vec2[msj] - vec1[msj] * vec2_i;
            }
            inv[msi * n_size + msa] -= vec2_i;
        }
        for msj in 0..n_size {
            inv[msa * n_size + msj] += vec2[msj];
        }
    }

    fn pseudo_random(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    #[test]
    fn slice_based_rank_one_update_matches_indexed_reference_bitwise() {
        let mut seed = 7;
        for n_size in [2, 6, 16] {
            for msa in [0, n_size / 2, n_size - 1] {
                let base: Vec<f64> = (0..n_size * n_size)
                    .map(|_| pseudo_random(&mut seed))
                    .collect();
                let slt: Vec<f64> = (0..n_size).map(|_| pseudo_random(&mut seed)).collect();
                let (mut expected, mut pf_expected) = (base.clone(), 1.25);
                reference_update_one_real(msa, &mut expected, &slt, n_size, &mut pf_expected);
                let (mut actual, mut pf_actual) = (base.clone(), 1.25);
                let (mut vec1, mut vec2) = (vec![0.0; n_size], vec![0.0; n_size]);
                update_one_real(
                    0,
                    msa,
                    &mut actual,
                    &slt,
                    &mut vec1,
                    &mut vec2,
                    n_size,
                    &mut pf_actual,
                );
                assert_eq!(pf_actual.to_bits(), pf_expected.to_bits());
                assert!(actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }

    #[test]
    fn fused_new_pf_m2_ip_matches_separate_kernels_bitwise() {
        let (n_site, n_elec) = (5, 2);
        let n_size = 2 * n_elec;
        let mut seed = 23;
        for n_qp in [1usize, 3, 4, 5, 8, 11] {
            let mut slater = SlaterElmFlat::<f64>::zeros(n_qp, n_site);
            for value in slater.as_mut_slice() {
                *value = pseudo_random(&mut seed);
            }
            let inv_stride = n_size * n_size + 1;
            let inv: Vec<f64> = (0..n_qp * inv_stride)
                .map(|_| pseudo_random(&mut seed))
                .collect();
            let pf: Vec<f64> = (0..n_qp).map(|_| pseudo_random(&mut seed)).collect();
            // One weight fewer than QPs once, to cover the `calculate_ip_real` guard.
            for n_weights in [n_qp, n_qp.saturating_sub(1)] {
                let weights: Vec<Complex64> = (0..n_weights)
                    .map(|_| Complex64::new(pseudo_random(&mut seed), 0.0))
                    .collect();
                let ele_idx = [4_i64, 0, 3, 1];
                for (ma, spin, new_site) in [(0usize, 0u8, 2_i64), (1, 1, 4), (0, 1, 0)] {
                    let msj = ma + spin as usize * n_elec;
                    let mut moved_idx = ele_idx;
                    moved_idx[msj] = new_site;
                    let mut new_pf = vec![0.0; n_qp];
                    calculate_new_pf_m2_real_flat(
                        ma,
                        spin,
                        &mut new_pf,
                        &moved_idx,
                        &slater,
                        &inv,
                        inv_stride,
                        &pf,
                        0,
                        n_qp,
                        n_site,
                        n_elec,
                    );
                    let mut expected = 0.0;
                    for (qp, value) in new_pf.iter().enumerate() {
                        if qp < weights.len() {
                            expected += weights[qp].re * value;
                        }
                    }
                    let actual = calculate_new_pf_m2_ip_real_flat(
                        ma,
                        spin,
                        (msj, new_site),
                        &ele_idx,
                        &slater,
                        &inv,
                        inv_stride,
                        &pf,
                        &weights,
                        n_site,
                        n_elec,
                    );
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "n_qp {n_qp}, weights {n_weights}"
                    );
                }
            }
        }
    }

    #[test]
    fn blocked_new_pf_m2_matches_per_qp_reference_for_every_qp_count() {
        let (n_site, n_elec) = (4, 2);
        let n_size = 2 * n_elec;
        let n_site2 = 2 * n_site;
        let mut seed = 11;
        for n_qp in [1usize, 3, 4, 5, 8, 9] {
            let mut slater = SlaterElmFlat::<f64>::zeros(n_qp, n_site);
            for value in slater.as_mut_slice() {
                *value = pseudo_random(&mut seed);
            }
            let inv_stride = n_size * n_size + 1;
            let inv: Vec<f64> = (0..n_qp * inv_stride)
                .map(|_| pseudo_random(&mut seed))
                .collect();
            let pf: Vec<f64> = (0..n_qp).map(|_| pseudo_random(&mut seed)).collect();
            let ele_idx = [2_i64, 0, 3, 1];
            for (ma, spin) in [(0usize, 0u8), (1, 1)] {
                let mut actual = vec![0.0; n_qp];
                calculate_new_pf_m2_real_flat(
                    ma,
                    spin,
                    &mut actual,
                    &ele_idx,
                    &slater,
                    &inv,
                    inv_stride,
                    &pf,
                    0,
                    n_qp,
                    n_site,
                    n_elec,
                );
                let msa = ma + spin as usize * n_elec;
                let rsa = ele_idx[msa] as usize + spin as usize * n_site;
                for qp in 0..n_qp {
                    let mut ratio = 0.0;
                    for msj in 0..n_size {
                        let rsj = ele_idx[msj] as usize + if msj < n_elec { 0 } else { n_site };
                        ratio +=
                            inv[qp * inv_stride + msa * n_size + msj] * slater.get(qp, rsa, rsj);
                    }
                    assert_eq!(
                        actual[qp].to_bits(),
                        (-ratio * pf[qp]).to_bits(),
                        "qp {qp}/{n_qp}"
                    );
                }
                assert!(n_site2 > rsa);
            }
        }
    }

    use super::*;

    #[test]
    fn fsz_bilinear_matches_archived_julia_scalar_reduction_values() {
        let inputs = include_str!("../../../../tests/fixtures/pfaffian_cg/two_hop_bilinear.txt");
        let expected = include_str!("../../../../tests/fixtures/real_fsz/bilinear.txt");
        let mut lines = inputs
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'));
        let mut differences = 0;
        for row in expected.lines().filter(|l| !l.starts_with('#')) {
            let mut fields = row.split_whitespace();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let bits = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            assert_eq!(lines.next().unwrap().parse::<usize>().unwrap(), n);
            let values: Vec<_> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect();
            let actual = two_hop_bilinear_fsz_real(
                &values[1..1 + n * n],
                0,
                n,
                &values[1 + n * n..1 + n * n + n],
                &values[1 + n * n + n..],
            );
            let expected = f64::from_bits(bits);
            // n squared products/reductions, with an absolute term for cancellation.
            let bound = 8.0 * (n * n) as f64 * f64::EPSILON;
            crate::numerical_comparison::assert_close(
                actual,
                expected,
                bound,
                bound,
                format!("size {n}"),
            );
            // Independent archived reductions still exercise distinct rounding paths.
            differences += usize::from(expected != values[0]);
        }
        assert!(lines.next().is_none());
        assert!(
            differences > 0,
            "fixtures must distinguish the scalar and vectorized paths"
        );
    }

    /// Historical Julia 1.13.1 `@turbo` values (four inner lanes, six outer accumulators, FMA):
    /// the C-order kernel differs from them only by reduction-order roundoff.
    #[test]
    fn two_hop_bilinear_is_within_reordering_bound_of_archived_julia_values() {
        let fixture = include_str!("../../../../tests/fixtures/pfaffian_cg/two_hop_bilinear.txt");
        let mut lines = fixture
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'));
        while let Some(line) = lines.next() {
            let n: usize = line.parse().unwrap();
            let values: Vec<f64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect();
            assert_eq!(values.len(), 1 + n * n + 2 * n);
            let actual = two_hop_bilinear_real(
                &values[1..1 + n * n],
                0,
                n,
                &values[1 + n * n..1 + n * n + n],
                &values[1 + n * n + n..],
            );
            let bound = 8.0 * (n * n) as f64 * f64::EPSILON;
            crate::numerical_comparison::assert_close(
                actual,
                values[0],
                bound,
                bound,
                format!("size {n}"),
            );
        }
    }

    /// The kernel equals the values of the authoritative C loops (compiled without FMA from
    /// `pfupdate_two_real.c:167-177`, `c_toolbox/two_hop_bilinear_449`) bit for bit.
    #[test]
    fn two_hop_bilinear_is_bit_identical_to_the_c_reduction() {
        let fixture = include_str!("../../../../tests/fixtures/pfaffian_cg/two_hop_bilinear.txt");
        let expected =
            include_str!("../../../../tests/fixtures/pfaffian_cg/two_hop_bilinear_c.txt");
        let mut lines = fixture
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'));
        let mut cases = 0;
        for row in expected.lines().filter(|l| !l.starts_with('#')) {
            let mut fields = row.split_whitespace();
            let n: usize = fields.next().unwrap().parse().unwrap();
            let bits = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            assert_eq!(lines.next().unwrap().parse::<usize>().unwrap(), n);
            let values: Vec<f64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect();
            let (m, a, b) = (
                &values[1..1 + n * n],
                &values[1 + n * n..1 + n * n + n],
                &values[1 + n * n + n..],
            );
            assert_eq!(
                two_hop_bilinear_real(m, 0, n, a, b).to_bits(),
                bits,
                "size {n}, case {cases}"
            );
            assert_eq!(two_hop_bilinear_fsz_real(m, 0, n, a, b).to_bits(), bits);
            cases += 1;
        }
        assert!(lines.next().is_none());
        assert!(cases >= 100, "{cases}");
    }

    /// The four-row interleaving is the plain nested loop, for every size and a base offset.
    #[test]
    fn two_hop_bilinear_interleaving_matches_the_plain_nested_loop_bitwise() {
        let mut seed = 99;
        for n in 1..=21 {
            let base = 3;
            let m: Vec<f64> = (0..base + n * n)
                .map(|_| pseudo_random(&mut seed))
                .collect();
            let a: Vec<f64> = (0..n).map(|_| pseudo_random(&mut seed)).collect();
            let b: Vec<f64> = (0..n).map(|_| pseudo_random(&mut seed)).collect();
            let mut plain = 0.0;
            for i in 0..n {
                let mut tmp = 0.0;
                for j in 0..n {
                    tmp += m[base + i * n + j] * a[j];
                }
                plain += b[i] * tmp;
            }
            assert_eq!(
                two_hop_bilinear_real(&m, base, n, &a, &b).to_bits(),
                plain.to_bits(),
                "n = {n}"
            );
        }
    }

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
