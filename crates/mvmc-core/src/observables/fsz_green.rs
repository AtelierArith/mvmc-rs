//! General FSZ four-fermion ratios, using explicit spin labels for each operator.

use mvmc_expert_parsers::utils::julia_exp::exp as julia_exp;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use super::{calculate_ip_complex, green_func1_fsz};
use crate::sampling::projection::{log_proj_ratio, update_proj_cnt};
use crate::sampling::updates::calculate_new_pf_m_two_fsz_complex_flat;
use crate::state::VmcOptimizationState;

/// Ratio for `c†(ri,s) c(rj,t) c†(rk,u) c(rl,v)` in an explicit-spin basis.
///
/// Ports Julia's `green_func2_fsz` and `green_func2_fsz2`, including all
/// coincident spin-site reductions. Both source families evaluate the supplied
/// complex Pfaffian/inverse buffers, including for real-valued Slater tables.
#[allow(clippy::too_many_arguments)]
pub fn green_func2_fsz(
    ri: usize,
    rj: usize,
    rk: usize,
    rl: usize,
    s: u8,
    t: u8,
    u: u8,
    v: u8,
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_proj_cnt: &[i64],
    ele_spn: &[i64],
) -> Complex64 {
    let ns = data.modpara.nsite as usize;
    let ne = data.modpara.nelec as usize;
    let nq = state.slater_matrix.pf_m.len();
    let xi = ri + s as usize * ns;
    let xj = rj + t as usize * ns;
    let xk = rk + u as usize * ns;
    let xl = rl + v as usize * ns;
    let zero = Complex64::new(0.0, 0.0);
    let one = |i, j, a, b, state: &mut VmcOptimizationState| {
        green_func1_fsz(
            i,
            j,
            a,
            b,
            ip,
            data,
            state,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            ele_spn,
        )
    };
    if xi == xj {
        if xj == xk {
            return if xk == xl {
                Complex64::new(ele_num[xi] as f64, 0.0)
            } else if ele_num[xi] == 1 {
                zero
            } else {
                one(rk, rl, u, v, state)
            };
        } else if xj == xl {
            return zero;
        } else if xk == xl {
            return Complex64::new((ele_num[xi] * ele_num[xk]) as f64, 0.0);
        } else {
            return if ele_num[xi] == 0 {
                zero
            } else {
                one(rk, rl, u, v, state)
            };
        }
    } else if xi == xk {
        return zero;
    } else if xi == xl {
        return if xj == xk {
            Complex64::new((ele_num[xi] * (1 - ele_num[xj])) as f64, 0.0)
        } else if ele_num[xi] == 0 {
            zero
        } else {
            -one(rk, rj, u, t, state)
        };
    } else if xj == xk {
        return if xk == xl {
            if ele_num[xj] == 0 {
                zero
            } else {
                one(ri, rj, s, t, state)
            }
        } else if ele_num[xj] == 1 {
            zero
        } else {
            one(ri, rl, s, v, state)
        };
    } else if xj == xl {
        return zero;
    } else if xk == xl {
        return if ele_num[xk] == 0 {
            zero
        } else {
            one(ri, rj, s, t, state)
        };
    }
    if ele_num[xi] == 1 || ele_num[xj] == 0 || ele_num[xk] == 1 || ele_num[xl] == 0 {
        return zero;
    }
    let mj = ele_cfg[xj];
    let ml = ele_cfg[xl];
    if mj < 0 || ml < 0 {
        return zero;
    }
    let (mj, ml) = (mj as usize, ml as usize);
    let mut idx = ele_idx.to_vec();
    let mut spins = ele_spn.to_vec();
    let mut num = ele_num.to_vec();
    let mut mid = vec![0; ele_proj_cnt.len()];
    let mut final_cnt = vec![0; ele_proj_cnt.len()];

    // Rightmost hop precedes the leftmost in both projection and Pfaffian work.
    idx[ml] = rk as i64;
    spins[ml] = u as i64;
    num[xl] = 0;
    num[xk] = 1;
    // The supported Gutzwiller/Jastrow source update depends on site charges,
    // so update_proj_cnt_fsz and update_proj_cnt have identical integer work.
    update_proj_cnt(rl as i64, rk as i64, u, &mut mid, ele_proj_cnt, &num, data);
    idx[mj] = ri as i64;
    spins[mj] = s as i64;
    num[xj] = 0;
    num[xi] = 1;
    update_proj_cnt(rj as i64, ri as i64, s, &mut final_cnt, &mid, &num, data);
    let ratio = super::with_rbm_ratio(
        julia_exp(log_proj_ratio(&final_cnt, ele_proj_cnt, data)),
        &num,
        ele_num,
        data,
    );
    let mut pf = vec![zero; nq];
    calculate_new_pf_m_two_fsz_complex_flat(
        ml,
        u,
        mj,
        s,
        &mut pf,
        &idx,
        &spins,
        &state.slater_matrix.slater_elm,
        state.slater_matrix.inv_m.as_slice(),
        (2 * ne).pow(2) + 1,
        &state.slater_matrix.pf_m,
        0,
        nq,
        ns,
        ne,
    );
    let numerator = ratio * calculate_ip_complex(&pf, 0, nq, data);
    crate::julia_complex::divide(numerator, ip).conj()
}
