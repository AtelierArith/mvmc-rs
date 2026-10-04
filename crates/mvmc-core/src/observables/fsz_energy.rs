//! Native serial FSZ Hamiltonian order and scalar/complex accumulator contracts.

use super::{
    green_func1_fsz_complex, green_func1_fsz_real, green_func2_fsz_complex, green_func2_fsz_real,
    spin_code,
};
use crate::c_timer::CTimer;
use crate::state::VmcOptimizationState;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

#[allow(clippy::too_many_arguments)]
pub(super) fn native_energy<const REAL: bool, const TIMED: bool>(
    ip: Complex64,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    idx: &[i64],
    cfg: &[i64],
    num: &[i64],
    cnt: &[i64],
    spins: &[i64],
    timer: &mut CTimer<TIMED>,
) -> Complex64 {
    let ns = data.modpara.nsite as usize;
    let valid = |sites: &[i64]| sites.iter().all(|&site| (0..ns as i64).contains(&site));
    let (up, down) = num.split_at(ns);
    let mut e = Complex64::new(0.0, 0.0);
    timer.start(70);
    for term in &data.coulomb_intra_terms {
        if valid(&[term.site]) {
            let i = term.site as usize;
            e.re += term.value * up[i] as f64 * down[i] as f64;
        }
    }
    for term in &data.coulomb_inter_terms {
        if valid(&[term.site1, term.site2]) {
            let (i, j) = (term.site1 as usize, term.site2 as usize);
            e.re += term.value * (up[i] + down[i]) as f64 * (up[j] + down[j]) as f64;
        }
    }
    for term in &data.hund_terms {
        if valid(&[term.site1, term.site2]) {
            let (i, j) = (term.site1 as usize, term.site2 as usize);
            e.re -= term.value * (up[i] * up[j] + down[i] * down[j]) as f64;
        }
    }
    timer.stop(70);
    timer.start(71);
    for term in &data.transfer_terms {
        if valid(&[term.site1, term.site2]) {
            let (i, j) = (term.site1 as usize, term.site2 as usize);
            let (s, t) = (spin_code(term.spin1), spin_code(term.spin2));
            if REAL {
                e.re -= term.value.re
                    * green_func1_fsz_real(
                        i, j, s, t, ip.re, data, state, idx, cfg, num, cnt, spins,
                    );
            } else {
                e -= term.value
                    * green_func1_fsz_complex(
                        i, j, s, t, ip, data, state, idx, cfg, num, cnt, spins,
                    );
            }
        }
    }
    timer.stop(71);
    timer.start(72);
    let green = |sites: [usize; 4], op_spins: [u8; 4]| {
        let [i, j, k, l] = sites;
        let [s, t, u, v] = op_spins;
        if REAL {
            Complex64::new(
                green_func2_fsz_real(
                    i, j, k, l, s, t, u, v, ip.re, data, state, idx, cfg, num, cnt, spins,
                ),
                0.0,
            )
        } else {
            green_func2_fsz_complex(
                i, j, k, l, s, t, u, v, ip, data, state, idx, cfg, num, cnt, spins,
            )
        }
    };
    for term in &data.pair_hop_terms {
        if valid(&[term.site1, term.site2]) {
            let (i, j) = (term.site1 as usize, term.site2 as usize);
            let g = green([i, j, i, j], [0, 0, 1, 1]);
            if REAL {
                e.re += term.value * g.re;
            } else {
                e += term.value * g;
            }
        }
    }
    for term in &data.exchange_terms {
        if valid(&[term.site1, term.site2]) {
            let (i, j) = (term.site1 as usize, term.site2 as usize);
            let mut g = green([i, j, j, i], [0, 0, 1, 1]);
            g += green([i, j, j, i], [1, 1, 0, 0]);
            if REAL {
                e.re += term.value * g.re;
            } else {
                e += term.value * g;
            }
        }
    }
    for term in &data.inter_all_terms {
        let sites = [term.site0, term.site1, term.site2, term.site3];
        if !valid(&sites) {
            continue;
        }
        let op_spins = [term.spin0, term.spin1, term.spin2, term.spin3];
        assert!(
            op_spins.iter().all(|spin| (0..=1).contains(spin)),
            "invalid InterAll spin index"
        );
        let g = green(sites.map(|s| s as usize), op_spins.map(|s| s as u8));
        if REAL {
            e.re += term.value.re * g.re;
        } else {
            e += term.value * g;
        }
    }
    timer.stop(72);
    // The serial C body adds its private accumulator to a zero reduction sum.
    Complex64::new(0.0, 0.0) + e
}
