//! Weighted average of stored per-sample optimisation data.
//!
//! Port target: `MVMCOptimizers.jl/src/weight_average.jl`.
//!
//! All three routines divide the accumulated buffers by `wc` (the
//! per-step weight count). The real / complex versions deliberately
//! exclude `sr_opt_o` from the normalisation, mirroring upstream
//! `weight_average_sr_opt!` which only rescales `sr_opt_oo` and
//! `sr_opt_ho`.

use num_complex::Complex64;

use crate::state::VmcOptimizationState;

/// `weight_average_we!(state)` mirror — divides the per-step energy
/// accumulators (`etot`, `etot2`, `sztot`, `sztot2`) by `wc`. No-op if
/// `wc` is numerically zero.
pub fn weight_average_we(state: &mut VmcOptimizationState) {
    let wc = state.energy.wc;
    if wc.norm() < 1.0e-15 {
        return;
    }
    let inv = Complex64::new(1.0, 0.0) / wc;
    state.energy.etot *= inv;
    state.energy.etot2 *= inv;
    state.energy.sztot *= inv;
    state.energy.sztot2 *= inv;
}

/// Complex `weight_average_sr_opt!(state)` mirror.
pub fn weight_average_sr_opt(state: &mut VmcOptimizationState) {
    let wc = state.energy.wc;
    if wc.norm() < 1.0e-15 {
        return;
    }
    let inv = Complex64::new(1.0, 0.0) / wc;
    for x in state.sr_opt.sr_opt_oo.iter_mut() {
        *x *= inv;
    }
    for x in state.sr_opt.sr_opt_ho.iter_mut() {
        *x *= inv;
    }
}

/// Real `weight_average_sr_opt_real!(state)` mirror. Touches the real
/// scratch only when present (skipped in all-complex mode).
pub fn weight_average_sr_opt_real(state: &mut VmcOptimizationState) {
    let wc = state.energy.wc;
    if wc.norm() < 1.0e-15 {
        return;
    }
    let inv = 1.0 / wc.re;
    if !state.sr_opt.sr_opt_oo_real.is_empty() {
        for x in state.sr_opt.sr_opt_oo_real.iter_mut() {
            *x *= inv;
        }
        for x in state.sr_opt.sr_opt_ho_real.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::EnergyData;

    fn dummy_state() -> VmcOptimizationState {
        VmcOptimizationState::zeros(2, 1, 0, 1, 1, 4, true, false)
    }

    #[test]
    fn weight_average_we_divides_by_wc() {
        let mut state = dummy_state();
        state.energy = EnergyData {
            wc: Complex64::new(4.0, 0.0),
            etot: Complex64::new(8.0, 0.0),
            etot2: Complex64::new(16.0, 0.0),
            sztot: Complex64::new(4.0, 0.0),
            sztot2: Complex64::new(2.0, 0.0),
        };
        weight_average_we(&mut state);
        assert_eq!(state.energy.etot, Complex64::new(2.0, 0.0));
        assert_eq!(state.energy.etot2, Complex64::new(4.0, 0.0));
        assert_eq!(state.energy.sztot, Complex64::new(1.0, 0.0));
        assert_eq!(state.energy.sztot2, Complex64::new(0.5, 0.0));
    }
}
