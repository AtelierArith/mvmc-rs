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
    // C `average.c:99-108` `omp parallel for`: one producer per element.
    crate::threading::for_each_mut(&mut state.sr_opt.sr_opt_oo, 4, |_, x| *x *= inv);
    crate::threading::for_each_mut(&mut state.sr_opt.sr_opt_ho, 4, |_, x| *x *= inv);
}

/// Normalize the active real SR buffers, preserving unused OO capacity.
///
/// C `WeightAverageSROpt_real` scales one contiguous OO+HO prefix: direct
/// uses `size*(size+1)` elements, CG uses `3*size`. Rust stores HO separately,
/// so the corresponding OO lengths are `size*size` and `2*size`; HO has `size`
/// entries. The caller performs collective reduction before this local scaling.
/// Real buffers are absent in all-complex mode. The existing small-weight
/// no-op guard is retained; it is not the unguarded C zero-weight contract.
pub fn weight_average_sr_opt_real(state: &mut VmcOptimizationState, nsrcg: bool) {
    let wc = state.energy.wc;
    if wc.norm() < 1.0e-15 {
        return;
    }
    let inv = 1.0 / wc.re;
    if !state.sr_opt.sr_opt_oo_real.is_empty() {
        let size = state.sr_opt.sr_opt_size;
        let active_oo = if nsrcg { 2 * size } else { size * size };
        crate::threading::for_each_mut(&mut state.sr_opt.sr_opt_oo_real[..active_oo], 2, |_, x| {
            *x *= inv
        });
        crate::threading::for_each_mut(&mut state.sr_opt.sr_opt_ho_real, 2, |_, x| *x *= inv);
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

    #[test]
    fn real_direct_weight_average_scales_only_c_active_prefix() {
        check_real_active_prefix(false, 2.0);
    }

    #[test]
    fn real_cg_weight_average_scales_only_c_active_prefix() {
        check_real_active_prefix(true, 4.0);
    }

    #[test]
    fn real_weight_average_matches_c_at_minimum_positive_runtime_count() {
        // C clears Wc to zero and adds w=1 per valid sample (including FSZ/BF).
        // Thus the smallest positive runtime count is one, not an exponential
        // importance weight. C's invW=1/Wc is exactly one at this boundary.
        check_real_active_prefix(false, 1.0);
        check_real_active_prefix(true, 1.0);
    }

    fn check_real_active_prefix(nsrcg: bool, weight: f64) {
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 3, 1, 1, false, false);
        let size = state.sr_opt.sr_opt_size;
        state.sr_opt.sr_opt_oo_real = (1..=size * (size + 2)).map(|i| i as f64).collect();
        state.sr_opt.sr_opt_ho_real = (0..size).map(|i| 101.0 + i as f64).collect();
        state.sr_opt.sr_opt_o_real.fill(37.0);
        state.sr_opt.sr_opt_o_store_real.fill(53.0);
        state.energy.wc = Complex64::new(weight, 0.0);
        let oo_before = state.sr_opt.sr_opt_oo_real.clone();
        let ho_before = state.sr_opt.sr_opt_ho_real.clone();
        let o_before = state.sr_opt.sr_opt_o_real.clone();
        let store_before = state.sr_opt.sr_opt_o_store_real.clone();
        // Size-one communicator semantics: already-local sums, one scalar
        // normalization. This helper has no reducer/MPI call to duplicate sums.
        weight_average_sr_opt_real(&mut state, nsrcg);
        let active = if nsrcg { 2 * size } else { size * size };
        assert_eq!(state.sr_opt.sr_opt_oo_real.len(), oo_before.len());
        assert_eq!(state.sr_opt.sr_opt_ho_real.len(), size);
        assert_eq!(
            state.sr_opt.sr_opt_oo_real[..active],
            oo_before[..active]
                .iter()
                .map(|v| v / weight)
                .collect::<Vec<_>>()
        );
        assert_eq!(state.sr_opt.sr_opt_oo_real[active..], oo_before[active..]);
        assert_eq!(
            state.sr_opt.sr_opt_ho_real,
            ho_before.iter().map(|v| v / weight).collect::<Vec<_>>()
        );
        assert_eq!(state.sr_opt.sr_opt_o_real, o_before);
        assert_eq!(state.sr_opt.sr_opt_o_store_real, store_before);
        assert_eq!(state.energy.wc, Complex64::new(weight, 0.0));
    }

    #[test]
    fn real_weight_average_keeps_existing_small_weight_guard_and_complex_noop() {
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 3, 1, 1, false, false);
        state.sr_opt.sr_opt_oo_real.fill(7.0);
        state.sr_opt.sr_opt_ho_real.fill(11.0);
        for weight in [0.0, 1.0e-16] {
            state.energy.wc = Complex64::new(weight, 0.0);
            weight_average_sr_opt_real(&mut state, false);
            assert!(state.sr_opt.sr_opt_oo_real.iter().all(|&v| v == 7.0));
            assert!(state.sr_opt.sr_opt_ho_real.iter().all(|&v| v == 11.0));
        }
        let mut complex = dummy_state();
        complex.energy.wc = Complex64::new(2.0, 0.0);
        let before = complex.sr_opt.sr_opt_oo.clone();
        weight_average_sr_opt_real(&mut complex, true);
        assert!(complex.sr_opt.sr_opt_oo_real.is_empty());
        assert_eq!(complex.sr_opt.sr_opt_oo, before);
    }
}
