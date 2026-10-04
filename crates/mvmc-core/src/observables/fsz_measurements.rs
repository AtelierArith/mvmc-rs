//! FSZ Green measurement lifecycle and ordered weighted accumulation.

use super::{
    green_func1_fsz, green_func1_fsz_complex, green_func2_fsz, green_func2_fsz_complex, spin_code,
};
use crate::c_timer::CTimer;
use crate::state::VmcOptimizationState;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

/// Measure and accumulate FSZ Green functions for one saved walker.
///
/// The caller supplies the complex Pfaffian/inverse shadow, including when
/// sampling uses real buffers. The interface follows Julia; supported C models
/// always use C's complex FSZ Green family and scalar weight arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn calculate_green_func_fsz(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    weight: f64,
    ip: Complex64,
    idx: &[i64],
    cfg: &[i64],
    num: &[i64],
    cnt: &[i64],
    spins: &[i64],
) {
    calculate_green_func_fsz_timed(
        data,
        state,
        weight,
        ip,
        idx,
        cfg,
        num,
        cnt,
        spins,
        &mut CTimer::<false>::new(),
    );
}

/// Measure FSZ Green functions while recording C's phases 50 through 53.
#[allow(clippy::too_many_arguments)]
pub fn calculate_green_func_fsz_timed<const TIMED: bool>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    weight: f64,
    ip: Complex64,
    idx: &[i64],
    cfg: &[i64],
    num: &[i64],
    cnt: &[i64],
    spins: &[i64],
    timer: &mut CTimer<TIMED>,
) {
    let phys = state
        .phys_quantities
        .as_ref()
        .expect("PhysicalQuantities must be initialized before FSZ measurements");
    assert_eq!(phys.local_cis_ajs.len(), data.green_one_terms.len());
    assert_eq!(phys.phys_cis_ajs.len(), data.green_one_terms.len());
    assert_eq!(
        phys.local_cis_ajs_ckt_alt_dc.len(),
        data.green_two_terms.len()
    );
    assert_eq!(
        phys.phys_cis_ajs_ckt_alt_dc.len(),
        data.green_two_terms.len()
    );
    assert_eq!(
        phys.phys_cis_ajs_ckt_alt.len(),
        data.green_two_ex_indices.len()
    );

    // Native C's FSZ Green family has no RBM factors. Retain the explicitly
    // Julia-derived RBM extension without treating it as native C parity.
    let one_kernel = if data.has_rbm_terms() {
        green_func1_fsz
    } else {
        green_func1_fsz_complex
    };
    let two_kernel = if data.has_rbm_terms() {
        green_func2_fsz
    } else {
        green_func2_fsz_complex
    };

    // Complete structural checks on the caller before any worker can enter.
    // RBM is a separate Julia extension; preserve its serial producer route.
    let parallel = !data.has_rbm_terms();
    if parallel {
        super::green_measurements::validate_green_sample(
            data,
            state,
            idx,
            cfg,
            num,
            cnt,
            Some(spins),
        );
    }
    let sample_state: &VmcOptimizationState = state;

    timer.start(50);
    let one_body = super::green_measurements::collect_green_values(
        data.green_one_terms.len(),
        parallel,
        |index| {
            let term = &data.green_one_terms[index];
            one_kernel(
                term.site1 as usize,
                term.site2 as usize,
                spin_code(term.spin1),
                spin_code(term.spin2),
                ip,
                data,
                sample_state,
                idx,
                cfg,
                num,
                cnt,
                spins,
            )
        },
    );
    timer.stop(50);
    timer.start(51);
    let direct = super::green_measurements::collect_green_values(
        data.green_two_terms.len(),
        parallel,
        |index| {
            let term = &data.green_two_terms[index];
            two_kernel(
                term.site1 as usize,
                term.site2 as usize,
                term.site3 as usize,
                term.site4 as usize,
                spin_code(term.spin1),
                spin_code(term.spin2),
                spin_code(term.spin3),
                spin_code(term.spin4),
                ip,
                data,
                sample_state,
                idx,
                cfg,
                num,
                cnt,
                spins,
            )
        },
    );
    let phys = state.phys_quantities.as_mut().expect("checked above");
    for (index, value) in direct.into_iter().enumerate() {
        phys.local_cis_ajs_ckt_alt_dc[index] = value;
        phys.phys_cis_ajs_ckt_alt_dc[index] += weight * value;
    }
    timer.stop(51);
    timer.start(52);
    for (index, value) in one_body.iter().copied().enumerate() {
        phys.local_cis_ajs[index] = value;
        phys.phys_cis_ajs[index] += weight * value;
    }
    timer.stop(52);
    timer.start(53);
    for (index, &(first, second)) in data.green_two_ex_indices.iter().enumerate() {
        phys.phys_cis_ajs_ckt_alt[index] += weight * one_body[first] * one_body[second].conj();
    }
    timer.stop(53);
}

/// Normalize accumulated FSZ Green measurements by the accumulated sample weight.
pub fn weight_average_green_func_fsz(state: &mut VmcOptimizationState) {
    let wc = state.energy.wc;
    if wc == Complex64::new(0.0, 0.0) {
        return;
    }
    // C average.c uses complex Wc: compute its native reciprocal once, then
    // multiply each vector element. Per-element division changes rounding.
    let inverse_weight = crate::c_complex::divide(Complex64::new(1.0, 0.0), wc);
    if let Some(phys) = state.phys_quantities.as_mut() {
        for values in [
            &mut phys.phys_cis_ajs,
            &mut phys.phys_cis_ajs_ckt_alt,
            &mut phys.phys_cis_ajs_ckt_alt_dc,
        ] {
            for value in values {
                *value *= inverse_weight;
            }
        }
    }
}
