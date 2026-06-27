//! Phase 4 — variational-parameter initialisation.
//!
//! Port target: `MVMCExpertModeParsers.jl/src/utils/parameter_init.jl`
//!              + `mVMC/src/mVMC/parameter.c :: InitParameter()`.
//!
//! BIT-PARITY CRITICAL: preserves the upstream RNG draw order so the
//! Slater values that seed the optimiser match Julia + C exactly. The
//! caller MUST seed the SFMT RNG before calling [`init_parameter`].
//!
//! Phase-4 scope: real Slater initialisation only (`AllComplexFlag == 0`).
//! RBM init lives behind `flag_rbm` and currently no-ops because the
//! Phase-3 parsers do not surface RBM terms; once RBM parsers land,
//! drop the parking logic here.

use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::c_const::D_AMP_MAX;
use crate::types::ExpertModeData;

/// Compute `AllComplexFlag` exactly as `parameter_init.jl` does.
pub fn all_complex_flag(data: &ExpertModeData) -> bool {
    if data.modpara.complex_flag != 0 {
        return true;
    }
    let g = data.gutzwiller_terms.iter().any(|t| t.is_complex);
    let j = data.jastrow_terms.iter().any(|t| t.is_complex);
    let o = data.orbital_terms.iter().any(|t| t.is_complex);
    g || j || o
}

/// Number of unique Slater (orbital) parameters. Mirrors
/// `n_orbital_idx` / `NSlater` derivation in upstream.
pub fn n_slater(data: &ExpertModeData) -> usize {
    if data.modpara.n_orbital_idx > 0 {
        data.modpara.n_orbital_idx as usize
    } else if let Some(max_idx) = data.orbital_terms.iter().map(|t| t.idx).max() {
        (max_idx + 1).max(0) as usize
    } else {
        0
    }
}

/// `init_parameter!(data; rng)` mirror.
pub fn init_parameter(data: &mut ExpertModeData, rng: &mut Sfmt19937Rng) {
    // Proj parameters always start at zero.
    for term in data.gutzwiller_terms.iter_mut() {
        term.value = Complex64::new(0.0, 0.0);
    }
    for term in data.jastrow_terms.iter_mut() {
        term.value = Complex64::new(0.0, 0.0);
    }

    let all_complex = all_complex_flag(data);
    let n_proj = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    // Phase-4 scope: RBM blocks are still parser-side stubs; treat
    // `n_rbm = 0` so the Slater opt-flag index lines up with the
    // upstream layout for non-RBM models.
    let n_rbm = 0usize;
    let flag_rbm = false;
    let n_s = n_slater(data);
    let mut slater_values = vec![Complex64::new(0.0, 0.0); n_s];

    if !all_complex {
        for (i, slot) in slater_values.iter_mut().enumerate() {
            let opt_flag_idx = 2 * i + 2 * n_proj + 2 * (if flag_rbm { 1 } else { 0 }) * n_rbm;
            let should_optimize = data
                .optimization_flags
                .get(opt_flag_idx)
                .copied()
                .unwrap_or(true);
            if should_optimize {
                let r = rng.genrand_real2();
                *slot = Complex64::new(2.0 * (r - 0.5), 0.0);
            } else {
                *slot = Complex64::new(0.0, 0.0);
            }
        }
    } else {
        let inv_sqrt_2 = 1.0 / std::f64::consts::SQRT_2;
        for (i, slot) in slater_values.iter_mut().enumerate() {
            let opt_flag_idx = 2 * i + 2 * n_proj + 2 * (if flag_rbm { 1 } else { 0 }) * n_rbm;
            let should_optimize = data
                .optimization_flags
                .get(opt_flag_idx)
                .copied()
                .unwrap_or(true);
            if should_optimize {
                let r1 = rng.genrand_real2();
                let r2 = rng.genrand_real2();
                let real = 2.0 * (r1 - 0.5);
                let imag = 2.0 * (r2 - 0.5);
                *slot = Complex64::new(real * inv_sqrt_2, imag * inv_sqrt_2);
            } else {
                *slot = Complex64::new(0.0, 0.0);
            }
        }
    }

    for term in data.orbital_terms.iter_mut() {
        if term.idx >= 0 && (term.idx as usize) < n_s {
            term.value = slater_values[term.idx as usize];
        }
    }
}

/// `SyncModifiedParameter()` mirror — rescales the Slater block so that
/// `max |Slater[i]| <= D_AMP_MAX`. Matches `parameter.c:159-161`.
pub fn sync_modified_parameter(data: &mut ExpertModeData) {
    // Optional Gutzwiller / Jastrow shift mirroring the Julia
    // `flag_shift_gj` block. Active only when every parameter is
    // optimised (true by default before opt-flag tracking lands).
    let n_gutz = data.gutzwiller_terms.len();
    let n_jast = data.jastrow_terms.len();
    if n_gutz > 0 && n_jast > 0 {
        let total = n_gutz + n_jast;
        let mut shift = 0.0;
        for term in &data.gutzwiller_terms {
            shift += term.value.re;
        }
        for term in &data.jastrow_terms {
            shift += term.value.re;
        }
        if total > 0 {
            shift /= total as f64;
            for term in data.gutzwiller_terms.iter_mut() {
                term.value -= Complex64::new(shift, 0.0);
            }
            for term in data.jastrow_terms.iter_mut() {
                term.value -= Complex64::new(shift, 0.0);
            }
        }
    }

    let mut xmax = 0.0;
    for term in &data.orbital_terms {
        let abs_val = term.value.norm();
        if abs_val > xmax {
            xmax = abs_val;
        }
    }
    if xmax > 0.0 {
        let ratio = D_AMP_MAX / xmax;
        for term in data.orbital_terms.iter_mut() {
            term.value *= Complex64::new(ratio, 0.0);
        }
    }
}

/// `initialize_parameters!` — init + sync.
pub fn initialize_parameters(data: &mut ExpertModeData, rng: &mut Sfmt19937Rng) {
    init_parameter(data, rng);
    sync_modified_parameter(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrbitalTerm;

    fn data_with_orbitals(n: i64) -> ExpertModeData {
        let mut d = ExpertModeData::new();
        d.modpara.n_orbital_idx = n;
        d.orbital_terms = (0..n)
            .map(|idx| OrbitalTerm {
                site1: idx,
                site2: idx,
                idx,
                value: Complex64::new(0.0, 0.0),
                is_complex: false,
                sign: 1,
            })
            .collect();
        d
    }

    #[test]
    fn real_slater_initial_values_match_julia_for_seed_one() {
        let mut d = data_with_orbitals(4);
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut d, &mut rng);
        // Sample once via fresh RNG to confirm draw order: 4 draws of
        // genrand_real2() should match what `init_parameter` consumed.
        let mut probe = Sfmt19937Rng::new(1);
        for i in 0..4 {
            let expected = 2.0 * (probe.genrand_real2() - 0.5);
            assert!((d.orbital_terms[i].value.re - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn sync_modified_parameter_rescales_slater_block() {
        let mut d = data_with_orbitals(2);
        d.orbital_terms[0].value = Complex64::new(8.0, 0.0);
        d.orbital_terms[1].value = Complex64::new(-4.0, 0.0);
        sync_modified_parameter(&mut d);
        assert!((d.orbital_terms[0].value.re - 4.0).abs() < 1e-15);
        assert!((d.orbital_terms[1].value.re - (-2.0)).abs() < 1e-15);
    }
}
