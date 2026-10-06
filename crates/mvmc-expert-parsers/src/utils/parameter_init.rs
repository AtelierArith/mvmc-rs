//! Phase 4 — variational-parameter initialisation.
//!
//! Port target: `MVMCExpertModeParsers.jl/src/utils/parameter_init.jl`
//!              + `mVMC/src/mVMC/parameter.c :: InitParameter()`.
//!
//! Preserve the upstream RNG draw order and exact subsequent RNG state.
//! Computed declared Slater values use explicit numerical bounds against C. The
//! caller MUST seed the SFMT RNG before calling `init_parameter`.
//!
//! Real/complex Slater initialization includes Gutzwiller/Jastrow/DH2/DH4 declarations.
//! RBM coefficients consume draws in canonical section/index order before Slater.

use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::c_const::D_AMP_MAX;
use crate::types::ExpertModeData;

/// Validate loaded declarations and compute C's raw header sum.
///
/// Coefficient values do not define the declaration. To replace loaded
/// declarations programmatically, clear both native metadata maps first.
pub fn all_complex_flag(data: &ExpertModeData) -> Result<bool, &'static str> {
    let mut sum = 0_i64;
    for (key, declaration) in [
        (
            "Gutzwiller",
            data.gutzwiller_terms.iter().any(|t| t.is_complex),
        ),
        ("Jastrow", data.jastrow_terms.iter().any(|t| t.is_complex)),
        ("DH2", data.doublon_holon_2site_complex),
        ("DH4", data.doublon_holon_4site_complex),
    ] {
        if let Some(&header) = data.native_complex_headers.get(key) {
            if data.native_complex_declarations.get(key) != Some(&declaration) {
                return Err("loaded complex declaration changed without clearing native metadata");
            }
            sum += i64::from(header);
        } else {
            sum += i64::from(declaration);
        }
    }
    let orbital = super::opt_flag::raw_orbital_complex_header(data);
    let declaration = data.orbital_terms.iter().any(|t| t.is_complex);
    if let Some(header) = orbital {
        if data.native_complex_declarations.get("Orbitals") != Some(&declaration) {
            return Err("loaded orbital declaration changed without clearing native metadata");
        }
        sum += header;
    } else {
        sum += i64::from(declaration);
    }
    Ok(sum != 0)
}

/// Number of unique Slater (orbital) parameters. Mirrors
/// `n_orbital_idx` / `NSlater` derivation in upstream.
pub fn n_slater(data: &ExpertModeData) -> usize {
    data.modpara.n_orbital_idx.max(0) as usize
}

/// `init_parameter!(data; rng)` mirror.
pub fn init_parameter(
    data: &mut ExpertModeData,
    rng: &mut Sfmt19937Rng,
) -> Result<(), &'static str> {
    // Validate before changing coefficients or consuming RNG draws.
    let all_complex = all_complex_flag(data)?;
    // Proj parameters always start at zero.
    for term in data.gutzwiller_terms.iter_mut() {
        term.value = Complex64::new(0.0, 0.0);
    }
    for term in data.jastrow_terms.iter_mut() {
        term.value = Complex64::new(0.0, 0.0);
    }
    data.doublon_holon_2site_params
        .fill(Complex64::new(0.0, 0.0));

    data.doublon_holon_4site_params
        .fill(Complex64::new(0.0, 0.0));

    let n_proj = data.projection_layout().n_proj;
    let sizes = data.rbm_section_sizes();
    let n_rbm = sizes.iter().sum::<usize>();
    let mut rbm_values = vec![Complex64::new(0.0, 0.0); n_rbm];
    let neurons = data
        .modpara
        .nneuron
        .wrapping_add(data.modpara.nneuron_charge)
        .wrapping_add(data.modpara.nneuron_spin)
        .wrapping_add(data.modpara.nneuron_general);
    let divisor = neurons as f64;
    for (i, slot) in rbm_values.iter_mut().enumerate() {
        // Unlike Slater, absent RBM flags are inactive and consume no draws.
        if data
            .optimization_flags
            .get(2 * (n_proj + i))
            .copied()
            .unwrap_or(0)
            > 0
        {
            if all_complex {
                let radius = 1e-2 * rng.genrand_real2();
                let phase = (2.0 * std::f64::consts::PI) * rng.genrand_real2();
                let sin = super::c_math::sin(phase);
                let cos = super::c_math::cos(phase);
                *slot = Complex64::new(radius * cos, radius * sin);
            } else {
                *slot = Complex64::new(0.01 * (rng.genrand_real2() - 0.5) / divisor, 0.0);
            }
        }
    }
    data.set_rbm_parameters(rbm_values);
    // C initializes every declared active slot, including unmapped slots.
    let n_s = n_slater(data);
    let mut slater_values = vec![Complex64::new(0.0, 0.0); n_s];

    if !all_complex {
        for (i, slot) in slater_values.iter_mut().enumerate() {
            let opt_flag_idx = 2 * i + 2 * n_proj + 2 * n_rbm;
            let should_optimize = data
                .optimization_flags
                .get(opt_flag_idx)
                .copied()
                .unwrap_or(1)
                > 0;
            if should_optimize {
                let r = rng.genrand_real2();
                *slot = Complex64::new(2.0 * (r - 0.5), 0.0);
            } else {
                *slot = Complex64::new(0.0, 0.0);
            }
        }
    } else {
        for (i, slot) in slater_values.iter_mut().enumerate() {
            let opt_flag_idx = 2 * i + 2 * n_proj + 2 * n_rbm;
            let should_optimize = data
                .optimization_flags
                .get(opt_flag_idx)
                .copied()
                .unwrap_or(1)
                > 0;
            if should_optimize {
                let r1 = rng.genrand_real2();
                let r2 = rng.genrand_real2();
                let real = 2.0 * (r1 - 0.5);
                let imag = 2.0 * (r2 - 0.5);
                *slot = Complex64::new(
                    real / std::f64::consts::SQRT_2,
                    imag / std::f64::consts::SQRT_2,
                );
            } else {
                *slot = Complex64::new(0.0, 0.0);
            }
        }
    }

    data.slater_params = slater_values;
    if !data.para_qp_opt_trans.is_empty() {
        data.opt_trans.clone_from(&data.para_qp_opt_trans);
    }
    Ok(())
}

/// Synchronize DH2/DH4/Gutzwiller/Jastrow real gauges when enabled, then rescale
/// Slater to `D_AMP_MAX`. Correlation shifts can be disabled as in Julia.
pub fn sync_modified_parameter(data: &mut ExpertModeData, shift_correlations: bool) {
    let layout = data.projection_layout();
    let real_flag = |i: usize| data.optimization_flags.get(2 * i).copied().unwrap_or(0) == 1;
    let all_gutz = layout.n_gutzwiller > 0 && (0..layout.n_gutzwiller).all(real_flag);
    let shift_dh2 = layout.n_dh2 > 0
        && all_gutz
        && (0..6 * layout.n_dh2).all(|i| real_flag(layout.dh2_offset + i));
    let shift_dh4 = layout.n_dh4 > 0
        && all_gutz
        && (0..10 * layout.n_dh4).all(|i| real_flag(layout.dh4_offset + i));
    if shift_correlations {
        let mut g_shift = 0.0;
        if shift_dh2 {
            let stride = 2 * layout.n_dh2;
            for group in 0..stride {
                if group + 2 * stride < data.doublon_holon_2site_params.len() {
                    let params = &mut data.doublon_holon_2site_params;
                    let shift = (params[group].re
                        + params[group + stride].re
                        + params[group + 2 * stride].re)
                        / 3.0;
                    params[group].re -= shift;
                    params[group + stride].re -= shift;
                    params[group + 2 * stride].re -= shift;
                    g_shift += shift;
                }
            }
        }
        if shift_dh4 {
            let stride = 2 * layout.n_dh4;
            let mut dh4_shift = 0.0;
            for group in 0..stride {
                if group + 4 * stride < data.doublon_holon_4site_params.len() {
                    let params = &mut data.doublon_holon_4site_params;
                    // Julia sums the five-bin generator in order, then adds
                    // the complete DH4 compensation to the DH2 compensation.
                    let mut shift = params[group].re;
                    for bin in 1..5 {
                        shift += params[group + bin * stride].re;
                    }
                    shift /= 5.0;
                    for bin in 0..5 {
                        params[group + bin * stride].re -= shift;
                    }
                    dh4_shift += shift;
                }
            }
            g_shift += dh4_shift;
        }
        if g_shift != 0.0 {
            for term in &mut data.gutzwiller_terms {
                term.value.re += g_shift;
            }
        }
    }
    // Optional Gutzwiller / Jastrow shift mirroring the Julia
    // `flag_shift_gj` block. Active only when every parameter is
    // optimized. Empty flags mean all active, as in Julia's local sync.
    let n_gutz = data.gutzwiller_terms.len();
    let n_jast = data.jastrow_terms.len();
    let all_active = data.optimization_flags.is_empty()
        || ((0..n_gutz).all(|i| super::opt_flag::is_gutzwiller_optimized(data, i))
            && (0..n_jast).all(|i| super::opt_flag::is_jastrow_optimized(data, i)));
    if shift_correlations && n_gutz > 0 && n_jast > 0 && all_active {
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
    for value in &data.slater_params {
        let abs_val = super::c_math::hypot(value.re, value.im);
        if abs_val > xmax {
            xmax = abs_val;
        }
    }
    if xmax > 0.0 {
        let ratio = D_AMP_MAX / xmax;
        for value in &mut data.slater_params {
            *value *= ratio;
        }
    }
}

/// `initialize_parameters!` — init + sync.
pub fn initialize_parameters(
    data: &mut ExpertModeData,
    rng: &mut Sfmt19937Rng,
) -> Result<(), &'static str> {
    init_parameter(data, rng)?;
    sync_modified_parameter(data, false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrbitalTerm;

    fn data_with_orbitals(n: i64) -> ExpertModeData {
        let mut d = ExpertModeData::new();
        d.modpara.n_orbital_idx = n;
        d.slater_params = vec![Complex64::new(0.0, 0.0); n as usize];
        d.orbital_terms = (0..n)
            .map(|idx| OrbitalTerm {
                site1: idx,
                site2: idx,
                idx,
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
        init_parameter(&mut d, &mut rng).unwrap();
        // Sample once via fresh RNG to confirm draw order: 4 draws of
        // genrand_real2() should match what `init_parameter` consumed.
        let mut probe = Sfmt19937Rng::new(1);
        for i in 0..4 {
            let expected = 2.0 * (probe.genrand_real2() - 0.5);
            assert!((d.slater_params[d.orbital_terms[i].idx as usize].re - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn sync_modified_parameter_rescales_slater_block() {
        let mut d = data_with_orbitals(2);
        d.slater_params[d.orbital_terms[0].idx as usize] = Complex64::new(8.0, 0.0);
        d.slater_params[d.orbital_terms[1].idx as usize] = Complex64::new(-4.0, 0.0);
        sync_modified_parameter(&mut d, true);
        assert!((d.slater_params[d.orbital_terms[0].idx as usize].re - 4.0).abs() < 1e-15);
        assert!((d.slater_params[d.orbital_terms[1].idx as usize].re - (-2.0)).abs() < 1e-15);
    }
}
