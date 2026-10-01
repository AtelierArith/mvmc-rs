//! Julia's C-compatible real/imaginary optimization flag layout.
//! Parameter indices passed to the Rust accessors are zero-based.

use super::parameter_init::{all_complex_flag, n_slater};
use crate::types::ExpertModeData;
use std::collections::BTreeMap;

/// Extend the component array with true defaults without changing existing flags.
pub fn ensure_optimization_flags_size(data: &mut ExpertModeData, n_components: usize) {
    if data.optimization_flags.len() < n_components {
        data.optimization_flags.resize(n_components, true);
    }
}

/// Apply the explicitly listed Gutzwiller/Jastrow flags in Julia's layout.
/// Noncomplex projection parameters have false imaginary flags when listed;
/// omitted entries keep their previous/default values.
pub fn set_projection_opt_flags(
    data: &mut ExpertModeData,
    gutzwiller_flags: &BTreeMap<i64, i64>,
    jastrow_flags: &BTreeMap<i64, i64>,
    gutzwiller_is_complex: bool,
    jastrow_is_complex: bool,
) {
    if gutzwiller_flags.is_empty() && jastrow_flags.is_empty() {
        return;
    }
    let n_gutz = data.projection_layout().n_gutzwiller;
    let n_proj = data.projection_layout().n_proj;
    ensure_optimization_flags_size(data, 2 * n_proj);
    for (flags, offset, complex) in [
        (gutzwiller_flags, 0, gutzwiller_is_complex),
        (jastrow_flags, n_gutz, jastrow_is_complex),
    ] {
        for (&idx, &flag) in flags {
            let Ok(idx) = usize::try_from(idx) else {
                continue;
            };
            let component = 2 * (offset + idx);
            if component < data.optimization_flags.len() {
                data.optimization_flags[component] = flag != 0;
                data.optimization_flags[component + 1] = complex && flag != 0;
            }
        }
    }
}

/// Apply orbital flags once every factor's parameter count is known.
/// Julia only assigns imaginary flags in complex mode; real orbital
/// imaginary flags retain the array's default/previous value.
pub fn set_orbital_opt_flags(data: &mut ExpertModeData, flags: &BTreeMap<i64, i64>) {
    if flags.is_empty() {
        return;
    }
    let n_proj = data.projection_layout().n_proj;
    let complex = all_complex_flag(data);
    ensure_optimization_flags_size(data, 2 * (n_proj + n_slater(data)));
    for (&idx, &flag) in flags {
        let Ok(idx) = usize::try_from(idx) else {
            continue;
        };
        let component = 2 * (n_proj + idx);
        if component < data.optimization_flags.len() {
            data.optimization_flags[component] = flag != 0;
            if complex {
                data.optimization_flags[component + 1] = flag != 0;
            }
        }
    }
}

/// Fold DH2/DH4 row-ordered flags into the final projection layout.
/// Real DH parameters always have false imaginary flags; defaults and flags
/// of other factors are preserved, including incomplete programmatic arrays.
pub fn set_dh_opt_flags(data: &mut ExpertModeData) {
    let layout = data.projection_layout();
    if layout.n_dh2 == 0 && layout.n_dh4 == 0 {
        return;
    }
    ensure_optimization_flags_size(data, 2 * layout.n_proj);
    for (index, &flag) in data.doublon_holon_2site_opt_flags.iter().enumerate() {
        let component = 2 * (layout.dh2_offset + index);
        if component < data.optimization_flags.len() {
            data.optimization_flags[component] = flag;
            data.optimization_flags[component + 1] = data.doublon_holon_2site_complex && flag;
        }
    }
    for (index, &flag) in data.doublon_holon_4site_opt_flags.iter().enumerate() {
        let component = 2 * (layout.dh4_offset + index);
        if component < data.optimization_flags.len() {
            data.optimization_flags[component] = flag;
            data.optimization_flags[component + 1] = data.doublon_holon_4site_complex && flag;
        }
    }
}

/// Component index for the real part of a zero-based Slater parameter.
pub fn get_slater_opt_flag_index(data: &ExpertModeData, slater_idx: usize) -> usize {
    2 * (data.projection_layout().n_proj + slater_idx)
}

/// Whether the Slater real component is active; missing entries return false.
pub fn is_slater_optimized(data: &ExpertModeData, slater_idx: usize) -> bool {
    data.optimization_flags
        .get(get_slater_opt_flag_index(data, slater_idx))
        .copied()
        .unwrap_or(false)
}

/// Whether the Gutzwiller real component is active; missing entries return false.
pub fn is_gutzwiller_optimized(data: &ExpertModeData, idx: usize) -> bool {
    data.optimization_flags
        .get(2 * idx)
        .copied()
        .unwrap_or(false)
}

/// Whether the Jastrow real component is active; missing entries return false.
pub fn is_jastrow_optimized(data: &ExpertModeData, idx: usize) -> bool {
    data.optimization_flags
        .get(2 * (data.projection_layout().n_gutzwiller + idx))
        .copied()
        .unwrap_or(false)
}
