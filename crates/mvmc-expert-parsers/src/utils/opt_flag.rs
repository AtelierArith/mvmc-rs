//! C integer real/imaginary optimization flag layout.
//! Parameter indices passed to the Rust accessors are zero-based.

use super::parameter_init::n_slater;
use crate::types::ExpertModeData;
use std::collections::BTreeMap;

/// Extend the component array with integer 1 defaults without changing existing flags.
pub fn ensure_optimization_flags_size(data: &mut ExpertModeData, n_components: usize) {
    if data.optimization_flags.len() < n_components {
        data.optimization_flags.resize(n_components, 1);
    }
}

/// Apply the explicitly listed Gutzwiller/Jastrow flags in the global component layout.
/// Noncomplex projection parameters have zero imaginary flags when listed;
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
                data.optimization_flags[component] = flag;
                data.optimization_flags[component + 1] = if complex { flag } else { 0 };
            }
        }
    }
}

/// Apply orbital flags once every factor's parameter count is known.
/// AP/General complex imaginary flags copy the raw real flag; P imaginary
/// flags receive the normalized orbital complex header independently. Unwritten
/// real-mode imaginary entries are deterministic zero, not allocator contents.
pub fn set_orbital_opt_flags(data: &mut ExpertModeData, flags: &BTreeMap<i64, i64>) {
    if flags.is_empty() {
        return;
    }
    let n_proj = data.projection_layout().n_proj + data.count_rbm_parameters();
    // C collapses the sum of orbital headers to 1 before all orbital readers.
    let complex = data.orbital_terms.iter().any(|term| term.is_complex);
    ensure_optimization_flags_size(
        data,
        2 * (n_proj + n_slater(data) + data.count_opt_trans_parameters()),
    );
    for (&idx, &flag) in flags {
        let Ok(idx) = usize::try_from(idx) else {
            continue;
        };
        let component = 2 * (n_proj + idx);
        if component < data.optimization_flags.len() {
            data.optimization_flags[component] = flag;
            data.optimization_flags[component + 1] = if data.i_flg_orbital_parallel == 1
                && idx >= data.n_orbital_anti_parallel as usize
            {
                i64::from(complex)
            } else if complex {
                flag
            } else {
                0
            };
        }
    }
}

/// Fold DH2/DH4 row-ordered flags into the final projection layout.
/// Real DH parameters always have zero imaginary flags; defaults and flags
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
            data.optimization_flags[component + 1] = if data.doublon_holon_2site_complex {
                flag
            } else {
                0
            };
        }
    }
    for (index, &flag) in data.doublon_holon_4site_opt_flags.iter().enumerate() {
        let component = 2 * (layout.dh4_offset + index);
        if component < data.optimization_flags.len() {
            data.optimization_flags[component] = flag;
            data.optimization_flags[component + 1] = if data.doublon_holon_4site_complex {
                flag
            } else {
                0
            };
        }
    }
}

/// Component index for the real part of a zero-based Slater parameter.
pub fn get_slater_opt_flag_index(data: &ExpertModeData, slater_idx: usize) -> usize {
    2 * (data.projection_layout().n_proj + data.count_rbm_parameters() + slater_idx)
}

/// Whether the Slater real component is eligible for SR (exactly 1); missing entries return false.
pub fn is_slater_optimized(data: &ExpertModeData, slater_idx: usize) -> bool {
    data.optimization_flags
        .get(get_slater_opt_flag_index(data, slater_idx))
        .copied()
        .unwrap_or(0)
        == 1
}

/// Whether the Gutzwiller real component is eligible for SR (exactly 1); missing entries return false.
pub fn is_gutzwiller_optimized(data: &ExpertModeData, idx: usize) -> bool {
    data.optimization_flags.get(2 * idx).copied().unwrap_or(0) == 1
}

/// Whether the Jastrow real component is eligible for SR (exactly 1); missing entries return false.
pub fn is_jastrow_optimized(data: &ExpertModeData, idx: usize) -> bool {
    data.optimization_flags
        .get(2 * (data.projection_layout().n_gutzwiller + idx))
        .copied()
        .unwrap_or(0)
        == 1
}

/// Apply RBM flags at a global parameter offset. Listed indices may cross
/// section boundaries as in Julia; only the final component array bounds apply.
pub fn set_rbm_opt_flags(
    data: &mut ExpertModeData,
    flags: &BTreeMap<i64, i64>,
    offset: usize,
    is_complex: bool,
) {
    if flags.is_empty() {
        return;
    }
    let n_para = data.projection_layout().n_proj
        + data.count_rbm_parameters()
        + n_slater(data)
        + data.count_opt_trans_parameters();
    ensure_optimization_flags_size(data, 2 * n_para);
    for (&idx, &flag) in flags {
        let Ok(idx) = usize::try_from(idx) else {
            continue;
        };
        let Some(component) = offset.checked_add(idx).and_then(|p| p.checked_mul(2)) else {
            continue;
        };
        if component < data.optimization_flags.len() {
            data.optimization_flags[component] = flag;
            data.optimization_flags[component + 1] = if is_complex { flag } else { 0 };
        }
    }
}

/// Activate real OptTrans components after projection, RBM and Slater parameters.
/// Imaginary components are fixed even when current weights are complex.
pub fn set_opt_trans_opt_flags(data: &mut ExpertModeData) {
    let count = data.count_opt_trans_parameters();
    if count == 0 {
        return;
    }
    let offset = data.projection_layout().n_proj + data.count_rbm_parameters() + n_slater(data);
    ensure_optimization_flags_size(data, 2 * (offset + count));
    for idx in offset..offset + count {
        data.optimization_flags[2 * idx] = 1;
        data.optimization_flags[2 * idx + 1] = 0;
    }
}

/// Apply the native C `GetInfoOptTrans` writes.
pub fn set_opt_trans_c_opt_flags(data: &mut ExpertModeData) {
    let count = data.count_opt_trans_parameters();
    if count == 0 {
        return;
    }
    data.c_opt_trans_flags = true;
    let fidx = data.projection_layout().n_proj + n_slater(data);
    let required = 2
        * (data.projection_layout().n_proj + data.count_rbm_parameters() + n_slater(data) + count);
    if data.optimization_flags.len() < required {
        data.optimization_flags.resize(required, 0);
    }
    for index in fidx..fidx + count {
        if let Some(flag) = data.optimization_flags.get_mut(index) {
            *flag = 1;
        }
    }
}
