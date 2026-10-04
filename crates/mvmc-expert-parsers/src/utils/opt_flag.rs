//! C integer real/imaginary optimization flag layout.
//! Parameter indices passed to the Rust accessors are zero-based.

use super::parameter_init::n_slater;
use crate::types::ExpertModeData;
use std::collections::BTreeMap;
use std::io::{self, Write};

pub(crate) fn raw_orbital_complex_header(data: &ExpertModeData) -> Option<i64> {
    let headers = [
        "Orbital",
        "OrbitalAntiParallel",
        "OrbitalParallel",
        "OrbitalGeneral",
    ]
    .into_iter()
    .filter_map(|key| data.native_complex_headers.get(key));
    headers.fold(None, |sum, &header| {
        Some(sum.unwrap_or(0) + i64::from(header))
    })
}

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
    // readdef.c: only a positive sum is normalized to one; negatives remain raw.
    let raw = raw_orbital_complex_header(data)
        .unwrap_or_else(|| i64::from(data.orbital_terms.iter().any(|term| term.is_complex)));
    let complex = if raw > 0 { 1 } else { raw };
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
                complex
            } else if complex > 0 {
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
            data.optimization_flags[component + 1] = if data
                .native_complex_headers
                .get("DH2")
                .map_or(data.doublon_holon_2site_complex, |&header| header > 0)
            {
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
            data.optimization_flags[component + 1] = if data
                .native_complex_headers
                .get("DH4")
                .map_or(data.doublon_holon_4site_complex, |&header| header > 0)
            {
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

/// Write real-component optimization status without changing parameters or flags.
///
/// Dense declared slots and Slater mapping rows are counted separately. G/J term
/// ordinals identify coefficient slots; their sites are representative metadata,
/// not exhaustive spatial mappings. Slater rows use their actual `idx`.
/// All indices in the output are zero-based. Missing flags are fixed; only the
/// raw real-component flag exactly equal to one is optimized.
///
/// # Errors
///
/// Report-local invalid indices, excess representative rows, negative Slater
/// width or arithmetic overflow return [`io::ErrorKind::InvalidInput`] before
/// writing. These checks do not alter loader/pack acceptance. Writer errors
/// propagate; output may be partial on IO failure, but data remains unchanged.
pub fn write_optimization_status<W: Write + ?Sized>(
    data: &ExpertModeData,
    writer: &mut W,
) -> io::Result<()> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid optimization report layout",
        )
    };
    let width = |declared: i64, fallback: usize| {
        if declared > 0 {
            usize::try_from(declared).map_err(|_| invalid())
        } else {
            Ok(fallback)
        }
    };
    // Do not call unchecked projection_layout/count_rbm/offset helpers here.
    let g = width(data.n_gutzwiller_idx, data.gutzwiller_terms.len())?;
    let j = width(data.n_jastrow_idx, data.jastrow_terms.len())?;
    if data.gutzwiller_terms.len() > g || data.jastrow_terms.len() > j {
        return Err(invalid());
    }
    let projection = g
        .checked_add(j)
        .and_then(|n| {
            data.doublon_holon_2site_indices
                .len()
                .checked_mul(6)
                .and_then(|dh| n.checked_add(dh))
        })
        .and_then(|n| {
            data.doublon_holon_4site_indices
                .len()
                .checked_mul(10)
                .and_then(|dh| n.checked_add(dh))
        })
        .ok_or_else(invalid)?;
    let rbm = data
        .rbm_section_widths
        .iter()
        .try_fold(0usize, |n, &w| n.checked_add(w))
        .ok_or_else(invalid)?;
    let slater_offset = projection.checked_add(rbm).ok_or_else(invalid)?;
    let slater = usize::try_from(data.modpara.n_orbital_idx).map_err(|_| invalid())?;
    slater_offset
        .checked_add(slater)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(invalid)?;
    for term in &data.orbital_terms {
        if usize::try_from(term.idx).map_or(true, |idx| idx >= slater) {
            return Err(invalid());
        }
    }
    let optimized =
        |offset: usize, idx: usize| data.optimization_flags.get(2 * (offset + idx)) == Some(&1);
    let status = |yes| if yes { "optimized" } else { "fixed" };
    writeln!(
        writer,
        "Optimization status (real component; optimized iff flag == 1)"
    )?;
    writeln!(writer, "Gutzwiller parameters={g}")?;
    for idx in 0..g {
        writeln!(writer, "  parameter={idx} {}", status(optimized(0, idx)))?;
    }
    writeln!(
        writer,
        "Gutzwiller representative_rows={} without_metadata={}",
        data.gutzwiller_terms.len(),
        g - data.gutzwiller_terms.len()
    )?;
    for (row, term) in data.gutzwiller_terms.iter().enumerate() {
        writeln!(
            writer,
            "  row={row} parameter={row} site={} {}",
            term.site,
            status(optimized(0, row))
        )?;
    }
    writeln!(writer, "Jastrow parameters={j}")?;
    for idx in 0..j {
        writeln!(writer, "  parameter={idx} {}", status(optimized(g, idx)))?;
    }
    writeln!(
        writer,
        "Jastrow representative_rows={} without_metadata={}",
        data.jastrow_terms.len(),
        j - data.jastrow_terms.len()
    )?;
    for (row, term) in data.jastrow_terms.iter().enumerate() {
        writeln!(
            writer,
            "  row={row} parameter={row} sites={},{} {}",
            term.site1,
            term.site2,
            status(optimized(g, row))
        )?;
    }
    let active = (0..slater)
        .filter(|&idx| optimized(slater_offset, idx))
        .count();
    writeln!(
        writer,
        "Slater parameters={slater} optimized={active} fixed={}",
        slater - active
    )?;
    for idx in 0..slater {
        if slater > 10 && (5..slater - 5).contains(&idx) {
            if idx == 5 {
                writeln!(writer, "  ...")?;
            }
            continue;
        }
        writeln!(
            writer,
            "  parameter={idx} {}",
            status(optimized(slater_offset, idx))
        )?;
    }
    let rows = data.orbital_terms.len();
    writeln!(writer, "Slater mapping_rows={rows}")?;
    for (row, term) in data.orbital_terms.iter().enumerate() {
        if rows > 10 && (5..rows - 5).contains(&row) {
            if row == 5 {
                writeln!(writer, "  ...")?;
            }
            continue;
        }
        writeln!(
            writer,
            "  row={row} parameter={} sites={},{} sign={} {}",
            term.idx,
            term.site1,
            term.site2,
            term.sign,
            status(optimized(slater_offset, term.idx as usize))
        )?;
    }
    Ok(())
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
