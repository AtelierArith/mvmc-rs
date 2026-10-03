//! Julia-compatible optional parameter overlays, applied in namelist order.
use super::file::{
    clean_line, julia_parse_float, julia_parse_int, parse_namelist_content, read_def_file,
    safe_parse_float, safe_parse_int, split_def_line,
};
use super::parameter_init::n_slater;
use crate::types::ExpertModeData;
use num_complex::Complex64;
use std::{
    collections::{BTreeMap, HashSet},
    io,
    path::Path,
};

/// Parse a permissive In*.def record, retaining Julia's numeric fallbacks.
/// Missing files produce an empty map; duplicate indices keep the last value.
pub fn parse_input_parameter_file(path: impl AsRef<Path>) -> io::Result<BTreeMap<i64, Complex64>> {
    let path = path.as_ref();
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    let content = read_def_file(path)?;
    let lines: Vec<_> = content.split('\n').collect();
    let mut params = BTreeMap::new();
    if lines.len() < 6 {
        return Ok(params);
    }
    let header = split_def_line(lines[1]);
    let expected = header
        .get(1)
        .map(|token| safe_parse_int(token, 0))
        .unwrap_or(0);
    for line in &lines[5..] {
        let tokens = split_def_line(line);
        if tokens.len() >= 3 {
            let index = safe_parse_int(tokens[0], -1);
            if index >= 0 {
                params.insert(
                    index,
                    Complex64::new(
                        safe_parse_float(tokens[1], 0.0),
                        safe_parse_float(tokens[2], 0.0),
                    ),
                );
            }
        }
    }
    if expected > 0 && params.len() != expected as usize {
        eprintln!(
            "warning: Parameter count mismatch in {}: expected {expected}, got {}",
            path.display(),
            params.len()
        );
    }
    Ok(params)
}

/// Validate a complete indexed overlay before returning any parameter values.
/// Header counts, indices, duplicates, row width and finite values are strict.
pub fn parse_indexed_input_parameter_file_strict(
    path: impl AsRef<Path>,
    expected_header_count: usize,
    expected_param_count: usize,
    label: &str,
) -> Result<Vec<Complex64>, String> {
    let path = path.as_ref();
    let content = read_def_file(path).map_err(|e| e.to_string())?;
    let lines: Vec<_> = content.split('\n').collect();
    let location = path.display();
    if lines.len() < 5 {
        return Err(format!("{label}: {location} must include 5 header lines"));
    }
    let header = split_def_line(lines[1]);
    let count = header
        .get(1)
        .ok_or_else(|| format!("{label}: {location}:2 missing count header"))?;
    let count =
        julia_parse_int(count).ok_or_else(|| format!("{location}:2: invalid count '{count}'"))?;
    if count < 0 || count as usize != expected_header_count {
        return Err(format!("{label}: header count mismatch in {location}: got {count}, expected {expected_header_count}"));
    }
    let mut params = vec![Complex64::new(0.0, 0.0); expected_param_count];
    let mut seen = vec![false; expected_param_count];
    let mut rows = 0;
    for (line_index, line) in lines.iter().enumerate().skip(5) {
        if clean_line(line).is_empty() {
            continue;
        }
        let line_number = line_index + 1;
        let tokens = split_def_line(line);
        if tokens.len() != 3 {
            return Err(format!(
                "{label}: {location}:{line_number} expected 'idx real imag'"
            ));
        }
        rows += 1;
        let index = julia_parse_int(tokens[0])
            .ok_or_else(|| format!("{location}:{line_number}: invalid index '{}'", tokens[0]))?;
        if index < 0 || index as usize >= expected_param_count {
            return Err(format!(
                "{label}: index {index} out of range [0, {}] in {location}:{line_number}",
                expected_param_count as i64 - 1
            ));
        }
        let index = index as usize;
        if seen[index] {
            return Err(format!(
                "{label}: duplicated index {index} in {location}:{line_number}"
            ));
        }
        let parse_float = |token: &str, field: &str| -> Result<f64, String> {
            let value = julia_parse_float(token)
                .ok_or_else(|| format!("{location}:{line_number}: invalid {field} '{token}'"))?;
            if !value.is_finite() {
                return Err(format!(
                    "{location}:{line_number}: non-finite {field} '{token}'"
                ));
            }
            Ok(value)
        };
        params[index] = Complex64::new(
            parse_float(tokens[1], "real value")?,
            parse_float(tokens[2], "imag value")?,
        );
        seen[index] = true;
    }
    if rows != expected_param_count {
        return Err(format!("{label}: row count mismatch in {location}: got {rows}, expected {expected_param_count}"));
    }
    Ok(params)
}

/// Read optional overlays after initial.def and before synchronization.
/// Missing referenced files are skipped; applied records follow namelist order.
/// Supported factors are Gutzwiller, Jastrow and normal/AP/Parallel/General
/// orbitals, strict DH2/DH4/OptTrans, and the nine indexed RBM sections.
/// Each strict overlay commits atomically;
/// earlier successful overlays remain applied if a later record fails.
pub fn read_input_parameters(
    data: &mut ExpertModeData,
    namelist_path: impl AsRef<Path>,
) -> Result<(), String> {
    let namelist_path = namelist_path.as_ref();
    let base_dir = namelist_path.parent().unwrap_or_else(|| Path::new("."));
    let content = read_def_file(namelist_path).map_err(|error| error.to_string())?;
    let mut overlays = Vec::new();
    let mut seen = HashSet::new();
    for (kind, filename) in parse_namelist_content(&content) {
        let kind = crate::canonical_namelist_keyword(&kind)
            .unwrap_or(&kind)
            .to_owned();
        if !kind.starts_with("In") {
            continue;
        }
        if overlay_order(&kind).is_some() && !seen.insert(kind.clone()) {
            return Err(format!(
                "duplicate keyword {kind} in {}",
                namelist_path.display()
            ));
        }
        overlays.push((overlay_order(&kind).unwrap_or(usize::MAX), kind, filename));
    }
    overlays.sort_by_key(|(order, _, _)| *order);
    for (_, kind, filename) in overlays {
        let path = base_dir.join(filename);
        if !path.is_file() {
            continue;
        }
        match kind.as_str() {
            "InGutzwiller" | "InJastrow" => {
                let params =
                    parse_input_parameter_file(&path).map_err(|error| error.to_string())?;
                for (index, value) in params {
                    if kind == "InGutzwiller" {
                        if let Some(term) = data.gutzwiller_terms.get_mut(index as usize) {
                            term.value = value;
                        }
                    } else if let Some(term) = data.jastrow_terms.get_mut(index as usize) {
                        term.value = value;
                    }
                }
            }
            "InOrbital" | "InOrbitalAntiParallel" | "InOrbitalGeneral" => {
                let params =
                    parse_input_parameter_file(&path).map_err(|error| error.to_string())?;
                for (index, value) in params {
                    if let Some(slot) = data.slater_params.get_mut(index as usize) {
                        *slot = value;
                    }
                }
            }
            "InOrbitalParallel" => {
                let count = n_slater(data);
                let offset = if data.i_flg_orbital_parallel != 1 {
                    count as i64
                } else if data.i_flg_orbital_anti_parallel != 1 {
                    0
                } else {
                    data.n_orbital_anti_parallel
                };
                let expected = count as i64 - offset;
                if expected <= 0 {
                    return Err(format!("InOrbitalParallel target parameter length mismatch: got {expected} from NOrbitalIdx={count} and offset={offset}"));
                }
                let params = parse_indexed_input_parameter_file_strict(
                    &path,
                    expected as usize,
                    expected as usize,
                    "InOrbitalParallel",
                )?;
                data.slater_params[offset as usize..count].copy_from_slice(&params);
            }
            "InDH2" => {
                let layout = data.projection_layout();
                let expected = 6 * layout.n_dh2;
                if data.doublon_holon_2site_params.len() != expected {
                    return Err(format!(
                        "InDH2 target parameter length mismatch: got {}, expected {expected}",
                        data.doublon_holon_2site_params.len()
                    ));
                }
                let params = parse_indexed_input_parameter_file_strict(
                    &path,
                    layout.n_dh2,
                    expected,
                    "InDH2",
                )?;
                data.doublon_holon_2site_params.copy_from_slice(&params);
            }
            "InDH4" => {
                let layout = data.projection_layout();
                let expected = 10 * layout.n_dh4;
                if data.doublon_holon_4site_params.len() != expected {
                    return Err(format!(
                        "InDH4 target parameter length mismatch: got {}, expected {expected}",
                        data.doublon_holon_4site_params.len()
                    ));
                }
                let params = parse_indexed_input_parameter_file_strict(
                    &path,
                    layout.n_dh4,
                    expected,
                    "InDH4",
                )?;
                data.doublon_holon_4site_params.copy_from_slice(&params);
            }
            "InOptTrans" => {
                let expected = data.count_opt_trans_parameters();
                if expected == 0 {
                    return Err(
                        "InOptTrans target parameter length mismatch: OptTrans is not active"
                            .into(),
                    );
                }
                let params = parse_indexed_input_parameter_file_strict(
                    &path,
                    expected,
                    expected,
                    "InOptTrans",
                )?;
                data.opt_trans.copy_from_slice(&params);
            }
            kind if kind.starts_with("InChargeRBM_")
                || kind.starts_with("InSpinRBM_")
                || kind.starts_with("InGeneralRBM_") =>
            {
                let params =
                    parse_input_parameter_file(&path).map_err(|error| error.to_string())?;
                if let Some(section) = crate::parsers::rbm::SECTION_NAMES
                    .iter()
                    .position(|&name| kind.strip_prefix("In") == Some(name))
                {
                    let widths = data.rbm_section_sizes();
                    let offset: usize = widths[..section].iter().sum();
                    for (&index, &value) in &params {
                        if index >= 0 && (index as usize) < widths[section] {
                            data.set_rbm_parameter(offset + index as usize, value);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// C's `GetFileName` stores each keyword in a fixed slot and the reader then
/// visits those slots in keyword order, independently of their textual order
/// in `namelist.def`. Keep the overlay phase on the same order.
fn overlay_order(kind: &str) -> Option<usize> {
    const ORDER: &[&str] = &[
        "InGutzwiller",
        "InJastrow",
        "InDH2",
        "InDH4",
        "InChargeRBM_HiddenLayer",
        "InChargeRBM_PhysLayer",
        "InChargeRBM_PhysHidden",
        "InSpinRBM_HiddenLayer",
        "InSpinRBM_PhysLayer",
        "InSpinRBM_PhysHidden",
        "InGeneralRBM_HiddenLayer",
        "InGeneralRBM_PhysLayer",
        "InGeneralRBM_PhysHidden",
        "InOrbital",
        "InOrbitalAntiParallel",
        "InOrbitalParallel",
        "InOrbitalGeneral",
        "InOptTrans",
    ];
    ORDER.iter().position(|candidate| *candidate == kind)
}
