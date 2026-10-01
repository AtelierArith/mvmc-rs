//! Canonical RBM index maps and permissive section splitting.

use crate::types::*;
use crate::utils::file::{safe_parse_int, split_def_line};
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

/// Canonical order of the nine RBM coefficient blocks.
pub const SECTION_NAMES: [&str; 9] = [
    "ChargeRBM_PhysLayer",
    "SpinRBM_PhysLayer",
    "GeneralRBM_PhysLayer",
    "ChargeRBM_HiddenLayer",
    "SpinRBM_HiddenLayer",
    "GeneralRBM_HiddenLayer",
    "ChargeRBM_PhysHidden",
    "SpinRBM_PhysHidden",
    "GeneralRBM_PhysHidden",
];

/// Extended result retaining header, flags and diagnostics even after a mapping error.
#[derive(Debug, Clone, PartialEq)]
pub struct RbmParseResult<T> {
    /// Whether all collected mapping rows parsed.
    pub success: bool,
    /// Atomic mapping result; absent on any mapping error.
    pub terms: Option<Vec<T>>,
    /// Declared width, or inferred maximum index plus one if nonpositive.
    pub n_rbm_idx: i64,
    /// Last listed nonnegative optimization flag for each index.
    pub opt_flags: BTreeMap<i64, i64>,
    /// Nonzero header ComplexType, including empty sections.
    pub is_complex_flag: bool,
    /// Joined mapping errors.
    pub error_message: String,
    /// Last collected mapping row's one-based line number; zero if none.
    pub line_number: usize,
}

fn parse<T: RbmParameter>(
    content: &str,
    context: &str,
    map_cols: usize,
    invalid: &str,
    make: impl Fn(&[i64], bool) -> T,
) -> RbmParseResult<T> {
    let lines: Vec<_> = content.split('\n').collect();
    let (mut n_rbm_idx, mut complex, mut start) = (0, false, 0);
    if lines.len() >= 3 {
        let second = split_def_line(lines[1]);
        let third = split_def_line(lines[2]);
        if second.len() >= 2 && third.len() >= 2 && third[0] == "ComplexType" {
            n_rbm_idx = safe_parse_int(second[1], 0);
            complex = safe_parse_int(third[1], 0) != 0;
            start = 5.min(lines.len());
        }
    }
    let mut in_flags = false;
    let mut seen = BTreeSet::new();
    let mut terms = Vec::new();
    let mut flags = BTreeMap::new();
    let mut errors = Vec::new();
    let mut line_number = 0;
    for (line, raw) in lines.iter().enumerate().skip(start) {
        let tokens = split_def_line(raw);
        if map_cols == 2 && !in_flags && tokens.len() == 2 {
            in_flags = !seen.insert(safe_parse_int(tokens[0], i64::MIN));
        } else if map_cols != 2 && !in_flags && tokens.len() == 2 {
            in_flags = true;
        }
        if in_flags {
            if tokens.len() >= 2 {
                let idx = safe_parse_int(tokens[0], -1);
                let flag = safe_parse_int(tokens[1], -1);
                if idx >= 0 && flag >= 0 {
                    flags.insert(idx, flag);
                }
            }
        } else if tokens.len() == map_cols {
            line_number = line + 1;
            let values: Vec<_> = tokens.iter().map(|t| safe_parse_int(t, -1)).collect();
            if values.iter().any(|&v| v < 0) {
                errors.push(format!("Line {line_number}: Error parsing {context} term: ErrorException(\"{invalid}\")"));
            } else {
                terms.push(make(&values, complex));
            }
        }
    }
    if n_rbm_idx <= 0 {
        if let Some(idx) = terms.iter().map(RbmParameter::idx).max() {
            n_rbm_idx = idx.wrapping_add(1);
        }
    }
    RbmParseResult {
        success: errors.is_empty(),
        terms: errors.is_empty().then_some(terms),
        n_rbm_idx,
        opt_flags: flags,
        is_complex_flag: complex,
        error_message: errors.join("; "),
        line_number,
    }
}

/// Parse ChargeRBM_PhysLayer, preserving Julia's extended diagnostics.
pub fn parse_charge_rbm_phys_layer_content(
    content: &str,
) -> RbmParseResult<ChargeRBMPhysLayerTerm> {
    parse(
        content,
        "ChargeRBM_PhysLayer",
        2,
        "invalid site/idx",
        |v, is_complex| ChargeRBMPhysLayerTerm {
            site: v[0],
            idx: v[1],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse SpinRBM_PhysLayer, preserving Julia's extended diagnostics.
pub fn parse_spin_rbm_phys_layer_content(content: &str) -> RbmParseResult<SpinRBMPhysLayerTerm> {
    parse(
        content,
        "SpinRBM_PhysLayer",
        2,
        "invalid site/idx",
        |v, is_complex| SpinRBMPhysLayerTerm {
            site: v[0],
            idx: v[1],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse GeneralRBM_PhysLayer, preserving Julia's extended diagnostics.
pub fn parse_general_rbm_phys_layer_content(
    content: &str,
) -> RbmParseResult<GeneralRBMPhysLayerTerm> {
    parse(
        content,
        "GeneralRBM_PhysLayer",
        3,
        "invalid site/spin/idx",
        |v, is_complex| GeneralRBMPhysLayerTerm {
            site: v[0],
            spin: v[1],
            idx: v[2],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse ChargeRBM_HiddenLayer, preserving Julia's extended diagnostics.
pub fn parse_charge_rbm_hidden_layer_content(
    content: &str,
) -> RbmParseResult<ChargeRBMHiddenLayerTerm> {
    parse(
        content,
        "ChargeRBM_HiddenLayer",
        2,
        "invalid hidden/idx",
        |v, is_complex| ChargeRBMHiddenLayerTerm {
            site: v[0],
            idx: v[1],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse SpinRBM_HiddenLayer, preserving Julia's extended diagnostics.
pub fn parse_spin_rbm_hidden_layer_content(
    content: &str,
) -> RbmParseResult<SpinRBMHiddenLayerTerm> {
    parse(
        content,
        "SpinRBM_HiddenLayer",
        2,
        "invalid hidden/idx",
        |v, is_complex| SpinRBMHiddenLayerTerm {
            site: v[0],
            idx: v[1],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse GeneralRBM_HiddenLayer, preserving Julia's extended diagnostics.
pub fn parse_general_rbm_hidden_layer_content(
    content: &str,
) -> RbmParseResult<GeneralRBMHiddenLayerTerm> {
    parse(
        content,
        "GeneralRBM_HiddenLayer",
        2,
        "invalid hidden/idx",
        |v, is_complex| GeneralRBMHiddenLayerTerm {
            site: v[0],
            idx: v[1],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse ChargeRBM_PhysHidden, preserving Julia's extended diagnostics.
pub fn parse_charge_rbm_phys_hidden_content(
    content: &str,
) -> RbmParseResult<ChargeRBMPhysHiddenTerm> {
    parse(
        content,
        "ChargeRBM_PhysHidden",
        3,
        "invalid site/hidden/idx",
        |v, is_complex| ChargeRBMPhysHiddenTerm {
            site1: v[0],
            site2: v[1],
            idx: v[2],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse SpinRBM_PhysHidden, preserving Julia's extended diagnostics.
pub fn parse_spin_rbm_phys_hidden_content(content: &str) -> RbmParseResult<SpinRBMPhysHiddenTerm> {
    parse(
        content,
        "SpinRBM_PhysHidden",
        3,
        "invalid site/hidden/idx",
        |v, is_complex| SpinRBMPhysHiddenTerm {
            site1: v[0],
            site2: v[1],
            idx: v[2],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}

/// Parse GeneralRBM_PhysHidden, preserving Julia's extended diagnostics.
pub fn parse_general_rbm_phys_hidden_content(
    content: &str,
) -> RbmParseResult<GeneralRBMPhysHiddenTerm> {
    parse(
        content,
        "GeneralRBM_PhysHidden",
        4,
        "invalid site/spin/hidden/idx",
        |v, is_complex| GeneralRBMPhysHiddenTerm {
            site1: v[0],
            spin: v[1],
            site2: v[2],
            idx: v[3],
            value: Complex64::new(0.0, 0.0),
            is_complex,
        },
    )
}
