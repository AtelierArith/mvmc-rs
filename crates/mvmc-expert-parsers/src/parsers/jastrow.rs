//! `jastrowidx.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/jastrow_parser.jl`.
//! Restricted to the round-trip subset: builds the `(site1, site2, idx)`
//! list, parses the `NJastrowIdx` / `ComplexType` header, and exposes
//! the symmetric `jastrow_idx[ri+1, rj+1]` matrix.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::types::JastrowTerm;
use crate::utils::file::{read_def_file, safe_parse_int, split_def_line};

/// Output of [`parse_jastrow_content`].
#[derive(Debug, Clone, Default)]
pub struct JastrowSection {
    /// `(site1, site2, idx)` terms.
    pub terms: Vec<JastrowTerm>,
    /// `NJastrowIdx`.
    pub n_jastrow_idx: i64,
    /// `ComplexType` flag.
    pub is_complex: bool,
    /// Per-parameter flags from the trailing OptFlag section.
    pub opt_flags: BTreeMap<i64, i64>,
}

/// Parse a `jastrowidx.def` file from disk.
pub fn parse_jastrow_def<P: AsRef<Path>>(path: P) -> io::Result<JastrowSection> {
    let content = read_def_file(path)?;
    Ok(parse_jastrow_content(&content))
}

/// Parse a `jastrowidx.def` payload from memory.
pub fn parse_jastrow_content(content: &str) -> JastrowSection {
    let lines: Vec<&str> = content.lines().collect();
    let (n_jastrow_idx, is_complex, has_header) = read_idx_header(&lines, "NJastrowIdx");

    let start_line = if has_header { 5 } else { 0 };
    let mut terms = Vec::new();
    let mut in_opt_section = false;
    let mut opt_flags = BTreeMap::new();
    for line in lines.get(start_line..).unwrap_or_default() {
        let tokens = split_def_line(line);
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() == 2 && !terms.is_empty() {
            in_opt_section = true;
        }
        if in_opt_section {
            if tokens.len() >= 2 {
                let idx = safe_parse_int(tokens[0], -1);
                let flag = safe_parse_int(tokens[1], -1);
                if idx >= 0 && flag >= 0 {
                    opt_flags.insert(idx, flag);
                }
            }
            continue;
        }
        if tokens.len() < 3 {
            continue;
        }
        let site1 = safe_parse_int(tokens[0], -1);
        let site2 = safe_parse_int(tokens[1], -1);
        if site1 < 0 || site2 < 0 {
            continue;
        }
        let _idx = safe_parse_int(tokens[2], -1); // recorded in JastrowTerm.value later
        terms.push(JastrowTerm {
            site1,
            site2,
            value: Complex64::new(0.0, 0.0),
            is_complex,
        });
    }

    JastrowSection {
        terms: if n_jastrow_idx > 0 && (terms.len() as i64) > n_jastrow_idx {
            terms.truncate(n_jastrow_idx as usize);
            terms
        } else {
            terms
        },
        n_jastrow_idx,
        is_complex,
        opt_flags,
    }
}

/// Build the symmetric `JastrowIdx[ri+1, rj+1] = idx` matrix from a
/// `jastrowidx.def` payload, in the same shape the upstream Julia
/// helper `build_jastrow_idx_matrix` returns: `Vec<Vec<i64>>` with the
/// outer dimension `nsite` and inner dimension `nsite`. Unset entries
/// stay at `-1`.
pub fn build_jastrow_idx_matrix(content: &str, nsite: usize) -> (Vec<Vec<i64>>, i64) {
    if nsite == 0 {
        return (Vec::new(), 0);
    }
    let mut matrix = vec![vec![-1i64; nsite]; nsite];
    let lines: Vec<&str> = content.lines().collect();
    let (n_jastrow_idx, _is_complex, _has_header) = read_idx_header(&lines, "NJastrowIdx");

    let start_line = 5usize;
    for line in lines.iter().skip(start_line) {
        let tokens = split_def_line(line);
        if tokens.len() == 2 {
            break; // entered the opt-flag block
        }
        if tokens.len() < 3 {
            continue;
        }
        let site1 = safe_parse_int(tokens[0], -1);
        let site2 = safe_parse_int(tokens[1], -1);
        let idx = safe_parse_int(tokens[2], -1);
        if site1 >= 0
            && (site1 as usize) < nsite
            && site2 >= 0
            && (site2 as usize) < nsite
            && idx >= 0
        {
            matrix[site1 as usize][site2 as usize] = idx;
            matrix[site2 as usize][site1 as usize] = idx;
        }
    }
    (matrix, n_jastrow_idx)
}

// Legacy Jastrow header reader; the Gutzwiller reader uses physical C headers.
fn read_idx_header(lines: &[&str], n_keyword: &str) -> (i64, bool, bool) {
    let mut n_idx = 0i64;
    let mut complex_type = 0i64;
    let mut has_header = false;

    if lines.len() > 1 {
        let tokens = split_def_line(lines[1]);
        if tokens.len() >= 2
            && (tokens[0] == n_keyword
                || (n_keyword == "NOrbitalIdx" && tokens[0].starts_with("NOrbital")))
        {
            n_idx = safe_parse_int(tokens[1], 0);
            has_header = true;
        }
    }
    if has_header && lines.len() > 2 {
        let tokens = split_def_line(lines[2]);
        if tokens.len() >= 2 && tokens[0] == "ComplexType" {
            complex_type = safe_parse_int(tokens[1], 0);
        }
    }
    (n_idx, complex_type != 0, has_header)
}
