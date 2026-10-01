//! Atomic C-compatible DH2 neighbor definitions from `doublon_holon_parser.jl`.
use std::{io, path::Path};

use crate::types::{DoublonHolon2SiteDefinition, DoublonHolon2SiteIndex};
use crate::utils::file::{clean_line, read_def_file};

/// Original parser status, diagnostic and last body-line number.
/// Failed sections retain no partially parsed definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dh2ParseResult {
    /// Complete definition, or None on any parsing error.
    pub data: Option<DoublonHolon2SiteDefinition>,
    /// First error, matching the canonical parser text.
    pub error_message: String,
    /// Last neighbor/flag body line reached; header/count failures leave zero.
    pub line_number: usize,
}

impl Dh2ParseResult {
    /// Whether the entire definition is valid.
    pub fn is_success(&self) -> bool {
        self.data.is_some()
    }
}

/// Read a strict DH2 file, distinguishing filesystem errors from format errors.
pub fn parse_doublon_holon_2site_def(
    path: impl AsRef<Path>,
    nsite: i64,
) -> io::Result<Dh2ParseResult> {
    Ok(parse_doublon_holon_2site_content(
        &read_def_file(path)?,
        nsite,
    ))
}

fn integer(token: &str, number: usize, field: &str) -> Result<i64, String> {
    token
        .parse()
        .map_err(|_| format!("line {number}: invalid integer for {field}: '{token}'"))
}

fn tokens(line: &str) -> Vec<&str> {
    clean_line(line).split_whitespace().collect()
}

fn header_value(lines: &[&str], number: usize, field: &str) -> Result<i64, String> {
    let parts = tokens(lines[number - 1]);
    let token = parts
        .get(1)
        .ok_or_else(|| format!("line {number}: expected '{field} <int>'"))?;
    integer(token, number, field)
}

fn check_site(site: i64, nsite: i64, number: usize, field: &str) -> Result<(), String> {
    if !(0..nsite).contains(&site) {
        return Err(format!(
            "line {number}: {field} site {site} out of range [0, {}]",
            nsite - 1
        ));
    }
    Ok(())
}

/// Parse exactly Nsite*NDH2 neighbor rows followed by 6*NDH2 optimization rows.
///
/// Neighbor rows may be unordered; definitions are stored by index and center.
/// Flags use row order and only validate (then ignore) their first column.
/// Header labels are not validated, matching the original positional format.
pub fn parse_doublon_holon_2site_content(content: &str, nsite: i64) -> Dh2ParseResult {
    let mut line_number = 0;
    let parse = (|| {
        if nsite <= 0 {
            return Err("Nsite must be positive before parsing DH2".into());
        }
        let lines: Vec<_> = content.split('\n').collect();
        if lines.len() < 5 {
            return Err("DH2 file must include 5 header lines".into());
        }
        let n_dh2 = header_value(&lines, 2, "NDoublonHolon2siteIdx")?;
        let complex_type = header_value(&lines, 3, "ComplexType")?;
        if n_dh2 < 0 {
            return Err("NDoublonHolon2siteIdx must be non-negative".into());
        }
        let rows: Vec<_> = lines
            .iter()
            .enumerate()
            .skip(5)
            .filter_map(|(i, line)| {
                let parts = tokens(line);
                (!parts.is_empty()).then_some((i + 1, parts))
            })
            .collect();
        // Julia Int arithmetic wraps; reject row-count mismatches before allocation.
        let expected_main = nsite.wrapping_mul(n_dh2);
        let expected_opt = 6_i64.wrapping_mul(n_dh2);
        let expected_total = expected_main.wrapping_add(expected_opt);
        if rows.len() as i64 != expected_total {
            return Err(format!(
                "DH2 row count mismatch: got {}, expected {expected_total} ({expected_main} neighbor rows + {expected_opt} opt rows)", rows.len()
            ));
        }
        let mut indices: Vec<_> = (0..n_dh2)
            .map(|_| DoublonHolon2SiteIndex {
                neighbors: vec![[-1; 2]; nsite as usize],
            })
            .collect();
        // Column-major (definition,site), matching Julia's seen matrix.
        let mut seen = vec![false; expected_main as usize];
        for (number, parts) in rows.iter().take(expected_main as usize) {
            line_number = *number;
            if parts.len() != 4 {
                return Err(format!("line {number}: expected 'i x0 x1 n'"));
            }
            let site = integer(parts[0], *number, "center site")?;
            let x0 = integer(parts[1], *number, "neighbor 0")?;
            let x1 = integer(parts[2], *number, "neighbor 1")?;
            let idx = integer(parts[3], *number, "DH2 index")?;
            check_site(site, nsite, *number, "center")?;
            check_site(x0, nsite, *number, "neighbor 0")?;
            check_site(x1, nsite, *number, "neighbor 1")?;
            if !(0..n_dh2).contains(&idx) {
                return Err(format!(
                    "line {number}: DH2 index {idx} out of range [0, {}]",
                    n_dh2 - 1
                ));
            }
            let seen_index = idx as usize + n_dh2 as usize * site as usize;
            if seen[seen_index] {
                return Err(format!(
                    "line {number}: duplicate DH2 row for index {idx} site {site}"
                ));
            }
            seen[seen_index] = true;
            indices[idx as usize].neighbors[site as usize] = [x0, x1];
        }
        // Julia findfirst walks the (definition,site) matrix in column-major order.
        if let Some(missing) = seen.iter().position(|&present| !present) {
            let idx = missing % n_dh2 as usize;
            let site = missing / n_dh2 as usize;
            return Err(format!(
                "missing DH2 neighbor row for index {idx} site {site}"
            ));
        }
        let mut opt_flags = Vec::with_capacity(expected_opt as usize);
        for (number, parts) in rows.iter().skip(expected_main as usize) {
            line_number = *number;
            if parts.len() != 2 {
                return Err(format!(
                    "line {number}: expected 'local_param_index opt_flag'"
                ));
            }
            integer(parts[0], *number, "ignored local parameter index")?;
            let flag = integer(parts[1], *number, "opt flag")?;
            if flag != 0 && flag != 1 {
                return Err(format!(
                    "line {number}: opt flag must be 0 or 1, got {flag}"
                ));
            }
            opt_flags.push(flag != 0);
        }
        Ok(DoublonHolon2SiteDefinition {
            indices,
            opt_flags,
            is_complex: complex_type != 0,
        })
    })();
    match parse {
        Ok(data) => Dh2ParseResult {
            data: Some(data),
            error_message: String::new(),
            line_number,
        },
        Err(error_message) => Dh2ParseResult {
            data: None,
            error_message,
            line_number,
        },
    }
}
