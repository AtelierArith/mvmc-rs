//! C AP/P declared widths and complete mapping/flag sections.
//! General's historical combined-coordinate reader remains scoped under #41.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use crate::types::OrbitalTerm;
use crate::utils::file::{read_def_file, safe_parse_int, split_def_line};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Orbital mapping geometry selected by the namelist keyword.
pub enum OrbitalKind {
    /// Ordered spatial pairs for opposite spins.
    AntiParallel,
    /// Upper-triangle spatial pairs, expanded into both equal-spin blocks.
    Parallel,
    /// General spin-site mappings; the six-column C port is tracked by #41.
    General,
}

/// Output of [`parse_orbital_content`].
#[derive(Debug, Clone, Default)]
pub struct OrbitalSection {
    /// `(site1, site2, idx, sign)` entries from the OrbitalIdx block.
    pub terms: Vec<OrbitalTerm>,
    /// `NOrbitalIdx` from the header (number of unique parameters).
    pub n_orbital_idx: i64,
    /// `ComplexType` flag.
    pub is_complex: bool,
    /// Per-parameter flags from the trailing OptFlag section.
    pub opt_flags: BTreeMap<i64, i64>,
}

/// Parse a `orbitalidx*.def` file from disk.
pub fn parse_orbital_def<P: AsRef<Path>>(
    path: P,
    nsite: i64,
    kind: OrbitalKind,
) -> io::Result<OrbitalSection> {
    let content = read_def_file(path)?;
    parse_orbital_content(&content, nsite, kind)
}

/// Parse a `orbitalidx*.def` payload from memory.
pub fn parse_orbital_content(
    content: &str,
    nsite: i64,
    kind: OrbitalKind,
) -> io::Result<OrbitalSection> {
    let lines: Vec<&str> = content.lines().collect();
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    if lines.len() < 5 {
        return Err(invalid("orbital definition requires five header lines"));
    }
    // ReadBuffIntCmpFlg reads physical lines 2/3, ignoring their labels.
    let n_orbital_idx = c_fields(lines[1])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .filter(|&width| width > 0)
        .ok_or_else(|| invalid("orbital declaration on line 2 must have a positive width"))?
        as i64;
    let is_complex = c_fields(lines[2])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .unwrap_or(0)
        > 0;
    if kind == OrbitalKind::General {
        return Ok(parse_general_rows(&lines[5..], n_orbital_idx, is_complex));
    }
    let nsite = usize::try_from(nsite)
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| invalid("AP/P orbital definitions require a positive Nsite"))?;
    if kind == OrbitalKind::Parallel && nsite < 2 {
        return Err(invalid(
            "parallel orbital definitions require at least two sites",
        ));
    }
    let n_mapping = if kind == OrbitalKind::Parallel {
        nsite.checked_mul(nsite - 1).map(|count| count / 2)
    } else {
        nsite.checked_mul(nsite)
    }
    .ok_or_else(|| invalid("orbital mapping count overflows"))?;
    let body = &lines[5..];
    if body.len() < n_mapping {
        return Err(invalid("incomplete orbital mapping section"));
    }
    let mut terms = Vec::new();
    // sscanf leaves the preceding sign in place for three-column mappings.
    let mut sign = 1;
    for line in &body[..n_mapping] {
        let fields: Vec<_> = c_fields(line).collect();
        let integer = |position: usize| {
            fields
                .get(position)
                .and_then(|field| field.parse::<i32>().ok())
                .map(i64::from)
                .ok_or_else(|| invalid("invalid integer in orbital mapping section"))
        };
        let site1 = integer(0)?;
        let site2 = integer(1)?;
        let idx = integer(2)?;
        if site1 < 0 || site2 < 0 || site1 >= nsite as i64 || site2 >= nsite as i64 {
            return Err(invalid("orbital mapping site is outside Nsite"));
        }
        if kind == OrbitalKind::Parallel && site1 >= site2 {
            return Err(invalid("parallel orbital mappings require site1 < site2"));
        }
        if idx < 0 || idx >= n_orbital_idx {
            return Err(invalid(
                "orbital parameter index is outside the declared width",
            ));
        }
        if fields.len() >= 4 {
            sign = integer(3)?;
        }
        terms.push(OrbitalTerm {
            site1,
            site2,
            idx,
            sign,
            is_complex,
        });
    }
    // GetInfoOpt scans whitespace-separated pairs to EOF, independent of line
    // breaks. Printed indices are consumed but flags are assigned by row order.
    let flags_text = body[n_mapping..].join("\n");
    let fields: Vec<_> = c_fields(&flags_text).collect();
    if fields.len() != 2 * n_orbital_idx as usize {
        return Err(invalid(
            "orbital flag section must contain exactly the declared number of pairs",
        ));
    }
    let mut opt_flags = BTreeMap::new();
    for (index, pair) in fields.chunks_exact(2).enumerate() {
        let _printed_index = pair[0]
            .parse::<i32>()
            .map_err(|_| invalid("invalid printed integer in orbital flag section"))?;
        let flag = pair[1]
            .parse::<i32>()
            .map_err(|_| invalid("invalid integer in orbital flag section"))?;
        opt_flags.insert(index as i64, i64::from(flag));
    }
    Ok(OrbitalSection {
        terms,
        n_orbital_idx,
        is_complex,
        opt_flags,
    })
}

fn c_fields(text: &str) -> impl Iterator<Item = &str> {
    text.split([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
        .filter(|field| !field.is_empty())
}

fn parse_general_rows(lines: &[&str], n_orbital_idx: i64, is_complex: bool) -> OrbitalSection {
    let mut terms = Vec::new();
    let mut processing_idx = true;
    let mut opt_flags = BTreeMap::new();
    for line in lines {
        let tokens = split_def_line(line);
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() == 2 {
            if processing_idx && !terms.is_empty() {
                processing_idx = false;
            }
            if !processing_idx {
                let idx = safe_parse_int(tokens[0], -1);
                let flag = safe_parse_int(tokens[1], 0);
                if idx >= 0 {
                    opt_flags.insert(idx, flag);
                }
            }
            continue;
        }
        if !processing_idx || tokens.len() < 3 {
            continue;
        }
        let site1 = safe_parse_int(tokens[0], -1);
        let site2 = safe_parse_int(tokens[1], -1);
        let idx = safe_parse_int(tokens[2], -1);
        if site1 < 0 || site2 < 0 || idx < 0 {
            continue;
        }
        let mut sign = 1i64;
        if tokens.len() >= 4 {
            let parsed = safe_parse_int(tokens[3], 1);
            if parsed == 1 || parsed == -1 {
                sign = parsed;
            }
        }
        terms.push(OrbitalTerm {
            site1,
            site2,
            idx,
            is_complex,
            sign,
        });
    }

    OrbitalSection {
        terms,
        n_orbital_idx,
        is_complex,
        opt_flags,
    }
}
