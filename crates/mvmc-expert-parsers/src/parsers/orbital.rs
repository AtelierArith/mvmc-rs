//! `orbitalidx.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/orbital_parser.jl`,
//! restricted to the round-trip subset.

use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::parsers::gutzwiller::read_idx_header;
use crate::types::OrbitalTerm;
use crate::utils::file::{read_def_file, safe_parse_int, split_def_line};

/// Output of [`parse_orbital_content`].
#[derive(Debug, Clone, Default)]
pub struct OrbitalSection {
    /// `(site1, site2, idx, sign)` entries from the OrbitalIdx block.
    pub terms: Vec<OrbitalTerm>,
    /// `NOrbitalIdx` from the header (number of unique parameters).
    pub n_orbital_idx: i64,
    /// `ComplexType` flag.
    pub is_complex: bool,
}

/// Parse a `orbitalidx*.def` file from disk.
pub fn parse_orbital_def<P: AsRef<Path>>(path: P) -> io::Result<OrbitalSection> {
    let content = read_def_file(path)?;
    Ok(parse_orbital_content(&content))
}

/// Parse a `orbitalidx*.def` payload from memory.
pub fn parse_orbital_content(content: &str) -> OrbitalSection {
    let lines: Vec<&str> = content.lines().collect();
    let (n_orbital_idx, is_complex, has_header) = read_idx_header(&lines, "NOrbitalIdx");

    let start_line = if has_header { 5 } else { 0 };
    let mut terms = Vec::new();
    let mut processing_idx = true;
    for line in &lines[start_line..] {
        let tokens = split_def_line(line);
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() == 2 {
            if processing_idx && !terms.is_empty() {
                processing_idx = false;
            }
            // OptFlag lines: ignored for round-trip.
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
            value: Complex64::new(0.0, 0.0),
            is_complex,
            sign,
        });
    }

    OrbitalSection {
        terms,
        n_orbital_idx,
        is_complex,
    }
}
