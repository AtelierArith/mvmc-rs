//! `coulombintra.def` / `coulombinter.def` parsers.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/coulomb_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::{CoulombInterTerm, CoulombIntraTerm};
use crate::utils::file::{
    clean_line, read_def_file, safe_parse_float, safe_parse_int, split_def_line,
};

/// Parse a `coulombintra.def` file from disk.
pub fn parse_coulomb_intra_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<CoulombIntraTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_coulomb_intra_content(&content))
}

/// Parse a `coulombintra.def` payload from memory.
pub fn parse_coulomb_intra_content(content: &str) -> Vec<CoulombIntraTerm> {
    let mut out = Vec::new();
    for line in content.lines() {
        let tokens = split_def_line(line);
        if tokens.len() < 2 {
            continue;
        }
        // Header lines like "NCoulombIntra 8" -- skip silently.
        let site = match tokens[0].parse::<i64>() {
            Ok(v) if v >= 0 => v,
            _ => continue,
        };
        let value = safe_parse_float(tokens[1], 0.0);
        out.push(CoulombIntraTerm { site, value });
    }
    out
}

/// Parse a `coulombinter.def` file from disk.
pub fn parse_coulomb_inter_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<CoulombInterTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_coulomb_inter_content(&content))
}

/// Parse a `coulombinter.def` payload from memory.
pub fn parse_coulomb_inter_content(content: &str) -> Vec<CoulombInterTerm> {
    let mut out = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut start_line = 0usize;
    if lines.len() > 5 {
        let first_data = clean_line(lines[5]);
        if !first_data.is_empty() {
            let tokens = split_def_line(first_data);
            if tokens.len() >= 3 {
                start_line = 5;
            }
        }
    }

    for line in &lines[start_line..] {
        let tokens = split_def_line(line);
        if tokens.len() < 3 {
            continue;
        }
        let site1 = safe_parse_int(tokens[0], -1);
        let site2 = safe_parse_int(tokens[1], -1);
        if site1 < 0 || site2 < 0 {
            continue;
        }
        let value = safe_parse_float(tokens[2], 0.0);
        out.push(CoulombInterTerm {
            site1,
            site2,
            value,
        });
    }
    out
}
