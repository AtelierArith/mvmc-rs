//! `trans.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/trans_parser.jl`.

use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::types::{Spin, TransferTerm};
use crate::utils::file::{read_def_file, safe_parse_float, safe_parse_int, split_def_line};

/// Parse a `trans.def` file from disk.
pub fn parse_trans_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<TransferTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_trans_content(&content))
}

/// Parse a `trans.def` payload from memory.
pub fn parse_trans_content(content: &str) -> Vec<TransferTerm> {
    let mut out = Vec::new();
    for line in content.lines() {
        let tokens = split_def_line(line);
        if tokens.len() < 6 {
            continue;
        }
        // Match the Julia parser: skip the line silently when any field
        // is malformed; the line counts (NTransfer) come from the
        // header but we don't enforce them.
        let site1 = safe_parse_int(tokens[0], -1);
        let spin1 = safe_parse_int(tokens[1], -1);
        let site2 = safe_parse_int(tokens[2], -1);
        let spin2 = safe_parse_int(tokens[3], -1);
        if site1 < 0 || site2 < 0 {
            continue;
        }
        let (s1, s2) = match (Spin::from_code(spin1), Spin::from_code(spin2)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let re = safe_parse_float(tokens[4], 0.0);
        let im = safe_parse_float(tokens[5], 0.0);
        out.push(TransferTerm {
            site1,
            spin1: s1,
            site2,
            spin2: s2,
            value: Complex64::new(re, im),
        });
    }
    out
}
