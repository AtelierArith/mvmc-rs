//! `locspn.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/locspin_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::LocSpinTerm;
use crate::utils::file::{read_def_file, safe_parse_int, split_def_line};

/// Number of header lines skipped before the `site spin_value` payload.
pub const IGNORE_LINES_IN_DEF: usize = 5;

/// Try to read `NlocalSpin` (or `NLocSpin`) from the first few lines
/// of a `locspn.def` payload. Mirrors the C-mVMC `ReadBuffInt` for that
/// header field.
pub fn read_nlocspin(content: &str) -> Option<i64> {
    for line in content.lines().take(5) {
        let tokens = split_def_line(line);
        if tokens.len() >= 2 && (tokens[0] == "NlocalSpin" || tokens[0] == "NLocSpin") {
            return Some(safe_parse_int(tokens[1], 0));
        }
    }
    None
}

/// Parse a `locspn.def` file from disk.
pub fn parse_locspin_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<LocSpinTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_locspin_content(&content))
}

/// Parse a `locspn.def` payload from memory.
pub fn parse_locspin_content(content: &str) -> Vec<LocSpinTerm> {
    let mut out = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if line_idx < IGNORE_LINES_IN_DEF {
            continue;
        }
        let tokens = split_def_line(line);
        if tokens.len() < 2 {
            continue;
        }
        let site = match tokens[0].parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let spin_value = match tokens[1].parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if site < 0 || spin_value < 0 {
            continue;
        }
        out.push(LocSpinTerm { site, spin_value });
    }
    out
}
