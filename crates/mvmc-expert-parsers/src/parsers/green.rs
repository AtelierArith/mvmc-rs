//! `greenone.def` / `greentwo.def` parsers.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/green_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::{GreenOneTerm, GreenTwoTerm, Spin};
use crate::utils::file::{clean_line, read_def_file, safe_parse_int, split_def_line};

const IGNORE_LINES_IN_DEF: usize = 5;

fn detect_start(lines: &[&str], min_tokens: usize) -> usize {
    if lines.len() > IGNORE_LINES_IN_DEF {
        let first_data = clean_line(lines[IGNORE_LINES_IN_DEF]);
        if !first_data.is_empty() {
            let tokens = split_def_line(first_data);
            if tokens.len() >= min_tokens {
                return IGNORE_LINES_IN_DEF;
            }
        }
    }
    0
}

/// Parse a `greenone.def` file from disk.
pub fn parse_green_one_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<GreenOneTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_green_one_content(&content))
}

/// Parse a `greenone.def` payload from memory.
pub fn parse_green_one_content(content: &str) -> Vec<GreenOneTerm> {
    let lines: Vec<&str> = content.lines().collect();
    let start = detect_start(&lines, 2);
    let mut out = Vec::new();
    for line in &lines[start..] {
        let tokens = split_def_line(line);
        if tokens.len() < 4 {
            continue;
        }
        let site1 = safe_parse_int(tokens[0], -1);
        let spin1 = safe_parse_int(tokens[1], -1);
        let site2 = safe_parse_int(tokens[2], -1);
        let spin2 = safe_parse_int(tokens[3], -1);
        if site1 < 0 || site2 < 0 {
            continue;
        }
        let s1 = match Spin::from_code(spin1) {
            Some(s) => s,
            None => continue,
        };
        let s2 = match Spin::from_code(spin2) {
            Some(s) => s,
            None => continue,
        };
        out.push(GreenOneTerm {
            site1,
            spin1: s1,
            site2,
            spin2: s2,
        });
    }
    out
}

/// Parse a `greentwo.def` file from disk.
pub fn parse_green_two_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<GreenTwoTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_green_two_content(&content))
}

/// Parse a `greentwo.def` payload from memory.
pub fn parse_green_two_content(content: &str) -> Vec<GreenTwoTerm> {
    let lines: Vec<&str> = content.lines().collect();
    let start = detect_start(&lines, 4);
    let mut out = Vec::new();
    for line in &lines[start..] {
        let tokens = split_def_line(line);
        if tokens.len() < 8 {
            continue;
        }
        let site1 = safe_parse_int(tokens[0], -1);
        let spin1 = safe_parse_int(tokens[1], -1);
        let site2 = safe_parse_int(tokens[2], -1);
        let spin2 = safe_parse_int(tokens[3], -1);
        let site3 = safe_parse_int(tokens[4], -1);
        let spin3 = safe_parse_int(tokens[5], -1);
        let site4 = safe_parse_int(tokens[6], -1);
        let spin4 = safe_parse_int(tokens[7], -1);
        if site1 < 0 || site2 < 0 || site3 < 0 || site4 < 0 {
            continue;
        }
        let spins = [spin1, spin2, spin3, spin4];
        let mut ok = true;
        let mut decoded = [Spin::Up; 4];
        for (slot, code) in spins.iter().enumerate() {
            match Spin::from_code(*code) {
                Some(s) => decoded[slot] = s,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        out.push(GreenTwoTerm {
            site1,
            spin1: decoded[0],
            site2,
            spin2: decoded[1],
            site3,
            spin3: decoded[2],
            site4,
            spin4: decoded[3],
        });
    }
    out
}
