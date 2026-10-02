//! `greenone.def` / `greentwo.def` parsers.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/green_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::{GreenOneTerm, GreenTwoExTerm, GreenTwoTerm, Spin};
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

/// Parse a strict `greentwoex.def` file, including its declared row count.
pub fn parse_green_two_ex_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<GreenTwoExTerm>> {
    let content = read_def_file(path)?;
    parse_green_two_ex_content(&content)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
}

/// Parse `greentwoex.def` content using C's exact eight-column and count rules.
pub fn parse_green_two_ex_content(content: &str) -> Result<Vec<GreenTwoExTerm>, String> {
    let lines: Vec<&str> = content.lines().collect();
    let header = lines
        .get(1)
        .map(|line| split_def_line(clean_line(line)))
        .unwrap_or_default();
    let count = header
        .get(1)
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            "greentwoex.def: missing or invalid header count on line 2 (expected `<keyword> <count>`)".to_owned()
        })?;
    let mut terms = Vec::with_capacity(count);
    let mut errors = Vec::new();
    for (line_index, line) in lines.iter().enumerate().skip(5) {
        let cleaned = clean_line(line);
        if cleaned.is_empty() {
            continue;
        }
        let tokens = split_def_line(cleaned);
        if tokens.len() != 8 {
            errors.push(format!(
                "Line {}: expected exactly 8 integer fields, got {}",
                line_index + 1,
                tokens.len()
            ));
            continue;
        }
        let mut values = [0_i64; 8];
        let mut valid = true;
        for (index, token) in tokens.iter().enumerate() {
            match token.parse::<i64>() {
                Ok(value) => values[index] = value,
                Err(_) => {
                    errors.push(format!(
                        "Line {}: field {} is not an integer: '{token}'",
                        line_index + 1,
                        index + 1
                    ));
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        let sites = [values[0], values[2], values[6], values[4]];
        let spins = [values[1], values[3], values[7], values[5]];
        if sites.iter().any(|site| *site < 0) {
            errors.push(format!(
                "Line {}: negative site index in {:?}",
                line_index + 1,
                values
            ));
            continue;
        }
        if spins.iter().any(|spin| !matches!(*spin, 0 | 1)) {
            errors.push(format!(
                "Line {}: spin must be 0 or 1 in {:?}",
                line_index + 1,
                values
            ));
            continue;
        }
        terms.push(GreenTwoExTerm {
            site1: values[0],
            spin1: Spin::from_code(values[1]).expect("validated spin"),
            site2: values[2],
            spin2: Spin::from_code(values[3]).expect("validated spin"),
            site3: values[6],
            spin3: Spin::from_code(values[7]).expect("validated spin"),
            site4: values[4],
            spin4: Spin::from_code(values[5]).expect("validated spin"),
        });
    }
    if errors.is_empty() && terms.len() != count {
        errors.push(format!(
            "greentwoex.def: header count {count} does not match parsed rows {}",
            terms.len()
        ));
    }
    if errors.is_empty() {
        Ok(terms)
    } else {
        Err(errors.join("; "))
    }
}
