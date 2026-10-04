//! `trans.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/trans_parser.jl`.

use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::types::{Spin, TransferTerm};
use crate::utils::file::{
    c_parse_float, read_def_file, safe_parse_float, safe_parse_int, split_def_line,
};

/// Read a declared C Transfer section, checking row count and site bounds.
///
/// This definition boundary differs from the headerless Julia convenience
/// [`parse_trans_content`]. Native `GetTransferInfo` reads decimal/hex `%lf`
/// values, checks sites and requires the declared count. A zero count ignores
/// the body. A valid five-field row retains the previous imaginary coefficient,
/// initially zero, matching the native loop's initialized persistent variable.
/// Malformed rows and invalid typed spins are safely rejected, not
/// claims about native unchecked `sscanf` or excess-row memory writes.
pub fn parse_trans_definition<P: AsRef<Path>>(
    path: P,
    nsite: i64,
) -> io::Result<Vec<TransferTerm>> {
    let content = read_def_file(path)?;
    let lines: Vec<_> = content.lines().collect();
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    let count = lines
        .get(1)
        .and_then(|line| line.split_ascii_whitespace().nth(1))
        .and_then(|token| token.parse::<usize>().ok())
        .ok_or_else(|| invalid("Trans: missing or invalid declared count"))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if lines.len() < 5 {
        return Err(invalid("Trans: incomplete five-line definition header"));
    }
    let mut terms = Vec::new();
    let mut previous_imaginary = 0.0;
    for line in &lines[5..] {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        if fields.len() < 5 {
            return Err(invalid(
                "Trans: expected at least five fields in each consumed row",
            ));
        }
        let mut coordinates = [0_i64; 4];
        for (value, field) in coordinates.iter_mut().zip(&fields[..4]) {
            *value = field
                .parse()
                .map_err(|_| invalid("Trans: invalid integer coordinate"))?;
        }
        let [site1, spin1, site2, spin2] = coordinates;
        if site1 < 0 || site2 < 0 || site1 >= nsite || site2 >= nsite {
            return Err(invalid("Trans: site coordinate out of range"));
        }
        let spin1 =
            Spin::from_code(spin1).ok_or_else(|| invalid("Trans: typed spin must be 0 or 1"))?;
        let spin2 =
            Spin::from_code(spin2).ok_or_else(|| invalid("Trans: typed spin must be 0 or 1"))?;
        let real =
            c_parse_float(fields[4]).ok_or_else(|| invalid("Trans: invalid real coefficient"))?;
        let imaginary = match fields.get(5) {
            Some(field) => c_parse_float(field)
                .ok_or_else(|| invalid("Trans: invalid imaginary coefficient"))?,
            None => previous_imaginary,
        };
        previous_imaginary = imaginary;
        terms.push(TransferTerm {
            site1,
            spin1,
            site2,
            spin2,
            value: Complex64::new(real, imaginary),
        });
        if terms.len() > count {
            return Err(invalid("Trans: more rows than the declared count"));
        }
    }
    if terms.len() != count {
        return Err(invalid("Trans: row count differs from declared count"));
    }
    Ok(terms)
}

/// Read the permissive Julia-helper payload from disk, without C count/bounds checks.
pub fn parse_trans_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<TransferTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_trans_content(&content))
}

/// Parse a headerless Julia convenience payload; not a validated C definition.
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
