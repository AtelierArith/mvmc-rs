//! `locspn.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/locspin_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::LocSpinTerm;
use crate::utils::file::{read_def_file, safe_parse_int, split_def_line};

/// Number of header lines skipped before the `site spin_value` payload.
pub const IGNORE_LINES_IN_DEF: usize = 5;

/// Complete native LocSpin definition, distinct from the permissive helper.
#[derive(Debug)]
pub struct LocSpinDefinition {
    /// Informational local-spin integer read from the second header line.
    pub nlocal_spin: i64,
    /// Complete site-indexed values; negative/nonbinary integers are retained.
    pub terms: Vec<LocSpinTerm>,
}

// Native %d consumes a signed decimal prefix, leaving the remainder for the
// next conversion. Failed conversions preserve prior initialized variables.
// Reject C-int overflow safely rather than copy scanf overflow behavior.
fn c_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

fn scan_int(input: &str) -> io::Result<Option<(i32, &str)>> {
    let input = input.trim_start_matches(c_space);
    let bytes = input.as_bytes();
    let sign = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mut end = sign;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == sign {
        return Ok(None);
    }
    let value = input[..end]
        .parse::<i32>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "LocSpin: C integer overflow"))?;
    Ok(Some((value, &input[end..])))
}

/// Read the native header and NSite physical rows before publishing any data.
///
/// C initialized/prior site/scalar carry and decimal-prefix conversions are
/// preserved whenever the resulting table is complete and in range. Duplicate
/// holes are rejected because native malloc leaves them indeterminate, not
/// because C enforces uniqueness. This does not approve a runnable model or
/// require the header to equal the number of values equal to one.
pub fn parse_locspin_definition<P: AsRef<Path>>(
    path: P,
    nsite: i64,
) -> io::Result<LocSpinDefinition> {
    let content = read_def_file(path)?;
    let lines: Vec<_> = content.lines().collect();
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let header = lines
        .get(1)
        .ok_or_else(|| invalid("LocSpin: missing header"))?;
    let header = header.trim_start_matches(c_space);
    let token_end = header.find(c_space).unwrap_or(header.len());
    let nlocal_spin = scan_int(&header[token_end..])?
        .ok_or_else(|| invalid("LocSpin: missing header integer"))?
        .0;
    let nsite = usize::try_from(nsite).map_err(|_| invalid("LocSpin: invalid NSite"))?;
    if lines.len() < IGNORE_LINES_IN_DEF || lines.len() - IGNORE_LINES_IN_DEF != nsite {
        return Err(invalid("LocSpin: physical body count must equal NSite"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(nsite)
        .map_err(|error| io::Error::new(io::ErrorKind::OutOfMemory, format!("LocSpin: {error}")))?;
    values.resize(nsite, None);
    let mut site = 0_i32;
    let mut scalar = 0_i32;
    for line in &lines[IGNORE_LINES_IN_DEF..] {
        // Native fgets has a 256-byte buffer. Oversized records are an explicit
        // bounded-reader safety boundary, not a claim about C chunk splitting.
        if line.len() >= 255 {
            return Err(invalid("LocSpin: oversized native record"));
        }
        if let Some((parsed_site, remainder)) = scan_int(line)? {
            site = parsed_site;
            if let Some((parsed_scalar, _)) = scan_int(remainder)? {
                scalar = parsed_scalar;
            }
        }
        let index = usize::try_from(site).map_err(|_| invalid("LocSpin: site out of range"))?;
        let value = values
            .get_mut(index)
            .ok_or_else(|| invalid("LocSpin: site out of range"))?;
        *value = Some(scalar);
    }
    if values.iter().any(Option::is_none) {
        return Err(invalid(
            "LocSpin: incomplete table leaves native malloc holes",
        ));
    }
    let terms = values
        .into_iter()
        .enumerate()
        .map(|(site, value)| LocSpinTerm {
            site: site as i64,
            spin_value: i64::from(value.expect("complete table checked")),
        })
        .collect();
    Ok(LocSpinDefinition {
        nlocal_spin: i64::from(nlocal_spin),
        terms,
    })
}

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
