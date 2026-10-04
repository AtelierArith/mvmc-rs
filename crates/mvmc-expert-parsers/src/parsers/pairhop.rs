//! Directed pair expansion and strict section errors from `pairhop_parser.jl`.
use std::io;
use std::path::Path;

use crate::types::PairHopTerm;
use crate::utils::file::{clean_line, read_def_file};

/// Accepted directed terms and all line errors, including partial failed results.
#[derive(Debug, Clone, PartialEq)]
pub struct PairHopSection {
    /// Each accepted input row contributes `(i,j)` followed by `(j,i)`.
    pub terms: Vec<PairHopTerm>,
    /// Julia's line-numbered errors; a nonempty list invalidates the section.
    pub errors: Vec<String>,
}

impl PairHopSection {
    /// Whether the complete section can replace the previous Hamiltonian payload.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Read C-declared PairHop rows with site bounds and native directed expansion.
///
/// Native `NPairHopping` is twice the declared input count. Each accepted row
/// emits forward then reverse, without a sign change, including diagonal rows.
/// The shared pair reader preserves C scalar initialization/conversion rules;
/// malformed inputs are rejected before any partial payload is published.
pub fn parse_pairhop_definition<P: AsRef<Path>>(
    path: P,
    nsite: i64,
) -> io::Result<Vec<PairHopTerm>> {
    let rows = super::coulomb::parse_coulomb_inter_definition(path, nsite)?;
    let length = rows.len().checked_mul(2).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "PairHop: expanded count overflow",
        )
    })?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(length)
        .map_err(|error| io::Error::new(io::ErrorKind::OutOfMemory, format!("PairHop: {error}")))?;
    for row in rows {
        terms.push(PairHopTerm {
            site1: row.site1,
            site2: row.site2,
            value: row.value,
        });
        terms.push(PairHopTerm {
            site1: row.site2,
            site2: row.site1,
            value: row.value,
        });
    }
    Ok(terms)
}

/// Read a permissive Julia-style section, separate from the C file boundary.
/// Read failures are distinct from section-format errors.
pub fn parse_pairhop_def<P: AsRef<Path>>(path: P) -> io::Result<PairHopSection> {
    Ok(parse_pairhop_content(&read_def_file(path)?))
}

/// Expand directed terms, retaining duplicates, same-site rows, and signed zeros.
///
/// Only a recognized five-line header is skipped. Extra columns are ignored,
/// the declared count does not truncate the payload, and malformed/negative-site
/// rows produce errors while parsing continues through the remaining rows.
pub fn parse_pairhop_content(content: &str) -> PairHopSection {
    let lines: Vec<_> = content.split('\n').collect();
    let start = if lines.len() > 5
        && (clean_line(lines[0]).starts_with('=') || clean_line(lines[1]).starts_with("NPairHopp"))
    {
        5
    } else {
        0
    };
    let mut result = PairHopSection {
        terms: Vec::new(),
        errors: Vec::new(),
    };
    for (index, line) in lines.iter().enumerate().skip(start) {
        let line = clean_line(line);
        if line.is_empty() {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        let number = index + 1;
        if parts.len() < 3 {
            result.errors.push(format!(
                "Line {number}: Invalid format, expected 'site1 site2 value'"
            ));
            continue;
        }
        let site1 = match parts[0].parse::<i64>() {
            Ok(value) => value,
            Err(_) => {
                result.errors.push(format!(
                    "Line {number}: Invalid site1 number '{}'",
                    parts[0]
                ));
                continue;
            }
        };
        let site2 = match parts[1].parse::<i64>() {
            Ok(value) => value,
            Err(_) => {
                result.errors.push(format!(
                    "Line {number}: Invalid site2 number '{}'",
                    parts[1]
                ));
                continue;
            }
        };
        let value = match parts[2].parse::<f64>() {
            Ok(value) => value,
            Err(_) => {
                result
                    .errors
                    .push(format!("Line {number}: Invalid value '{}'", parts[2]));
                continue;
            }
        };
        if site1 < 0 {
            result.errors.push(format!(
                "Line {number}: Site1 number must be non-negative, got {site1}"
            ));
            continue;
        }
        if site2 < 0 {
            result.errors.push(format!(
                "Line {number}: Site2 number must be non-negative, got {site2}"
            ));
            continue;
        }
        result.terms.push(PairHopTerm {
            site1,
            site2,
            value,
        });
        result.terms.push(PairHopTerm {
            site1: site2,
            site2: site1,
            value,
        });
    }
    result
}
