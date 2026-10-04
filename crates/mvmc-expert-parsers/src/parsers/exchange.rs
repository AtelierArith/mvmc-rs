//! `exchange.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/exchange_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::ExchangeTerm;
use crate::utils::file::{
    clean_line, read_def_file, safe_parse_float, safe_parse_int, split_def_line,
};

/// Read C-declared Exchange pairs with supplied site bounds.
///
/// C uses the same `ReadPairDValue` routine as CoulombInter. Coordinates,
/// scalar values and row order are retained without sign or symmetry changes.
/// Diagonal pairs are allowed; omitted scalar fields retain initialized/prior
/// values. Nonfinite values are parser results, not runnable-model approval.
pub fn parse_exchange_definition<P: AsRef<Path>>(
    path: P,
    nsite: i64,
) -> io::Result<Vec<ExchangeTerm>> {
    Ok(super::coulomb::parse_coulomb_inter_definition(path, nsite)?
        .into_iter()
        .map(|term| ExchangeTerm {
            site1: term.site1,
            site2: term.site2,
            value: term.value,
        })
        .collect())
}

/// Read a permissive headerless convenience payload without C count/bounds checks.
pub fn parse_exchange_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<ExchangeTerm>> {
    let content = read_def_file(path)?;
    Ok(parse_exchange_content(&content))
}

/// Parse an `exchange.def` payload from memory.
pub fn parse_exchange_content(content: &str) -> Vec<ExchangeTerm> {
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
        out.push(ExchangeTerm {
            site1,
            site2,
            value,
        });
    }
    out
}
