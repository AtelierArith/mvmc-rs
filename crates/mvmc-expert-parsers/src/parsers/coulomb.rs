//! `coulombintra.def` / `coulombinter.def` parsers.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/coulomb_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::{CoulombInterTerm, CoulombIntraTerm, ParsingContext, ParsingDiagnostic};
use crate::utils::file::{
    c_parse_float, clean_line, read_def_file, safe_parse_float, safe_parse_int, split_def_line,
};

/// Read C-declared CoulombIntra records with supplied site bounds.
/// A zero count ignores the body; omitted scalar fields retain the native
/// initialized/prior value. Malformed rows are safely rejected before publication.
pub fn parse_coulomb_intra_definition<P: AsRef<Path>>(
    path: P,
    nsite: i64,
) -> io::Result<Vec<CoulombIntraTerm>> {
    Ok(read_definition(path.as_ref(), nsite, 1)?
        .into_iter()
        .map(|(sites, value)| CoulombIntraTerm {
            site: sites[0],
            value,
        })
        .collect())
}

/// Read C-declared CoulombInter records, preserving row and coordinate order.
/// This is not the permissive headerless Julia content helper. Native
/// `ReadPairDValue` accepts diagonal pairs; no invented off-diagonal rule applies.
pub fn parse_coulomb_inter_definition<P: AsRef<Path>>(
    path: P,
    nsite: i64,
) -> io::Result<Vec<CoulombInterTerm>> {
    Ok(read_definition(path.as_ref(), nsite, 2)?
        .into_iter()
        .map(|(sites, value)| CoulombInterTerm {
            site1: sites[0],
            site2: sites[1],
            value,
        })
        .collect())
}

fn read_definition(
    path: &Path,
    nsite: i64,
    site_fields: usize,
) -> io::Result<Vec<([i64; 2], f64)>> {
    let content = read_def_file(path)?;
    let lines: Vec<_> = content.lines().collect();
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let count = lines
        .get(1)
        .and_then(|line| line.split_ascii_whitespace().nth(1))
        .and_then(|field| field.parse::<usize>().ok())
        .ok_or_else(|| invalid("Coulomb: missing or invalid declared count"))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if lines.len() < 5 {
        return Err(invalid("Coulomb: incomplete positive-count header"));
    }
    let mut rows = Vec::new();
    let mut previous_value = 0.0;
    for line in &lines[5..] {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        if fields.len() < site_fields {
            return Err(invalid("Coulomb: missing site coordinates"));
        }
        let mut sites = [0; 2];
        for (site, field) in sites[..site_fields].iter_mut().zip(&fields[..site_fields]) {
            *site = field
                .parse::<i64>()
                .map_err(|_| invalid("Coulomb: invalid integer coordinate"))?;
            if *site < 0 || *site >= nsite {
                return Err(invalid("Coulomb: site coordinate out of range"));
            }
        }
        let value = match fields.get(site_fields) {
            Some(field) => {
                c_parse_float(field).ok_or_else(|| invalid("Coulomb: invalid coefficient"))?
            }
            None => previous_value,
        };
        previous_value = value;
        rows.push((sites, value));
        if rows.len() > count {
            return Err(invalid("Coulomb: excess records"));
        }
    }
    if rows.len() != count {
        return Err(invalid("Coulomb: row count differs from declaration"));
    }
    Ok(rows)
}

/// Read a permissive Julia-helper CoulombIntra payload without C count/bounds checks.
pub fn parse_coulomb_intra_def<P: AsRef<Path>>(
    path: P,
) -> io::Result<(Vec<CoulombIntraTerm>, ParsingContext)> {
    let content = read_def_file(path.as_ref())?;
    let mut context = ParsingContext::new(path.as_ref());
    let terms = parse_coulomb_intra_content(&content, &mut context);
    Ok((terms, context))
}

/// Parse a `coulombintra.def` payload from memory.
/// Starts a new observation in `context`, retaining its filename but clearing
/// previous diagnostics. Like Julia's split, an empty input and a trailing
/// newline each include an empty physical line. Negative sites are warning-only
/// skipped rows; malformed coefficients retain the existing safe zero default.
/// This helper does not apply C declared-count or Nsite admission checks.
pub fn parse_coulomb_intra_content(
    content: &str,
    context: &mut ParsingContext,
) -> Vec<CoulombIntraTerm> {
    context.line_number = 0;
    context.errors.clear();
    context.warnings.clear();
    let mut out = Vec::new();
    for (index, line) in content.split('\n').enumerate() {
        context.line_number = index + 1;
        let tokens = split_def_line(line);
        if tokens.len() < 2 {
            continue;
        }
        // Header lines like "NCoulombIntra 8" -- skip silently.
        let site = match tokens[0].parse::<i64>() {
            Ok(v) if v >= 0 => v,
            Ok(v) => {
                context.warnings.push(ParsingDiagnostic {
                    line_number: context.line_number,
                    message: format!("Negative site index: {v}"),
                });
                continue;
            }
            _ => continue,
        };
        let value = safe_parse_float(tokens[1], 0.0);
        out.push(CoulombIntraTerm { site, value });
    }
    out
}

/// Read a permissive Julia-helper CoulombInter payload without C count/bounds checks.
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
