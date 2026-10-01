//! `interall.def` parser, matching Julia's `interall_parser.jl`.

use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::types::InterAllTerm;
use crate::utils::file::{clean_line, read_def_file};

/// Read all general interaction terms from an `interall.def` file.
pub fn parse_interall_def<P: AsRef<Path>>(path: P) -> io::Result<Vec<InterAllTerm>> {
    Ok(parse_interall_content(&read_def_file(path)?))
}

/// Parse four site/spin pairs and a real/imaginary coupling from each data row.
///
/// Julia skips malformed rows and any cleaned line containing ASCII letters,
/// including scientific notation. Extra numeric columns are ignored, duplicates
/// are retained, and the declared header count does not truncate the payload.
pub fn parse_interall_content(content: &str) -> Vec<InterAllTerm> {
    let mut terms = Vec::new();
    for line in content.lines() {
        let line = clean_line(line);
        if line.is_empty() || line.starts_with('=') || line.bytes().any(|b| b.is_ascii_alphabetic())
        {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < 10 {
            continue;
        }
        let term = (|| {
            let re = parts[8].parse::<f64>().ok()?;
            let im = parts[9].parse::<f64>().ok()?;
            Some(InterAllTerm {
                site0: parts[0].parse().ok()?,
                spin0: parts[1].parse().ok()?,
                site1: parts[2].parse().ok()?,
                spin1: parts[3].parse().ok()?,
                site2: parts[4].parse().ok()?,
                spin2: parts[5].parse().ok()?,
                site3: parts[6].parse().ok()?,
                spin3: parts[7].parse().ok()?,
                value: Complex64::new(re, im),
                is_complex: im.abs() > 1e-14,
            })
        })();
        if let Some(term) = term {
            terms.push(term);
        }
    }
    terms
}
