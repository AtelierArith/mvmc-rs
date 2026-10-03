//! Authoritative C Gutzwiller headers, complete site mappings and raw flags.
use crate::types::GutzwillerTerm;
use crate::utils::file::read_def_file;
use num_complex::Complex64;
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::Path;

/// Declared coefficients and C site-index/flag assignments.
#[derive(Debug, Clone, Default)]
pub struct GutzwillerSection {
    /// One coefficient slot per declared parameter, including unused slots.
    pub terms: Vec<GutzwillerTerm>,
    /// Declared coefficient count.
    pub n_gutzwiller_idx: i64,
    /// Positive complex header enables imaginary flag writes.
    pub is_complex: bool,
    /// Site-to-parameter lookup; duplicate mappings overwrite in read order.
    pub site_idx_map: HashMap<i64, i64>,
    /// Raw flags assigned by pair order, ignoring printed labels.
    pub opt_flags: BTreeMap<i64, i64>,
}

/// Read a complete C Gutzwiller definition for the supplied geometry.
pub fn parse_gutzwiller_def<P: AsRef<Path>>(path: P, nsite: i64) -> io::Result<GutzwillerSection> {
    parse_gutzwiller_content(&read_def_file(path)?, nsite)
}

/// Parse five physical header lines, Nsite mapping pairs and declared flag pairs.
/// Unsafe C indices and malformed scans receive bounded Rust diagnostics.
pub fn parse_gutzwiller_content(content: &str, nsite: i64) -> io::Result<GutzwillerSection> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    let lines: Vec<_> = content.lines().collect();
    if lines.len() < 5 {
        return Err(invalid("Gutzwiller definition requires five header lines"));
    }
    let n_gutzwiller_idx = c_fields(lines[1])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .filter(|&width| width > 0)
        .ok_or_else(|| invalid("Gutzwiller declaration on line 2 must have a positive width"))?
        as i64;
    let is_complex = c_fields(lines[2])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .unwrap_or(0)
        > 0;
    let nsite = usize::try_from(nsite)
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| invalid("Gutzwiller definitions require a positive Nsite"))?;
    let expected_fields = nsite
        .checked_add(n_gutzwiller_idx as usize)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| invalid("Gutzwiller field count overflows"))?;
    let body = lines[5..].join("\n");
    let fields: Vec<_> = c_fields(&body).collect();
    if fields.len() != expected_fields {
        return Err(invalid(
            "Gutzwiller requires Nsite mapping pairs and exactly the declared flag pairs",
        ));
    }
    let mut indices = vec![0; nsite];
    let integer = |field: &str| {
        field
            .parse::<i32>()
            .map(i64::from)
            .map_err(|_| invalid("invalid integer in Gutzwiller definition"))
    };
    for pair in fields[..2 * nsite].as_chunks::<2>().0.iter() {
        let site = integer(pair[0])?;
        let idx = integer(pair[1])?;
        if site < 0 || site >= nsite as i64 {
            return Err(invalid("Gutzwiller site is outside Nsite"));
        }
        if idx < 0 || idx >= n_gutzwiller_idx {
            return Err(invalid(
                "Gutzwiller parameter index is outside the declared width",
            ));
        }
        indices[site as usize] = idx;
    }
    let mut opt_flags = BTreeMap::new();
    for (index, pair) in fields[2 * nsite..].as_chunks::<2>().0.iter().enumerate() {
        let _printed_index = integer(pair[0])?;
        opt_flags.insert(index as i64, integer(pair[1])?);
    }
    let terms = (0..n_gutzwiller_idx)
        .map(|idx| GutzwillerTerm {
            site: indices.iter().position(|&value| value == idx).unwrap_or(0) as i64,
            // Preserve the pre-initialization placeholder convention. InitParameter
            // resets every declared Gutzwiller coefficient to zero.
            value: Complex64::new(idx as f64, 0.0),
            is_complex,
        })
        .collect();
    Ok(GutzwillerSection {
        terms,
        n_gutzwiller_idx,
        is_complex,
        site_idx_map: indices
            .into_iter()
            .enumerate()
            .map(|(site, idx)| (site as i64, idx))
            .collect(),
        opt_flags,
    })
}

fn c_fields(text: &str) -> impl Iterator<Item = &str> {
    text.split([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
        .filter(|field| !field.is_empty())
}
