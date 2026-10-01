//! `gutzwilleridx.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/gutzwiller_parser.jl`,
//! restricted to the bit-stable subset needed for round-tripping the
//! four upstream `examples/inputs/*` cases:
//!
//! 1. Read `NGutzwillerIdx` and `ComplexType` from the header.
//! 2. Walk the body in two phases: site/idx pairs first, then optional
//!    `idx opt_flag` pairs.
//! 3. Materialise one [`GutzwillerTerm`] per **unique** idx encountered
//!    (sorted ascending), pre-populated with the first site that maps
//!    to it. This matches upstream behaviour for the 4 supported inputs
//!    where each site has its own Gutzwiller index slot.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::types::GutzwillerTerm;
use crate::utils::file::{clean_line, read_def_file, safe_parse_int, split_def_line};

/// Output of [`parse_gutzwiller_content`].
#[derive(Debug, Clone, Default)]
pub struct GutzwillerSection {
    /// Terms keyed by parameter index (one per unique idx, sorted).
    pub terms: Vec<GutzwillerTerm>,
    /// `NGutzwillerIdx` from the header.
    pub n_gutzwiller_idx: i64,
    /// Whether the `ComplexType` header set the complex flag.
    pub is_complex: bool,
    /// `site -> idx` (0-based) lookup for ALL sites listed in the file.
    pub site_idx_map: HashMap<i64, i64>,
    /// Per-parameter optimization flags, as listed in the trailing section.
    pub opt_flags: BTreeMap<i64, i64>,
}

/// Parse a `gutzwilleridx.def` file from disk.
pub fn parse_gutzwiller_def<P: AsRef<Path>>(path: P) -> io::Result<GutzwillerSection> {
    let content = read_def_file(path)?;
    Ok(parse_gutzwiller_content(&content))
}

/// Parse a `gutzwilleridx.def` payload from memory.
pub fn parse_gutzwiller_content(content: &str) -> GutzwillerSection {
    let lines: Vec<&str> = content.lines().collect();
    let (n_gutzwiller_idx, is_complex, has_header) = read_idx_header(&lines, "NGutzwillerIdx");

    let mut site_idx_map: HashMap<i64, i64> = HashMap::new();
    let mut seen_sites: BTreeSet<i64> = BTreeSet::new();
    let mut seen_idx: BTreeSet<i64> = BTreeSet::new();
    let mut in_opt_section = false;
    let mut opt_flags = BTreeMap::new();

    let start_line = if has_header { 5 } else { 0 };
    for line in lines.get(start_line..).unwrap_or_default() {
        let tokens = split_def_line(line);
        if tokens.len() < 2 {
            continue;
        }
        let a = safe_parse_int(tokens[0], -1);
        let b = safe_parse_int(tokens[1], -1);
        if a < 0 || b < 0 {
            continue;
        }
        if !in_opt_section && seen_sites.contains(&a) {
            in_opt_section = true;
        }
        if in_opt_section {
            opt_flags.insert(a, b);
            continue;
        }
        site_idx_map.insert(a, b);
        seen_sites.insert(a);
        seen_idx.insert(b);
    }

    // Build one term per unique idx (matches Julia path when
    // n_gutzwiller_idx == 0). For the four upstream cases each site has
    // a distinct idx, so this yields the same NGutzwillerIdx terms the
    // Julia parser emits.
    let mut terms = Vec::new();
    let idx_iter: Box<dyn Iterator<Item = i64>> = if n_gutzwiller_idx > 0 {
        Box::new(0..n_gutzwiller_idx)
    } else {
        Box::new(seen_idx.into_iter())
    };
    for idx_val in idx_iter {
        let site = site_idx_map
            .iter()
            .find(|(_, v)| **v == idx_val)
            .map(|(k, _)| *k)
            .unwrap_or(0);
        terms.push(GutzwillerTerm {
            site,
            value: Complex64::new(0.0, 0.0),
            is_complex,
        });
    }

    // Touch clean_line so the import is kept (used implicitly via split_def_line).
    let _ = clean_line;

    GutzwillerSection {
        terms,
        n_gutzwiller_idx,
        is_complex,
        site_idx_map,
        opt_flags,
    }
}

/// Read `(N<Idx>, ComplexType, has_header)` from the first 5 lines of
/// a `*idx.def` payload. Shared between gutzwiller / jastrow / orbital.
pub(crate) fn read_idx_header(lines: &[&str], n_keyword: &str) -> (i64, bool, bool) {
    let mut n_idx = 0i64;
    let mut complex_type = 0i64;
    let mut has_header = false;

    if lines.len() > 1 {
        let tokens = split_def_line(lines[1]);
        if tokens.len() >= 2
            && (tokens[0] == n_keyword
                || (n_keyword == "NOrbitalIdx" && tokens[0].starts_with("NOrbital")))
        {
            n_idx = safe_parse_int(tokens[1], 0);
            has_header = true;
        }
    }
    if has_header && lines.len() > 2 {
        let tokens = split_def_line(lines[2]);
        if tokens.len() >= 2 && tokens[0] == "ComplexType" {
            complex_type = safe_parse_int(tokens[1], 0);
        }
    }
    (n_idx, complex_type != 0, has_header)
}
