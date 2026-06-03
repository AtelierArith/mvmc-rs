//! `qptransidx.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/qptrans_parser.jl`
//! merged with the `build_qp_trans_mappings!` helper from
//! `utils/orbital_qptrans_utils.jl`. The Rust API exposes the data as a
//! `Vec<QPTransEntry>` so the caller does not have to juggle parallel
//! arrays.

use std::io;
use std::path::Path;

use num_complex::Complex64;

use crate::types::QPTransEntry;
use crate::utils::file::{read_def_file, safe_parse_float, safe_parse_int, split_def_line};

const IGNORE_LINES_IN_DEF: usize = 5;

/// Output of [`parse_qptrans_content`].
#[derive(Debug, Clone, Default)]
pub struct QPTransSection {
    /// `NQPTrans`.
    pub n_qp_trans: i64,
    /// One entry per translation operator.
    pub entries: Vec<QPTransEntry>,
}

/// Parse a `qptransidx.def` file from disk.
pub fn parse_qptrans_def<P: AsRef<Path>>(path: P, nsite: i64) -> io::Result<QPTransSection> {
    let content = read_def_file(path)?;
    Ok(parse_qptrans_content(&content, nsite))
}

/// Parse a `qptransidx.def` payload from memory. `nsite` comes from
/// `modpara.def` and dimensions the per-translation site map.
pub fn parse_qptrans_content(content: &str, nsite: i64) -> QPTransSection {
    let lines: Vec<&str> = content.lines().collect();
    let mut section = QPTransSection::default();

    if lines.len() <= IGNORE_LINES_IN_DEF {
        return section;
    }

    // Header line: `NQPTrans N`.
    let header_tokens = split_def_line(lines[1]);
    if header_tokens.len() < 2 || header_tokens[0] != "NQPTrans" {
        return section;
    }
    let n_qp_trans = safe_parse_int(header_tokens[1], 0);
    if n_qp_trans <= 0 || nsite <= 0 {
        section.n_qp_trans = n_qp_trans;
        return section;
    }
    section.n_qp_trans = n_qp_trans;

    // Pre-allocate one entry per translation, with default site-identity / +1 sign.
    section.entries = (0..n_qp_trans)
        .map(|_| QPTransEntry {
            weight: Complex64::new(0.0, 0.0),
            site_map: vec![-1; nsite as usize],
            site_sign: vec![1; nsite as usize],
        })
        .collect();

    // Read the NQPTrans `idx weight` lines.
    let mut consumed_weight = 0usize;
    let mut cursor = IGNORE_LINES_IN_DEF;
    while consumed_weight < n_qp_trans as usize && cursor < lines.len() {
        let tokens = split_def_line(lines[cursor]);
        cursor += 1;
        if tokens.len() < 2 {
            continue;
        }
        let idx = safe_parse_int(tokens[0], -1);
        let weight = safe_parse_float(tokens[1], 0.0);
        if idx >= 0 && (idx as usize) < section.entries.len() {
            section.entries[idx as usize].weight = Complex64::new(weight, 0.0);
            consumed_weight += 1;
        }
    }

    // Remaining lines are `(mpidx, j, itmp[, sgn])` quadruples.
    while cursor < lines.len() {
        let tokens = split_def_line(lines[cursor]);
        cursor += 1;
        if tokens.len() < 3 {
            continue;
        }
        let mpidx = safe_parse_int(tokens[0], -1);
        let j = safe_parse_int(tokens[1], -1);
        let itmp = safe_parse_int(tokens[2], -1);
        let sgn = if tokens.len() >= 4 {
            safe_parse_int(tokens[3], 1)
        } else {
            1
        };
        if mpidx < 0
            || j < 0
            || itmp < 0
            || (mpidx as i64) >= n_qp_trans
            || (j as i64) >= nsite
            || (itmp as i64) >= nsite
        {
            continue;
        }
        let entry = &mut section.entries[mpidx as usize];
        entry.site_map[j as usize] = itmp;
        entry.site_sign[j as usize] = sgn;
    }

    section
}
