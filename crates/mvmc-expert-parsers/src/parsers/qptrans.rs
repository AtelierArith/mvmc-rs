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
use crate::utils::c_numeric::Scan;
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

/// Read the C TransSym definition boundary, separately from the permissive
/// content helper. Weights reset their scan fields per physical row; mapping
/// fields retain their native initialized/prior values. Public signs stay raw;
/// [`QPTransEntry::boundary_sign`] applies the effective periodic/AP policy.
///
/// Missing weights, forward origins or inverse targets would read uninitialized
/// C storage. Rust safely rejects these, bounds/overflow and truncated records
/// before publishing a section; C itself does not diagnose all of these cases.
pub fn parse_qptrans_def<P: AsRef<Path>>(path: P, nsite: i64) -> io::Result<QPTransSection> {
    let content = read_def_file(path)?;
    read_definition(&content, nsite)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("TransSym: {message}"))
}

fn integer(scan: &mut Scan<'_>) -> io::Result<Option<i32>> {
    scan.integer().map_err(|()| invalid("integer overflow"))
}

fn allocated<T: Clone>(length: usize, value: T) -> io::Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| invalid("allocation size cannot be represented or reserved"))?;
    result.resize(length, value);
    Ok(result)
}

fn read_definition(content: &str, nsite: i64) -> io::Result<QPTransSection> {
    // Unlike split_def_line, this preserves blank/comment physical records.
    // Reject long records rather than emulate native fgets chunking/stale EOF.
    let lines: Vec<_> = content.split_terminator('\n').collect();
    if lines.len() < IGNORE_LINES_IN_DEF
        || lines[..IGNORE_LINES_IN_DEF]
            .iter()
            .any(|line| line.len() >= 255)
    {
        return Err(invalid(
            "incomplete five-line header or overlong physical record",
        ));
    }
    let mut header = Scan::new(lines[1].as_bytes());
    if !header.word() {
        return Err(invalid("missing count header token"));
    }
    let count = integer(&mut header)?.ok_or_else(|| invalid("missing declared count"))?;
    let count = usize::try_from(count).map_err(|_| invalid("negative declared count"))?;
    if count == 0 {
        return Ok(QPTransSection::default());
    }
    let sites = usize::try_from(nsite)
        .ok()
        .filter(|&sites| sites > 0 && sites <= i32::MAX as usize)
        .ok_or_else(|| invalid("positive-count definition requires a valid Nsite"))?;
    let map_count = count
        .checked_mul(sites)
        .ok_or_else(|| invalid("mapping count overflow"))?;
    let body = &lines[IGNORE_LINES_IN_DEF..];
    if body.iter().any(|line| line.len() >= 255) {
        return Err(invalid("overlong consumed physical record"));
    }
    if body.len()
        != count
            .checked_add(map_count)
            .ok_or_else(|| invalid("record count overflow"))?
    {
        return Err(invalid(
            "physical weight/mapping counts differ from declaration",
        ));
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| invalid("entry allocation failed"))?;
    for _ in 0..count {
        entries.push(QPTransEntry {
            weight: Complex64::new(0.0, 0.0),
            site_map: allocated(sites, -1)?,
            site_sign: allocated(sites, 0)?,
        });
    }
    let mut weights_written = allocated(count, false)?;
    let mut previous_target = 0;
    for line in &body[..count] {
        // GetInfoTransSym resets all three fields before each sscanf.
        let (mut index, mut real, mut imaginary) = (0, 0.0, 0.0);
        let mut scan = Scan::new(line.as_bytes());
        if let Some(value) = integer(&mut scan)? {
            index = value;
            if let Some(value) = scan.float() {
                real = value;
                if let Some(value) = scan.float() {
                    imaginary = value;
                }
            }
        }
        let index = usize::try_from(index)
            .ok()
            .filter(|&index| index < count)
            .ok_or_else(|| invalid("weight index out of range"))?;
        entries[index].weight = Complex64::new(real, imaginary);
        weights_written[index] = true;
        previous_target = index as i32;
    }
    if weights_written.contains(&false) {
        return Err(invalid("unwritten weight slot is not defined C input"));
    }
    // C's weight loop leaves i == NArray; j/sign are initially zero and itmp
    // is the last scanned weight index. Mapping sscanf stops at first failure.
    let mut fields = [count as i32, 0, previous_target, 0];
    let mut origins_written = allocated(map_count, false)?;
    let mut targets_written = allocated(map_count, false)?;
    for line in &body[count..] {
        let mut scan = Scan::new(line.as_bytes());
        for field in &mut fields {
            match integer(&mut scan)? {
                Some(value) => *field = value,
                None => break,
            }
        }
        let translation = usize::try_from(fields[0])
            .ok()
            .filter(|&value| value < count)
            .ok_or_else(|| invalid("translation index out of range"))?;
        let origin = usize::try_from(fields[1])
            .ok()
            .filter(|&value| value < sites)
            .ok_or_else(|| invalid("origin site out of range"))?;
        let target = usize::try_from(fields[2])
            .ok()
            .filter(|&value| value < sites)
            .ok_or_else(|| invalid("target site out of range"))?;
        entries[translation].site_map[origin] = target as i64;
        entries[translation].site_sign[origin] = fields[3] as i64;
        origins_written[translation * sites + origin] = true;
        targets_written[translation * sites + target] = true;
    }
    if origins_written.contains(&false) || targets_written.contains(&false) {
        return Err(invalid(
            "unwritten forward/inverse slot is not defined C input",
        ));
    }
    Ok(QPTransSection {
        n_qp_trans: count as i64,
        entries,
    })
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
        let imaginary = tokens
            .get(2)
            .map_or(0.0, |token| safe_parse_float(token, 0.0));
        if idx >= 0 && (idx as usize) < section.entries.len() {
            section.entries[idx as usize].weight = Complex64::new(weight, imaginary);
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
        if mpidx < 0 || j < 0 || itmp < 0 || mpidx >= n_qp_trans || j >= nsite || itmp >= nsite {
            continue;
        }
        let entry = &mut section.entries[mpidx as usize];
        entry.site_map[j as usize] = itmp;
        entry.site_sign[j as usize] = sgn;
    }

    section
}
