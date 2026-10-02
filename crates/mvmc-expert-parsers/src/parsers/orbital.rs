//! C AP/P/General declared widths and complete mapping/flag sections.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use crate::types::OrbitalTerm;
use crate::utils::file::read_def_file;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Orbital mapping geometry selected by the namelist keyword.
pub enum OrbitalKind {
    /// Ordered spatial pairs for opposite spins.
    AntiParallel,
    /// Upper-triangle spatial pairs, expanded into both equal-spin blocks.
    Parallel,
    /// Six-column General mappings, converted to combined spin-site coordinates.
    General,
}

/// Output of [`parse_orbital_content`].
#[derive(Debug, Clone, Default)]
pub struct OrbitalSection {
    /// `(site1, site2, idx, sign)` entries from the OrbitalIdx block.
    pub terms: Vec<OrbitalTerm>,
    /// `NOrbitalIdx` from the header (number of unique parameters).
    pub n_orbital_idx: i64,
    /// `ComplexType` flag.
    pub is_complex: bool,
    /// Per-parameter flags from the trailing OptFlag section.
    pub opt_flags: BTreeMap<i64, i64>,
}

/// Parse a `orbitalidx*.def` file from disk.
pub fn parse_orbital_def<P: AsRef<Path>>(
    path: P,
    nsite: i64,
    kind: OrbitalKind,
) -> io::Result<OrbitalSection> {
    let content = read_def_file(path)?;
    parse_orbital_content(&content, nsite, kind)
}

/// Parse a `orbitalidx*.def` payload from memory.
pub fn parse_orbital_content(
    content: &str,
    nsite: i64,
    kind: OrbitalKind,
) -> io::Result<OrbitalSection> {
    let lines: Vec<&str> = content.lines().collect();
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    if lines.len() < 5 {
        return Err(invalid("orbital definition requires five header lines"));
    }
    // ReadBuffIntCmpFlg reads physical lines 2/3, ignoring their labels.
    let n_orbital_idx = c_fields(lines[1])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .filter(|&width| width > 0)
        .ok_or_else(|| invalid("orbital declaration on line 2 must have a positive width"))?
        as i64;
    let is_complex = c_fields(lines[2])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .unwrap_or(0)
        > 0;
    let nsite = usize::try_from(nsite)
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| invalid("orbital definitions require a positive Nsite"))?;
    if kind == OrbitalKind::Parallel && nsite < 2 {
        return Err(invalid(
            "parallel orbital definitions require at least two sites",
        ));
    }
    let n_mapping = match kind {
        OrbitalKind::Parallel => nsite.checked_mul(nsite - 1).map(|count| count / 2),
        OrbitalKind::AntiParallel => nsite.checked_mul(nsite),
        OrbitalKind::General => nsite
            .checked_mul(nsite)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_sub(nsite)),
    }
    .ok_or_else(|| invalid("orbital mapping count overflows"))?;
    let body = &lines[5..];
    if body.len() < n_mapping {
        return Err(invalid("incomplete orbital mapping section"));
    }
    let mut terms = Vec::new();
    // sscanf retains initialized/preceding optional values: AP/P sign, and
    // General's index/sign following the four required coordinate fields.
    let mut sign = 1;
    let mut general_idx = 0;
    for line in &body[..n_mapping] {
        let fields: Vec<_> = c_fields(line).collect();
        let integer = |position: usize| {
            fields
                .get(position)
                .and_then(|field| field.parse::<i32>().ok())
                .map(i64::from)
                .ok_or_else(|| invalid("invalid integer in orbital mapping section"))
        };
        let (mut site1, mut site2, idx) = if kind == OrbitalKind::General {
            if fields.len() >= 5 {
                general_idx = integer(4)?;
            }
            (integer(0)?, integer(2)?, general_idx)
        } else {
            (integer(0)?, integer(1)?, integer(2)?)
        };
        if site1 < 0 || site2 < 0 || site1 >= nsite as i64 || site2 >= nsite as i64 {
            return Err(invalid("orbital mapping site is outside Nsite"));
        }
        if kind == OrbitalKind::Parallel && site1 >= site2 {
            return Err(invalid("parallel orbital mappings require site1 < site2"));
        }
        if kind == OrbitalKind::General {
            let spin1 = integer(1)?;
            let spin2 = integer(3)?;
            if !(0..=1).contains(&spin1) || !(0..=1).contains(&spin2) {
                return Err(invalid("General orbital spin must be zero or one"));
            }
            site1 += spin1 * nsite as i64;
            site2 += spin2 * nsite as i64;
            if site1 >= site2 {
                return Err(invalid(
                    "General orbital mappings require site1 + spin1*Nsite < site2 + spin2*Nsite",
                ));
            }
            if fields.len() >= 6 {
                sign = integer(5)?;
            }
        }
        if idx < 0 || idx >= n_orbital_idx {
            return Err(invalid(
                "orbital parameter index is outside the declared width",
            ));
        }
        if kind != OrbitalKind::General && fields.len() >= 4 {
            sign = integer(3)?;
        }
        terms.push(OrbitalTerm {
            site1,
            site2,
            idx,
            sign,
            is_complex,
        });
    }
    // GetInfoOpt scans whitespace-separated pairs to EOF, independent of line
    // breaks. Printed indices are consumed but flags are assigned by row order.
    let flags_text = body[n_mapping..].join("\n");
    let fields: Vec<_> = c_fields(&flags_text).collect();
    if fields.len() != 2 * n_orbital_idx as usize {
        return Err(invalid(
            "orbital flag section must contain exactly the declared number of pairs",
        ));
    }
    let mut opt_flags = BTreeMap::new();
    for (index, pair) in fields.chunks_exact(2).enumerate() {
        let _printed_index = pair[0]
            .parse::<i32>()
            .map_err(|_| invalid("invalid printed integer in orbital flag section"))?;
        let flag = pair[1]
            .parse::<i32>()
            .map_err(|_| invalid("invalid integer in orbital flag section"))?;
        opt_flags.insert(index as i64, i64::from(flag));
    }
    Ok(OrbitalSection {
        terms,
        n_orbital_idx,
        is_complex,
        opt_flags,
    })
}

fn c_fields(text: &str) -> impl Iterator<Item = &str> {
    text.split([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
        .filter(|field| !field.is_empty())
}
