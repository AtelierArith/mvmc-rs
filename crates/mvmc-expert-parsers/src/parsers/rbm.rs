//! Authoritative C RBM declarations, complete geometry and ordered raw flags.
use std::{collections::BTreeMap, io};

/// Canonical order of the nine RBM coefficient blocks.
pub const SECTION_NAMES: [&str; 9] = [
    "ChargeRBM_PhysLayer",
    "SpinRBM_PhysLayer",
    "GeneralRBM_PhysLayer",
    "ChargeRBM_HiddenLayer",
    "SpinRBM_HiddenLayer",
    "GeneralRBM_HiddenLayer",
    "ChargeRBM_PhysHidden",
    "SpinRBM_PhysHidden",
    "GeneralRBM_PhysHidden",
];

/// Final spatial assignments and complete coefficient declarations.
#[derive(Debug)]
pub struct RbmSection {
    /// Positive width read from the second physical header line.
    pub width: usize,
    /// Positive local complex flag enables imaginary optimization flags.
    pub is_complex: bool,
    /// Raw flags in pair order; printed labels do not choose slots.
    pub opt_flags: BTreeMap<i64, i64>,
    /// Final coordinate rows followed by their coefficient index.
    /// Duplicate coordinates keep the last assignment, as in C.
    pub mappings: Vec<Vec<i64>>,
}

/// Read one of the nine C RBM sections for its physical/hidden geometry.
/// Section indices follow SECTION_NAMES. Missing sections have zero width;
/// an explicitly supplied section requires a positive declared width.
pub fn parse_rbm_content(
    content: &str,
    section: usize,
    nsite: i64,
    nhidden: i64,
) -> io::Result<RbmSection> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    if section >= SECTION_NAMES.len() {
        return Err(invalid("unknown RBM section"));
    }
    let lines: Vec<_> = content.lines().collect();
    if lines.len() < 5 {
        return Err(invalid(
            "RBM definition requires five physical header lines",
        ));
    }
    let width = fields(lines[1])
        .nth(1)
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid("RBM declaration requires a positive width"))?
        as usize;
    let is_complex = fields(lines[2])
        .nth(1)
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0)
        > 0;
    let dimension = |value| {
        usize::try_from(value)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| invalid("RBM geometry requires a positive dimension"))
    };
    let nsite = dimension(nsite)?;
    let bounds = match section {
        0 | 1 => vec![nsite],
        2 => vec![nsite, 2],
        3..=5 => vec![dimension(nhidden)?],
        6 | 7 => vec![nsite, dimension(nhidden)?],
        8 => vec![nsite, 2, dimension(nhidden)?],
        _ => unreachable!(),
    };
    let mapping_count = bounds
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| invalid("RBM mapping count overflows"))?;
    let mapping_fields = mapping_count
        .checked_mul(bounds.len() + 1)
        .ok_or_else(|| invalid("RBM mapping field count overflows"))?;
    let expected = width
        .checked_mul(2)
        .and_then(|v| v.checked_add(mapping_fields))
        .ok_or_else(|| invalid("RBM field count overflows"))?;
    let body = lines[5..].join("\n");
    let values: Vec<_> = fields(&body).collect();
    if values.len() != expected {
        return Err(invalid(
            "RBM requires complete spatial mappings and exactly the declared flag pairs",
        ));
    }
    let integer = |text: &str| {
        text.parse::<i32>()
            .map(i64::from)
            .map_err(|_| invalid("invalid integer in RBM definition"))
    };
    let mut assignments = BTreeMap::new();
    for row in values[..mapping_fields].chunks_exact(bounds.len() + 1) {
        let coordinates: Vec<i64> = row[..bounds.len()]
            .iter()
            .map(|v| integer(v))
            .collect::<Result<_, _>>()?;
        if coordinates
            .iter()
            .zip(&bounds)
            .any(|(&v, &bound)| v < 0 || v as usize >= bound)
        {
            return Err(invalid("RBM coordinate is outside the declared geometry"));
        }
        let index = integer(row[bounds.len()])?;
        if index < 0 || index as usize >= width {
            return Err(invalid(
                "RBM coefficient index is outside the declared width",
            ));
        }
        assignments.insert(coordinates, index);
    }
    let mut opt_flags = BTreeMap::new();
    for (index, pair) in values[mapping_fields..]
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
    {
        integer(pair[0])?; // C ignores the printed label.
        opt_flags.insert(index as i64, integer(pair[1])?);
    }
    // C's final arrays are traversed with spin blocks before physical sites.
    // Sort General rows by spin, site and hidden coordinate accordingly.
    let mut mappings: Vec<_> = assignments
        .into_iter()
        .map(|(mut coords, index)| {
            coords.push(index);
            coords
        })
        .collect();
    if section == 2 || section == 8 {
        mappings.sort_by_key(|v| (v[1], v[0], if section == 8 { v[2] } else { 0 }));
    }
    Ok(RbmSection {
        width,
        is_complex,
        opt_flags,
        mappings,
    })
}

fn fields(text: &str) -> impl Iterator<Item = &str> {
    text.split([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
        .filter(|v| !v.is_empty())
}
