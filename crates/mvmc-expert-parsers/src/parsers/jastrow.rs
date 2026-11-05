//! Authoritative C 1.4.0 Jastrow headers, complete symmetric mappings and ordered raw flags.
use crate::types::JastrowTerm;
use crate::utils::file::read_def_file;
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

/// Declared coefficients, directional site indices and flag assignments.
#[derive(Debug, Clone, Default)]
pub struct JastrowSection {
    /// Validated raw signed C header, before the local >0 test.
    pub complex_type: i32,
    /// One coefficient slot per declared parameter, including unmapped slots.
    pub terms: Vec<JastrowTerm>,
    /// Declared coefficient count.
    pub n_jastrow_idx: i64,
    /// Positive local complex header enables imaginary flag writes.
    pub is_complex: bool,
    /// Raw flags assigned by pair order, ignoring printed labels.
    pub opt_flags: BTreeMap<i64, i64>,
    /// Symmetric site-to-parameter table (mVMC 1.4.0); the diagonal is -1.
    pub idx_matrix: Vec<Vec<i64>>,
}

/// Read a complete C Jastrow definition for the supplied geometry.
pub fn parse_jastrow_def<P: AsRef<Path>>(path: P, nsite: i64) -> io::Result<JastrowSection> {
    parse_jastrow_content(&read_def_file(path)?, nsite)
}

/// Parse five physical headers, Nsite*(Nsite-1) directed triples and flag pairs.
/// Unsafe indices and malformed C scans receive bounded Rust diagnostics.
pub fn parse_jastrow_content(content: &str, nsite: i64) -> io::Result<JastrowSection> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    let lines: Vec<_> = content.lines().collect();
    if lines.len() < 5 {
        return Err(invalid("Jastrow definition requires five header lines"));
    }
    let n_jastrow_idx = c_fields(lines[1])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .filter(|&width| width > 0)
        .ok_or_else(|| invalid("Jastrow declaration on line 2 must have a positive width"))?
        as i64;
    let complex_type = c_fields(lines[2])
        .nth(1)
        .and_then(|field| field.parse::<i32>().ok())
        .ok_or_else(|| invalid("Jastrow ComplexType must be a C integer"))?;
    let is_complex = complex_type > 0;
    let nsite = usize::try_from(nsite)
        .ok()
        .filter(|&count| count >= 2)
        .ok_or_else(|| invalid("Jastrow definitions require at least two sites"))?;
    let n_mapping = nsite
        .checked_mul(nsite - 1)
        .ok_or_else(|| invalid("Jastrow mapping count overflows"))?;
    let mapping_fields = n_mapping
        .checked_mul(3)
        .ok_or_else(|| invalid("Jastrow mapping field count overflows"))?;
    let expected_fields = (n_jastrow_idx as usize)
        .checked_mul(2)
        .and_then(|count| count.checked_add(mapping_fields))
        .ok_or_else(|| invalid("Jastrow field count overflows"))?;
    let body = lines[5..].join("\n");
    let fields: Vec<_> = c_fields(&body).collect();
    if fields.len() != expected_fields {
        return Err(invalid(
            "Jastrow requires Nsite*(Nsite-1) directed triples and exactly the declared flag pairs",
        ));
    }
    let integer = |field: &str| {
        field
            .parse::<i32>()
            .map(i64::from)
            .map_err(|_| invalid("invalid integer in Jastrow definition"))
    };
    // mVMC 1.4.0 GetInfoJastrow (readdef.c:2477-2587): -1 on the diagonal, -2
    // for an unread pair; every directed pair exactly once, with the same class
    // for (i,j) and (j,i). The table is then symmetric.
    const UNREAD: i64 = -2;
    let mut idx_matrix: Vec<Vec<i64>> = (0..nsite)
        .map(|i| {
            (0..nsite)
                .map(|j| if i == j { -1 } else { UNREAD })
                .collect()
        })
        .collect();
    for (row, triple) in fields[..mapping_fields]
        .as_chunks::<3>()
        .0
        .iter()
        .enumerate()
    {
        let site1 = integer(triple[0])?;
        let site2 = integer(triple[1])?;
        let idx = integer(triple[2])?;
        let row = row + 1;
        if site1 == site2 {
            return Err(invalid(&format!(
                "Jastrow mappings require distinct sites ([Condition] i neq j at row {row})"
            )));
        }
        if site1 < 0 || site2 < 0 || site1 >= nsite as i64 || site2 >= nsite as i64 {
            return Err(invalid(&format!(
                "Jastrow site is outside Nsite (site index is incorrect at row {row})"
            )));
        }
        if idx < 0 || idx >= n_jastrow_idx {
            return Err(invalid(&format!(
                "Jastrow parameter index {idx} is outside the declared width [0, {n_jastrow_idx}) \
                 at row {row}"
            )));
        }
        let (i, j) = (site1 as usize, site2 as usize);
        if idx_matrix[i][j] != UNREAD {
            return Err(invalid(&format!(
                "Jastrow pair ({i},{j}) is duplicated at row {row}"
            )));
        }
        idx_matrix[i][j] = idx;
        let reverse = idx_matrix[j][i];
        if reverse != UNREAD && reverse != idx {
            let (a, b) = (i.min(j), i.max(j));
            return Err(invalid(&format!(
                "Jastrow pair ({a},{b}) has inconsistent classes at row {row}"
            )));
        }
    }
    // Nsite*(Nsite-1) distinct directed pairs cover the table, so nothing is
    // missing here; C checks it explicitly after the loop.
    debug_assert!(idx_matrix.iter().flatten().all(|&idx| idx != UNREAD));
    let mut opt_flags = BTreeMap::new();
    for (index, pair) in fields[mapping_fields..]
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
    {
        let _printed_index = integer(pair[0])?;
        opt_flags.insert(index as i64, integer(pair[1])?);
    }
    // Terms represent the dense coefficient array, not individual mapping rows.
    // Site metadata is a deterministic representative; kernels use idx_matrix.
    let mut representatives = vec![(0, 1); n_jastrow_idx as usize];
    let mut assigned = vec![false; n_jastrow_idx as usize];
    for (site1, row) in idx_matrix.iter().enumerate() {
        for (site2, &idx) in row.iter().enumerate() {
            if idx >= 0 && !assigned[idx as usize] {
                representatives[idx as usize] = (site1 as i64, site2 as i64);
                assigned[idx as usize] = true;
            }
        }
    }
    let terms = representatives
        .into_iter()
        .map(|(site1, site2)| JastrowTerm {
            site1,
            site2,
            value: Complex64::new(0.0, 0.0),
            is_complex,
        })
        .collect();
    Ok(JastrowSection {
        complex_type,
        terms,
        n_jastrow_idx,
        is_complex,
        opt_flags,
        idx_matrix,
    })
}

fn c_fields(text: &str) -> impl Iterator<Item = &str> {
    text.split([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
        .filter(|field| !field.is_empty())
}
