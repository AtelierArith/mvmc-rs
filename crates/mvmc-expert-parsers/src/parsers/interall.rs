//! Authoritative C InterAll header, bounded scan carry and term order.
use crate::types::InterAllTerm;
use crate::utils::{c_numeric::Scan, file::read_def_file};
use num_complex::Complex64;
use std::io;
use std::path::Path;

/// Read a C InterAll definition with the lattice bounds and TwoSz policy.
pub fn parse_interall_def<P: AsRef<Path>>(
    path: P,
    nsite: i64,
    two_sz: i64,
) -> io::Result<Vec<InterAllTerm>> {
    parse_interall_content(&read_def_file(path)?, nsite, two_sz)
}

/// Match ReadBuffInt, five fgets calls and GetInfoInterAll's initialized carry.
/// Comments and blank chunks count as terms; numeric prefixes stop at the first
/// failed conversion. Unsafe integer/spin ranges receive bounded diagnostics.
pub fn parse_interall_content(
    content: &str,
    nsite: i64,
    two_sz: i64,
) -> io::Result<Vec<InterAllTerm>> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut chunks = c_chunks(content.as_bytes());
    chunks
        .next()
        .ok_or_else(|| invalid("InterAll requires a count header on line 2"))?;
    let header = chunks
        .next()
        .ok_or_else(|| invalid("InterAll requires a count header on line 2"))?;
    let mut scan = Scan::new(header);
    let width = if scan.word() {
        scan.integer()
            .map_err(|_| invalid("InterAll count exceeds the C integer range"))?
            .unwrap_or(0)
    } else {
        0
    };
    if width < 0 {
        return Err(invalid("InterAll count must be nonnegative"));
    }
    if width == 0 {
        return Ok(Vec::new());
    }
    if nsite <= 0 || nsite > i64::from(i32::MAX) {
        return Err(invalid("InterAll requires a positive C-range Nsite"));
    }
    for _ in 0..3 {
        chunks.next();
    }
    let mut fields = [0_i64; 8];
    let mut real = 0.0;
    let mut imag = 0.0;
    let mut terms = Vec::new();
    for chunk in chunks {
        if terms.len() >= width as usize {
            return Err(invalid("InterAll has more rows than its declared count"));
        }
        let mut scan = Scan::new(chunk);
        let mut complete = true;
        for field in &mut fields {
            match scan
                .integer()
                .map_err(|_| invalid("InterAll index exceeds the C integer range"))?
            {
                Some(value) => *field = i64::from(value),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if complete {
            if let Some(value) = scan.float() {
                real = value;
                if let Some(value) = scan.float() {
                    imag = value;
                }
            }
        }
        if [0, 2, 4, 6]
            .iter()
            .any(|&index| fields[index] < 0 || fields[index] >= nsite)
        {
            return Err(invalid("InterAll site is outside Nsite"));
        }
        if [1, 3, 5, 7]
            .iter()
            .any(|&index| !(0..=1).contains(&fields[index]))
        {
            return Err(invalid("InterAll spin must be zero or one"));
        }
        if two_sz != -1 && (fields[1] != fields[3] || fields[5] != fields[7]) {
            return Err(invalid(
                "InterAll fixed TwoSz requires spin-conserving pairs",
            ));
        }
        terms.push(InterAllTerm {
            site0: fields[0],
            spin0: fields[1],
            site1: fields[2],
            spin1: fields[3],
            site2: fields[4],
            spin2: fields[5],
            site3: fields[6],
            spin3: fields[7],
            // C constructs dRe + I*dIm. Preserve zero signs and NaN propagation.
            value: Complex64::new(real + 0.0 * imag, imag),
            is_complex: imag.abs() > 1e-14,
        });
    }
    if terms.len() != width as usize {
        return Err(invalid("InterAll row count differs from its declaration"));
    }
    Ok(terms)
}

fn c_chunks(mut text: &[u8]) -> impl Iterator<Item = &[u8]> {
    std::iter::from_fn(move || {
        if text.is_empty() {
            return None;
        }
        let capacity = text.len().min(255);
        let count = text[..capacity]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(capacity, |index| index + 1);
        let (chunk, rest) = text.split_at(count);
        text = rest;
        Some(chunk)
    })
}
