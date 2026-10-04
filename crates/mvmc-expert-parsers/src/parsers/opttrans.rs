//! Strict optimized-translation definitions, committed only after validation.
use crate::types::ExpertModeData;
use crate::utils::file::{
    clean_line, julia_parse_float, julia_parse_int, read_def_file, split_def_line,
};
use num_complex::Complex64;
use std::{io, path::Path};

/// Complete `ParaQPOptTrans`, site mappings and boundary signs.
#[derive(Debug, Clone, PartialEq)]
pub struct OptTransDefinition {
    /// Initial real weights in sector-index order.
    pub parameters: Vec<Complex64>,
    /// Zero-based destination site for each sector and input site.
    pub site_maps: Vec<Vec<i64>>,
    /// Per-site signs; periodic boundary conditions force all signs to +1.
    pub site_signs: Vec<Vec<i64>>,
}

fn integer(token: &str, path: &str, line: usize, field: &str) -> Result<i64, String> {
    julia_parse_int(token).ok_or_else(|| format!("{path}:{line}: invalid {field} '{token}'"))
}

/// Parse the canonical five-header-line format without changing existing data.
/// Extra row tokens and repeated destinations are accepted as in Julia.
pub fn parse_opttrans_content(
    content: &str,
    nsite: i64,
    nmp_trans: i64,
    path: &str,
) -> Result<OptTransDefinition, String> {
    if nsite <= 0 {
        return Err(format!(
            "OptTrans requires ModPara/Nsite to be parsed before {path}"
        ));
    }
    let lines: Vec<_> = content.split('\n').collect();
    if lines.len() <= 5 {
        return Err(format!("{path} must include 5 header lines"));
    }
    let header = split_def_line(lines[1]);
    let count = header
        .get(1)
        .ok_or_else(|| format!("{path}:2 missing NQPOptTrans header"))?;
    let count = integer(count, path, 2, "NQPOptTrans")?;
    if count < 1 {
        return Err(format!("NQPOptTrans should be larger than 0 in {path}"));
    }
    let n = count as usize;
    let mut parameters = vec![Complex64::new(0.0, 0.0); n];
    let mut seen = vec![false; n];
    let mut cursor = 5;
    for _ in 0..n {
        while cursor < lines.len() && clean_line(lines[cursor]).is_empty() {
            cursor += 1;
        }
        if cursor == lines.len() {
            return Err(format!(
                "{path} ended before ParaQPOptTrans block was complete"
            ));
        }
        let line = cursor + 1;
        let tokens = split_def_line(lines[cursor]);
        if tokens.len() < 2 {
            return Err(format!("{path}:{line} expected 'idx value'"));
        }
        let idx = integer(tokens[0], path, line, "ParaQPOptTrans index")?;
        if !(0..count).contains(&idx) {
            return Err(format!(
                "{path}:{line} ParaQPOptTrans index {idx} out of range [0, {}]",
                count - 1
            ));
        }
        if seen[idx as usize] {
            return Err(format!(
                "{path}:{line} duplicated ParaQPOptTrans index {idx}"
            ));
        }
        let value = julia_parse_float(tokens[1]).ok_or_else(|| {
            format!(
                "{path}:{line}: invalid ParaQPOptTrans value '{}'",
                tokens[1]
            )
        })?;
        if !value.is_finite() {
            return Err(format!(
                "{path}:{line}: non-finite ParaQPOptTrans value '{}'",
                tokens[1]
            ));
        }
        parameters[idx as usize] = Complex64::new(value, 0.0);
        seen[idx as usize] = true;
        cursor += 1;
    }
    let mut site_maps = vec![vec![-1; nsite as usize]; n];
    let mut site_signs = vec![vec![1; nsite as usize]; n];
    let mut seen = vec![vec![false; nsite as usize]; n];
    for (index, text) in lines.iter().enumerate().skip(cursor) {
        let tokens = split_def_line(text);
        if tokens.is_empty() {
            continue;
        }
        let line = index + 1;
        if tokens.len() < 4 {
            return Err(format!("{path}:{line} expected 'optidx site mapped sign'"));
        }
        let sector = integer(tokens[0], path, line, "QPOptTrans index")?;
        let site = integer(tokens[1], path, line, "site index")?;
        let mapped = integer(tokens[2], path, line, "mapped site index")?;
        let sign = integer(tokens[3], path, line, "QPOptTrans sign")?;
        for (value, limit, label) in [
            (sector, count, "QPOptTrans index"),
            (site, nsite, "site index"),
            (mapped, nsite, "mapped site index"),
        ] {
            if !(0..limit).contains(&value) {
                return Err(format!(
                    "{path}:{line} {label} {value} out of range [0, {}]",
                    limit - 1
                ));
            }
        }
        if sign != 1 && sign != -1 {
            return Err(format!("{path}:{line} sign must be +1 or -1"));
        }
        if seen[sector as usize][site as usize] {
            return Err(format!(
                "{path}:{line} duplicated QPOptTrans entry optidx={sector} site={site}"
            ));
        }
        seen[sector as usize][site as usize] = true;
        site_maps[sector as usize][site as usize] = mapped;
        site_signs[sector as usize][site as usize] = if nmp_trans >= 0 { 1 } else { sign };
    }
    for (sector, sites) in seen.iter().enumerate() {
        for (site, &present) in sites.iter().enumerate() {
            if !present {
                return Err(format!(
                    "{path} missing QPOptTrans entry optidx={sector} site={site}"
                ));
            }
        }
    }
    Ok(OptTransDefinition {
        parameters,
        site_maps,
        site_signs,
    })
}

/// Replace an active definition atomically, retaining the prior state on error.
pub fn parse_opttrans_def(data: &mut ExpertModeData, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    let label = path.display().to_string();
    if data.modpara.nsite <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("OptTrans requires ModPara/Nsite to be parsed before {label}"),
        ));
    }
    let definition = parse_opttrans_content(
        &read_def_file(path)?,
        data.modpara.nsite,
        data.modpara.nmp_trans,
        &label,
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    data.n_qp_opt_trans = definition.parameters.len() as i64;
    data.opt_trans = definition.parameters.clone();
    data.para_qp_opt_trans = definition.parameters;
    data.qp_opt_trans = definition.site_maps;
    data.qp_opt_trans_sgn = definition.site_signs;
    Ok(())
}
