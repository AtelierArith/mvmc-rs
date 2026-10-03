//! Complete C initial-parameter records, with the final record taking precedence.
use std::{fs, io, path::Path};

use mvmc_expert_parsers::{
    utils::{file::c_parse_float, parameter_init::n_slater},
    ExpertModeData,
};
use num_complex::Complex64;

fn load_para_triples(data: &mut ExpertModeData, text: &str) -> Result<usize, String> {
    let mut values = Vec::new();
    for (index, token) in text
        .split([' ', '\t', '\n', '\r', '\x0b', '\x0c'])
        .filter(|token| !token.is_empty())
        .enumerate()
    {
        let value = c_parse_float(token)
            .ok_or_else(|| format!("non-numeric token '{token}' at field {}", index + 1))?;
        values.push(value);
    }
    // C's EOF loop applies no values for an empty file.
    if values.is_empty() {
        return Ok(0);
    }
    let layout = data.projection_layout();
    let n_orbital = n_slater(data);
    let n_rbm = data.count_rbm_parameters();
    let n_opt_trans = data.count_opt_trans_parameters();
    let n_parameters = layout.n_proj + n_rbm + n_orbital + n_opt_trans;
    let expected = 6 + 3 * n_parameters;
    if values.len() < expected {
        return Err(format!(
            "too short: got {} floats, expected {expected} (6 + 3*(NProj={} + NRBM={n_rbm} + NSlater={n_orbital} + NOptTrans={n_opt_trans}))",
            values.len(), layout.n_proj
        ));
    }
    let extra = values.len() % expected;
    if extra > 0 {
        return Err(if n_opt_trans == 0 && extra.is_multiple_of(3) {
            format!(
                "OptTrans-style block of {} triples but OptTrans is not active",
                extra / 3
            )
        } else if extra.is_multiple_of(3) {
            format!("{extra} trailing floats (file likely malformed)")
        } else {
            format!(
                "{extra} trailing floats (not a whole number of triples; file likely malformed)"
            )
        });
    }

    // C applies each complete record; only the last one remains. Parse the
    // complete file before mutation so malformed inputs have bounded errors.
    let values = &values[values.len() - expected..];
    let parameter = |index: usize| {
        let real = values[6 + 3 * index];
        let imag = values[7 + 3 * index];
        // C constructs tmp_real + tmp_comp*I, rather than assigning its two
        // components directly. Preserve signed zeros and 0*infinity's NaN.
        Complex64::new(real + 0.0 * imag, imag)
    };
    for (index, term) in data
        .gutzwiller_terms
        .iter_mut()
        .take(layout.n_gutzwiller)
        .enumerate()
    {
        term.value = parameter(index);
    }
    for (index, term) in data
        .jastrow_terms
        .iter_mut()
        .take(layout.n_jastrow)
        .enumerate()
    {
        term.value = parameter(layout.jastrow_offset + index);
    }
    for (index, value) in data
        .doublon_holon_2site_params
        .iter_mut()
        .take(6 * layout.n_dh2)
        .enumerate()
    {
        *value = parameter(layout.dh2_offset + index);
    }
    for (index, value) in data
        .doublon_holon_4site_params
        .iter_mut()
        .take(10 * layout.n_dh4)
        .enumerate()
    {
        *value = parameter(layout.dh4_offset + index);
    }
    data.set_rbm_parameters(
        (0..n_rbm)
            .map(|index| parameter(layout.n_proj + index))
            .collect(),
    );
    data.slater_params = (0..n_orbital)
        .map(|index| parameter(layout.n_proj + n_rbm + index))
        .collect();
    for (index, value) in data.opt_trans.iter_mut().enumerate() {
        *value = parameter(layout.n_proj + n_rbm + n_orbital + index);
    }
    Ok(n_parameters)
}

/// Load an optional initial.def overlay without changing parameters on rejection.
///
/// Missing files and invalid records emit a warning and return `Ok(false)`,
/// Complete records follow C: the final record wins, empty files leave data
/// unchanged, and numeric range/nonfinite values are loaded as C does. Rust
/// rejects malformed records before mutation; C's unchecked scans do not define
/// a useful malformed-input contract. No RNG draws or normalization occur.
/// DH2/DH4 follow Jastrow, then the nine RBM sections, Slater and OptTrans.
pub fn read_initial_def(data: &mut ExpertModeData, path: impl AsRef<Path>) -> io::Result<bool> {
    let path = path.as_ref();
    if !path.is_file() {
        eprintln!("warning: initial.def not found (path: {})", path.display());
        return Ok(false);
    }
    match load_para_triples(data, &fs::read_to_string(path)?) {
        Ok(_) => Ok(true),
        Err(reason) => {
            eprintln!(
                "warning: read_initial_def!: {reason}; refusing to apply initial.def (path: {})",
                path.display()
            );
            Ok(false)
        }
    }
}

/// Strictly load optimized parameters, returning the number of consumed slots.
///
/// Rejects missing files, malformed tokens and incomplete records. Complete
/// records follow C conversion and last-record precedence. Empty files and
/// parameterless records return zero; rejected records leave data unchanged.
/// The layout is six diagnostics followed by projection/RBM/Slater/OptTrans triples.
pub fn read_opt_para_file(
    data: &mut ExpertModeData,
    path: impl AsRef<Path>,
) -> Result<usize, String> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(format!(
            "read_opt_para_file!: file not found: {}",
            path.display()
        ));
    }
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let consumed = load_para_triples(data, &text)
        .map_err(|reason| format!("read_opt_para_file!: {reason} (path: {})", path.display()))?;
    Ok(consumed)
}
