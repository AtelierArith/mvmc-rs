//! Atomic loaders for Julia's six-diagnostic-fields plus parameter-triples format.
use std::{fs, io, path::Path};

use mvmc_expert_parsers::{utils::parameter_init::n_slater, ExpertModeData};
use num_complex::Complex64;

fn load_para_triples(data: &mut ExpertModeData, text: &str) -> Result<usize, String> {
    let mut values = Vec::new();
    for (index, token) in text.split_whitespace().enumerate() {
        let value = token
            .parse::<f64>()
            .map_err(|_| format!("non-numeric token '{token}' at field {}", index + 1))?;
        if !value.is_finite() {
            return Err(format!("non-finite token '{token}' at field {}", index + 1));
        }
        values.push(value);
    }
    let layout = data.projection_layout();
    let n_orbital = n_slater(data);
    let n_rbm = data.count_rbm_parameters();
    let n_parameters = layout.n_proj + n_rbm + n_orbital;
    let expected = 6 + 3 * n_parameters;
    if values.len() < expected {
        return Err(format!(
            "too short: got {} floats, expected {expected} (6 + 3*(NProj={} + NRBM={n_rbm} + NSlater={n_orbital} + NOptTrans=0))",
            values.len(), layout.n_proj
        ));
    }
    let extra = values.len() - expected;
    if extra > 0 {
        return Err(if extra % 3 == 0 {
            format!(
                "OptTrans-style block of {} triples but OptTrans is not active",
                extra / 3
            )
        } else {
            format!(
                "{extra} trailing floats (not a whole number of triples; file likely malformed)"
            )
        });
    }

    // All tokens, including diagnostics and gradients, are valid before mutation.
    let parameter = |index: usize| Complex64::new(values[6 + 3 * index], values[7 + 3 * index]);
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
    let sizes = data.rbm_section_sizes();
    let mut offsets = [layout.n_proj; 9];
    for i in 1..9 {
        offsets[i] = offsets[i - 1] + sizes[i - 1];
    }
    data.visit_rbm_terms_mut(|section, term| {
        let index = term.idx();
        if index >= 0 && (index as usize) < sizes[section] {
            term.set_value(parameter(offsets[section] + index as usize));
        }
    });
    for term in &mut data.orbital_terms {
        if term.idx >= 0 && (term.idx as usize) < n_orbital {
            term.value = parameter(layout.n_proj + n_rbm + term.idx as usize);
        }
    }
    Ok(n_parameters)
}

/// Load an optional initial.def overlay without changing parameters on rejection.
///
/// Missing files and invalid records emit a warning and return `Ok(false)`,
/// matching Julia's recoverable contract. Other file-reading failures propagate
/// as I/O errors, as Julia's `read` does. No RNG draws or normalization occur.
/// DH2/DH4 follow Jastrow, then the nine RBM sections and Slater; OptTrans is pending.
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
/// Rejects missing files, malformed tokens, non-finite values, incorrect record
/// lengths and models with no parameters. Rejected records leave data unchanged.
/// The supported layout is six diagnostics followed by projection/RBM/Slater triples.
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
    if consumed == 0 {
        return Err(format!("read_opt_para_file!: no parameters consumed (no Gutzwiller/Jastrow/Slater terms) from {}", path.display()));
    }
    Ok(consumed)
}
