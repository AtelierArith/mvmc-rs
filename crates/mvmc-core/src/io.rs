//! `zvo_out.dat` / `zqp_opt.dat` / `zvo_var.dat` writers.
//!
//! Port target: `MVMCOptimizers.jl/src/data_io.jl`. The byte-for-byte
//! tolerance vs. C `%.18e` is documented in `docs/PORTING_PLAN.md`
//! Risks section -- short version: parse-then-compare for `zvo_out.dat`
//! (integration tests already do this), exact byte-diff for
//! `zqp_opt.dat` via a small `format_c_double` helper.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use mvmc_expert_parsers::ExpertModeData;

use crate::state::{OptDataPoint, VmcOptimizationState};

/// Store a synchronized parameter snapshot with the just-measured energy.
/// Slater coefficients follow C's declared index order, including unmapped
/// slots. Gaps in the zero-based sample index are empty snapshots.
pub fn store_opt_data(data: &ExpertModeData, state: &mut VmcOptimizationState, sample_idx: usize) {
    let parameters = data
        .gutzwiller_terms
        .iter()
        .map(|term| term.value)
        .chain(data.jastrow_terms.iter().map(|term| term.value))
        .chain(data.slater_params.iter().copied())
        .collect();
    if state.opt_data.len() <= sample_idx {
        state.opt_data.resize_with(sample_idx + 1, || OptDataPoint {
            energy: num_complex::Complex64::new(0.0, 0.0),
            parameters: Vec::new(),
        });
    }
    state.opt_data[sample_idx] = OptDataPoint {
        energy: state.energy.etot,
        parameters,
    };
}

/// Format a `f64` the way C’s `"% .18e"` does (sign-or-space
/// followed by a normalised mantissa with 18 fractional digits).
///
/// Rust’s `format!("{:+.18e}", ...)` already covers the digit count
/// and exponent, so the helper just patches the sign byte (`+` -> ` `)
/// and pads the exponent to at least two digits (C uses two by default
/// for IEEE-754 doubles).
pub fn format_c_double(value: f64) -> String {
    let mut s = format!("{:+.18e}", value);
    if let Some(stripped) = s.strip_prefix('+') {
        s = format!(" {stripped}");
    }
    // Pad single-digit exponents to two digits so the byte width matches C.
    if let Some(e_pos) = s.find('e') {
        let (mantissa, exp_with_e) = s.split_at(e_pos);
        let exp = &exp_with_e[1..];
        let (sign, digits) = if let Some(rest) = exp.strip_prefix('-') {
            ("-", rest)
        } else if let Some(rest) = exp.strip_prefix('+') {
            ("+", rest)
        } else {
            ("+", exp)
        };
        let digits_padded = if digits.len() < 2 {
            format!("{:0>2}", digits)
        } else {
            digits.to_string()
        };
        s = format!("{mantissa}e{sign}{digits_padded}");
    }
    s
}

/// Resolve `filename` against `output_dir`, creating the directory if
/// supplied. Mirrors `_output_path` in upstream Julia.
pub fn output_path(filename: &str, output_dir: Option<&Path>) -> io::Result<PathBuf> {
    if let Some(dir) = output_dir {
        fs::create_dir_all(dir)?;
        Ok(dir.join(filename))
    } else {
        Ok(PathBuf::from(filename))
    }
}

/// Append (or overwrite, when `step == 0`) one `zvo_out.dat` row.
/// Mirrors `output_data!(data, state, step; output_dir=...)`.
pub fn output_data(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    step: usize,
    output_dir: Option<&Path>,
) -> io::Result<()> {
    let head = if data.modpara.c_data_file_head.is_empty() {
        "zvo".to_string()
    } else {
        data.modpara.c_data_file_head.clone()
    };

    let etot = state.energy.etot;
    let etot2 = state.energy.etot2;
    let variance = if etot.norm() > 1.0e-14 {
        crate::julia_complex::divide(etot2 - etot * etot, etot * etot).re
    } else {
        0.0
    };
    let sztot = state.energy.sztot.re;
    let sztot2 = state.energy.sztot2.re;

    let mode_first = step == 0;

    let out_path = output_path(&format!("{head}_out.dat"), output_dir)?;
    let mut out_file = open_step_file(&out_path, mode_first)?;
    writeln!(
        out_file,
        "{} {}  {} {} {} {}",
        format_c_double(etot.re),
        format_c_double(etot.im),
        format_c_double(etot2.re),
        format_c_double(variance),
        // sztot / sztot2 are written without the leading space in the
        // upstream format string ("%.18e" rather than "% .18e").
        format_c_double(sztot).trim_start(),
        format_c_double(sztot2).trim_start(),
    )?;

    let var_path = output_path(&format!("{head}_var.dat"), output_dir)?;
    let mut var_file = open_step_file(&var_path, mode_first)?;
    write!(
        var_file,
        "{} {} 0.0 {} {} 0.0 ",
        format_c_double(etot.re),
        format_c_double(etot.im),
        format_c_double(etot2.re),
        format_c_double(etot2.im),
    )?;
    for term in &data.gutzwiller_terms {
        write!(
            var_file,
            "{} {} 0.0 ",
            format_c_double(term.value.re),
            format_c_double(term.value.im),
        )?;
    }
    for term in &data.jastrow_terms {
        write!(
            var_file,
            "{} {} 0.0 ",
            format_c_double(term.value.re),
            format_c_double(term.value.im),
        )?;
    }
    for value in &data.slater_params {
        write!(
            var_file,
            "{} {} 0.0 ",
            format_c_double(value.re),
            format_c_double(value.im),
        )?;
    }
    writeln!(var_file)?;

    Ok(())
}

/// Write the final `zqp_opt.dat` snapshot.
pub fn output_opt_data(data: &ExpertModeData, output_dir: Option<&Path>) -> io::Result<()> {
    let head = if data.modpara.c_para_file_head.is_empty() {
        "zqp".to_string()
    } else {
        data.modpara.c_para_file_head.clone()
    };
    let path = output_path(&format!("{head}_opt.dat"), output_dir)?;
    let mut f = File::create(&path)?;
    for term in &data.gutzwiller_terms {
        writeln!(
            f,
            "{} {} ",
            format_c_double(term.value.re),
            format_c_double(term.value.im)
        )?;
    }
    for term in &data.jastrow_terms {
        writeln!(
            f,
            "{} {} ",
            format_c_double(term.value.re),
            format_c_double(term.value.im)
        )?;
    }
    for value in &data.slater_params {
        writeln!(
            f,
            "{} {} ",
            format_c_double(value.re),
            format_c_double(value.im)
        )?;
    }
    drop(f);
    output_parameter_block(
        &head,
        "gutzwiller",
        "NGutzwillerIdx",
        data.gutzwiller_terms.iter().map(|term| term.value),
        output_dir,
    )?;
    output_parameter_block(
        &head,
        "jastrow",
        "NJastrowIdx",
        data.jastrow_terms.iter().map(|term| term.value),
        output_dir,
    )?;
    output_parameter_block(
        &head,
        "orbital",
        "NOrbitalIdx",
        data.slater_params.iter().copied(),
        output_dir,
    )?;
    Ok(())
}

fn output_parameter_block(
    head: &str,
    suffix: &str,
    label: &str,
    values: impl ExactSizeIterator<Item = num_complex::Complex64>,
    output_dir: Option<&Path>,
) -> io::Result<()> {
    if values.len() == 0 {
        return Ok(());
    }
    let path = output_path(&format!("{head}_{suffix}_opt.dat"), output_dir)?;
    let mut file = File::create(path)?;
    writeln!(file, "===============================")?;
    writeln!(file, "{label} {}", values.len())?;
    writeln!(file, "===============================")?;
    writeln!(file, "===============================")?;
    for (index, value) in values.enumerate() {
        writeln!(
            file,
            "{index} {} {} ",
            format_c_double(value.re),
            format_c_double(value.im)
        )?;
    }
    Ok(())
}

fn open_step_file(path: &Path, overwrite: bool) -> io::Result<File> {
    if overwrite {
        File::create(path)
    } else {
        OpenOptions::new().create(true).append(true).open(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_c_double_matches_c_pattern() {
        assert_eq!(format_c_double(1.5), " 1.500000000000000000e+00");
        assert_eq!(format_c_double(-2.0), "-2.000000000000000000e+00");
        assert_eq!(format_c_double(0.0), " 0.000000000000000000e+00");
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;
    use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};
    use num_complex::Complex64;

    #[test]
    fn history_keeps_declared_index_order_unmapped_slots_and_owned_snapshots() {
        let mut data = ExpertModeData::new();
        data.gutzwiller_terms.push(GutzwillerTerm {
            site: 0,
            value: Complex64::new(1.0, 2.0),
            is_complex: true,
        });
        data.jastrow_terms.push(JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(3.0, 4.0),
            is_complex: true,
        });
        data.modpara.n_orbital_idx = 3;
        data.slater_params = vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
        ];
        for idx in [1, 0, 1] {
            data.orbital_terms.push(OrbitalTerm {
                site1: 0,
                site2: 1,
                idx,
                is_complex: true,
                sign: 1,
            });
        }
        let mut state = VmcOptimizationState::zeros(2, 1, 2, 4, 1, 1, true, false);
        state.energy.etot = Complex64::new(-1.0, 0.25);
        store_opt_data(&data, &mut state, 2);
        assert_eq!(state.opt_data.len(), 3);
        assert!(state.opt_data[0].parameters.is_empty());
        assert_eq!(state.opt_data[0].energy, Complex64::new(0.0, 0.0));
        assert_eq!(state.opt_data[2].energy, state.energy.etot);
        let expected = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
        ];
        assert_eq!(state.opt_data[2].parameters, expected);
        data.slater_params[data.orbital_terms[0].idx as usize] = Complex64::new(9.0, 0.0);
        store_opt_data(&data, &mut state, 3);
        assert_eq!(state.opt_data[2].parameters, expected);
        assert_eq!(state.opt_data[3].parameters[3], Complex64::new(9.0, 0.0));
        store_opt_data(&data, &mut state, 0);
        assert_eq!(state.opt_data.len(), 4);
    }
}
