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

use crate::state::VmcOptimizationState;

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
        ((etot2 - etot * etot) / (etot * etot)).re
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
        format!("{:.18e}", sztot),
        format!("{:.18e}", sztot2),
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
    for term in &data.orbital_terms {
        write!(
            var_file,
            "{} {} 0.0 ",
            format_c_double(term.value.re),
            format_c_double(term.value.im),
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
    for term in &data.orbital_terms {
        writeln!(
            f,
            "{} {} ",
            format_c_double(term.value.re),
            format_c_double(term.value.im)
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
