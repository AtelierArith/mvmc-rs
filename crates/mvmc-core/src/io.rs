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
use num_complex::Complex64;

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
        .chain(data.rbm_params.iter().copied())
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
    for value in &data.rbm_params {
        write!(
            var_file,
            "{} {} 0.0 ",
            format_c_double(value.re),
            format_c_double(value.im),
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

/// Write one PhysCal Green-function sample using C's indexed file names.
pub fn output_phys_data(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    sample: usize,
    output_dir: Option<&Path>,
) -> io::Result<()> {
    let Some(phys) = state.phys_quantities.as_ref() else {
        return Ok(());
    };
    let head = if data.modpara.c_data_file_head.is_empty() {
        "zvo"
    } else {
        data.modpara.c_data_file_head.as_str()
    };
    let index = data.modpara.n_data_idx_start.max(0) as usize + sample;
    let write_rows = |suffix: &str, rows: Vec<String>| -> io::Result<()> {
        let path = output_path(&format!("{head}_{suffix}_{index:03}.dat"), output_dir)?;
        let mut file = File::create(path)?;
        for row in rows {
            writeln!(file, "{row}")?;
        }
        Ok(())
    };
    let one_rows = data
        .green_one_terms
        .iter()
        .zip(&phys.phys_cis_ajs)
        .map(|(term, value)| {
            format!(
                "{} {} {} {} {} {}",
                term.site1,
                crate::observables::spin_code(term.spin1),
                term.site2,
                crate::observables::spin_code(term.spin2),
                format_c_double(value.re),
                format_c_double(value.im)
            )
        })
        .collect();
    write_rows("cisajs", one_rows)?;
    let ex_rows = data
        .green_two_ex_terms
        .iter()
        .zip(&phys.phys_cis_ajs_ckt_alt)
        .map(|(_term, value)| {
            format!(
                "{} {}",
                format_c_double(value.re),
                format_c_double(value.im)
            )
        })
        .collect();
    write_rows("cisajscktaltex", ex_rows)?;
    let direct_rows = data
        .green_two_terms
        .iter()
        .zip(&phys.phys_cis_ajs_ckt_alt_dc)
        .map(|(term, value)| {
            format!(
                "{} {} {} {} {} {} {} {} {} {}",
                term.site1,
                crate::observables::spin_code(term.spin1),
                term.site2,
                crate::observables::spin_code(term.spin2),
                term.site3,
                crate::observables::spin_code(term.spin3),
                term.site4,
                crate::observables::spin_code(term.spin4),
                format_c_double(value.re),
                format_c_double(value.im)
            )
        })
        .collect();
    write_rows("cisajscktalt", direct_rows)?;

    if data.modpara.lanczos_mode > 0 {
        let qqqq_path = output_path(&format!("{head}_ls_qqqq_{index:03}.dat"), output_dir)?;
        let mut qqqq_file = File::create(qqqq_path)?;
        for value in &phys.phys_lanczos_qqqq {
            write!(qqqq_file, "{}  ", format_c_double(value.re))?;
        }
        writeln!(qqqq_file)?;

        let (energy, variance, alpha) =
            match crate::lanczos::lanczos_energy(&phys.phys_lanczos_qqqq) {
                Ok(result) => (result.energy, result.variance, result.alpha),
                Err(_) => (f64::NAN, f64::NAN, f64::NAN),
            };
        let ls_path = output_path(&format!("{head}_ls_out_{index:03}.dat"), output_dir)?;
        let mut ls_file = File::create(ls_path)?;
        writeln!(
            ls_file,
            "{}  {}  {}",
            format_c_double(energy),
            format_c_double(variance),
            format_c_double(alpha)
        )?;

        if data.modpara.lanczos_mode > 1 {
            let one_values = lanczos_phys_values(
                &phys.phys_lanczos_qqqq,
                &phys.phys_lanczos_qcisajsq,
                data.green_one_terms.len(),
                alpha,
            );
            let one_rows = data
                .green_one_terms
                .iter()
                .zip(one_values)
                .map(|(term, value)| {
                    format!(
                        "{} {} {} {} {} {}",
                        term.site1,
                        crate::observables::spin_code(term.spin1),
                        term.site2,
                        crate::observables::spin_code(term.spin2),
                        format_c_double(value.re),
                        format_c_double(value.im)
                    )
                })
                .collect();
            write_rows("ls_cisajs", one_rows)?;

            let direct_values = lanczos_phys_values(
                &phys.phys_lanczos_qqqq,
                &phys.phys_lanczos_qcisajscktaltq_dc,
                data.green_two_terms.len(),
                alpha,
            );
            let direct_rows = data
                .green_two_terms
                .iter()
                .zip(direct_values)
                .map(|(term, value)| {
                    format!(
                        "{} {} {} {} {} {} {} {} {} {}",
                        term.site1,
                        crate::observables::spin_code(term.spin1),
                        term.site2,
                        crate::observables::spin_code(term.spin2),
                        term.site3,
                        crate::observables::spin_code(term.spin3),
                        term.site4,
                        crate::observables::spin_code(term.spin4),
                        format_c_double(value.re),
                        format_c_double(value.im)
                    )
                })
                .collect();
            write_rows("ls_cisajscktalt", direct_rows)?;

            let factored_values = lanczos_phys_values(
                &phys.phys_lanczos_qqqq,
                &phys.phys_lanczos_qcisajscktaltq,
                data.green_two_ex_indices.len(),
                alpha,
            );
            let factored_rows = factored_values
                .into_iter()
                .map(|value| {
                    format!(
                        "{} {}",
                        format_c_double(value.re),
                        format_c_double(value.im)
                    )
                })
                .collect();
            write_rows("ls_cisajscktaltex", factored_rows)?;
        }
    }
    Ok(())
}

fn lanczos_phys_values(
    qqqq: &[Complex64],
    qphysq: &[Complex64],
    nphys: usize,
    alpha: f64,
) -> Vec<Complex64> {
    if qqqq.len() < 4 || qphysq.len() != 4 * nphys {
        return vec![Complex64::new(f64::NAN, f64::NAN); nphys];
    }
    let h1 = qqqq[2];
    let h2_1 = qqqq[3];
    let dnorm = (Complex64::new(1.0, 0.0) + 2.0 * alpha * h1 + alpha * alpha * h2_1).re;
    (0..nphys)
        .map(|index| {
            (qphysq[index]
                + alpha * (qphysq[nphys + index] + qphysq[2 * nphys + index])
                + alpha * alpha * qphysq[3 * nphys + index])
                / dnorm
        })
        .collect()
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
    for value in &data.rbm_params {
        writeln!(
            f,
            "{} {} ",
            format_c_double(value.re),
            format_c_double(value.im)
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
    let mut offset = 0;
    for (section, width) in data.rbm_section_sizes().into_iter().enumerate() {
        let (suffix, label) = RBM_OUTPUT_BLOCKS[section];
        output_parameter_block(
            &head,
            suffix,
            label,
            data.rbm_params[offset..offset + width].iter().copied(),
            output_dir,
        )?;
        offset += width;
    }
    output_parameter_block(
        &head,
        "orbital",
        "NOrbitalIdx",
        data.slater_params.iter().copied(),
        output_dir,
    )?;
    Ok(())
}

const RBM_OUTPUT_BLOCKS: [(&str, &str); 9] = [
    ("chargeRBM_physlayer", "NChargeRBM_PhysLayerIdx"),
    ("spinRBM_physlayer", "NSpinRBM_PhysLayerIdx"),
    ("generalRBM_physlayer", "NGeneralRBM_PhysLayerIdx"),
    ("chargeRBM_hiddenlayer", "NChargeRBM_HiddenLayerIdx"),
    ("spinRBM_hiddenlayer", "NSpinRBM_HiddenLayerIdx"),
    ("generalRBM_hiddenlayer", "NGeneralRBM_HiddenLayerIdx"),
    ("chargeRBM_physhidden", "NChargeRBM_PhysHiddenIdx"),
    ("spinRBM_physhidden", "NSpinRBM_PhysHiddenIdx"),
    ("generalRBM_physhidden", "NGeneralRBM_PhysHiddenIdx"),
];

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
    writeln!(file, "======================")?;
    writeln!(file, "{label} {}", values.len())?;
    writeln!(file, "======================")?;
    writeln!(file, "======================")?;
    writeln!(file, "======================")?;
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
    use mvmc_expert_parsers::{GreenOneTerm, GreenTwoExTerm, GreenTwoTerm, Spin};
    use num_complex::Complex64;

    #[test]
    fn format_c_double_matches_c_pattern() {
        assert_eq!(format_c_double(1.5), " 1.500000000000000000e+00");
        assert_eq!(format_c_double(-2.0), "-2.000000000000000000e+00");
        assert_eq!(format_c_double(0.0), " 0.000000000000000000e+00");
    }

    #[test]
    fn phys_data_uses_c_indexed_green_file_names_and_rows() {
        let mut data = ExpertModeData::new();
        data.modpara.c_data_file_head = "zvo".to_string();
        data.modpara.n_data_idx_start = 7;
        data.modpara.lanczos_mode = 1;
        data.green_one_terms.push(GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Down,
        });
        data.green_two_ex_terms.push(GreenTwoExTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Down,
            site3: 2,
            spin3: Spin::Up,
            site4: 3,
            spin4: Spin::Down,
        });
        data.green_two_terms.push(GreenTwoTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Down,
            site3: 2,
            spin3: Spin::Up,
            site4: 3,
            spin4: Spin::Down,
        });
        let mut state = VmcOptimizationState::zeros(4, 1, 0, 0, 1, 1, true, false);
        let mut phys = crate::state::PhysicalQuantities::zeros(1, 1, 1);
        phys.phys_cis_ajs[0] = Complex64::new(1.5, -2.0);
        phys.phys_cis_ajs_ckt_alt[0] = Complex64::new(3.0, 4.0);
        phys.phys_cis_ajs_ckt_alt_dc[0] = Complex64::new(-5.0, 6.0);
        phys.phys_lanczos_qqqq[2] = Complex64::new(1.0, 0.0);
        phys.phys_lanczos_qqqq[3] = Complex64::new(2.0, 0.0);
        phys.phys_lanczos_qqqq[10] = Complex64::new(3.0, 0.0);
        phys.phys_lanczos_qqqq[11] = Complex64::new(4.0, 0.0);
        phys.phys_lanczos_qqqq[15] = Complex64::new(5.0, 0.0);
        state.phys_quantities = Some(phys);
        let output_dir =
            std::env::temp_dir().join(format!("mvmc-phys-output-{}", std::process::id()));
        let _ = fs::remove_dir_all(&output_dir);
        output_phys_data(&data, &state, 2, Some(&output_dir)).unwrap();
        let one = fs::read_to_string(output_dir.join("zvo_cisajs_009.dat")).unwrap();
        assert!(one.contains("0 0 1 1  1.500000000000000000e+00 -2.000000000000000000e+00"));
        let ex = fs::read_to_string(output_dir.join("zvo_cisajscktaltex_009.dat")).unwrap();
        assert!(ex.contains(" 3.000000000000000000e+00  4.000000000000000000e+00"));
        assert!(output_dir.join("zvo_cisajscktalt_009.dat").exists());
        let qqqq = fs::read_to_string(output_dir.join("zvo_ls_qqqq_009.dat")).unwrap();
        assert!(qqqq.split_whitespace().count() == 16);
        let ls = fs::read_to_string(output_dir.join("zvo_ls_out_009.dat")).unwrap();
        assert_eq!(ls.split_whitespace().count(), 3);
        let _ = fs::remove_dir_all(output_dir);
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
