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
        .projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect();
    if state.opt_data.len() <= sample_idx {
        state.opt_data.resize_with(sample_idx + 1, || OptDataPoint {
            energy: num_complex::Complex64::new(0.0, 0.0),
            energy_squared: num_complex::Complex64::new(0.0, 0.0),
            parameters: Vec::new(),
        });
    }
    state.opt_data[sample_idx] = OptDataPoint {
        energy: state.energy.etot,
        energy_squared: state.energy.etot2,
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
    // C vmcmain.c:655–657 writes every Para[0..NPara] slot, not mapped
    // coefficient families. DH2/DH4 and inactive/reserved slots are data;
    // OptTrans follows the declared Slater block.
    for value in data
        .projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
    {
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

/// Write one PhysCal sample using C's indexed, per-sample truncated files.
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
    // C accepts a signed NDataIdxStart and formats idx with %03d (e.g. -01).
    // Reject overflow before creating files instead of clamping or wrapping.
    let index = i64::try_from(sample)
        .ok()
        .and_then(|sample| data.modpara.n_data_idx_start.checked_add(sample))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "PhysCal output index overflow")
        })?;
    // C InitFilePhysCal opens out/var with "w" for EVERY indexed sample;
    // outputData writes all declared coefficients, including DH/RBM/OptTrans.
    // Do not route this through the Julia optimizer's shared append files.
    let energy = &state.energy;
    let variance = crate::c_complex::divide(
        energy.etot2 - energy.etot * energy.etot,
        energy.etot * energy.etot,
    )
    .re;
    let mut out = File::create(output_path(
        &format!("{head}_out_{index:03}.dat"),
        output_dir,
    )?)?;
    writeln!(
        out,
        "{} {}  {} {} {} {}",
        format_c_double(energy.etot.re),
        format_c_double(energy.etot.im),
        format_c_double(energy.etot2.re),
        format_c_double(variance),
        format_c_double(energy.sztot.re).trim_start(),
        format_c_double(energy.sztot2.re).trim_start()
    )?;
    let mut var = File::create(output_path(
        &format!("{head}_var_{index:03}.dat"),
        output_dir,
    )?)?;
    write!(
        var,
        "{} {} 0.0 {} {} 0.0 ",
        format_c_double(energy.etot.re),
        format_c_double(energy.etot.im),
        format_c_double(energy.etot2.re),
        format_c_double(energy.etot2.im)
    )?;
    for value in data
        .projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
    {
        write!(
            var,
            "{} {} 0.0 ",
            format_c_double(value.re),
            format_c_double(value.im)
        )?;
    }
    writeln!(var)?;
    let write_rows = |suffix: &str, rows: Vec<String>, terminal_blank: bool| -> io::Result<()> {
        let path = output_path(&format!("{head}_{suffix}_{index:03}.dat"), output_dir)?;
        let mut file = File::create(path)?;
        for row in rows {
            writeln!(file, "{row}")?;
        }
        if terminal_blank {
            writeln!(file)?;
        }
        Ok(())
    };
    let one_rows = data
        .green_one_terms
        .iter()
        .zip(&phys.phys_cis_ajs)
        .map(|(term, value)| {
            format!(
                "{} {} {} {} {}  {} ",
                term.site1,
                crate::observables::spin_code(term.spin1),
                term.site2,
                crate::observables::spin_code(term.spin2),
                format_c_double(value.re),
                format_c_double(value.im)
            )
        })
        .collect::<Vec<_>>();
    if !one_rows.is_empty() {
        write_rows("cisajs", one_rows, true)?;
    }
    let ex_pairs = data
        .green_two_ex_terms
        .iter()
        .zip(&phys.phys_cis_ajs_ckt_alt)
        .map(|(_term, value)| {
            format!(
                "{}  {} ",
                format_c_double(value.re),
                format_c_double(value.im)
            )
        })
        .collect::<Vec<_>>();
    // C vmcmain.c:671–675 emits pairs in term order, newline after the loop.
    if !ex_pairs.is_empty() {
        write_rows("cisajscktaltex", vec![ex_pairs.concat()], false)?;
    }
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
        .collect::<Vec<_>>();
    if !direct_rows.is_empty() {
        write_rows("cisajscktalt", direct_rows, true)?;
    }

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
        write!(
            ls_file,
            "{}  {}  {}  ",
            format_c_double(energy),
            format_c_double(variance),
            format_c_double(alpha)
        )?;

        if data.modpara.lanczos_mode > 1 {
            let complex = crate::run::get_all_complex_flag(data);
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
                        "{} {} {} {} {} {} ",
                        term.site1,
                        crate::observables::spin_code(term.spin1),
                        term.site2,
                        crate::observables::spin_code(term.spin2),
                        format_c_double(value.re),
                        if complex {
                            format_c_double(value.im)
                        } else {
                            "0.0".to_owned()
                        }
                    )
                })
                .collect();
            write_rows("ls_cisajs", one_rows, true)?;

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
                        if complex {
                            format_c_double(value.im)
                        } else {
                            "0.0".to_owned()
                        }
                    )
                })
                .collect();
            write_rows("ls_cisajscktalt", direct_rows, true)?;

            let factored_values = lanczos_phys_values(
                &phys.phys_lanczos_qqqq,
                &phys.phys_lanczos_qcisajscktaltq,
                data.green_two_ex_indices.len(),
                alpha,
            );
            let factored_pairs = factored_values
                .into_iter()
                .map(|value| {
                    format!(
                        "{} {} ",
                        format_c_double(value.re),
                        if complex {
                            format_c_double(value.im)
                        } else {
                            "0.0".to_owned()
                        }
                    )
                })
                .collect::<Vec<_>>();
            // C InitFilePhysCal (initfile.c:130–132) opens this even at zero
            // count. physcal_lanczos.c:141/262 unconditionally emits '\n'
            // after the real/complex pair loop: zero terms means ONE newline.
            write_rows("ls_cisajscktaltex", vec![factored_pairs.concat()], false)?;
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

/// C avevar.c OutputOptData: chronological means and sample deviations.
/// The caller supplies a complete, explicitly bounded optimization window.
pub fn output_opt_data(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    output_dir: Option<&Path>,
) -> io::Result<()> {
    let window = &state.opt_data;
    let width = data.count_variational_parameters();
    if window.is_empty() || window.iter().any(|point| point.parameters.len() != width) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "incomplete optimization window",
        ));
    }
    let head = if data.modpara.c_para_file_head.is_empty() {
        "zqp".to_string()
    } else {
        data.modpara.c_para_file_head.clone()
    };
    let path = output_path(&format!("{head}_opt.dat"), output_dir)?;
    let mut f = File::create(&path)?;
    let value = |point: &OptDataPoint, index: usize| match index {
        0 => point.energy,
        1 => point.energy_squared,
        _ => point.parameters[index - 2],
    };
    if window.len() == 1 {
        // Upstream deliberately emits real/zero PAIRS, not triples, and no auxiliary files.
        for index in 0..width + 2 {
            write!(
                f,
                "{} {} ",
                format_c_double(value(&window[0], index).re),
                format_c_double(0.0)
            )?;
        }
        writeln!(f)?;
        return Ok(());
    }
    let moments = (0..width + 2)
        .map(|index| {
            let mut mean = Complex64::new(0.0, 0.0);
            for point in window {
                mean += value(point, index);
            }
            mean /= window.len() as f64;
            let mut variance = 0.0;
            for point in window {
                let delta = value(point, index) - mean;
                variance += (delta * delta.conj()).re;
            }
            (mean, (variance / (window.len() as f64 - 1.0)).sqrt())
        })
        .collect::<Vec<_>>();
    for (mean, deviation) in &moments {
        write!(
            f,
            "{} {} {} ",
            format_c_double(mean.re),
            format_c_double(mean.im),
            format_c_double(*deviation)
        )?;
    }
    writeln!(f)?;
    let layout = data.projection_layout();
    let mut blocks = vec![
        (
            "gutzwiller",
            "NGutzwillerIdx",
            layout.n_gutzwiller,
            layout.n_gutzwiller,
        ),
        ("jastrow", "NJastrowIdx", layout.n_jastrow, layout.n_jastrow),
        (
            "doublonHolon2site",
            "NDoublonHolon2siteIdx",
            layout.n_dh2,
            6 * layout.n_dh2,
        ),
        (
            "doublonHolon4site",
            "NDoublonHolon4siteIdx",
            layout.n_dh4,
            10 * layout.n_dh4,
        ),
    ];
    for (section, width) in data.rbm_section_sizes().into_iter().enumerate() {
        let (suffix, label) = RBM_OUTPUT_BLOCKS[section];
        blocks.push((suffix, label, width, width));
    }
    let slater = data.slater_params.len();
    if data.i_flg_orbital_general == 0 {
        blocks.push(("orbital", "NOrbitalIdx", slater, slater));
    } else if data.i_flg_orbital_parallel != 0 {
        let anti = usize::try_from(data.n_orbital_anti_parallel)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "negative orbital width"))?;
        let parallel = slater
            .checked_sub(anti)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid orbital width"))?;
        blocks.push(("orbitalAntiParallel", "NOrbitalAntiParallelIdx", anti, anti));
        blocks.push(("orbitalParallel", "NOrbitalParallelIdx", parallel, parallel));
    } else {
        blocks.push(("orbital_general", "NOrbitalIdx", slater, slater));
    }
    blocks.push((
        "trans",
        "NQPOptTrans",
        data.opt_trans.len(),
        data.opt_trans.len(),
    ));
    let mut offset = 2;
    for (suffix, label, declared, count) in blocks {
        output_parameter_block(
            &head,
            suffix,
            label,
            declared,
            moments[offset..offset + count]
                .iter()
                .map(|(mean, _)| *mean),
            output_dir,
        )?;
        offset += count;
    }
    debug_assert_eq!(offset, moments.len());
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
    declared: usize,
    values: impl ExactSizeIterator<Item = num_complex::Complex64>,
    output_dir: Option<&Path>,
) -> io::Result<()> {
    if values.len() == 0 {
        return Ok(());
    }
    let path = output_path(&format!("{head}_{suffix}_opt.dat"), output_dir)?;
    let mut file = File::create(path)?;
    const PARAMETER_BLOCK_SEPARATOR: &str = "======================";
    writeln!(file, "{PARAMETER_BLOCK_SEPARATOR}")?;
    writeln!(file, "{label}  {declared}")?;
    writeln!(file, "{PARAMETER_BLOCK_SEPARATOR}")?;
    writeln!(file, "{PARAMETER_BLOCK_SEPARATOR}")?;
    writeln!(file, "{PARAMETER_BLOCK_SEPARATOR}")?;
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

    pub(super) fn test_output_dir(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        for _ in 0..10_000 {
            let index = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("mvmc-io-{label}-{}-{index}", std::process::id(),));
            match fs::create_dir(&path) {
                Ok(()) => return path,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create exclusive IO test directory: {error}"),
            }
        }
        panic!("cannot allocate exclusive IO test directory");
    }

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
        let output_dir = test_output_dir("phys-output");
        output_phys_data(&data, &state, 2, Some(&output_dir)).unwrap();
        let one = fs::read_to_string(output_dir.join("zvo_cisajs_009.dat")).unwrap();
        // Literal fixed-input expectations follow the C fprintf templates;
        // these byte checks do not assert bitwise computed numerical parity.
        let formatting = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/physcal_181/formatting");
        assert_eq!(
            one.as_bytes(),
            fs::read(formatting.join("one.dat")).unwrap()
        );
        let ex = fs::read_to_string(output_dir.join("zvo_cisajscktaltex_009.dat")).unwrap();
        assert_eq!(
            ex.as_bytes(),
            fs::read(formatting.join("factored.dat")).unwrap()
        );
        let direct = fs::read_to_string(output_dir.join("zvo_cisajscktalt_009.dat")).unwrap();
        assert_eq!(
            direct.as_bytes(),
            fs::read(formatting.join("direct.dat")).unwrap()
        );
        let qqqq = fs::read_to_string(output_dir.join("zvo_ls_qqqq_009.dat")).unwrap();
        assert!(qqqq.split_whitespace().count() == 16);
        let ls = fs::read_to_string(output_dir.join("zvo_ls_out_009.dat")).unwrap();
        assert_eq!(ls.split_whitespace().count(), 3);
        assert!(ls.ends_with("  "));
        assert!(!ls.contains('\n'));
        let _ = fs::remove_dir_all(output_dir);
    }

    #[test]
    fn physcal_indexed_out_var_truncate_each_sample_and_keep_empty_green_contract() {
        let mut data = ExpertModeData::new();
        data.modpara.n_data_idx_start = 7;
        data.modpara.n_orbital_idx = 4;
        // Slots 2/3 have no spatial mappings but belong to C's declared block.
        data.slater_params = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 4, 1, 1, true, false);
        state.phys_quantities = Some(crate::state::PhysicalQuantities::zeros(0, 0, 0));
        state.energy.etot = Complex64::new(1.0, 1.0);
        state.energy.etot2 = Complex64::new(2.0, 0.0);
        state.energy.sztot = Complex64::new(0.5, 0.0);
        state.energy.sztot2 = Complex64::new(0.25, 0.0);
        let dir = test_output_dir("physcal-indexed-lifecycle");
        output_phys_data(&data, &state, 0, Some(&dir)).unwrap();
        let values = |path: PathBuf| {
            fs::read_to_string(path)
                .unwrap()
                .split_whitespace()
                .map(|token| token.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        // (2 - (1+i)^2)/(1+i)^2 = -1-i: no real-only variance shortcut.
        assert_eq!(
            values(dir.join("zvo_out_007.dat")),
            [1.0, 1.0, 2.0, -1.0, 0.5, 0.25]
        );
        assert_eq!(
            fs::read(dir.join("zvo_out_007.dat")).unwrap(),
            fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/fixtures/physcal_181/formatting/out.dat")
            )
            .unwrap()
        );
        assert_eq!(
            values(dir.join("zvo_var_007.dat")),
            [
                1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0,
                0.0, 0.0
            ]
        );
        let first = fs::read(dir.join("zvo_out_007.dat")).unwrap();
        state.energy.etot = Complex64::new(2.0, 0.0);
        output_phys_data(&data, &state, 1, Some(&dir)).unwrap();
        assert_eq!(fs::read(dir.join("zvo_out_007.dat")).unwrap(), first);
        assert_eq!(values(dir.join("zvo_out_008.dat"))[0], 2.0);
        // Every sample truncates its own files, even when replayed out of order.
        output_phys_data(&data, &state, 0, Some(&dir)).unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("zvo_out_007.dat"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        assert_eq!(values(dir.join("zvo_out_007.dat"))[0], 2.0);
        for suffix in ["out", "var", "cisajs", "cisajscktalt", "cisajscktaltex"] {
            assert!(!dir.join(format!("zvo_{suffix}.dat")).exists());
        }
        for suffix in ["cisajs", "cisajscktalt", "cisajscktaltex"] {
            assert!(
                !dir.join(format!("zvo_{suffix}_007.dat")).exists(),
                "C skips zero-count normal {suffix}"
            );
        }
        data.modpara.lanczos_mode = 2;
        output_phys_data(&data, &state, 0, Some(&dir)).unwrap();
        for suffix in ["ls_cisajs", "ls_cisajscktalt", "ls_cisajscktaltex"] {
            assert_eq!(
                fs::read(dir.join(format!("zvo_{suffix}_007.dat"))).unwrap(),
                b"\n"
            );
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn physcal_preserves_signed_c_index_and_rejects_overflow_before_writes() {
        let mut data = ExpertModeData::new();
        data.modpara.n_data_idx_start = -1;
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
        state.phys_quantities = Some(crate::state::PhysicalQuantities::zeros(0, 0, 0));
        let dir = test_output_dir("physcal-signed-index");
        output_phys_data(&data, &state, 0, Some(&dir)).unwrap();
        assert!(dir.join("zvo_out_-01.dat").is_file());
        assert!(dir.join("zvo_var_-01.dat").is_file());
        assert!(!dir.join("zvo_out_000.dat").exists());
        output_phys_data(&data, &state, 1, Some(&dir)).unwrap();
        assert!(dir.join("zvo_out_000.dat").is_file());
        let before = fs::read_dir(&dir).unwrap().count();
        data.modpara.n_data_idx_start = i64::MAX;
        assert_eq!(
            output_phys_data(&data, &state, 1, Some(&dir))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), before);
        let _ = fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod history_tests {
    use super::tests::test_output_dir;
    use super::*;
    use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};
    use num_complex::Complex64;

    #[test]
    fn optimization_windows_match_independent_c_prefix_outputs() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ctest_model_prefixes/heisenberg_chain_real");
        for prefix in [1, 2, 3, 50] {
            let fixture = fixtures.join(format!("step-{prefix}"));
            let input =
                std::fs::read_to_string(fixture.join("c-window-declared-input.txt")).unwrap();
            let mut lines = input.lines();
            let header = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            let mut data = ExpertModeData::new();
            data.n_gutzwiller_idx = header[2] as i64;
            data.n_jastrow_idx = header[3] as i64;
            data.modpara.n_orbital_idx = header[15] as i64;
            data.slater_params = vec![Complex64::new(0.0, 0.0); header[15]];
            let mut state = VmcOptimizationState::zeros(6, 3, 2, header[1], 1, 1, false, false);
            state.opt_data = lines
                .map(|line| {
                    let values = line
                        .split_whitespace()
                        .map(|s| s.parse::<f64>().unwrap())
                        .collect::<Vec<_>>();
                    let (pairs, remainder) = values.as_chunks::<2>();
                    assert!(remainder.is_empty(), "complete complex fixture records");
                    let values = pairs
                        .iter()
                        .map(|v| Complex64::new(v[0], v[1]))
                        .collect::<Vec<_>>();
                    assert_eq!(values.len(), header[1] + 2);
                    OptDataPoint {
                        energy: values[0],
                        energy_squared: values[1],
                        parameters: values[2..].to_vec(),
                    }
                })
                .collect();
            assert_eq!(state.opt_data.len(), header[0]);
            let out = test_output_dir(&format!("c-window-{prefix}"));
            output_opt_data(&data, &state, Some(&out)).unwrap();
            let actual = std::fs::read_to_string(out.join("zqp_opt.dat")).unwrap();
            let expected = std::fs::read_to_string(fixture.join("zqp_c_window_opt.dat")).unwrap();
            assert_eq!(actual.lines().count(), 1);
            let parse = |s: &str| {
                s.split_whitespace()
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>()
            };
            let actual = parse(&actual);
            let expected = parse(&expected);
            assert_eq!(actual.len(), expected.len());
            for (column, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-14_f64.max(1e-14 * a.abs().max(e.abs())),
                    "prefix {prefix} column {column}: {a} != {e}"
                );
            }
            if header[0] == 1 {
                assert_eq!(std::fs::read_dir(&out).unwrap().count(), 1);
            } else {
                let block = std::fs::read_to_string(out.join("zqp_orbital_opt.dat")).unwrap();
                assert_eq!(
                    &block.lines().take(5).collect::<Vec<_>>(),
                    &[
                        "======================",
                        "NOrbitalIdx  12",
                        "======================",
                        "======================",
                        "======================"
                    ]
                );
            }
            std::fs::remove_dir_all(out).unwrap();
        }
    }

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
