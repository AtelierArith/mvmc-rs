//! Port of `ComplexUHF/output.c` (files, number formats and orbital output).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::definition::{OrbitalOutputMode, UhfDefinition};
use crate::hf::HfState;
use crate::linalg;
use crate::UhfError;

/// C `%.<prec>lf`.
pub fn c_fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { "nan" }.to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    format!("{x:.prec$}")
}

/// C `% .18e` (space flag, two-digit signed exponent).
pub fn c_space_exp18(x: f64) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { " nan" }.to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { " inf" }.to_string();
    }
    let text = format!("{x:.18e}");
    let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sign = if exponent < 0 { '-' } else { '+' };
    let lead = if mantissa.starts_with('-') { "" } else { " " };
    format!("{lead}{mantissa}e{sign}{:02}", exponent.abs())
}

fn write_file(path: &Path, content: &str) -> Result<(), UhfError> {
    fs::write(path, content).map_err(|source| UhfError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Resolve `{head}_suffix` against `base`.
pub fn data_path(base: &Path, head: &str, suffix: &str) -> PathBuf {
    base.join(format!("{head}{suffix}"))
}

/// `output()`: result, eigenvalues, gap, Green function and orbital files.
pub fn output(
    def: &UhfDefinition,
    state: &HfState,
    rng: &mut Sfmt19937Rng,
    base: &Path,
) -> Result<(), UhfError> {
    let head = &def.data_file_head;
    let path = data_path(base, head, "_result.dat");
    write_file(
        &path,
        &format!(
            " energy {} \n num    {} \n",
            c_fixed(state.energy, 10),
            c_fixed(state.num, 10)
        ),
    )?;
    println!("Energy and num are outputted to {head}_result.dat.");

    let mut text = String::new();
    for i in 0..def.nsite * 2 {
        let _ = writeln!(text, " {}  {} ", i + 1, c_fixed(state.eigen_values[i], 10));
    }
    write_file(&data_path(base, head, "_eigen.dat"), &text)?;
    println!("Eigenvalues are outputted to {head}_eigen.dat.");

    let gap = state.eigen_values[def.nsize] - state.eigen_values[def.nsize - 1];
    write_file(
        &data_path(base, head, "_gap.dat"),
        &format!("  {} \n", c_fixed(gap, 10)),
    )?;
    println!("Energy gaps are outputted to {head}_gap.dat.");

    cal_cisajs(def, state, base)?;

    if def.orbital_flag {
        make_orbital_file(def, state, rng, base)?;
    }
    Ok(())
}

fn cal_cisajs(def: &UhfDefinition, state: &HfState, base: &Path) -> Result<(), UhfError> {
    let ns = def.nsite;
    let n2 = state.n2;
    let mut text = String::new();
    for site_1 in 0..ns {
        for spin_1 in 0..2 {
            for site_2 in 0..ns {
                for spin_2 in 0..2 {
                    let tmp = state.g[(site_1 + ns * spin_1) * n2 + (site_2 + ns * spin_2)];
                    let _ = writeln!(
                        text,
                        " {site_1:4} {spin_1:4} {site_2:4} {spin_2:4} {} {}",
                        c_fixed(tmp.re, 10),
                        c_fixed(tmp.im, 10)
                    );
                }
            }
        }
    }
    let name = format!("{}_UHF_cisajs.dat", def.data_file_head);
    write_file(&base.join(&name), &text)?;
    println!("Onebody Green's functions are outputted to {name}.");
    Ok(())
}

fn cm(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// `Child_OutputOptData`: five header lines then `index re im` records.
fn child_output_opt_data(
    path: &Path,
    keyword: &str,
    params: &[Complex64],
    count: usize,
) -> Result<(), UhfError> {
    let mut text = String::new();
    text.push_str("======================\n");
    let _ = writeln!(text, "{keyword}  {count}");
    text.push_str("======================\n");
    text.push_str("======================\n");
    text.push_str("======================\n");
    for (i, value) in params.iter().take(count).enumerate() {
        let _ = writeln!(
            text,
            "{i} {} {} ",
            c_space_exp18(value.re),
            c_space_exp18(value.im)
        );
    }
    write_file(path, &text)
}

fn make_orbital_file(
    def: &UhfDefinition,
    state: &HfState,
    rng: &mut Sfmt19937Rng,
    base: &Path,
) -> Result<(), UhfError> {
    if def.n_orbital_idx == 0 {
        return Ok(());
    }
    let ns = def.nsite;
    let n2 = state.n2;
    let Some(mode) = def.orbital_mode else {
        return Ok(());
    };
    if mode == OrbitalOutputMode::AntiParallel {
        // for anti-parallel, re-diagonalise each spin block
        let block = |row0: usize, col0: usize| -> Vec<Complex64> {
            let mut m = vec![Complex64::new(0.0, 0.0); ns * ns];
            for i in 0..ns {
                for j in 0..ns {
                    m[i * ns + j] = state.ham[(i + row0) * n2 + (j + col0)];
                }
            }
            m
        };
        let transpose_vectors = |vec: &[Complex64]| -> Vec<Complex64> {
            let mut slt = vec![Complex64::new(0.0, 0.0); ns * ns];
            for k in 0..ns {
                for l in 0..ns {
                    slt[l * ns + k] = vec[k * ns + l];
                }
            }
            slt
        };
        let (_, vec_up) = linalg::zheev_all(ns, &block(0, 0))?;
        let slt_u = transpose_vectors(&vec_up);
        let (_, vec_dn) = linalg::zheev_all(ns, &block(ns, ns))?;
        let slt_d = transpose_vectors(&vec_dn);
        let mut ap_fij = vec![Complex64::new(0.0, 0.0); ns * ns];
        for i in 0..ns {
            for j in 0..ns {
                let mut tmp = Complex64::new(0.0, 0.0);
                for n in 0..def.ne.max(0) as usize {
                    tmp += cm(slt_u[i * ns + n], slt_d[j * ns + n]);
                }
                ap_fij[i * ns + j] = tmp;
            }
        }
        output_anti_parallel(def, &ap_fij, ns, false, rng, base, "fij")?;
    } else {
        let mut fij = vec![Complex64::new(0.0, 0.0); n2 * n2];
        let nsize = def.nsize;
        for ispin in 0..2 {
            for jspin in 0..2 {
                for i in 0..ns {
                    for j in 0..ns {
                        let isite = i + ispin * ns;
                        let jsite = j + jspin * ns;
                        let mut value = Complex64::new(0.0, 0.0);
                        let mut n = 0;
                        while n < 2 * def.ne.max(0) as usize {
                            let a = cm(
                                state.r_slt[isite * nsize + n].conj(),
                                state.r_slt[jsite * nsize + n + 1].conj(),
                            );
                            let b = cm(
                                state.r_slt[isite * nsize + n + 1].conj(),
                                state.r_slt[jsite * nsize + n].conj(),
                            );
                            value += a - b;
                            n += 2;
                        }
                        fij[isite * n2 + jsite] = value;
                    }
                }
            }
        }
        if mode == OrbitalOutputMode::AntiParallelAndParallel {
            output_anti_parallel(def, &fij, n2, true, rng, base, "Fij")?;
            output_parallel(def, &fij, rng, base)?;
        } else {
            output_general(def, &fij, rng, base)?;
        }
    }
    Ok(())
}

struct Accumulator {
    param: Vec<Complex64>,
    count: Vec<i32>,
}

impl Accumulator {
    fn new(n: usize) -> Self {
        Self {
            param: vec![Complex64::new(0.0, 0.0); n],
            count: vec![0; n],
        }
    }

    fn add(&mut self, def: &UhfDefinition, isite: usize, jsite: usize, value: Complex64) {
        let n2 = 2 * def.nsite;
        let idx = def.orbital_idx[isite * n2 + jsite];
        let sgn = def.orbital_sgn[isite * n2 + jsite];
        if idx != -1 {
            let idx = idx as usize;
            // complex * int: C scales each component (real operand has no imaginary part)
            let sign = f64::from(sgn);
            self.param[idx] += Complex64::new(value.re * sign, value.im * sign);
            self.count[idx] += 1;
        }
    }

    /// Average over the contributing entries and add the SFMT noise.
    fn finish(
        &mut self,
        range: std::ops::Range<usize>,
        def: &UhfDefinition,
        rng: &mut Sfmt19937Rng,
    ) {
        for i in range {
            let count = f64::from(self.count[i]);
            self.param[i] = Complex64::new(self.param[i].re / count, self.param[i].im / count);
            self.param[i].re += rng.genrand_real2() * 10.0_f64.powf(-f64::from(def.eps_int_slater));
        }
    }
}

fn fij_line(i: usize, j: usize, value: Complex64) -> String {
    format!(
        " {i} {j} {} {}\n",
        c_fixed(value.re, 6),
        c_fixed(value.im, 6)
    )
}

/// `OutputAntiParallel` (full `2N` matrix) and `OutputAntiParallel_2` (`N x N`).
fn output_anti_parallel(
    def: &UhfDefinition,
    fij: &[Complex64],
    stride: usize,
    full: bool,
    rng: &mut Sfmt19937Rng,
    base: &Path,
    label: &str,
) -> Result<(), UhfError> {
    let ns = def.nsite;
    let mut acc = Accumulator::new(def.n_orbital_idx);
    let mut text = String::new();
    for i in 0..ns {
        for j in 0..ns {
            let (isite, jsite) = (i, j + ns);
            let value = if full {
                fij[isite * stride + jsite]
            } else {
                fij[i * stride + j]
            };
            text.push_str(&fij_line(i, j, value));
            acc.add(def, isite, jsite, value);
        }
    }
    let head = &def.para_file_head;
    write_file(&data_path(base, head, "_AP_Fij.dat"), &text)?;
    acc.finish(0..def.n_orbital_ap, def, rng);
    let name = format!("{head}_APOrbital_opt.dat");
    child_output_opt_data(
        &base.join(&name),
        "NOrbitalAP",
        &acc.param,
        def.n_orbital_ap,
    )?;
    println!("{label} for mVMC are outputted to {name}.");
    Ok(())
}

fn output_parallel(
    def: &UhfDefinition,
    fij: &[Complex64],
    rng: &mut Sfmt19937Rng,
    base: &Path,
) -> Result<(), UhfError> {
    let ns = def.nsite;
    let n2 = 2 * ns;
    let ini = def.n_orbital_ap;
    let fin = def.n_orbital_ap + def.n_orbital_p;
    let mut acc = Accumulator::new(def.n_orbital_idx);
    let mut text = String::new();
    for i in 0..ns {
        for j in i + 1..ns {
            for spin in 0..2 {
                let (isite, jsite) = (i + spin * ns, j + spin * ns);
                let value = fij[isite * n2 + jsite];
                text.push_str(&fij_line(isite, jsite, value));
                acc.add(def, isite, jsite, value);
            }
        }
    }
    let head = &def.para_file_head;
    write_file(&data_path(base, head, "_P_Fij.dat"), &text)?;
    acc.finish(ini..fin, def, rng);
    for i in ini..fin {
        acc.param[i - ini] = acc.param[i];
    }
    let name = format!("{head}_POrbital_opt.dat");
    child_output_opt_data(&base.join(&name), "NOrbitalP", &acc.param, def.n_orbital_p)?;
    println!("Fij for mVMC are outputted to {name}.");
    Ok(())
}

fn output_general(
    def: &UhfDefinition,
    fij: &[Complex64],
    rng: &mut Sfmt19937Rng,
    base: &Path,
) -> Result<(), UhfError> {
    let ns = def.nsite;
    let n2 = 2 * ns;
    let mut acc = Accumulator::new(def.n_orbital_idx);
    let mut text = String::new();
    for ispin in 0..2 {
        for jspin in 0..2 {
            for i in 0..ns {
                for j in 0..ns {
                    let (isite, jsite) = (i + ispin * ns, j + jspin * ns);
                    let value = fij[isite * n2 + jsite];
                    text.push_str(&fij_line(isite, jsite, value));
                    acc.add(def, isite, jsite, value);
                }
            }
        }
    }
    let head = &def.para_file_head;
    write_file(&data_path(base, head, "_General_Fij.dat"), &text)?;
    acc.finish(0..def.n_orbital_idx, def, rng);
    let name = format!("{head}_GeneralOrbital_opt.dat");
    child_output_opt_data(
        &base.join(&name),
        "NOrbitalIdx",
        &acc.param,
        def.n_orbital_idx,
    )?;
    println!("Fij for mVMC are outputted to {name}.");
    Ok(())
}
