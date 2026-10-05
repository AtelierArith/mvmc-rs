//! Pure-Rust port of the mVMC C ComplexUHF initial-orbital tool.
//!
//! Authoritative source: `extern/mVMC-1.3.0/src/ComplexUHF/` (`UHFmain.c`,
//! `readdef.c`, `initial.c`, `makeham.c`, `diag.c`, `green.c`,
//! `cal_energy.c`, `output.c`). The tool builds a random (or `Initial`-file)
//! Green matrix, iterates `makeham` -> `diag` (LAPACK `zheev`) -> `green`
//! (BLAS `zgemm`) -> `cal_energy` with linear mixing until the residual drops
//! below `0.1^EPS`, then writes the observables and the `f_ij` initial-orbital
//! files that `vmc.out` reads through `InOrbital`.
//!
//! The SFMT draw order of C is preserved: `(2*Nsite)^2` draws for the
//! random start (row-major) when no `Initial` records are declared, then one
//! draw per orbital parameter while writing orbital files.
//!
//! Usage from the command line: `mvmc uhf namelist.def [OptParaFile]`.

pub mod definition;
pub mod hf;
pub mod linalg;
pub mod output;
pub mod scan;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

pub use definition::{read_definition, OrbitalOutputMode, UhfDefinition};
pub use output::{c_fixed, c_space_exp18};

/// Errors of the ComplexUHF port.
#[derive(Debug, thiserror::Error)]
pub enum UhfError {
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// File involved.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// Invalid or incomplete definition files.
    #[error("{0}")]
    Definition(String),
    /// LAPACK reported a failure.
    #[error("LAPACK: {0}")]
    Lapack(String),
}

/// Run-time options.
#[derive(Debug, Clone, Default)]
pub struct UhfOptions {
    /// Directory against which definition files and outputs are resolved.
    /// Empty means the current directory, exactly like the C tool.
    pub base_dir: PathBuf,
}

/// Result of one ComplexUHF run.
#[derive(Debug, Clone)]
pub struct UhfReport {
    /// `true` when the residual dropped below `eps` within `IterationMax`.
    pub converged: bool,
    /// Loop counter at exit (`i` of `UHFmain.c`): the converging step, or `IterationMax`.
    pub steps: i32,
    /// Final total energy.
    pub energy: f64,
    /// Final particle number.
    pub num: f64,
    /// Last residual.
    pub rest: f64,
    /// Green function `G[a][b]` row-major of dimension `2*Nsite` (after mixing, as output).
    pub green: Vec<Complex64>,
    /// Eigenvalues of the last diagonalisation.
    pub eigen_values: Vec<f64>,
    /// Parsed definition.
    pub definition: UhfDefinition,
}

/// `tmp_eps`: `eps_int` successive multiplications by `0.1` as in `UHFmain.c`.
pub fn convergence_threshold(eps_int: i32) -> f64 {
    let mut eps = 1.0_f64;
    for _ in 0..eps_int {
        eps *= 0.1;
    }
    eps
}

fn append_check(path: &Path, text: &str) -> Result<(), UhfError> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| UhfError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(text.as_bytes())
        .map_err(|source| UhfError::Io {
            path: path.to_path_buf(),
            source,
        })
}

/// Run the ComplexUHF calculation for `namelist` (`UHFmain.c`).
///
/// A non-converged run still writes the output files (C behaviour) and is
/// reported with `converged == false`; the C process exits with `-1` then.
pub fn run(namelist: &Path, options: &UhfOptions) -> Result<UhfReport, UhfError> {
    let base = options.base_dir.as_path();
    let def = read_definition(namelist, base)?;
    println!("LARGE ALLOCATE FINISH !");
    let eps = convergence_threshold(def.eps_int);
    println!("########Input parameters ###########");
    println!("tmp_eps={eps:.6} ");
    println!("eps_int={} ", def.eps_int);
    println!("mix={:.6} ", def.mix);
    println!("print={} ", def.print);
    println!("#################################### ");

    // initialize Mersenne Twister
    let mut rng = Sfmt19937Rng::new(def.rnd_seed as u32);
    let mut state = hf::HfState::new(&def);
    hf::initial(&def, &mut state, &mut rng);

    let check_path = output::data_path(base, &def.data_file_head, "_check.dat");
    std::fs::write(
        &check_path,
        "#step,residue,       energy,           # of electrons\n",
    )
    .map_err(|source| UhfError::Io {
        path: check_path.clone(),
        source,
    })?;

    println!("\n########Start: Hartree-Fock calculation ###########");
    println!("stp, residue, energy");
    let mut step = 0;
    while step < def.iteration_max {
        hf::makeham(&def, &mut state);
        hf::diag(&def, &mut state)?;
        hf::green(&def, &mut state);
        hf::cal_energy(&def, &mut state);
        let line = format!(
            " {}  {} {} {}",
            step,
            c_fixed(state.rest, 12),
            c_fixed(state.energy, 12),
            c_fixed(state.num, 6)
        );
        println!("{line}");
        append_check(&check_path, &format!("{line}\n"))?;
        if state.rest < eps {
            break;
        }
        step += 1;
    }

    let converged = step < def.iteration_max;
    if converged {
        println!("\nHartree-Fock calculation is finished at {step} step. \n");
        println!("########Finish: Hartree-Fock calculation ###########");
        println!("\n########Start: Calculation of Physical Quantities ###########");
        output::output(&def, &state, &mut rng, base)?;
        println!("########Finish: Calculation of Physical Quantities ###########");
    } else {
        println!("\n!! Hartree-Fock calculation is not finished at {step}  step!! ");
        println!("\n!! Green functions at the last step are output (these are not converged )!!! ");
        output::output(&def, &state, &mut rng, base)?;
    }
    Ok(UhfReport {
        converged,
        steps: step,
        energy: state.energy,
        num: state.num,
        rest: state.rest,
        green: state.g.clone(),
        eigen_values: state.eigen_values.clone(),
        definition: def,
    })
}
