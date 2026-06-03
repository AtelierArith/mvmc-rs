//! Phase 5 regression gate: 50-step `zvo_out.dat` comparison vs Julia-mVMC.
//!
//! The Julia-mVMC implementation has already been validated against the C-mVMC
//! reference to within 1e-10 (10-step gate). This test drives the Rust port
//! for 50 SR steps on all four upstream models and checks that each `zvo_out`
//! row falls within 1e-8 of the Julia reference (absolute or relative).
//!
//! Tolerance rationale:
//!   * Over 50 steps the SR parameter updates compound floating-point
//!     differences across BLAS calls (dpotrf/dpotrs) so the accumulated
//!     drift grows to ~1e-8, bounded by the OpenBLAS summation order.
//!   * Columns 3 and 4 (<H²>, variance) inherit an extra 10× slack as in
//!     the Julia integration tests.
//!
//! The reference files live at:
//!   reference/<model>/zvo_out_first50.dat
//! and were produced by `tools/dump_zvo_50step_reference.jl` against the same
//! RNG seed (RndSeed=1) to guarantee a deterministic comparison.

use std::fs;
use std::path::{Path, PathBuf};

mod support;
use support::julia_mvmc_root;

const N_STEPS: usize = 50;
const TOL_DEFAULT: f64 = 1e-8;
const TOL_LOOSE: f64 = 1e-7; // cols 3 & 4: squared/derived quantities
const LOOSE_COLS: &[usize] = &[2, 3]; // 0-indexed

struct Model {
    name: &'static str,
    input_subdir: &'static str,
}

fn all_models() -> &'static [Model] {
    &[
        Model {
            name: "heisenberg_chain_real",
            input_subdir: "heisenberg_chain_real",
        },
        Model {
            name: "heisenberg_chain_cmp",
            input_subdir: "heisenberg_chain_cmp",
        },
        Model {
            name: "heisenberg_chain_fsz",
            input_subdir: "heisenberg_chain_fsz",
        },
        Model {
            name: "hubbard_chain_real",
            input_subdir: "hubbard_chain_real",
        },
    ]
}

fn run_and_compare(julia: &Path, model: &Model) -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join(model.input_subdir)
        .join("namelist.def");
    let reference = repo_root
        .join("reference")
        .join(model.input_subdir)
        .join("zvo_out_first50.dat");

    if !namelist.is_file() {
        return Err(format!("namelist missing: {}", namelist.display()));
    }
    if !reference.is_file() {
        return Err(format!(
            "50-step reference missing: {} — regenerate it and place it under reference/<model>/",
            reference.display()
        ));
    }

    let out_dir = tempdir(model.name);
    mvmc_core::run_para_opt_from_namelist(&namelist, N_STEPS, Some(1), Some(&out_dir))
        .map_err(|e| format!("{}: run failed: {e}", model.name))?;

    let got_str = fs::read_to_string(out_dir.join("zvo_out.dat"))
        .map_err(|e| format!("{}: zvo_out.dat unreadable: {e}", model.name))?;
    let ref_str = fs::read_to_string(&reference)
        .map_err(|e| format!("{}: reference unreadable: {e}", model.name))?;

    let got_rows = parse_rows(&got_str);
    let ref_rows = parse_rows(&ref_str);

    if got_rows.len() != N_STEPS {
        return Err(format!(
            "{}: expected {} rows, got {}",
            model.name,
            N_STEPS,
            got_rows.len()
        ));
    }
    if ref_rows.len() != N_STEPS {
        return Err(format!(
            "{}: reference has {} rows, expected {}",
            model.name,
            ref_rows.len(),
            N_STEPS
        ));
    }

    for (row, (got_row, ref_row)) in got_rows.iter().zip(ref_rows.iter()).enumerate() {
        if got_row.len() != ref_row.len() {
            return Err(format!(
                "{}: row {} column count mismatch (got {} expected {})",
                model.name,
                row,
                got_row.len(),
                ref_row.len()
            ));
        }
        for (col, (&g, &r)) in got_row.iter().zip(ref_row.iter()).enumerate() {
            let tol = if LOOSE_COLS.contains(&col) {
                TOL_LOOSE
            } else {
                TOL_DEFAULT
            };
            let tol = tol.max(r.abs() * tol);
            let diff = (g - r).abs();
            if diff > tol {
                return Err(format!(
                    "{}: row {} col {}: got {g:.18e}, ref {r:.18e}, diff {diff:.3e}, tol {tol:.3e}",
                    model.name, row, col,
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn heisenberg_chain_real_50step() {
    run_model_test("heisenberg_chain_real");
}

#[test]
fn heisenberg_chain_cmp_50step() {
    run_model_test("heisenberg_chain_cmp");
}

#[test]
fn heisenberg_chain_fsz_50step() {
    run_model_test("heisenberg_chain_fsz");
}

#[test]
fn hubbard_chain_real_50step() {
    run_model_test("hubbard_chain_real");
}

fn run_model_test(model_name: &str) {
    let Some(julia) = julia_mvmc_root() else {
        eprintln!("skipping {model_name} 50-step: Julia-mVMC checkout not found");
        return;
    };
    let model = all_models()
        .iter()
        .find(|m| m.name == model_name)
        .expect("unknown model");
    if let Err(e) = run_and_compare(&julia, model) {
        panic!("{e}");
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn parse_rows(content: &str) -> Vec<Vec<f64>> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<f64>().expect("numeric token"))
                .collect()
        })
        .collect()
}

fn tempdir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("mvmc-phase5-50step-{}-{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("create temp dir");
    p
}
