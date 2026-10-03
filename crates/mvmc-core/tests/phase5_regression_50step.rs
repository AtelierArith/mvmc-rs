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
//!
//! Historical fixture gate, not fresh Julia/C verification. Run explicitly with
//! `MVMC_RS_PHASE5_50STEP=1` and `--run-ignored only`; no oracle is invoked.

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
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let namelist = julia
        .join("examples")
        .join("inputs")
        .join(model.input_subdir)
        .join("namelist.def");
    let reference = repo_root
        .join("reference")
        .join(model.input_subdir)
        .join("zvo_out_first50.dat");

    require_fixtures(&namelist, &reference);

    let out_dir = tempdir(model.name);
    mvmc_core::run_para_opt_from_namelist(
        &namelist,
        mvmc_core::RunConfig {
            nsmp: Some(N_STEPS as i64),
            seed: Some(1),
            output_dir: Some(out_dir.clone()),
            ..mvmc_core::RunConfig::new(N_STEPS as i64, "real")
        },
    )
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
#[ignore = "long historical fixture gate: MVMC_RS_PHASE5_50STEP required"]
fn heisenberg_chain_real_50step() {
    run_model_test("heisenberg_chain_real");
}

#[test]
#[ignore = "long historical fixture gate: MVMC_RS_PHASE5_50STEP required"]
fn heisenberg_chain_cmp_50step() {
    run_model_test("heisenberg_chain_cmp");
}

#[test]
#[ignore = "long historical fixture gate: MVMC_RS_PHASE5_50STEP required"]
fn heisenberg_chain_fsz_50step() {
    run_model_test("heisenberg_chain_fsz");
}

#[test]
#[ignore = "long historical fixture gate: MVMC_RS_PHASE5_50STEP required"]
fn hubbard_chain_real_50step() {
    run_model_test("hubbard_chain_real");
}

fn run_model_test(model_name: &str) {
    support::require_gate("phase5-50step", "MVMC_RS_PHASE5_50STEP");
    let julia = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("phase5-50step", "Julia-mVMC checkout not found")
    });
    let model = all_models()
        .iter()
        .find(|m| m.name == model_name)
        .expect("unknown model");
    if let Err(e) = run_and_compare(&julia, model) {
        panic!("{e}");
    }
}

fn require_fixtures(namelist: &Path, reference: &Path) {
    for path in [namelist, reference] {
        if !path.is_file() {
            support::missing_fixture("phase5-50step", path.display().to_string());
        }
    }
}

#[test]
fn selected_long_gates_fail_on_missing_selector_checkout_or_namelist() {
    let empty_checkout = tempdir("reporting-negative");
    for model in all_models() {
        for (selector, root, status) in [
            (None, empty_checkout.clone(), "NotRun"),
            (Some(""), empty_checkout.clone(), "NotRun"),
            (Some("skip"), empty_checkout.clone(), "ExplicitSkip"),
            (
                Some("1"),
                empty_checkout.join("absent-checkout"),
                "MissingFixture",
            ),
            (Some("1"), empty_checkout.clone(), "MissingFixture"),
        ] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--ignored",
                    "--exact",
                    &format!("{}_50step", model.name),
                    "--nocapture",
                ])
                .env_remove("MVMC_RS_PHASE5_50STEP")
                .env("JULIA_MVMC_ROOT", root);
            if let Some(selector) = selector {
                command.env("MVMC_RS_PHASE5_50STEP", selector);
            }
            let output = command.output().unwrap();
            assert!(
                !output.status.success(),
                "{} must fail with {status}",
                model.name
            );
            assert!(String::from_utf8_lossy(&output.stderr).contains(status));
        }
    }
    fs::remove_dir(empty_checkout).unwrap();
}

#[test]
fn selected_long_gate_missing_expected_reference_cannot_pass() {
    // Use this source file as an existing namelist stand-in: preflight must
    // reject the absent expectation before parsing or numerical work.
    let existing = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/phase5_regression_50step.rs");
    let empty = tempdir("missing-reference");
    let missing = empty.join("zvo_out_first50.dat");
    let error = std::panic::catch_unwind(|| require_fixtures(&existing, &missing)).unwrap_err();
    let message = error.downcast_ref::<String>().unwrap();
    assert!(message.contains("missing fixture") && message.contains("zvo_out_first50.dat"));
    fs::remove_dir(empty).unwrap();
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
