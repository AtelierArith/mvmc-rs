//! Phase5 long baseline: twenty direct-SR steps, effective window twenty,
//! seed one, original four public inputs. Reviewed direct20 references are
//! required; historical fifty-step artifacts are never used as a fallback.
//! Existing numeric bounds are retained, not inferred from run length or
//! from a blanket claim about BLAS drift. This is mixed reference coverage,
//! not full C executable/MPI parity. No oracle is invoked by Cargo.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "support/ctest_provenance.rs"]
mod ctest_provenance;
mod support;
use support::julia_mvmc_root;

const N_STEPS: usize = 20;
const OUTPUT_COLUMNS: usize = 6;
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
        .join("tests/fixtures/phase5_direct20")
        .join(model.input_subdir)
        .join("zvo_out.dat");

    require_fixtures(&namelist, &reference);
    let provenance = reference.with_file_name("provenance.txt");
    if !provenance.is_file() {
        support::missing_fixture("phase5-20step", provenance.display().to_string());
    }
    let metadata = fs::read_to_string(provenance).map_err(|e| e.to_string())?;
    validate_metadata(&metadata)?;
    let manifest = reference.with_file_name("inputs.sha256");
    if !manifest.is_file() {
        support::missing_fixture("phase5-20step", manifest.display().to_string());
    }
    ctest_provenance::verify_inputs(&manifest, namelist.parent().unwrap());
    // Match the public runner's parser, not its non-overriding mode label.
    let data = mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist, false)
        .map_err(|error| format!("{}: input preflight: {error}", model.name))?;
    validate_model(model, &data)?;

    let owned = tempdir(model.name);
    let out_dir = owned.0.clone();
    let summary = mvmc_core::run_para_opt_from_namelist(
        &namelist,
        mvmc_core::RunConfig {
            nsmp: Some(N_STEPS as i64),
            seed: Some(1),
            output_dir: Some(out_dir.clone()),
            ..mvmc_core::RunConfig::new(N_STEPS as i64, model_mode(model))
        },
    )
    .map_err(|e| format!("{}: run failed: {e}", model.name))?;
    if summary.effective_nsteps != N_STEPS || summary.effective_nsmp != N_STEPS {
        return Err(format!("{}: effective step/window mismatch", model.name));
    }

    let got_str = fs::read_to_string(out_dir.join("zvo_out.dat"))
        .map_err(|e| format!("{}: zvo_out.dat unreadable: {e}", model.name))?;
    let ref_str = fs::read_to_string(&reference)
        .map_err(|e| format!("{}: reference unreadable: {e}", model.name))?;

    let got_rows = parse_rows(&got_str);
    let ref_rows = parse_rows(&ref_str);

    compare_rows(model.name, &got_rows, &ref_rows)
}

fn validate_metadata(metadata: &str) -> Result<(), String> {
    for required in [
        "Julia=1.13.1",
        "reviewed_fork=62b0f97f076fb55c71c3ab0caa041a9adff94e04",
        "nsteps=20 nsmp=20 seed=1 NSRCG=0 NStore=1",
    ] {
        if metadata
            .lines()
            .filter(|line| line.trim() == required)
            .count()
            != 1
        {
            return Err(format!(
                "direct20 provenance requires one exact line: {required}"
            ));
        }
    }

    Ok(())
}

fn model_mode(model: &Model) -> &'static str {
    match model.name {
        "heisenberg_chain_cmp" => "cmp",
        "heisenberg_chain_fsz" => "fsz",
        _ => "real",
    }
}

fn validate_model(model: &Model, data: &mvmc_core::ExpertModeData) -> Result<(), String> {
    let complex = mvmc_core::get_all_complex_flag(data).unwrap();
    let fsz = data.i_flg_orbital_general != 0;
    if data.modpara.nsrcg != 0
        || data.modpara.nstore_o != 1
        || complex != (model_mode(model) != "real")
        || fsz != (model_mode(model) == "fsz")
    {
        return Err(format!("{}: parsed direct/store1 model contract mismatch: NSRCG={} NStore={} complex={complex} FSZ={fsz}", model.name, data.modpara.nsrcg, data.modpara.nstore_o));
    }
    Ok(())
}

#[test]
fn preflight_rejects_metadata_substrings_duplicates_and_wrong_model_settings() {
    let metadata = "Julia=1.13.1\nreviewed_fork=62b0f97f076fb55c71c3ab0caa041a9adff94e04\nnsteps=20 nsmp=20 seed=1 NSRCG=0 NStore=1\n";
    assert!(validate_metadata(metadata).is_ok());
    for invalid in [
        metadata.replace("Julia=", "not_Julia="),
        format!("{metadata}Julia=1.13.1\n"),
        metadata.replace("NSRCG=0", "NSRCG=1"),
        metadata.replace("nsteps=20", "nsteps=50"),
    ] {
        assert!(validate_metadata(&invalid).is_err());
    }
    for model in all_models() {
        let mut data = mvmc_core::ExpertModeData::new();
        data.modpara.nstore_o = 1;
        data.complex_flags = vec![i64::from(model_mode(model) != "real")];
        data.i_flg_orbital_general = i64::from(model_mode(model) == "fsz");
        assert!(validate_model(model, &data).is_ok());
        data.modpara.nsrcg = 1;
        assert!(validate_model(model, &data).is_err());
        data.modpara.nsrcg = 0;
        data.modpara.nstore_o = 0;
        assert!(validate_model(model, &data).is_err());
        data.modpara.nstore_o = 1;
        data.complex_flags[0] ^= 1;
        assert!(validate_model(model, &data).is_err());
        data.complex_flags[0] ^= 1;
        data.i_flg_orbital_general ^= 1;
        assert!(validate_model(model, &data).is_err());
    }
}

fn compare_rows(model: &str, got_rows: &[Vec<f64>], ref_rows: &[Vec<f64>]) -> Result<(), String> {
    for (side, rows) in [("actual", got_rows), ("reference", ref_rows)] {
        let width = rows.first().map_or(0, Vec::len);
        if width != OUTPUT_COLUMNS || rows.iter().any(|row| row.len() != OUTPUT_COLUMNS) {
            return Err(format!(
                "{model}: {side} requires {OUTPUT_COLUMNS} columns in every row"
            ));
        }
        for (row, values) in rows.iter().enumerate() {
            for (col, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!("{model}: {side} nonfinite row {row} col {col}"));
                }
            }
        }
    }

    if got_rows.len() != N_STEPS {
        return Err(format!(
            "{}: expected {} rows, got {}",
            model,
            N_STEPS,
            got_rows.len()
        ));
    }
    if ref_rows.len() != N_STEPS {
        return Err(format!(
            "{}: reference has {} rows, expected {}",
            model,
            ref_rows.len(),
            N_STEPS
        ));
    }

    for (row, (got_row, ref_row)) in got_rows.iter().zip(ref_rows.iter()).enumerate() {
        if got_row.len() != ref_row.len() {
            return Err(format!(
                "{}: row {} column count mismatch (got {} expected {})",
                model,
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
                    model, row, col,
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn numeric_comparison_rejects_nonfinite_on_either_side_and_invalid_shapes() {
    let valid = vec![vec![1.0; 6]; N_STEPS];
    assert!(compare_rows("negative", &valid, &valid).is_ok());
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for col in 0..6 {
            let mut invalid = valid.clone();
            invalid[0][col] = value;
            assert!(compare_rows("negative", &invalid, &valid)
                .unwrap_err()
                .contains("actual nonfinite"));
            assert!(compare_rows("negative", &valid, &invalid)
                .unwrap_err()
                .contains("reference nonfinite"));
        }
    }
    for invalid in [vec![], vec![vec![]; N_STEPS], {
        let mut rows = valid.clone();
        rows[1].pop();
        rows
    }] {
        assert!(compare_rows("negative", &invalid, &valid).is_err());
        assert!(compare_rows("negative", &valid, &invalid).is_err());
    }
}

#[test]
fn numeric_comparison_requires_six_columns_on_each_side() {
    let valid = vec![vec![1.0; OUTPUT_COLUMNS]; N_STEPS];
    for width in [1, 5, 7] {
        let invalid = vec![vec![1.0; width]; N_STEPS];
        assert!(compare_rows("negative", &invalid, &valid)
            .unwrap_err()
            .contains("actual requires 6 columns"));
        assert!(compare_rows("negative", &valid, &invalid)
            .unwrap_err()
            .contains("reference requires 6 columns"));
        assert!(compare_rows("negative", &invalid, &invalid).is_err());
    }
}

#[test]
#[ignore = "long direct20 fixture gate: MVMC_RS_PHASE5_20STEP required"]
fn heisenberg_chain_real_20step() {
    run_model_test("heisenberg_chain_real");
}

#[test]
#[ignore = "long direct20 fixture gate: MVMC_RS_PHASE5_20STEP required"]
fn heisenberg_chain_cmp_20step() {
    run_model_test("heisenberg_chain_cmp");
}

#[test]
#[ignore = "long direct20 fixture gate: MVMC_RS_PHASE5_20STEP required"]
fn heisenberg_chain_fsz_20step() {
    run_model_test("heisenberg_chain_fsz");
}

#[test]
#[ignore = "long direct20 fixture gate: MVMC_RS_PHASE5_20STEP required"]
fn hubbard_chain_real_20step() {
    run_model_test("hubbard_chain_real");
}

fn run_model_test(model_name: &str) {
    support::require_gate("phase5-20step", "MVMC_RS_PHASE5_20STEP");
    let julia = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("phase5-20step", "Julia-mVMC checkout not found")
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
            support::missing_fixture("phase5-20step", path.display().to_string());
        }
    }
}

#[test]
fn selected_long_gates_fail_on_missing_selector_checkout_or_namelist() {
    let owned = tempdir("reporting-negative");
    let empty_checkout = &owned.0;
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
                    &format!("{}_20step", model.name),
                    "--nocapture",
                ])
                .env_remove("MVMC_RS_PHASE5_20STEP")
                .env("JULIA_MVMC_ROOT", root);
            if let Some(selector) = selector {
                command.env("MVMC_RS_PHASE5_20STEP", selector);
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
}

#[test]
fn selected_long_gate_missing_expected_reference_cannot_pass() {
    // Use this source file as an existing namelist stand-in: preflight must
    // reject the absent expectation before parsing or numerical work.
    let existing = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/phase5_regression_20step.rs");
    let empty = tempdir("missing-reference");
    let missing = empty.0.join("zvo_out.dat");
    let error = std::panic::catch_unwind(|| require_fixtures(&existing, &missing)).unwrap_err();
    let message = error.downcast_ref::<String>().unwrap();
    assert!(message.contains("missing fixture") && message.contains("zvo_out.dat"));
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

struct OwnedTempDir(PathBuf);

impl Drop for OwnedTempDir {
    fn drop(&mut self) {
        // Only paths acquired by exclusive create_dir reach this guard.
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("owned output cleanup {}: {error}", self.0.display());
        }
    }
}

fn tempdir(tag: &str) -> OwnedTempDir {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    loop {
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mvmc-phase5-20step-{tag}-{}-{sequence}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return OwnedTempDir(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create owned output {}: {error}", path.display()),
        }
    }
}
