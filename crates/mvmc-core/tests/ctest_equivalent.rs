//! Rust counterpart of Julia-mVMC's `test/integration/ctest_equivalent.jl`.
//!
//! Current long acceptance uses 20 steps and a matched 20-sample window.
//! Independent reference generation is opt-in. Select fixtures with
//! `MVMC_RS_CTEST_MODELS=heisenberg_chain_real,hubbard_chain_real`.

use std::fs;
use std::path::Path;

use mvmc_core::{run_para_opt_from_namelist, RunConfig};

#[path = "support/ctest_provenance.rs"]
mod ctest_provenance;
#[path = "support/fixture_status.rs"]
mod fixture_status;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
mod support;
use support::{julia_mvmc_root, report_gate, require_gate, GateStatus};

const ABSOLUTE_FLOOR: f64 = 1.0e-8;

#[derive(Clone, Copy)]
struct Model {
    fixture: &'static str,
    mode: &'static str,
    // Fixture availability is not a claim that its numerical gate passed.
    prefix_reference_available: bool,
    reason: &'static str,
}

const MODELS: &[Model] = &[
    Model {
        fixture: "heisenberg_chain_real",
        mode: "real",
        prefix_reference_available: true,
        reason: "",
    },
    Model {
        fixture: "hubbard_chain_real",
        mode: "real",
        prefix_reference_available: true,
        reason: "new canonical initial.def fixtures; archived no-overlay prefixes are distinct",
    },
    Model {
        fixture: "heisenberg_chain_cmp",
        mode: "cmp",
        prefix_reference_available: true,
        reason: "",
    },
    Model {
        fixture: "heisenberg_chain_fsz",
        mode: "fsz",
        prefix_reference_available: true,
        reason: "new canonical initial.def fixtures; archived no-overlay prefixes are distinct",
    },
    Model {
        fixture: "hubbard_chain_cmp",
        mode: "cmp",
        prefix_reference_available: true,
        reason: "canonical 1/2/3/50 fixtures; numerical gate is separate",
    },
    Model {
        fixture: "hubbard_chain_fsz",
        mode: "fsz",
        prefix_reference_available: true,
        reason: "canonical native-C-energy/Julia FSZ 1/2/3/50 fixtures",
    },
    Model {
        fixture: "kondo_chain_real",
        mode: "real",
        prefix_reference_available: true,
        reason: "canonical local-spin optimization 1/2/3/50 fixtures",
    },
    Model {
        fixture: "kondo_chain_cmp",
        mode: "cmp",
        prefix_reference_available: true,
        reason: "canonical complex local-spin optimization 1/2/3/50 fixtures",
    },
    Model {
        fixture: "kondo_chain_stot1_cmp",
        mode: "cmp",
        prefix_reference_available: true,
        reason: "canonical Stot=1 optimization 1/2/3/50 fixtures",
    },
    Model {
        fixture: "general_rbm_cmp",
        mode: "cmp",
        prefix_reference_available: true,
        reason: "mixed C-counter/Julia SR reference, not full C executable parity",
    },
    Model {
        fixture: "hubbard_tetragonal_real",
        mode: "real",
        prefix_reference_available: true,
        reason: "canonical tetragonal optimization 1/2/3/50 fixtures",
    },
    Model {
        fixture: "hubbard_tetragonal_momentum_projection_real",
        mode: "real",
        prefix_reference_available: true,
        reason: "canonical momentum-projection optimization 1/2/3/50 fixtures",
    },
    Model {
        fixture: "kondo_chain_fsz",
        mode: "fsz",
        prefix_reference_available: true,
        reason: "canonical native-C-energy/Julia FSZ Kondo 1/2/3/50 fixtures",
    },
];

fn read_values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
        .split_whitespace()
        .map(|token| {
            token.parse::<f64>().unwrap_or_else(|error| {
                panic!("invalid number {token:?} in {}: {error}", path.display())
            })
        })
        .collect()
}

fn passes(calculated: f64, expected: f64, sigma: f64) -> bool {
    if !calculated.is_finite() || !expected.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return false;
    }
    let difference = (calculated - expected).abs();
    !(difference >= 3.0 * sigma && difference >= ABSOLUTE_FLOOR)
}

fn report_classified(name: &str, error: &fixture_status::GateError) {
    let status = match error.status {
        fixture_status::Status::MissingFixture => GateStatus::MissingFixture,
        fixture_status::Status::Unsupported => GateStatus::Unsupported,
        fixture_status::Status::Failure => GateStatus::Failure,
    };
    report_gate(name, status, &error.detail);
}

fn record_failed_case<'a>(
    name: &'a str,
    payload: &(dyn std::any::Any + Send),
    failures: &mut Vec<&'a str>,
) -> fixture_status::Status {
    let error = fixture_status::caught_error(payload);
    report_classified(name, &error);
    failures.push(name);
    error.status
}

fn preflight_input_closure(expected: &Path, input_root: &Path) {
    let manifest = fixture_status::read_member(expected, "inputs.sha256")
        .unwrap_or_else(|error| fixture_status::raise(error));
    fixture_status::require_manifest_members(&manifest, input_root)
        .unwrap_or_else(|error| fixture_status::raise(error));
    let namelist = fixture_status::read_member(input_root, "namelist.def")
        .unwrap_or_else(|error| fixture_status::raise(error));
    let namelist = std::str::from_utf8(&namelist).expect("independent namelist UTF8");
    for (_, file) in mvmc_expert_parsers::utils::file::parse_namelist_content(namelist) {
        fixture_status::require_files(input_root, &[file.as_str()])
            .unwrap_or_else(|error| fixture_status::raise(error));
    }
}

#[test]
#[ignore = "20-step reference gate: select MVMC_RS_CTEST_MODELS and use --run-ignored only"]
fn rust_ctest_equivalent_selected_models() {
    if let Err(std::env::VarError::NotUnicode(_)) = std::env::var("MVMC_RS_CTEST_MODELS") {
        let error = fixture_status::GateError {
            status: fixture_status::Status::Unsupported,
            detail: "nonUnicode explicit MVMC_RS_CTEST_MODELS selector".into(),
        };
        report_classified("ctest-equivalent", &error);
        fixture_status::raise(error);
    }
    require_gate("ctest-equivalent", "MVMC_RS_CTEST_MODELS");
    let filter = std::env::var("MVMC_RS_CTEST_MODELS").expect("selection was present");
    let supported: Vec<_> = MODELS.iter().map(|model| model.fixture).collect();
    let requested = fixture_status::selection(&filter, &supported).unwrap_or_else(|error| {
        report_classified("ctest-equivalent", &error);
        fixture_status::raise(error)
    });

    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("ctest-equivalent", "Julia-mVMC checkout not found")
    });
    for model in MODELS {
        if !requested.contains(&model.fixture) {
            report_gate(
                model.fixture,
                GateStatus::NotRun,
                "long ctest not selected; unverified",
            );
        }
    }
    let mut failures = Vec::new();
    for name in requested {
        let model = MODELS
            .iter()
            .find(|model| model.fixture == name)
            .unwrap_or_else(|| {
                support::unsupported("ctest-equivalent", format!("unknown ctest model {name:?}"));
            });
        report_gate(name, GateStatus::NotRun,
            &format!("independent prefix gate is separate, not executed by long summary: {}; fixture_available={}",
                model.reason, model.prefix_reference_available));
        let outcome = std::panic::catch_unwind(|| {
            let fixture = root.join("test/integration/reference").join(model.fixture);
            let namelist = fixture.join("inputs/namelist.def");
            let references = fixture_status::oracle_root(
                std::env::var("MVMC_RS_CTEST_ORACLE_ROOT"),
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/fixtures/ctest_model_prefixes"),
            )
            .unwrap_or_else(|error| fixture_status::raise(error));
            let expected = references.join(name).join("step-20");
            let expected_output = expected.join("zvo_out.dat");
            fixture_status::require_files(
                &root,
                &[format!(
                    "test/integration/reference/{}/inputs/namelist.def",
                    model.fixture
                )
                .as_str()],
            )
            .unwrap_or_else(|error| fixture_status::raise(error));
            fixture_status::require_files(&references, &["provenance.txt"])
                .unwrap_or_else(|error| fixture_status::raise(error));
            let required: Vec<_> = [
                "inputs.sha256",
                "status.txt",
                "model-settings.txt",
                "zvo_out.dat",
            ]
            .iter()
            .map(|file| format!("{name}/step-20/{file}"))
            .collect();
            let required: Vec<_> = required.iter().map(String::as_str).collect();
            fixture_status::require_files(&references, &required)
                .unwrap_or_else(|error| fixture_status::raise(error));
            let provenance = fs::read_to_string(references.join("provenance.txt")).unwrap();
            assert!(provenance.split_whitespace().any(|v| v == "Julia=1.13.1"));
            assert!(
                provenance.contains("62b0f97f076fb55c71c3ab0caa041a9adff94e04"),
                "reviewed reference source identity required for fresh20"
            );
            assert!(provenance.contains("Manifest-v1.13.toml") && provenance.contains("sha256="));
            let input_root = namelist.parent().unwrap();
            preflight_input_closure(&expected, input_root);
            ctest_provenance::verify_inputs(
                &expected.join("inputs.sha256"),
                namelist.parent().unwrap(),
            );
            assert_eq!(
                fs::read_to_string(expected.join("status.txt"))
                    .unwrap()
                    .trim(),
                "0"
            );
            let settings = fs::read_to_string(expected.join("model-settings.txt")).unwrap();
            assert!(settings
                .split_whitespace()
                .any(|v| v == "effective_NSROptItrStep=20"));
            assert!(settings
                .split_whitespace()
                .any(|v| v == "effective_NSROptItrSmp=20"));
            assert!(settings
                .split_whitespace()
                .any(|v| v == "override=both_no_clamp"));
            let expected_text = fs::read_to_string(&expected_output).unwrap();
            let rows: Vec<Vec<f64>> = expected_text
                .lines()
                .map(|line| {
                    line.split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect()
                })
                .collect();
            assert_eq!(rows.len(), 20, "independent 20-step output rows");
            assert!(rows
                .iter()
                .all(|row| row.len() >= 2 && row.iter().all(|v| v.is_finite())));
            let ref_mean: Vec<f64> = (0..2)
                .map(|column| rows.iter().map(|row| row[column]).sum::<f64>() / 20.0)
                .collect();
            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let output = std::env::temp_dir()
                .join(format!("mvmc-rs-ctest-{name}-{}-{id}", std::process::id()));
            let parsed = mvmc_expert_parsers::parse_expert_mode_files(&namelist)
                .unwrap_or_else(|error| panic!("{name}: parse failed: {error}"));
            assert!(settings
                .split_whitespace()
                .any(|v| v == format!("RndSeed={}", parsed.modpara.rnd_seed)));
            let config = RunConfig {
                nsmp: Some(20),
                seed: None,
                output_dir: Some(output.clone()),
                ..RunConfig::new(20, model.mode)
            };
            fs::create_dir(&output).expect("exclusive owned output directory creation");
            let result = run_para_opt_from_namelist(&namelist, config)
                .unwrap_or_else(|error| panic!("{name}: Rust run failed: {error}"));
            assert_eq!(result.status, 0, "{name}: status");
            for (index, &expected_mean) in ref_mean.iter().enumerate() {
                numerical_comparison::assert_close(
                    result.ctest_values[index],
                    expected_mean,
                    1e-11,
                    1e-11,
                    format!("{name}: independent 20-step summary column {index}"),
                );
            }
            fs::remove_dir_all(&output)
                .expect("remove only this gate's exclusively created directory");
            report_gate(
                "ctest-equivalent",
                GateStatus::Pass,
                &format!("{name}: native 20-step/window20 independent summary only"),
            );
        });
        if let Err(payload) = outcome {
            record_failed_case(name, payload.as_ref(), &mut failures);
        }
    }
    assert!(
        failures.is_empty(),
        "failed long ctest models: {failures:?}"
    );
}

/// Upstream C/Julia ctest contract at the upstream run length.
///
/// Mirrors `extern/mVMC-1.3.0/test/python/runtest.py` and Julia's
/// `ctest_equivalent.jl`: run the unmodified input (`NSROptItrStep`/`NSROptItrSmp`
/// from `modpara.def`, seed from the input) and compare the first two window-averaged
/// summary values with `ref_mean.dat`/`ref_std.dat`. A model fails only when
/// `|difference| >= 3*ref_std` and `|difference| >= 1e-8`. References are the C
/// shipped `test/python/data/<Model>/ref` files (see
/// `tests/fixtures/ctest_upstream_reference/PROVENANCE.md`). This is a statistical
/// model-level contract; it is not deterministic trajectory parity.
#[test]
#[ignore = "upstream-length ctest gate: select MVMC_RS_CTEST_UPSTREAM_MODELS and use --run-ignored only"]
fn rust_ctest_upstream_rule_selected_models() {
    if let Err(std::env::VarError::NotUnicode(_)) = std::env::var("MVMC_RS_CTEST_UPSTREAM_MODELS") {
        let error = fixture_status::GateError {
            status: fixture_status::Status::Unsupported,
            detail: "nonUnicode explicit MVMC_RS_CTEST_UPSTREAM_MODELS selector".into(),
        };
        report_classified("ctest-upstream", &error);
        fixture_status::raise(error);
    }
    require_gate("ctest-upstream", "MVMC_RS_CTEST_UPSTREAM_MODELS");
    let mut filter = std::env::var("MVMC_RS_CTEST_UPSTREAM_MODELS").expect("selection present");
    let supported: Vec<_> = MODELS.iter().map(|model| model.fixture).collect();
    if filter.trim() == "all" {
        filter = supported.join(",");
    }
    let requested = fixture_status::selection(&filter, &supported).unwrap_or_else(|error| {
        report_classified("ctest-upstream", &error);
        fixture_status::raise(error)
    });
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("ctest-upstream", "Julia-mVMC checkout not found")
    });
    let references =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctest_upstream_reference");
    for model in MODELS {
        if !requested.contains(&model.fixture) {
            report_gate(
                model.fixture,
                GateStatus::NotRun,
                "upstream ctest not selected; unverified",
            );
        }
    }
    let mut failures = Vec::new();
    for name in requested {
        let model = MODELS.iter().find(|model| model.fixture == name).unwrap();
        let outcome = std::panic::catch_unwind(|| {
            let input_root = root
                .join("test/integration/reference")
                .join(model.fixture)
                .join("inputs");
            let namelist = input_root.join("namelist.def");
            fixture_status::require_files(&input_root, &["namelist.def"])
                .unwrap_or_else(|error| fixture_status::raise(error));
            let required = [
                format!("{name}/ref_mean.dat"),
                format!("{name}/ref_std.dat"),
                format!("{name}/inputs.sha256"),
            ];
            let required: Vec<_> = required.iter().map(String::as_str).collect();
            fixture_status::require_files(&references, &required)
                .unwrap_or_else(|error| fixture_status::raise(error));
            preflight_input_closure(&references.join(name), &input_root);
            ctest_provenance::verify_inputs(
                &references.join(name).join("inputs.sha256"),
                &input_root,
            );
            let ref_mean = read_values(&references.join(name).join("ref_mean.dat"));
            let ref_std = read_values(&references.join(name).join("ref_std.dat"));
            assert!(ref_mean.len() >= 2 && ref_std.len() >= 2);
            let parsed = mvmc_expert_parsers::parse_expert_mode_files(&namelist)
                .unwrap_or_else(|error| panic!("{name}: parse failed: {error}"));
            let nsteps = parsed.modpara.nsr_opt_itr_step;
            let nsmp = parsed.modpara.nsr_opt_itr_smp;
            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let output = std::env::temp_dir().join(format!(
                "mvmc-rs-ctest-upstream-{name}-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&output).expect("exclusive owned output directory creation");
            let config = RunConfig {
                nsmp: Some(nsmp),
                seed: None,
                output_dir: Some(output.clone()),
                ..RunConfig::new(nsteps, model.mode)
            };
            let started = std::time::Instant::now();
            let result = run_para_opt_from_namelist(&namelist, config)
                .unwrap_or_else(|error| panic!("{name}: Rust run failed: {error}"));
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(result.status, 0, "{name}: status");
            assert_eq!(result.effective_nsteps as i64, nsteps);
            assert_eq!(result.effective_nsmp as i64, nsmp);
            let mut failed_columns = Vec::new();
            for column in 0..2 {
                let calculated = result.ctest_values[column];
                let difference = (calculated - ref_mean[column]).abs();
                let ok = passes(calculated, ref_mean[column], ref_std[column]);
                let ratio = if ref_std[column] > 0.0 {
                    difference / ref_std[column]
                } else if difference == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                };
                eprintln!(
                    "ctest-upstream result model={name} column={column} nsteps={nsteps} nsmp={nsmp} calculated={calculated:.17e} expected={:.17e} sigma={:.17e} diff={difference:.17e} diff_over_sigma={ratio:.6} ok={ok} seconds={elapsed:.1}",
                    ref_mean[column], ref_std[column],
                );
                if !ok {
                    failed_columns.push(column);
                }
            }
            fs::remove_dir_all(&output)
                .expect("remove only this gate's exclusively created directory");
            assert!(
                failed_columns.is_empty(),
                "{name}: upstream ctest rule failed in columns {failed_columns:?}"
            );
            report_gate(
                "ctest-upstream",
                GateStatus::Pass,
                &format!(
                    "{name}: upstream rule (>=3 sigma and >=1e-8) at nsteps={nsteps} nsmp={nsmp}"
                ),
            );
        });
        if let Err(payload) = outcome {
            record_failed_case(name, payload.as_ref(), &mut failures);
        }
    }
    assert!(
        failures.is_empty(),
        "failed upstream ctest models: {failures:?}"
    );
}

#[test]
fn upstream_references_cover_every_model_with_well_formed_values() {
    let references =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctest_upstream_reference");
    let table = fs::read_to_string(references.join("models.tsv")).unwrap();
    let listed: Vec<&str> = table
        .lines()
        .map(|line| line.split('\t').next().unwrap())
        .collect();
    let mut expected: Vec<&str> = MODELS.iter().map(|model| model.fixture).collect();
    expected.sort_unstable();
    assert_eq!(
        listed, expected,
        "models.tsv must list exactly the 13 ctest models"
    );
    for model in MODELS {
        let mean = read_values(&references.join(model.fixture).join("ref_mean.dat"));
        let std = read_values(&references.join(model.fixture).join("ref_std.dat"));
        assert_eq!(mean.len(), std.len(), "{}: ref lengths", model.fixture);
        assert!(
            mean.len() >= 2,
            "{}: need two compared values",
            model.fixture
        );
        assert!(mean.iter().all(|v| v.is_finite()));
        assert!(std.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(references
            .join(model.fixture)
            .join("inputs.sha256")
            .is_file());
    }
}

#[test]
fn ctest_failure_requires_both_thresholds() {
    assert!(passes(1.0, 1.0 + 0.9e-8, 0.0));
    assert!(passes(1.0, 1.0 + 2.0e-8, 1.0e-8));
    assert!(!passes(1.0, 1.0 + 3.1e-8, 1.0e-8));
    assert!(!passes(f64::NAN, 1.0, 1.0));
    assert!(!passes(1.0, f64::INFINITY, 1.0));
    assert!(!passes(1.0, 1.0, -1.0));
}

#[test]
fn classified_reporting_keeps_terminal_aggregate_failure() {
    let mut failures = Vec::new();
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("gate-reporting-{}-{id}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let owned = Owned(dir);
    let missing = fixture_status::require_files(&owned.0, &["status.txt"]).unwrap_err();
    let unsupported = fixture_status::selection("unknown", &["real"]).unwrap_err();
    fs::create_dir(owned.0.join("status.txt")).unwrap();
    let corrupt = fixture_status::require_files(&owned.0, &["status.txt"]).unwrap_err();
    for (name, expected) in [
        ("missing", missing),
        ("unsupported", unsupported),
        ("corrupt", corrupt),
    ] {
        let expected_status = expected.status;
        let payload = std::panic::catch_unwind(|| fixture_status::raise(expected)).unwrap_err();
        assert_eq!(
            record_failed_case(name, payload.as_ref(), &mut failures),
            expected_status
        );
    }
    fs::remove_dir(owned.0.join("status.txt")).unwrap();
    fs::write(owned.0.join("status.txt"), b"1").unwrap();
    let payload = std::panic::catch_unwind(|| {
        assert_eq!(fs::read(owned.0.join("status.txt")).unwrap(), b"0")
    })
    .unwrap_err();
    assert_eq!(
        record_failed_case("changed", payload.as_ref(), &mut failures),
        fixture_status::Status::Failure
    );
    assert_eq!(failures, ["missing", "unsupported", "corrupt", "changed"]);
    assert!(
        !failures.is_empty(),
        "classified errors must not authorize selected gate success"
    );
}

#[test]
fn transitive_inputs_preserve_missing_and_changed_hash_classification() {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ctest-transitive-{}-{id}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let owned = Owned(dir);
    let input = owned.0.join("inputs");
    let expected = owned.0.join("expected");
    fs::create_dir(&input).unwrap();
    fs::create_dir(&expected).unwrap();
    fs::write(
        input.join("namelist.def"),
        b"ModPara modpara.def\nOrbital orbital.def\n",
    )
    .unwrap();
    fs::write(input.join("modpara.def"), b"abc").unwrap();
    fs::write(input.join("orbital.def"), b"").unwrap();
    // Fixed metadata hashes: standard abc/empty vectors and independent host
    // sha256sum of the literal two-line namelist. No numerical expected values.
    let manifest = b"028f1f7130456e9fcac5b923116c6fd2a1b1c8c1b281f19a18ab5de012eaf23d namelist.def\nba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad modpara.def\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 orbital.def\n";
    fs::write(expected.join("inputs.sha256"), manifest).unwrap();
    preflight_input_closure(&expected, &input);
    ctest_provenance::verify_inputs(&expected.join("inputs.sha256"), &input);
    let mut failures = Vec::new();
    fs::remove_file(input.join("orbital.def")).unwrap();
    let payload =
        std::panic::catch_unwind(|| preflight_input_closure(&expected, &input)).unwrap_err();
    assert_eq!(
        record_failed_case("missing-member", payload.as_ref(), &mut failures),
        fixture_status::Status::MissingFixture
    );
    fs::write(input.join("orbital.def"), b"changed").unwrap();
    preflight_input_closure(&expected, &input);
    let payload = std::panic::catch_unwind(|| {
        ctest_provenance::verify_inputs(&expected.join("inputs.sha256"), &input)
    })
    .unwrap_err();
    assert_eq!(
        record_failed_case("changed-hash", payload.as_ref(), &mut failures),
        fixture_status::Status::Failure
    );
    // Referenced, existing input omitted from hash manifest remains verifier Failure.
    fs::write(input.join("orbital.def"), b"").unwrap();
    let manifest = std::str::from_utf8(manifest)
        .unwrap()
        .lines()
        .take(2)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(expected.join("inputs.sha256"), manifest).unwrap();
    preflight_input_closure(&expected, &input);
    let payload = std::panic::catch_unwind(|| {
        ctest_provenance::verify_inputs(&expected.join("inputs.sha256"), &input)
    })
    .unwrap_err();
    assert_eq!(
        record_failed_case("unhashed-reference", payload.as_ref(), &mut failures),
        fixture_status::Status::Failure
    );
    fs::remove_file(input.join("orbital.def")).unwrap();
    let payload =
        std::panic::catch_unwind(|| preflight_input_closure(&expected, &input)).unwrap_err();
    assert_eq!(
        record_failed_case(
            "missing-namelist-reference",
            payload.as_ref(),
            &mut failures
        ),
        fixture_status::Status::MissingFixture
    );
    fs::remove_file(expected.join("inputs.sha256")).unwrap();
    let payload =
        std::panic::catch_unwind(|| preflight_input_closure(&expected, &input)).unwrap_err();
    assert_eq!(
        record_failed_case("missing-metadata", payload.as_ref(), &mut failures),
        fixture_status::Status::MissingFixture
    );
    assert_eq!(failures.len(), 5);
}

#[test]
#[ignore = "reference inventory: requires the Julia reference checkout"]
fn inventory_all_thirteen_ctest_input_contracts() {
    let root = julia_mvmc_root().expect("Julia reference checkout required");
    assert_eq!(MODELS.len(), 13);
    let mut names = std::collections::BTreeSet::new();
    for model in MODELS {
        assert!(names.insert(model.fixture), "duplicate model");
        let fixture = root.join("test/integration/reference").join(model.fixture);
        let namelist = fixture.join("inputs/namelist.def");
        let data = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap();
        let contract = mvmc_core::validation::validate_para_opt(&data);
        eprintln!("inventory {}: {:?}; steps={} nsmp={} orbital_rows={} declared_slater={}; deterministic coverage: {}",
            model.fixture, contract, data.modpara.nsr_opt_itr_step,
            data.modpara.nsr_opt_itr_smp, data.orbital_terms.len(),
            mvmc_expert_parsers::utils::parameter_init::n_slater(&data),
            if model.prefix_reference_available { "existing model gate; not executed by inventory" } else { model.reason });
        for file in ["ctest_ref/ref_mean.dat", "ctest_ref/ref_std.dat"] {
            let values = read_values(&fixture.join(file));
            assert!(values.len() >= 2 && values.iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
#[ignore = "execution audit, not parity: runs one step for all 13 historical inputs"]
fn audit_thirteen_ctest_one_step_execution_paths() {
    let root = julia_mvmc_root().expect("Julia reference checkout required");
    for model in MODELS {
        let namelist = root
            .join("test/integration/reference")
            .join(model.fixture)
            .join("inputs/namelist.def");
        let config = RunConfig {
            nsmp: Some(1),
            seed: None,
            ..RunConfig::new(1, model.mode)
        };
        match run_para_opt_from_namelist(&namelist, config) {
            Ok(result) => {
                assert_eq!(result.status, 0, "{}", model.fixture);
                assert!(result.ctest_values.iter().all(|value| value.is_finite()));
                eprintln!("execution audit {}: one step executed; deterministic and long parity UNVERIFIED", model.fixture);
            }
            Err(error) => eprintln!(
                "execution audit {}: REJECTED: {error}; parity UNVERIFIED",
                model.fixture
            ),
        }
    }
}
