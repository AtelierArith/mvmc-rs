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

#[test]
#[ignore = "20-step reference gate: select MVMC_RS_CTEST_MODELS and use --run-ignored only"]
fn rust_ctest_equivalent_selected_models() {
    require_gate("ctest-equivalent", "MVMC_RS_CTEST_MODELS");
    let filter = std::env::var("MVMC_RS_CTEST_MODELS").expect("selection was present");
    let requested: Vec<_> = if filter.trim() == "all" {
        MODELS.iter().map(|model| model.fixture).collect()
    } else {
        filter
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect()
    };
    if requested.is_empty() {
        report_gate(
            "ctest-equivalent",
            GateStatus::Failure,
            "selection listed no models",
        );
        panic!("MVMC_RS_CTEST_MODELS selected no models");
    }

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
            let references = std::env::var("MVMC_RS_CTEST_ORACLE_ROOT")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| {
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tests/fixtures/ctest_model_prefixes")
                });
            let expected = references.join(name).join("step-20");
            let expected_output = expected.join("zvo_out.dat");
            if !namelist.is_file() || !expected_output.is_file() {
                support::missing_fixture(
                    "ctest-equivalent",
                    format!("{name}: fresh independent 20-step reference missing; historical long/50-step output cannot substitute"),
                );
            }
            assert!(
                references.join("provenance.txt").is_file(),
                "independent provenance required"
            );
            let provenance = fs::read_to_string(references.join("provenance.txt")).unwrap();
            assert!(provenance.split_whitespace().any(|v| v == "Julia=1.13.1"));
            assert!(
                provenance.contains("62b0f97f076fb55c71c3ab0caa041a9adff94e04"),
                "reviewed reference source identity required for fresh20"
            );
            assert!(provenance.contains("Manifest-v1.13.toml") && provenance.contains("sha256="));
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
        if outcome.is_err() {
            report_gate(
                name,
                GateStatus::Failure,
                "native long ctest failed; see error above",
            );
            failures.push(name);
        }
    }
    assert!(
        failures.is_empty(),
        "failed long ctest models: {failures:?}"
    );
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
