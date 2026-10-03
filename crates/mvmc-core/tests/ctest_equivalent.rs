//! Rust counterpart of Julia-mVMC's `test/integration/ctest_equivalent.jl`.
//!
//! The test is opt-in because the pinned C ctest workloads run hundreds or
//! thousands of SR steps. Select fixtures with
//! `MVMC_RS_CTEST_MODELS=heisenberg_chain_real,hubbard_chain_real`.

use std::fs;
use std::path::Path;

use mvmc_core::{run_para_opt_from_namelist, RunConfig};

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
#[ignore = "long reference gate: select MVMC_RS_CTEST_MODELS and use --run-ignored only"]
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
                report_gate(
                    "ctest-equivalent",
                    GateStatus::Failure,
                    &format!("unknown ctest model {name:?}"),
                );
                panic!("unknown ctest model {name:?}");
            });
        report_gate(name, GateStatus::NotRun,
            &format!("independent prefix gate is separate, not executed by long summary: {}; fixture_available={}",
                model.reason, model.prefix_reference_available));
        let outcome = std::panic::catch_unwind(|| {
            let fixture = root.join("test/integration/reference").join(model.fixture);
            let namelist = fixture.join("inputs/namelist.def");
            let mean_path = fixture.join("ctest_ref/ref_mean.dat");
            let std_path = fixture.join("ctest_ref/ref_std.dat");
            if !namelist.is_file() || !mean_path.is_file() || !std_path.is_file() {
                support::missing_fixture(
                    "ctest-equivalent",
                    format!("{name}: namelist or ctest reference is missing"),
                );
            }
            let ref_mean = read_values(&mean_path);
            let ref_std = read_values(&std_path);
            assert!(
                ref_mean.len() >= 2 && ref_std.len() >= 2,
                "{name}: C refs need two values"
            );
            let output =
                std::env::temp_dir().join(format!("mvmc-rs-ctest-{name}-{}", std::process::id()));
            let parsed = mvmc_expert_parsers::parse_expert_mode_files(&namelist)
                .unwrap_or_else(|error| panic!("{name}: parse failed: {error}"));
            let config = RunConfig {
                nsmp: Some(parsed.modpara.nsr_opt_itr_smp),
                seed: None,
                output_dir: Some(output.clone()),
                ..RunConfig::new(parsed.modpara.nsr_opt_itr_step, model.mode)
            };
            let result = run_para_opt_from_namelist(&namelist, config)
                .unwrap_or_else(|error| panic!("{name}: Rust run failed: {error}"));
            assert_eq!(result.status, 0, "{name}: status");
            for index in 0..2 {
                assert!(
                    passes(result.ctest_values[index], ref_mean[index], ref_std[index]),
                    "{name}: ctest column {index}: calculated={} expected={} std={}",
                    result.ctest_values[index],
                    ref_mean[index],
                    ref_std[index]
                );
            }
            let _ = fs::remove_dir_all(output);
            report_gate(
                "ctest-equivalent",
                GateStatus::Pass,
                &format!("{name}: native long statistical summary only"),
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
