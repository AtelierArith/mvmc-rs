//! Rust counterpart of Julia-mVMC's `test/integration/ctest_equivalent.jl`.
//!
//! The test is opt-in because the pinned C ctest workloads run hundreds or
//! thousands of SR steps. Select fixtures with
//! `MVMC_RS_CTEST_MODELS=heisenberg_chain_real,hubbard_chain_real`.

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::{run_para_opt_from_namelist, RunConfig};

const ABSOLUTE_FLOOR: f64 = 1.0e-8;

#[derive(Clone, Copy)]
struct Model {
    fixture: &'static str,
    mode: &'static str,
    supported: bool,
    reason: &'static str,
}

const MODELS: &[Model] = &[
    Model {
        fixture: "heisenberg_chain_real",
        mode: "real",
        supported: true,
        reason: "",
    },
    Model {
        fixture: "hubbard_chain_real",
        mode: "real",
        supported: true,
        reason: "",
    },
    Model {
        fixture: "heisenberg_chain_cmp",
        mode: "cmp",
        supported: true,
        reason: "",
    },
    Model {
        fixture: "heisenberg_chain_fsz",
        mode: "fsz",
        supported: true,
        reason: "",
    },
    Model {
        fixture: "hubbard_chain_cmp",
        mode: "cmp",
        supported: false,
        reason: "no committed deterministic 50-step gate for this model",
    },
    Model {
        fixture: "hubbard_chain_fsz",
        mode: "fsz",
        supported: false,
        reason: "no committed deterministic 50-step gate for this model",
    },
    Model {
        fixture: "kondo_chain_real",
        mode: "real",
        supported: false,
        reason: "Kondo Hamiltonian parity is not implemented",
    },
    Model {
        fixture: "kondo_chain_cmp",
        mode: "cmp",
        supported: false,
        reason: "Kondo Hamiltonian parity is not implemented",
    },
    Model {
        fixture: "kondo_chain_stot1_cmp",
        mode: "cmp",
        supported: false,
        reason: "Kondo Hamiltonian parity is not implemented",
    },
    Model {
        fixture: "general_rbm_cmp",
        mode: "cmp",
        supported: false,
        reason: "GeneralRBM ctest parity is not yet enabled",
    },
    Model {
        fixture: "hubbard_tetragonal_real",
        mode: "real",
        supported: false,
        reason: "tetragonal fixture has no deterministic Rust gate",
    },
    Model {
        fixture: "hubbard_tetragonal_momentum_projection_real",
        mode: "real",
        supported: false,
        reason: "momentum-projection fixture has no deterministic Rust gate",
    },
    Model {
        fixture: "kondo_chain_fsz",
        mode: "fsz",
        supported: false,
        reason: "Kondo Hamiltonian parity is not implemented",
    },
];

fn root() -> PathBuf {
    std::env::var_os("JULIA_MVMC_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC")
        })
}

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
    let difference = (calculated - expected).abs();
    !(difference >= 3.0 * sigma && difference >= ABSOLUTE_FLOOR)
}

#[test]
fn rust_ctest_equivalent_selected_models() {
    let Ok(filter) = std::env::var("MVMC_RS_CTEST_MODELS") else {
        eprintln!("skipping Rust ctest-equivalent harness; set MVMC_RS_CTEST_MODELS");
        return;
    };
    let requested: Vec<_> = filter
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect();
    assert!(
        !requested.is_empty(),
        "MVMC_RS_CTEST_MODELS selected no models"
    );

    let root = root();
    for name in requested {
        let model = MODELS
            .iter()
            .find(|model| model.fixture == name)
            .unwrap_or_else(|| panic!("unknown ctest model {name:?}"));
        if !model.supported {
            eprintln!("ctest model {name} unsupported: {}", model.reason);
            continue;
        }
        let fixture = root.join("test/integration/reference").join(model.fixture);
        let namelist = fixture.join("inputs/namelist.def");
        let ref_mean = read_values(&fixture.join("ctest_ref/ref_mean.dat"));
        let ref_std = read_values(&fixture.join("ctest_ref/ref_std.dat"));
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
            seed: Some(1),
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
    }
}

#[test]
fn ctest_failure_requires_both_thresholds() {
    assert!(passes(1.0, 1.0 + 0.9e-8, 0.0));
    assert!(passes(1.0, 1.0 + 2.0e-8, 1.0e-8));
    assert!(!passes(1.0, 1.0 + 3.1e-8, 1.0e-8));
}
