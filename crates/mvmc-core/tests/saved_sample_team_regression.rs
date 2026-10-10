//! Numerical and exact sampling contracts for a QP team smaller than its pool.
//! Uses checked-in Expert inputs and fixed parameters; no external oracle runtime.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mvmc_core::{run, state::VmcOptimizationState, threading, RunConfig};
use num_complex::Complex64;
use serde_json::Value;
use sfmt19937::Sfmt19937Rng;

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

fn child(role: &str, threads: usize) -> Value {
    let output = fresh_scratch();
    let record_path = output.0.join("record.json");
    let mut process = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "saved_sample_numeric_child",
            "--nocapture",
        ])
        .env("SAVED_SAMPLE_ROLE", role)
        .env("SAVED_SAMPLE_RECORD", &record_path)
        .env("MVMC_RS_INNER_THREADS", threads.to_string())
        .env("MVMC_RS_INNER_THRESHOLD", "1")
        .env("MVMC_RS_SR_BACKEND", "c-order")
        .env("MVMC_RS_MEASURE_PF_BACKEND", "calc-m-all")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(status) = process.try_wait().unwrap() {
            assert!(status.success(), "{role}/{threads}: numerical child failed");
            break;
        }
        if Instant::now() >= deadline {
            process.kill().unwrap();
            process.wait().unwrap();
            panic!("{role}/{threads}: numerical child timed out");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    serde_json::from_slice(&std::fs::read(record_path).unwrap()).unwrap()
}

#[test]
fn partial_qp_team_preserves_opt_and_physcal_numerics_and_full_rng() {
    for role in ["opt", "physcal"] {
        let serial = child(role, 1);
        let parallel = child(role, 16);
        assert_eq!(
            serial["exact"], parallel["exact"],
            "{role}: RNG/discrete state changed"
        );
        let expected = serial["numerical"].as_array().unwrap();
        let actual = parallel["numerical"].as_array().unwrap();
        assert!(!expected.is_empty());
        assert_eq!(
            actual.len(),
            expected.len(),
            "{role}: numerical shape changed"
        );
        // The dispatch keeps each scalar calculation and ordered accumulation.
        // Use the repository's short SR / repeatability abs+rel envelope;
        // this independently exact RNG/control assertion cannot be relaxed by it.
        numerical_comparison::assert_values_close(
            actual.iter().map(|value| value.as_f64().unwrap()),
            expected.iter().map(|value| value.as_f64().unwrap()),
            1e-11,
            1e-11,
            role,
        );
    }
}

fn snapshot(state: &VmcOptimizationState, rng: &Sfmt19937Rng) -> Value {
    let (words, index) = rng.state_snapshot();
    let electron = &state.electron_config;
    let mut values = Vec::new();
    let mut add_complex = |value: Complex64| {
        values.extend([value.re, value.im]);
    };
    for value in [
        state.energy.wc,
        state.energy.etot,
        state.energy.etot2,
        state.energy.sztot,
        state.energy.sztot2,
    ] {
        add_complex(value);
    }
    for point in &state.opt_data {
        add_complex(point.energy);
        add_complex(point.energy_squared);
        for &parameter in &point.parameters {
            add_complex(parameter);
        }
    }
    if let Some(physical) = &state.phys_quantities {
        for &value in physical
            .phys_cis_ajs
            .iter()
            .chain(&physical.phys_cis_ajs_ckt_alt)
            .chain(&physical.phys_cis_ajs_ckt_alt_dc)
        {
            add_complex(value);
        }
    }
    values.extend(&state.slater_matrix.pf_m_real);
    values.extend(state.slater_matrix.inv_m_real.as_slice());
    values.extend(&state.sr_opt.sr_opt_oo_real);
    values.extend(&state.sr_opt.sr_opt_ho_real);
    serde_json::json!({
        "exact": {"rng_words": words.as_slice(), "rng_index": index,
            "rng_words_consumed": rng.words_consumed(),
            "ele_idx": electron.ele_idx, "ele_cfg": electron.ele_cfg,
            "ele_num": electron.ele_num, "ele_spn": electron.ele_spn,
            "ele_proj_cnt": electron.ele_proj_cnt, "counter": electron.counter},
        "numerical": values,
    })
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn fresh_scratch() -> Scratch {
    (0..)
        .find_map(|index| {
            let path = std::env::temp_dir().join(format!(
                "mvmc-saved-sample-team-{}-{index}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(Scratch(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("owned scratch: {error}"),
            }
        })
        .unwrap()
}

#[test]
#[ignore = "process-scoped numerical helper"]
fn saved_sample_numeric_child() {
    let role = std::env::var("SAVED_SAMPLE_ROLE").unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/hubbard_chain_dh_real");
    let scratch = fresh_scratch();
    let inputs = scratch.0.join("inputs");
    std::fs::create_dir(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    let path = inputs.join("modpara.def");
    let original = std::fs::read_to_string(&path).unwrap();
    let mut rewritten = String::new();
    for line in original.lines() {
        let key = line.split_whitespace().next().unwrap_or("");
        // Retain the fixture's eight Gaussian QP planes (NMPTrans=-1).
        // Only owned input files are edited for the short Opt prefix.
        let replacement = match (role.as_str(), key) {
            ("opt", "NVMCCalMode") => Some("NVMCCalMode 0"),
            ("opt", "NSROptItrStep") => Some("NSROptItrStep 3"),
            ("opt", "NSROptItrSmp") => Some("NSROptItrSmp 3"),
            ("opt", "NVMCSample") => Some("NVMCSample 60"),
            _ => None,
        };
        rewritten.push_str(replacement.unwrap_or(line));
        rewritten.push('\n');
    }
    std::fs::write(path, rewritten).unwrap();
    threading::install(|| {
        let record = if role == "physcal" {
            let preparation = run::prepare_phys_cal_from_namelist(
                inputs.join("namelist.def"),
                source.join("zqp_opt.dat"),
                "real",
                Some(1),
            )
            .unwrap();
            let result = run::vmc_phys_cal_to_dir(preparation, scratch.0.join("physcal")).unwrap();
            assert_eq!(result.iterations, 1);
            assert_eq!(result.state.slater_matrix.pf_m_real.len(), 8);
            snapshot(&result.state, &result.final_rng)
        } else {
            assert_eq!(role, "opt");
            let mut config = RunConfig::new(3, "real");
            config.seed = Some(1);
            config.nsmp = Some(3);
            config.output_dir = Some(scratch.0.join("opt"));
            let (summary, state, rng) = run::run_para_opt_from_namelist_observed(
                inputs.join("namelist.def"),
                config,
                &mvmc_core::SingleProcessReducer,
            )
            .unwrap();
            assert_eq!(summary.effective_nsteps, 3);
            assert_eq!(state.slater_matrix.pf_m_real.len(), 8);
            snapshot(&state, &rng)
        };
        std::fs::write(
            std::env::var("SAVED_SAMPLE_RECORD").unwrap(),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
    });
}
