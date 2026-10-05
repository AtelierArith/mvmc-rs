//! Issue #360: C-only OpenMP regions ported to the inner Rayon pool.
//!
//! Every region keeps a serial path for one worker and uses fixed-order
//! reductions otherwise, so results are required to be bit-identical for
//! 1/2/4 workers (threshold 1 forces the pooled path on these small inputs).
//! Each worker setting runs in its own child process because the pool and its
//! configuration are frozen per process. Runner jobs compare every output file
//! byte for byte, which includes the RNG-driven sample trajectory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use mvmc_core::sampling::updates::*;
use mvmc_core::threading::{inner_thread_config, start_observation};
use mvmc_core::{RunConfig, SlaterElmFlat};
use num_complex::Complex64 as C;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Child output: label -> value, one `R|label|value` line each.
type Records = BTreeMap<String, String>;

fn child(job: &str, workers: usize) -> Records {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "issue360_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("ISSUE360_CHILD", job)
        .env("MVMC_RS_INNER_THREADS", workers.to_string())
        .env("MVMC_RS_INNER_THRESHOLD", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{job} workers={workers}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut records = Records::new();
    for line in String::from_utf8(output.stdout).unwrap().lines() {
        if let Some(rest) = line.strip_prefix("R|") {
            let (label, value) = rest.split_once('|').unwrap();
            assert!(records.insert(label.to_owned(), value.to_owned()).is_none());
        }
    }
    assert!(!records.is_empty(), "{job}: child produced no records");
    records
}

/// Results must agree exactly; counters (`obs.*`) describe the dispatch.
fn assert_worker_invariant(job: &str, expect_pooled: &[&str]) {
    let serial = child(job, 1);
    for workers in [2usize, 4] {
        let pooled = child(job, workers);
        let strip = |records: &Records| -> Records {
            records
                .iter()
                .filter(|(k, _)| !k.starts_with("obs."))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        assert_eq!(
            strip(&serial),
            strip(&pooled),
            "{job}: workers={workers} differs from the serial result"
        );
        for counter in expect_pooled {
            let value: usize = pooled[&format!("obs.{counter}")].parse().unwrap();
            assert!(
                value > 0,
                "{job}: workers={workers} never used the pooled path ({counter})"
            );
        }
    }
    for key in serial.keys().filter(|k| k.starts_with("obs.parallel_")) {
        assert_eq!(
            serial[key], "0",
            "{job}: workers=1 must stay serial ({key})"
        );
    }
}

#[test]
fn qp_update_kernels_are_worker_invariant() {
    assert_worker_invariant("qp-kernels", &["parallel_qp_items"]);
}

#[test]
fn optimization_with_gutzwiller_jastrow_dh_is_worker_invariant() {
    assert_worker_invariant(
        "opt:hubbard_chain_dh_real:0",
        &[
            "parallel_qp_items",
            "parallel_entry_items",
            "parallel_region_items",
        ],
    );
}

#[test]
fn optimization_with_direct_store_cg_is_worker_invariant() {
    assert_worker_invariant(
        "opt:hubbard_chain_dh_real:1",
        &[
            "parallel_qp_items",
            "parallel_entry_items",
            "parallel_region_items",
        ],
    );
}

#[test]
fn optimization_with_rbm_and_opttrans_is_worker_invariant() {
    assert_worker_invariant(
        "opt:hubbard_chain_dh_rbm_opttrans:0",
        &[
            "parallel_qp_items",
            "parallel_entry_items",
            "parallel_region_items",
        ],
    );
}

#[test]
fn all_hamiltonian_terms_are_worker_invariant() {
    assert_worker_invariant(
        "opt:hubbard_all_terms_lanczos1:0",
        &[
            "parallel_qp_items",
            "parallel_entry_items",
            "parallel_term_items",
            "parallel_region_items",
        ],
    );
}

#[test]
fn interall_energy_is_worker_invariant_in_optimization() {
    assert_worker_invariant(
        "opt-interall:hubbard_all_terms_lanczos1:0",
        &[
            "parallel_qp_items",
            "parallel_entry_items",
            "parallel_term_items",
            "parallel_region_items",
        ],
    );
}

#[test]
fn interall_energy_is_worker_invariant_with_cg() {
    assert_worker_invariant(
        "opt-interall:hubbard_chain_dh_real:1",
        &[
            "parallel_qp_items",
            "parallel_entry_items",
            "parallel_region_items",
        ],
    );
}

#[test]
fn interall_energy_is_worker_invariant_in_physcal() {
    assert_worker_invariant(
        "physcal-interall:hubbard_chain_dh_real",
        &["parallel_qp_items", "parallel_region_items"],
    );
}

#[test]
fn lanczos_one_hamiltonian_terms_are_worker_invariant() {
    assert_worker_invariant(
        "physcal:hubbard_all_terms_lanczos1",
        &["parallel_qp_items", "parallel_region_items"],
    );
}

#[test]
fn lanczos_two_green_terms_are_worker_invariant() {
    assert_worker_invariant(
        "physcal:hubbard_all_terms_lanczos2",
        &["parallel_qp_items", "parallel_region_items"],
    );
}

#[ignore = "child process of the worker-invariance tests"]
#[test]
fn issue360_child() {
    let job = std::env::var("ISSUE360_CHILD").expect("child job");
    let workers = inner_thread_config().threads;
    let observer = start_observation();
    if job == "qp-kernels" {
        qp_kernels();
    } else if let Some(rest) = job.strip_prefix("opt:") {
        let (fixture, cg) = rest.split_once(':').unwrap();
        optimization(fixture, cg == "1", false);
    } else if let Some(rest) = job.strip_prefix("opt-interall:") {
        let (fixture, cg) = rest.split_once(':').unwrap();
        optimization(fixture, cg == "1", true);
    } else if let Some(fixture) = job.strip_prefix("physcal-interall:") {
        physcal(fixture, true);
    } else {
        physcal(job.strip_prefix("physcal:").expect("known job"), false);
    }
    let snapshot = observer.finish();
    record("obs.parallel_calls", snapshot.parallel_calls);
    record("obs.parallel_qp_items", snapshot.parallel_qp_items);
    record("obs.parallel_term_items", snapshot.parallel_term_items);
    record("obs.parallel_region_items", snapshot.parallel_region_items);
    record("obs.parallel_entry_items", snapshot.parallel_entry_items);
    if workers > 1 {
        assert!(snapshot.distinct_workers > 0);
    }
}

fn record(label: &str, value: impl std::fmt::Display) {
    println!("R|{label}|{value}");
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn optimization(fixture: &str, cg: bool, interall: bool) {
    let source = repo_root()
        .join("tests/fixtures/physcal_181/")
        .join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue360-{}-{fixture}-{}-{}",
        std::process::id(),
        u8::from(cg),
        u8::from(interall)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    if interall {
        add_interall(&inputs);
    }
    // Optimization mode with a few short SR steps; the sample loop stays serial.
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let overrides = [
        ("NVMCCalMode", "0"),
        ("NSROptItrStep", "3"),
        ("NSROptItrSmp", "3"),
        ("NVMCSample", "60"),
        ("NVMCWarmUp", "10"),
        ("NSRCG", if cg { "1" } else { "0" }),
        ("NStore", if cg { "1" } else { "0" }),
        ("NDataIdxStart", "1"),
        ("NDataQtySmp", "1"),
        ("NLanczosMode", "0"),
    ];
    let mut rewritten = String::new();
    let mut seen = Vec::new();
    for line in modpara.lines() {
        let key = line.split_whitespace().next().unwrap_or("");
        match overrides.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)) {
            Some((k, v)) => {
                seen.push(*k);
                rewritten.push_str(&format!("{k} {v}\n"));
            }
            None => {
                rewritten.push_str(line);
                rewritten.push('\n');
            }
        }
    }
    for (k, v) in overrides {
        if !seen.contains(&k) {
            rewritten.push_str(&format!("{k} {v}\n"));
        }
    }
    std::fs::write(inputs.join("modpara.def"), rewritten).unwrap();
    let output = root.join("output");
    let mut config = RunConfig::new(3, "real");
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    config.enable_opt_trans = Some(fixture.contains("opttrans"));
    config.mode = if fixture.contains("rbm") {
        "cmp"
    } else {
        "real"
    }
    .to_owned();
    let summary = mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config)
        .unwrap_or_else(|e| panic!("{fixture}: {e}"));
    let _ = summary;
    hash_outputs(&output);
}

// ---- direct QP kernels ------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 + 0.05
    }
    fn complex(&mut self) -> C {
        C::new(self.next(), self.next())
    }
}

fn bits_real(values: &[f64]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        hash = (hash ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn bits_complex(values: &[C]) -> String {
    let flat: Vec<f64> = values.iter().flat_map(|z| [z.re, z.im]).collect();
    bits_real(&flat)
}

fn qp_kernels() {
    const N_SITE: usize = 5;
    const N_ELEC: usize = 3;
    const N_SIZE: usize = 2 * N_ELEC;
    const N_QP: usize = 7;
    const STRIDE: usize = N_SIZE * N_SIZE + 1;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut slater_c = SlaterElmFlat::<C>::zeros(N_QP, N_SITE);
    let mut slater_r = SlaterElmFlat::<f64>::zeros(N_QP, N_SITE);
    for qp in 0..N_QP {
        for row in 0..2 * N_SITE {
            for col in 0..2 * N_SITE {
                slater_c.set(qp, row, col, rng.complex());
                slater_r.set(qp, row, col, rng.next());
            }
        }
    }
    let inv_c: Vec<C> = (0..N_QP * STRIDE).map(|_| rng.complex()).collect();
    let inv_r: Vec<f64> = (0..N_QP * STRIDE).map(|_| rng.next()).collect();
    let pf_c: Vec<C> = (0..N_QP).map(|_| rng.complex()).collect();
    let pf_r: Vec<f64> = (0..N_QP).map(|_| rng.next()).collect();
    let ele_idx: Vec<i64> = vec![0, 2, 4, 1, 2, 3];
    let ele_spn: Vec<i64> = vec![0, 1, 0, 1, 0, 1];
    let (qs, qe) = (1usize, N_QP); // exercise a non-zero start

    // One-electron ratios.
    for (label, spin) in [("a", 0u8), ("b", 1u8)] {
        let mut out = vec![C::default(); N_QP];
        calculate_new_pf_m2_complex_flat(
            1, spin, &mut out, &ele_idx, &slater_c, &inv_c, STRIDE, &pf_c, qs, qe, N_SITE, N_ELEC,
        );
        record(&format!("m2.cmp.{label}"), bits_complex(&out));
        let mut out = vec![0.0; N_QP];
        calculate_new_pf_m2_real_flat(
            1, spin, &mut out, &ele_idx, &slater_r, &inv_r, STRIDE, &pf_r, qs, qe, N_SITE, N_ELEC,
        );
        record(&format!("m2.real.{label}"), bits_real(&out));
        let mut out = vec![C::default(); N_QP];
        calculate_new_pf_m2_fsz_complex_flat(
            1, spin, &mut out, &ele_idx, &ele_spn, &slater_c, &inv_c, STRIDE, &pf_c, qs, qe,
            N_SITE, N_ELEC,
        );
        record(&format!("m2.fsz.cmp.{label}"), bits_complex(&out));
        let mut out = vec![0.0; N_QP];
        calculate_new_pf_m2_fsz_real_flat(
            1, spin, &mut out, &ele_idx, &ele_spn, &slater_r, &inv_r, STRIDE, &pf_r, qs, qe,
            N_SITE, N_ELEC,
        );
        record(&format!("m2.fsz.real.{label}"), bits_real(&out));
    }

    // Two-electron ratios, including the coincident-index fallback (ma == mb).
    for (label, ma, mb) in [("distinct", 0usize, 2usize), ("same", 1, 1)] {
        let mut out = vec![C::default(); N_QP];
        calculate_new_pf_m_two2_complex_flat(
            ma, 0, mb, 0, &mut out, &ele_idx, &slater_c, &inv_c, STRIDE, &pf_c, qs, qe, N_SITE,
            N_ELEC,
        );
        record(&format!("two2.cmp.{label}"), bits_complex(&out));
        for scalar in [false, true] {
            let mut out = vec![0.0; N_QP];
            if scalar {
                calculate_new_pf_m_two2_real_flat::<true>(
                    ma, 0, mb, 0, &mut out, &ele_idx, &slater_r, &inv_r, STRIDE, &pf_r, qs, qe,
                    N_SITE, N_ELEC,
                );
            } else {
                calculate_new_pf_m_two2_real_flat::<false>(
                    ma, 0, mb, 0, &mut out, &ele_idx, &slater_r, &inv_r, STRIDE, &pf_r, qs, qe,
                    N_SITE, N_ELEC,
                );
            }
            record(&format!("two2.real{scalar}.{label}"), bits_real(&out));
        }
        let mut out = vec![C::default(); N_QP];
        calculate_new_pf_m_two_fsz_complex_flat(
            ma, 0, mb, 1, &mut out, &ele_idx, &ele_spn, &slater_c, &inv_c, STRIDE, &pf_c, qs, qe,
            N_SITE, N_ELEC,
        );
        record(&format!("two.fsz.cmp.{label}"), bits_complex(&out));
        let mut out = vec![0.0; N_QP];
        calculate_new_pf_m_two_fsz_real_flat(
            ma, 0, mb, 1, &mut out, &ele_idx, &ele_spn, &slater_r, &inv_r, STRIDE, &pf_r, qs, qe,
            N_SITE, N_ELEC,
        );
        record(&format!("two.fsz.real.{label}"), bits_real(&out));
    }

    // Rank-one Pfaffian/inverse updates (mutating).
    let mut inv = inv_c.clone();
    let mut pf = pf_c.clone();
    update_m_all_complex_flat(
        1, 0, &ele_idx, &slater_c, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE, N_ELEC,
    );
    record("update.cmp.inv", bits_complex(&inv));
    record("update.cmp.pf", bits_complex(&pf));
    let mut inv = inv_r.clone();
    let mut pf = pf_r.clone();
    update_m_all_real_flat(
        1, 1, &ele_idx, &slater_r, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE, N_ELEC,
    );
    record("update.real.inv", bits_real(&inv));
    record("update.real.pf", bits_real(&pf));
    let mut inv = inv_c.clone();
    let mut pf = pf_c.clone();
    update_m_all_fsz_complex_flat(
        1, 0, &ele_idx, &ele_spn, &slater_c, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE, N_ELEC,
    );
    record("update.fsz.cmp.inv", bits_complex(&inv));
    record("update.fsz.cmp.pf", bits_complex(&pf));
    let mut inv = inv_r.clone();
    let mut pf = pf_r.clone();
    update_m_all_fsz_real_flat(
        1, 0, &ele_idx, &ele_spn, &slater_r, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE, N_ELEC,
    );
    record("update.fsz.real.inv", bits_real(&inv));
    record("update.fsz.real.pf", bits_real(&pf));

    // Two-electron updates (mutating).
    let mut inv = inv_c.clone();
    let mut pf = pf_c.clone();
    update_m_all_two_complex_flat(
        0, 0, 2, 0, 1, 3, &ele_idx, &slater_c, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE, N_ELEC,
    );
    record("update2.cmp.inv", bits_complex(&inv));
    record("update2.cmp.pf", bits_complex(&pf));
    let mut inv = inv_r.clone();
    let mut pf = pf_r.clone();
    update_m_all_two_real_flat(
        0, 0, 2, 0, 1, 3, &ele_idx, &slater_r, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE, N_ELEC,
    );
    record("update2.real.inv", bits_real(&inv));
    record("update2.real.pf", bits_real(&pf));
    let mut inv = inv_r.clone();
    let mut pf = pf_r.clone();
    update_m_all_two_fsz_real_flat(
        0, 0, 2, 1, 1, 3, &ele_idx, &ele_spn, &slater_r, &mut inv, STRIDE, &mut pf, qs, qe, N_SITE,
        N_ELEC,
    );
    record("update2.fsz.real.inv", bits_real(&inv));
    record("update2.fsz.real.pf", bits_real(&pf));
}

/// Append an InterAll definition (density-density and spin-pair-exchange terms that
/// conserve particle number and Sz) to the copied inputs.
fn add_interall(inputs: &Path) {
    std::fs::write(
        inputs.join("interall.def"),
        "======================\nNInterAll 3\n======================\n\
         ======================\n======================\n\
         0 0 0 0 1 1 1 1 0.30 0.0\n\
         0 0 1 0 1 1 0 1 0.10 0.0\n\
         2 0 3 0 3 1 2 1 -0.05 0.0\n",
    )
    .unwrap();
    let mut namelist = std::fs::read_to_string(inputs.join("namelist.def")).unwrap();
    if !namelist.ends_with('\n') {
        namelist.push('\n');
    }
    namelist.push_str("InterAll interall.def\n");
    std::fs::write(inputs.join("namelist.def"), namelist).unwrap();
}

fn hash_outputs(output: &Path) {
    let mut names: Vec<_> = std::fs::read_dir(output)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert!(!names.is_empty());
    for name in names {
        // Wall-clock content (timers, per-step time stamps) is not a result.
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        let bytes = std::fs::read(output.join(&name)).unwrap();
        record(&format!("file.{name}"), format!("{:016x}", fnv(&bytes)));
    }
}

/// Fixed-parameter PhysCal on a copied fixture (Lanczos 1/2 fixtures exercise the
/// Lanczos Hamiltonian and Green-function term regions).
fn physcal(fixture: &str, interall: bool) {
    let source = repo_root()
        .join("tests/fixtures/physcal_181/")
        .join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue360-phys-{}-{fixture}-{}",
        std::process::id(),
        u8::from(interall)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    if interall {
        add_interall(&inputs);
    }
    std::fs::copy(source.join("zqp_opt.dat"), root.join("zqp_opt.dat")).unwrap();
    let preparation = mvmc_core::prepare_phys_cal_from_namelist(
        inputs.join("namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap_or_else(|e| panic!("{fixture}: {e}"));
    let output = root.join("output");
    mvmc_core::vmc_phys_cal_to_dir(preparation, &output)
        .unwrap_or_else(|e| panic!("{fixture}: {e}"));
    hash_outputs(&output);
}
