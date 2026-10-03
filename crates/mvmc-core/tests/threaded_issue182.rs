//! Process-scoped worker matrix: no shared environment mutation between tests.
//! Kernel expectations are analytic or archived independent Julia values.
//! Runner comparisons are Rust worker invariance, not fresh C/Julia parity.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
mod support;

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::path::Path;
use std::process::Command;
use std::sync::Barrier;

use mvmc_core::threading::{
    inner_parallel_enabled, inner_thread_config, inner_worker_count, install, start_observation,
};
use mvmc_core::{
    calc_m_all_complex, calc_m_all_real, ExpertModeData, InvMColMajor, OptimizationOptions,
    SingleProcessReducer, SlaterElmFlat, SlaterMatrixData, ThreadedPfaPackWorkspace,
    VmcOptimizationState,
};
use mvmc_expert_parsers::utils::{parameter_init::init_parameter, qp_weight::init_qp_weight};
use num_complex::Complex64 as C;
use rayon::prelude::*;
use sfmt19937::Sfmt19937Rng;

const BOUND: f64 = 512.0 * f64::EPSILON;
const SIZES: [usize; 3] = [31, 32, 33];
const INDEPENDENT_PREFIX_CASES: [(&str, &str); 5] = [
    ("heisenberg_chain_real", "heisenberg_chain_real"),
    ("heisenberg_chain_cmp", "heisenberg_chain_cmp"),
    ("heisenberg_chain_fsz", "heisenberg_chain_fsz"),
    ("hubbard_chain_real", "hubbard_chain_real"),
    ("general_rbm_cmp_cg", "general_rbm_cmp"),
];
const INDEPENDENT_PHYSCAL_CASES: [(&str, &str); 4] = [
    ("heisenberg_chain_real", "real"),
    ("heisenberg_chain_cmp", "cmp"),
    ("heisenberg_chain_fsz", "fsz"),
    ("hubbard_chain_real", "real"),
];

fn discrete(label: &str, value: impl Debug) {
    println!("D|{label}|{value:?}");
}

fn numerical(label: &str, values: impl IntoIterator<Item = f64>) {
    let values = values
        .into_iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>();
    println!("N|{label}|{}", values.join(" "));
}

fn complex(label: &str, values: &[C]) {
    numerical(label, values.iter().flat_map(|z| [z.re, z.im]));
}

fn child(job: &str, workers: usize, threshold: usize) -> BTreeMap<String, (char, String)> {
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "issue182_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("ISSUE182_CHILD", job)
        .env("MVMC_RS_INNER_THREADS", workers.to_string())
        .env("MVMC_RS_INNER_THRESHOLD", threshold.to_string())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{job}: workers={workers}, threshold={threshold}\n{}\n{}",
        String::from_utf8_lossy(&result.stdout)
            .lines()
            .filter(|line| !line.starts_with("D|") && !line.starts_with("N|"))
            .map(|line| line.chars().take(240).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n"),
        String::from_utf8_lossy(&result.stderr)
    );
    if job.ends_with("-observed") || job.starts_with("independent-") {
        eprint!("{}", String::from_utf8_lossy(&result.stderr));
    }
    let mut records = BTreeMap::new();
    for line in String::from_utf8(result.stdout).unwrap().lines() {
        if line.starts_with("D|") || line.starts_with("N|") {
            let mut fields = line.splitn(3, '|');
            let kind = fields.next().unwrap().chars().next().unwrap();
            let label = fields.next().unwrap().to_owned();
            assert!(records
                .insert(label, (kind, fields.next().unwrap().to_owned()))
                .is_none());
        }
    }
    assert!(
        !records.is_empty(),
        "child must produce comparison evidence"
    );
    records
}

fn compare(expected: &BTreeMap<String, (char, String)>, actual: &BTreeMap<String, (char, String)>) {
    assert_eq!(
        expected.keys().collect::<Vec<_>>(),
        actual.keys().collect::<Vec<_>>()
    );
    let mut max_difference = 0.0_f64;
    for (label, (kind, value)) in actual {
        let (expected_kind, reference) = &expected[label];
        assert_eq!(kind, expected_kind, "{label}");
        if *kind == 'D' {
            assert_eq!(value, reference, "discrete contract: {label}");
        } else {
            let parse = |s: &str| {
                s.split_whitespace()
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>()
            };
            let values = parse(value);
            let references = parse(reference);
            assert_eq!(values.len(), references.len(), "{label}");
            for (&a, &e) in values.iter().zip(&references) {
                assert!(a.is_finite() && e.is_finite(), "{label}: nonfinite result");
                max_difference = max_difference.max((a - e).abs());
            }
            numerical_comparison::assert_values_close(values, references, BOUND, BOUND, label);
        }
    }
    eprintln!(
        "issue182 compared {} records; maximum numerical difference {max_difference:.5e}",
        actual.len()
    );
}

#[test]
fn qp_threshold_workers_scratch_and_lifecycle_match_independent_expectations() {
    let reference = child("kernels", 1, 32);
    for workers in [2, 4] {
        compare(&reference, &child("kernels", workers, 32));
    }
    // Configuration threshold override and worker capping are separate from
    // the production default of 32; both must preserve numerical results.
    compare(&reference, &child("kernels", 4, 1));
    compare(&reference, &child("kernels", 4, 34));
}

#[test]
fn same_worker_configuration_repeats_kernel_records() {
    // Acceptance is repeatability within one implementation/configuration.
    // Each child has independent state, fixed seeds, threshold 32, and BLAS
    // threads fixed to one; kernels exercise sizes 31/32/33 on both sides of
    // the threshold. Numeric records use the existing bounded comparison.
    for workers in [1, 2, 4] {
        let first = child("kernels", workers, 32);
        compare(&first, &child("kernels", workers, 32));
    }
}

#[test]
fn observation_classifies_actual_work_without_changing_kernel_records() {
    for workers in [1, 2, 4] {
        let baseline = child("kernels", workers, 32);
        let observed = child("kernels-observed", workers, 32);
        compare(&baseline, &observed);
    }
}

#[test]
#[ignore = "optional runner matrix; set MVMC_RS_THREADED_182=1; reads historical model inputs"]
fn runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal() {
    support::require_gate("threaded-issue182", "MVMC_RS_THREADED_182");
    require_runner_fixtures();
    let reference = child("runners", 1, 32);
    for workers in [1, 2, 4] {
        let baseline = if workers == 1 {
            reference.clone()
        } else {
            child("runners", workers, 32)
        };
        compare(&reference, &baseline);
        compare(&baseline, &child("runners-observed", workers, 32));
    }
}

#[test]
#[ignore = "optional independent SR prefix proof; set MVMC_RS_THREADED_182=1"]
fn independent_runner_prefixes_match_full_normalized_pre_sr_arrays() {
    support::require_gate("threaded-issue182-prefixes", "MVMC_RS_THREADED_182");
    require_runner_fixtures();
    let reference = child("independent-prefixes", 1, 32);
    for workers in [2, 4] {
        compare(&reference, &child("independent-prefixes", workers, 32));
    }
}

#[test]
#[ignore = "optional independent two-frame PhysCal reference; set MVMC_RS_THREADED_182=1"]
fn independent_physcal_workers_match_saved_rng_and_ordered_outputs() {
    support::require_gate("threaded-issue182-physcal", "MVMC_RS_THREADED_182");
    require_independent_physcal_fixtures();
    let reference = child("independent-physcal", 1, 32);
    for workers in [1, 2, 4] {
        let baseline = if workers == 1 {
            reference.clone()
        } else {
            child("independent-physcal", workers, 32)
        };
        compare(&reference, &baseline);
        compare(&baseline, &child("independent-physcal", workers, 32));
    }
}

#[test]
#[ignore = "optional synthetic transfer threshold activation; MVMC_RS_THREADED_182=1"]
fn transfer_site_executes_actual_jobs_at_threshold() {
    support::require_gate("threaded-issue182-transfer", "MVMC_RS_THREADED_182");
    require_runner_fixtures();
    let reference = child("transfer-site", 1, 32);
    for workers in [2, 4] {
        compare(&reference, &child("transfer-site", workers, 32));
    }
}

#[test]
#[ignore = "opt-in reviewed CG long baseline: 20 steps/window 20; MVMC_RS_THREADED_182=1"]
fn reviewed_cg_twenty_step_workers_match_and_repeat() {
    support::require_gate("threaded-issue182-long20", "MVMC_RS_THREADED_182");
    require_runner_fixtures();
    let root = runner_fixture_root().join("tests/fixtures/reviewed_cg_62b/canonical_general_rbm");
    for name in [
        "step-20/provenance.txt",
        "step-20/c-window-input.txt",
        "step-20-status.txt",
        "step-20-configs.txt",
        "step-20-rng.txt",
        "step-20-energy.txt",
        "step-20-parameters.txt",
    ] {
        let path = root.join(name);
        if !path.is_file() {
            support::missing_fixture("threaded-issue182-long20", path.display().to_string());
        }
    }
    let reference = child("reviewed-cg-long20", 1, 32);
    for workers in [1, 2, 4] {
        let first = if workers == 1 {
            reference.clone()
        } else {
            child("reviewed-cg-long20", workers, 32)
        };
        compare(&reference, &first);
        compare(&first, &child("reviewed-cg-long20", workers, 32));
    }
}

#[test]
#[ignore = "opt-in independent adapted C-real FSZ public fixture; MVMC_RS_THREADED_182=1"]
fn independent_real_fsz_workers_match_public_pre_sr_and_rng() {
    support::require_gate("threaded-issue182-real-fsz", "MVMC_RS_THREADED_182");
    require_independent_real_fsz_fixtures();
    let reference = child("independent-real-fsz", 1, 32);
    for workers in [1, 2, 4] {
        let baseline = if workers == 1 {
            reference.clone()
        } else {
            child("independent-real-fsz", workers, 32)
        };
        compare(&reference, &baseline);
        // Same input/seed and exact same worker/backend configuration in a
        // fresh process. Primitive RNG identity remains exact for the same
        // draw order/count. Any numerically dependent trajectory divergence
        // requires separate first-cause and decision-threshold evidence.
        compare(&baseline, &child("independent-real-fsz", workers, 32));
    }
}

fn require_runner_fixtures() {
    let repo = runner_fixture_root();
    let references = repo.join("tests/fixtures/ctest_model_prefixes");
    let mut required = vec![references.join("provenance.txt")];
    for model in [
        "heisenberg_chain_real",
        "heisenberg_chain_cmp",
        "heisenberg_chain_fsz",
        "hubbard_chain_real",
    ] {
        let root = repo.join(format!(
            "extern/Julia-mVMC/test/integration/reference/{model}"
        ));
        required.push(root.join("inputs/namelist.def"));
        required.push(root.join("physcal_ref/inputs/namelist.def"));
        required.push(root.join("physcal_ref/zqp_opt.dat"));
        for name in [
            "status.txt",
            "configs.txt",
            "rng.txt",
            "parameters.txt",
            "energy.txt",
            "sr_oo.txt",
            "sr_ho.txt",
        ] {
            required.push(references.join(model).join("step-1").join(name));
        }
    }
    for path in required {
        if !path.is_file() {
            support::missing_fixture("threaded-issue182", path.display().to_string());
        }
    }
    for (model, input_model) in INDEPENDENT_PREFIX_CASES {
        for name in [
            "status.txt",
            "configs.txt",
            "rng.txt",
            "parameters.txt",
            "energy.txt",
            "sr_oo.txt",
            "sr_ho.txt",
            "model-settings.txt",
        ] {
            let path = references.join(model).join("step-1").join(name);
            if !path.is_file() {
                support::missing_fixture("threaded-issue182", path.display().to_string());
            }
        }
        let input = repo.join(format!(
            "extern/Julia-mVMC/test/integration/reference/{input_model}/inputs/namelist.def"
        ));
        if !input.is_file() {
            support::missing_fixture("threaded-issue182", input.display().to_string());
        }
    }
    let reviewed = repo.join("tests/fixtures/reviewed_cg_62b/canonical_general_rbm");
    for name in [
        "step-1/provenance.txt",
        "step-1/c-window-input.txt",
        "step-1-status.txt",
        "step-1-configs.txt",
        "step-1-rng.txt",
        "step-1-energy.txt",
        "step-1-parameters.txt",
    ] {
        let path = reviewed.join(name);
        if !path.is_file() {
            support::missing_fixture("threaded-issue182-reviewed-cg", path.display().to_string());
        }
    }
}

fn runner_fixture_root() -> std::path::PathBuf {
    if let Ok(root) = std::env::var("MVMC_RS_THREADED_FIXTURE_ROOT") {
        support::require_gate("threaded-issue182", "MVMC_RS_THREADED_FIXTURE_ROOT");
        root.into()
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
}

#[test]
fn selected_runner_missing_fixture_root_fails_before_matrix_execution() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal",
            "--nocapture",
        ])
        .env("MVMC_RS_THREADED_182", "1")
        .env(
            "MVMC_RS_THREADED_FIXTURE_ROOT",
            std::env::temp_dir().join(format!("mvmc-issue182-absent-root-{}", std::process::id())),
        )
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("MissingFixture"));
}

#[test]
#[ignore = "internal process-scoped matrix worker; ISSUE182_CHILD required"]
fn issue182_child() {
    let job = std::env::var("ISSUE182_CHILD").expect("internal worker requires ISSUE182_CHILD");
    let config = inner_thread_config();
    for size in [0, 1, 2, 4, 31, 32, 33, 34] {
        let enabled = config.threads > 1 && size >= config.threshold;
        assert_eq!(inner_parallel_enabled(size), enabled);
        assert_eq!(
            inner_worker_count(size),
            if enabled { config.threads.min(size) } else { 1 }
        );
    }
    // Only this single-test child mutates its own environment, before starting
    // worker threads. Configuration and subsequent pool creation stay frozen.
    std::env::set_var("MVMC_RS_INNER_THREADS", "99");
    std::env::set_var("MVMC_RS_INNER_THRESHOLD", "99");
    assert_eq!(inner_thread_config(), config);
    assert_eq!(install(rayon::current_num_threads), config.threads);
    assert_eq!(
        install(|| install(rayon::current_num_threads)),
        config.threads
    );
    assert!(std::panic::catch_unwind(|| install(|| panic!("issue182 lifecycle probe"))).is_err());
    assert_eq!(install(rayon::current_num_threads), config.threads);
    install(|| {
        (0..config.threads).into_par_iter().for_each(|_| {
            assert!(std::thread::current()
                .name()
                .unwrap()
                .starts_with("mvmc-inner-"));
        })
    });
    if job == "kernels" || job == "kernels-observed" {
        scratch_isolation(config.threads);
        for size in SIZES {
            let observer = (job == "kernels-observed").then(start_observation);
            analytic_qp_ranges(size);
            archived_qp_ranges(size);
            independent_sr_and_copies(size);
            if let Some(observer) = observer {
                let snapshot = observer.finish();
                assert!(snapshot.executed_qp_items > 0);
                if inner_parallel_enabled(size) {
                    assert!(snapshot.parallel_qp_items > 0);
                    assert!(snapshot.parallel_calls > 0);
                    assert!(snapshot.worker_entries > 0);
                    assert!(snapshot.distinct_workers > 0);
                    assert!(snapshot.worker_ids.iter().all(|&id| id < config.threads));
                } else {
                    assert_eq!(snapshot.parallel_qp_items, 0);
                }
                eprintln!(
                    "actual kernel observation workers={} size={size}: {snapshot:?}",
                    config.threads
                );
            }
        }
        // Complex loops have even component counts. Parameter sizes 15/16/17
        // expose 30/32/34 components and 28/30/32 remaining rows, including
        // the mixed serial/parallel boundary at parameter size 16.
        for size in [15, 16, 17] {
            independent_sr_and_copies(size);
        }
    } else if job == "reviewed-cg-long20" {
        reviewed_cg_long20(&fresh_output_root());
    } else if job == "transfer-site" {
        transfer_site(&fresh_output_root());
    } else if job == "independent-prefixes" {
        independent_runner_prefixes(&fresh_output_root(), true);
    } else if job == "independent-physcal" {
        independent_physcal(&fresh_output_root());
    } else if job == "independent-real-fsz" {
        independent_real_fsz(&fresh_output_root());
    } else {
        assert!(matches!(job.as_str(), "runners" | "runners-observed"));
        let observer = (job == "runners-observed").then(start_observation);
        runner_matrix();
        if let Some(observer) = observer {
            let snapshot = observer.finish();
            assert!(snapshot.serial_qp_items > 0);
            assert!(snapshot.executed_term_items > 0);
            if config.threads > 1 {
                assert!(snapshot.parallel_qp_items > 0);
                assert!(snapshot.parallel_term_items > 0);
                assert!(
                    snapshot.distinct_workers > 0 && snapshot.distinct_workers <= config.threads
                );
            } else {
                assert_eq!(snapshot.parallel_calls, 0);
                assert_eq!(snapshot.worker_entries, 0);
            }
            eprintln!(
                "actual runner observation workers={}: {snapshot:?}",
                config.threads
            );
        }
    }
}

fn scratch_isolation(workers: usize) {
    let pool = ThreadedPfaPackWorkspace::new(4, 1);
    pool.ensure_capacity(workers);
    let barrier = Barrier::new(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers {
            let pool = &pool;
            let barrier = &barrier;
            handles.push(scope.spawn(move || {
                let mut scratch = pool.take();
                scratch.buf_m_real.fill(worker as f64 + 1.0);
                let pointer = scratch.buf_m_real.as_ptr() as usize;
                barrier.wait();
                assert!(scratch.buf_m_real.iter().all(|&v| v == worker as f64 + 1.0));
                pool.release(scratch);
                pointer
            }));
        }
        let mut pointers = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>();
        pointers.sort_unstable();
        pointers.dedup();
        assert_eq!(
            pointers.len(),
            workers,
            "simultaneously borrowed scratch must not alias"
        );
    });
    assert_eq!(
        pool.len(),
        workers,
        "all scratch returned after scoped joins"
    );
}

fn observed_qp_call(label: &str, size: usize, operation: impl FnOnce()) {
    if std::env::var("ISSUE182_CHILD").as_deref() != Ok("kernels-observed") {
        operation();
        return;
    }
    let observer = start_observation();
    operation();
    let snapshot = observer.finish();
    assert_eq!(snapshot.executed_qp_items, size, "{label} actual QP jobs");
    if inner_parallel_enabled(size) {
        assert_eq!(snapshot.parallel_qp_items, size);
        assert_eq!(snapshot.serial_qp_items, 0);
        assert_eq!(snapshot.worker_entries, size);
        assert!(!snapshot.worker_ids.is_empty());
        assert!(snapshot
            .worker_ids
            .iter()
            .all(|&id| id < inner_thread_config().threads));
    } else {
        assert_eq!(snapshot.serial_qp_items, size);
        assert_eq!(snapshot.parallel_qp_items, 0);
        assert_eq!(snapshot.worker_entries, 0);
    }
    eprintln!("actual site {label} QP_length={size}: {snapshot:?}");
}

fn analytic_qp_ranges(size: usize) {
    // Dense 4x4 skew A: Pf(A)=a*f-b*e+c*d. Its inverse follows
    // from six cofactors, independently of the production factorization.
    // Distinct QP coefficients expose misplaced range/chunk publication.
    let count = size + 2;
    let mut real = SlaterElmFlat::<f64>::zeros(count, 4);
    let mut cmp = SlaterElmFlat::<C>::zeros(count, 4);
    for qp in 0..count {
        let a = 1.0 + qp as f64 / 64.0;
        let b = 2.0 + qp as f64 / 32.0;
        for (i, j, z) in [
            (0, 1, C::new(a, 0.25)),
            (4, 5, C::new(b, -0.125)),
            (0, 4, C::new(0.5, 0.0)),
            (0, 5, C::new(0.25, 0.0)),
            (1, 4, C::new(-0.125, 0.0)),
            (1, 5, C::new(0.375, 0.0)),
        ] {
            real.set(qp, i, j, z.re);
            real.set(qp, j, i, -z.re);
            cmp.set(qp, i, j, z);
            cmp.set(qp, j, i, -z);
        }
    }
    let sentinel = 1234.0;
    let pool = ThreadedPfaPackWorkspace::new(4, 1);
    let mut ri = InvMColMajor::<f64>::zeros(count, 2);
    let mut ci = InvMColMajor::<C>::zeros(count, 2);
    ri.as_mut_slice().fill(sentinel);
    ci.as_mut_slice().fill(C::new(sentinel, 0.0));
    let mut rp = vec![sentinel; count];
    let mut cp = vec![C::new(sentinel, 0.0); count];
    for _ in 0..2 {
        observed_qp_call("calculate_m_all.jl:879 normal real QP", size, || {
            calc_m_all_real(
                &[0, 1, 0, 1],
                &real,
                &mut ri,
                &mut rp,
                1,
                size + 1,
                4,
                2,
                &pool,
            )
            .unwrap()
        });
        observed_qp_call("calculate_m_all.jl:440 normal complex QP", size, || {
            calc_m_all_complex(
                &[0, 1, 0, 1],
                &cmp,
                &mut ci,
                &mut cp,
                1,
                size + 1,
                4,
                2,
                &pool,
            )
            .unwrap()
        });
    }
    for qp in 1..size + 1 {
        let a = C::new(1.0 + qp as f64 / 64.0, 0.25);
        let b = C::new(2.0 + qp as f64 / 32.0, -0.125);
        let real_pf = a.re * b.re - 0.5 * 0.375 + 0.25 * (-0.125);
        let complex_pf = a * b - C::new(0.5 * 0.375 - 0.25 * (-0.125), 0.0);
        numerical_comparison::assert_close(rp[qp], real_pf, BOUND, BOUND, "analytic real pf");
        numerical_comparison::assert_values_close(
            [cp[qp].re, cp[qp].im],
            [complex_pf.re, complex_pf.im],
            BOUND,
            BOUND,
            "analytic complex pf",
        );
        for col in 0..4 {
            for row in 0..4 {
                let cofactor = match (row.min(col), row.max(col)) {
                    (0, 1) => b,
                    (0, 2) => C::new(-0.375, 0.0),
                    (0, 3) => C::new(-0.125, 0.0),
                    (1, 2) => C::new(0.25, 0.0),
                    (1, 3) => C::new(-0.5, 0.0),
                    (2, 3) => a,
                    _ => C::new(0.0, 0.0),
                };
                let sign = if row > col { -1.0 } else { 1.0 };
                let expected_r = sign * cofactor.re / real_pf;
                let expected_c = cofactor * sign / complex_pf;
                numerical_comparison::assert_close(
                    ri.get(qp, row, col),
                    expected_r,
                    BOUND,
                    BOUND,
                    "analytic real inverse",
                );
                let z = ci.get(qp, row, col);
                numerical_comparison::assert_values_close(
                    [z.re, z.im],
                    [expected_c.re, expected_c.im],
                    BOUND,
                    BOUND,
                    "analytic complex inverse",
                );
            }
        }
        assert_eq!(ri.pad_slot(qp), sentinel);
        assert_eq!(ci.pad_slot(qp), C::new(sentinel, 0.0));
    }
    for qp in [0, size + 1] {
        assert_eq!(rp[qp], sentinel);
        assert_eq!(cp[qp], C::new(sentinel, 0.0));
        assert!(ri.qp_matrix_slice(qp).iter().all(|&v| v == sentinel));
        assert!(ci
            .qp_matrix_slice(qp)
            .iter()
            .all(|&v| v == C::new(sentinel, 0.0)));
    }
    // FSZ is currently serial regardless of the worker setting. Verify its
    // spin-selected assembly and real-shadow publication, without pretending
    // the worker setting enables parallel FSZ computation.
    let mut fsz = SlaterMatrixData::zeros(count, 4, 2, false);
    fsz.slater_elm
        .as_mut_slice()
        .copy_from_slice(cmp.as_slice());
    fsz.inv_m.as_mut_slice().fill(C::new(sentinel, 0.0));
    fsz.inv_m_real.as_mut_slice().fill(sentinel);
    fsz.pf_m.fill(C::new(sentinel, 0.0));
    fsz.pf_m_real.fill(sentinel);
    mvmc_core::pfaffian::calc_m_all_fsz_real(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &mut fsz,
        1,
        size + 1,
        4,
        2,
        &pool,
    )
    .unwrap();
    numerical_comparison::assert_values_close(
        fsz.inv_m.as_slice().iter().flat_map(|z| [z.re, z.im]),
        ci.as_slice().iter().flat_map(|z| [z.re, z.im]),
        BOUND,
        BOUND,
        "analytic FSZ inverse",
    );
    numerical_comparison::assert_values_close(
        fsz.pf_m.iter().flat_map(|z| [z.re, z.im]),
        cp.iter().flat_map(|z| [z.re, z.im]),
        BOUND,
        BOUND,
        "analytic FSZ pf",
    );
    for (qp, expected_pf) in cp.iter().enumerate().skip(1).take(size) {
        numerical_comparison::assert_close(
            fsz.pf_m_real[qp],
            expected_pf.re,
            BOUND,
            BOUND,
            "FSZ real shadow",
        );
        numerical_comparison::assert_values_close(
            fsz.inv_m_real.qp_matrix_slice(qp).iter().copied(),
            ci.qp_matrix_slice(qp).iter().map(|z| z.re),
            BOUND,
            BOUND,
            "FSZ inverse real shadow",
        );
        assert_eq!(fsz.inv_m_real.pad_slot(qp), sentinel);
    }
    for qp in [0, size + 1] {
        assert_eq!(fsz.pf_m_real[qp], sentinel);
        assert!(fsz
            .inv_m_real
            .qp_matrix_slice(qp)
            .iter()
            .all(|&v| v == sentinel));
    }
    complex(&format!("kernel-{size}-fsz-pf"), &fsz.pf_m);
    numerical(
        &format!("kernel-{size}-fsz-real-shadow"),
        fsz.inv_m_real.as_slice().iter().copied(),
    );
    // A later-QP nonfinite Pfaffian must publish neither successful earlier
    // planes nor real shadows. Failure identity and sentinel preservation are
    // discrete contracts, not numerical approximations.
    fsz.slater_elm.set(size, 0, 1, C::new(f64::NAN, 0.0));
    fsz.slater_elm.set(size, 1, 0, C::new(f64::NAN, 0.0));
    fsz.pf_m.fill(C::new(sentinel, 0.0));
    fsz.pf_m_real.fill(sentinel);
    fsz.inv_m.as_mut_slice().fill(C::new(sentinel, 0.0));
    fsz.inv_m_real.as_mut_slice().fill(sentinel);
    let error = mvmc_core::pfaffian::calc_m_all_fsz_real(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &mut fsz,
        1,
        size + 1,
        4,
        2,
        &pool,
    )
    .unwrap_err();
    assert!(
        matches!(error, mvmc_core::pfaffian::CalcMAllError::NonFinitePfaffian { qp } if qp == size)
    );
    assert!(fsz.pf_m.iter().all(|&v| v == C::new(sentinel, 0.0)));
    assert!(fsz.pf_m_real.iter().all(|&v| v == sentinel));
    assert!(fsz
        .inv_m
        .as_slice()
        .iter()
        .all(|&v| v == C::new(sentinel, 0.0)));
    assert!(fsz.inv_m_real.as_slice().iter().all(|&v| v == sentinel));
    discrete(
        &format!("kernel-{size}-fsz-transaction"),
        format!("{error:?}"),
    );
    numerical(&format!("kernel-{size}-pf-real"), rp);
    complex(&format!("kernel-{size}-pf-complex"), &cp);
    numerical(
        &format!("kernel-{size}-inverse-real"),
        ri.as_slice().iter().copied(),
    );
    complex(&format!("kernel-{size}-inverse-complex"), ci.as_slice());
    let good_real_inverse = ri.as_slice().to_vec();
    let good_complex_inverse = ci.as_slice().to_vec();
    // QP 3 and the final QP belong to different chunks with 2/4 workers.
    // Indexed collection followed by serial error selection must consistently
    // report QP 3, irrespective of which worker finishes first.
    fsz.slater_elm.set(3, 0, 1, C::new(f64::NAN, 0.0));
    fsz.slater_elm.set(3, 1, 0, C::new(f64::NAN, 0.0));
    for repeat in 0..8 {
        let error = mvmc_core::pfaffian::calc_m_all_fsz_real(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &mut fsz,
            1,
            size + 1,
            4,
            2,
            &pool,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            mvmc_core::pfaffian::CalcMAllError::NonFinitePfaffian { qp: 3 }
        ));
        assert!(fsz.pf_m.iter().all(|&v| v == C::new(sentinel, 0.0)));
        assert!(fsz.pf_m_real.iter().all(|&v| v == sentinel));
        assert!(fsz
            .inv_m
            .as_slice()
            .iter()
            .all(|&v| v == C::new(sentinel, 0.0)));
        assert!(fsz.inv_m_real.as_slice().iter().all(|&v| v == sentinel));
        discrete(
            &format!("kernel-{size}-fsz-multi-error-{repeat}"),
            format!("{error:?}"),
        );
    }
    for qp in [3, size] {
        for i in 0..8 {
            for j in 0..8 {
                real.set(qp, i, j, 0.0);
                cmp.set(qp, i, j, C::new(0.0, 0.0));
            }
        }
    }
    for repeat in 0..8 {
        ri.as_mut_slice().fill(sentinel);
        ci.as_mut_slice().fill(C::new(sentinel, 0.0));
        cp.fill(C::new(sentinel, 0.0));
        let error = calc_m_all_real(
            &[0, 1, 0, 1],
            &real,
            &mut ri,
            &mut vec![0.0; count],
            1,
            size + 1,
            4,
            2,
            &pool,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            mvmc_core::pfaffian::CalcMAllError::AllZero { qp: 3 }
        ));
        discrete(
            &format!("kernel-{size}-real-multi-error-{repeat}"),
            format!("{error:?}"),
        );
        let error = calc_m_all_complex(
            &[0, 1, 0, 1],
            &cmp,
            &mut ci,
            &mut cp,
            1,
            size + 1,
            4,
            2,
            &pool,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            mvmc_core::pfaffian::CalcMAllError::AllZero { qp: 3 }
        ));
        discrete(
            &format!("kernel-{size}-complex-multi-error-{repeat}"),
            format!("{error:?}"),
        );
        // C matrix.c writes each assembled inverse before checking failure,
        // and publishes PfM after successful factorization. Preserve the
        // existing Rust serial boundary; do not invent normal full rollback.
        if !inner_parallel_enabled(size) {
            let stride = ri.as_slice().len() / count;
            for qp in 1..3 {
                numerical_comparison::assert_values_close(
                    ri.qp_matrix_slice(qp).iter().copied(),
                    good_real_inverse[qp * stride..qp * stride + 16]
                        .iter()
                        .copied(),
                    BOUND,
                    BOUND,
                    "serial real successful prefix",
                );
                numerical_comparison::assert_values_close(
                    ci.qp_matrix_slice(qp).iter().flat_map(|z| [z.re, z.im]),
                    good_complex_inverse[qp * stride..qp * stride + 16]
                        .iter()
                        .flat_map(|z| [z.re, z.im]),
                    BOUND,
                    BOUND,
                    "serial complex successful prefix",
                );
            }
            assert!(ri.qp_matrix_slice(3).iter().all(|&v| v == 0.0));
            assert!(ci.qp_matrix_slice(3).iter().all(|&v| v == C::new(0.0, 0.0)));
        } else {
            // This is the pre-existing Rust worker-local publication boundary,
            // not a newly claimed C/OpenMP rollback guarantee.
            assert!(ri.as_slice().iter().all(|&v| v == sentinel));
            assert!(ci.as_slice().iter().all(|&v| v == C::new(sentinel, 0.0)));
        }
        assert!(ri.qp_matrix_slice(4).iter().all(|&v| v == sentinel));
        assert!(ci
            .qp_matrix_slice(4)
            .iter()
            .all(|&v| v == C::new(sentinel, 0.0)));
        assert_eq!(cp[3], C::new(sentinel, 0.0));
    }
    let capacity = pool.len();
    let error = calc_m_all_real(
        &[99, 1, 0, 1],
        &real,
        &mut ri,
        &mut [0.0; 35][..count],
        1,
        size + 1,
        4,
        2,
        &pool,
    )
    .unwrap_err();
    discrete(&format!("kernel-{size}-invalid-site"), format!("{error:?}"));
    assert_eq!(pool.len(), capacity, "error paths return borrowed scratch");
}

fn observed_entry_call(label: &str, lengths: &[usize], operation: impl FnOnce()) {
    if std::env::var("ISSUE182_CHILD").as_deref() != Ok("kernels-observed") {
        operation();
        return;
    }
    let observer = start_observation();
    operation();
    let snapshot = observer.finish();
    let parallel: usize = lengths
        .iter()
        .copied()
        .filter(|&n| inner_parallel_enabled(n))
        .sum();
    let serial: usize = lengths.iter().sum::<usize>() - parallel;
    assert_eq!(
        snapshot.parallel_entry_items, parallel,
        "{label} actual parallel jobs"
    );
    assert_eq!(
        snapshot.serial_entry_items, serial,
        "{label} actual serial jobs"
    );
    assert_eq!(
        snapshot.executed_qp_items, 0,
        "{label} must not execute QP jobs"
    );
    assert_eq!(
        snapshot.executed_term_items, 0,
        "{label} must not execute transfer jobs"
    );
    if parallel > 0 {
        assert_eq!(
            snapshot.worker_entries, parallel,
            "{label} actual worker entries"
        );
        assert!(!snapshot.worker_ids.is_empty());
        assert!(snapshot
            .worker_ids
            .iter()
            .all(|&id| id < inner_thread_config().threads));
    } else {
        assert_eq!(snapshot.worker_entries, 0);
    }
    eprintln!("actual site {label} work_lengths={lengths:?}: {snapshot:?}");
}

fn independent_sr_and_copies(size: usize) {
    use mvmc_core::observables::{
        calculate_oo, calculate_oo_real, calculate_oo_store, calculate_oo_store_real,
        finalize_oo_store, finalize_oo_store_real, StoreFinalization,
    };
    let real = (0..size)
        .map(|i| (i as f64 - 9.0) / 16.0)
        .collect::<Vec<_>>();
    let mut copied = vec![C::new(99.0, 99.0); size + 2];
    observed_entry_call("threading.jl:66 real-to-complex", &[size], || {
        mvmc_core::threading::copy_real_to_complex(&mut copied, &real)
    });
    for i in 0..size {
        assert_eq!(copied[i], C::new(real[i], 0.0));
    }
    assert_eq!(&copied[size..], &[C::new(99.0, 99.0); 2]);
    let mut back = vec![99.0; size + 2];
    observed_entry_call("threading.jl:86 complex-to-real", &[size], || {
        mvmc_core::threading::copy_complex_realpart(&mut back, &copied[..size])
    });
    assert_eq!(&back[..size], real.as_slice());
    assert_eq!(&back[size..], &[99.0; 2]);
    let mut empty = [];
    mvmc_core::threading::copy_real_to_complex(&mut empty, &real);
    numerical(&format!("copies-{size}"), back);

    // Dyadic coefficients and weights make the short scalar expectation
    // independently representable; no Rust kernel generates this oracle.
    let mut oo = vec![0.0; size * (size + 2)];
    let mut ho = vec![0.0; size];
    let mut stored = vec![77.0; size * 4];
    for sample in 0..3 {
        observed_entry_call(
            "vmc_main_cal.jl:3462/3481 real OO/HO",
            &[size, size],
            || calculate_oo_real(&mut oo, &mut ho, &real, 4.0, 0.5, size),
        );
        let mut ignored_ho = vec![0.0; size];
        observed_entry_call("vmc_main_cal.jl:3555 real store/HO", &[size], || {
            calculate_oo_store_real(
                &mut ignored_ho,
                &mut stored,
                &real,
                4.0,
                0.5,
                sample + 1,
                size,
            )
        });
    }
    for j in 0..size {
        for i in 0..size {
            numerical_comparison::assert_close(
                oo[i + j * size],
                12.0 * real[i] * real[j],
                BOUND,
                BOUND,
                "analytic real OO",
            );
        }
    }
    for i in 0..size {
        numerical_comparison::assert_close(ho[i], 6.0 * real[i], BOUND, BOUND, "analytic real HO");
    }
    assert_eq!(&stored[..size], vec![77.0; size].as_slice());
    for diagonal_only in [false, true] {
        let mut final_oo = vec![88.0; size * (size + 2)];
        let diagonal_lengths = [size];
        observed_entry_call(
            if diagonal_only {
                "vmc_main_cal.jl:3671 real CG finalizer"
            } else {
                "real full Gram BLAS backend (not Julia @threads site)"
            },
            if diagonal_only {
                &diagonal_lengths
            } else {
                &[]
            },
            || {
                finalize_oo_store_real(
                    &mut final_oo,
                    &stored,
                    size,
                    3,
                    StoreFinalization {
                        sample_start: 1,
                        diagonal_only,
                    },
                )
            },
        );
        if diagonal_only {
            for i in 0..size {
                numerical_comparison::assert_close(
                    final_oo[i],
                    6.0 * real[i],
                    BOUND,
                    BOUND,
                    "analytic CG mean",
                );
                numerical_comparison::assert_close(
                    final_oo[size + i],
                    12.0 * real[i] * real[i],
                    BOUND,
                    BOUND,
                    "analytic CG diagonal",
                );
            }
            assert!(final_oo[2 * size..].iter().all(|&v| v == 88.0));
        } else {
            numerical_comparison::assert_values_close(
                final_oo[..size * size].iter().copied(),
                oo[..size * size].iter().copied(),
                BOUND,
                BOUND,
                "analytic real Gram",
            );
            assert!(final_oo[size * size..].iter().all(|&v| v == 88.0));
        }
        numerical(&format!("sr-{size}-real-final-{diagonal_only}"), final_oo);
    }
    numerical(&format!("sr-{size}-real-oo"), oo);
    numerical(&format!("sr-{size}-real-ho"), ho);
    let width = 2 * size;
    let values = (0..width)
        .map(|i| C::new((i as f64 - 9.0) / 16.0, (i % 3) as f64 / 8.0))
        .collect::<Vec<_>>();
    let mut coo = vec![C::new(0.0, 0.0); width * (width + 2)];
    let mut cho = vec![C::new(0.0, 0.0); width];
    let mut cstore = vec![C::new(77.0, 0.0); width * 4];
    for sample in 0..3 {
        observed_entry_call(
            "vmc_main_cal.jl:3406/3423 complex OO/HO",
            &[width, width - 2],
            || calculate_oo(&mut coo, &mut cho, &values, 4.0, C::new(0.5, -0.25), size),
        );
        let mut ignored_ho = vec![C::new(0.0, 0.0); width];
        observed_entry_call("vmc_main_cal.jl:3516 complex store/HO", &[width], || {
            calculate_oo_store(
                &mut ignored_ho,
                &mut cstore,
                &values,
                4.0,
                C::new(0.5, -0.25),
                sample + 1,
                size,
            )
        });
    }
    for i in 2..width {
        for j in 0..width {
            let expected = values[j] * values[i].conj() * 12.0;
            let actual = coo[i * width + j];
            numerical_comparison::assert_values_close(
                [actual.re, actual.im],
                [expected.re, expected.im],
                BOUND,
                BOUND,
                "analytic complex OO",
            );
        }
    }
    for i in 0..width {
        let expected = C::new(6.0, -3.0) * values[i];
        numerical_comparison::assert_values_close(
            [cho[i].re, cho[i].im],
            [expected.re, expected.im],
            BOUND,
            BOUND,
            "analytic complex HO",
        );
    }
    for diagonal_only in [false, true] {
        let mut final_oo = vec![C::new(88.0, 0.0); width * (width + 2)];
        observed_entry_call(
            if diagonal_only {
                "vmc_main_cal.jl:3590 complex CG finalizer"
            } else {
                "vmc_main_cal.jl:3630 complex full Gram"
            },
            &[width],
            || {
                finalize_oo_store(
                    &mut final_oo,
                    &cstore,
                    size,
                    3,
                    StoreFinalization {
                        sample_start: 1,
                        diagonal_only,
                    },
                )
            },
        );
        let active = if diagonal_only {
            2 * width
        } else {
            width * width
        };
        for index in 0..active {
            let expected = if diagonal_only {
                if index < width {
                    values[index] * 6.0
                } else {
                    C::new(values[index - width].norm_sqr() * 12.0, 0.0)
                }
            } else {
                values[index / width] * values[index % width].conj() * 12.0
            };
            let z = final_oo[index];
            numerical_comparison::assert_values_close(
                [z.re, z.im],
                [expected.re, expected.im],
                BOUND,
                BOUND,
                "analytic complex Gram/CG",
            );
        }
        assert!(final_oo[active..].iter().all(|&v| v == C::new(88.0, 0.0)));
        complex(
            &format!("sr-{size}-complex-final-{diagonal_only}"),
            &final_oo,
        );
    }
    complex(&format!("sr-{size}-complex-oo"), &coo);
    complex(&format!("sr-{size}-complex-ho"), &cho);
}

fn fixture_values(kind: &str, component: &str) -> Vec<Vec<f64>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../tests/fixtures/calc_m_all/{kind}_ns4_ne2_nqp3_seed1234.{component}.txt"
    ));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            line.split_whitespace()
                .map(str::parse::<f64>)
                .collect::<Result<Vec<_>, _>>()
                .ok()
        })
        .collect()
}

fn archived_qp_ranges(size: usize) {
    for kind in ["real", "complex"] {
        let slater = fixture_values(kind, "slater");
        let pf = fixture_values(kind, "pfaffian");
        let inverse = fixture_values(kind, "inverse");
        let indices = fixture_values(kind, "ele_idx")
            .iter()
            .map(|v| v[0] as i64)
            .collect::<Vec<_>>();
        assert_eq!(indices.len(), 4);
        assert_eq!(slater.len(), 3 * 64);
        assert_eq!(inverse.len(), 3 * 16);
        let mut sr = SlaterElmFlat::<f64>::zeros(size, 4);
        let mut sc = SlaterElmFlat::<C>::zeros(size, 4);
        for qp in 0..size {
            for row in 0..8 {
                for col in 0..8 {
                    let z = &slater[(qp % 3) * 64 + row * 8 + col];
                    sr.set(qp, row, col, z[0]);
                    sc.set(qp, row, col, C::new(z[0], *z.get(1).unwrap_or(&0.0)));
                }
            }
        }
        let pool = ThreadedPfaPackWorkspace::new(4, 1);
        let mut ri = InvMColMajor::<f64>::zeros(size, 2);
        let mut ci = InvMColMajor::<C>::zeros(size, 2);
        let mut rp = vec![0.0; size];
        let mut cp = vec![C::new(0.0, 0.0); size];
        if kind == "real" {
            calc_m_all_real(&indices, &sr, &mut ri, &mut rp, 0, size, 4, 2, &pool).unwrap();
        } else {
            calc_m_all_complex(&indices, &sc, &mut ci, &mut cp, 0, size, 4, 2, &pool).unwrap();
        }
        for qp in 0..size {
            let actual = if kind == "real" {
                vec![rp[qp]]
            } else {
                vec![cp[qp].re, cp[qp].im]
            };
            numerical_comparison::assert_values_close(
                actual,
                pf[qp % 3].iter().copied(),
                BOUND,
                BOUND,
                "archived Julia pf",
            );
            for element in 0..16 {
                let actual = if kind == "real" {
                    vec![ri.qp_matrix_slice(qp)[element]]
                } else {
                    let z = ci.qp_matrix_slice(qp)[element];
                    vec![z.re, z.im]
                };
                numerical_comparison::assert_values_close(
                    actual,
                    inverse[(qp % 3) * 16 + element].iter().copied(),
                    BOUND,
                    BOUND,
                    "archived Julia inverse",
                );
            }
        }
        discrete(
            &format!("independent-{kind}-{size}"),
            "archived planes verified",
        );
    }
}

fn parameters(data: &ExpertModeData) -> Vec<C> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_params.iter().copied())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn mapped_parameters(data: &ExpertModeData) -> Vec<C> {
    let mut rbm = Vec::new();
    // The read-only visitor enumerates declared coefficients, not spatial
    // mappings. Match the oracle's nine mapped sections without mutating data.
    let mut mapped_data = data.clone();
    mapped_data.visit_rbm_terms_mut(|_, term| rbm.push(term.value()));
    data.gutzwiller_terms
        .iter()
        .map(|term| term.value)
        .chain(data.jastrow_terms.iter().map(|term| term.value))
        .chain(data.doublon_holon_2site_params.iter().copied())
        .chain(data.doublon_holon_4site_params.iter().copied())
        .chain(rbm)
        .chain(
            data.orbital_terms
                .iter()
                .map(|term| data.slater_params[term.idx as usize]),
        )
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn emit_state(label: &str, data: &ExpertModeData, state: &VmcOptimizationState) {
    discrete(&format!("{label}-configuration"), &state.electron_config);
    discrete(&format!("{label}-flags"), &data.optimization_flags);
    complex(&format!("{label}-parameters"), &parameters(data));
    complex(
        &format!("{label}-energy"),
        &[
            state.energy.wc,
            state.energy.etot,
            state.energy.etot2,
            state.energy.sztot,
            state.energy.sztot2,
        ],
    );
    complex(&format!("{label}-pf"), &state.slater_matrix.pf_m);
    complex(
        &format!("{label}-inverse"),
        state.slater_matrix.inv_m.as_slice(),
    );
    numerical(
        &format!("{label}-inverse-real"),
        state.slater_matrix.inv_m_real.as_slice().iter().copied(),
    );
    numerical(
        &format!("{label}-pf-real"),
        state.slater_matrix.pf_m_real.iter().copied(),
    );
    complex(&format!("{label}-oo"), &state.sr_opt.sr_opt_oo);
    numerical(
        &format!("{label}-oo-real"),
        state.sr_opt.sr_opt_oo_real.iter().copied(),
    );
    complex(&format!("{label}-ho"), &state.sr_opt.sr_opt_ho);
    numerical(
        &format!("{label}-ho-real"),
        state.sr_opt.sr_opt_ho_real.iter().copied(),
    );
    complex(&format!("{label}-store"), &state.sr_opt.sr_opt_o_store);
    numerical(
        &format!("{label}-store-real"),
        state.sr_opt.sr_opt_o_store_real.iter().copied(),
    );
    if let Some(phys) = &state.phys_quantities {
        complex(&format!("{label}-onebody"), &phys.phys_cis_ajs);
        complex(&format!("{label}-twobody"), &phys.phys_cis_ajs_ckt_alt);
    }
}

fn fresh_output_root() -> std::path::PathBuf {
    // Retain isolated outputs for inspection; never overwrite fixture or shared
    // working-directory zvo files (including another agent's long-run output).
    let output_root = loop {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("mvmc-issue182-{}-{id}", std::process::id()));
        match std::fs::create_dir(&path) {
            Ok(()) => break path,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("create fresh output {}: {e}", path.display()),
        }
    };
    eprintln!("issue182 owned outputs: {}", output_root.display());
    output_root
}

fn runner_matrix() {
    let output_root = fresh_output_root();
    independent_runner_prefixes(&output_root, false);
    for (model, real_fsz) in [
        ("heisenberg_chain_real", false),
        ("heisenberg_chain_cmp", false),
        ("heisenberg_chain_fsz", false),
        ("heisenberg_chain_fsz", true),
        ("hubbard_chain_real", false),
    ] {
        let root = runner_fixture_root().join("extern/Julia-mVMC");
        for size in SIZES {
            for (store, cg) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let mut data = mvmc_expert_parsers::parse_expert_mode_files(root.join(format!(
                    "test/integration/reference/{model}/inputs/namelist.def"
                )))
                .unwrap();
                assert!(
                    data.inter_all_terms.is_empty(),
                    "InterAll is owned separately"
                );
                threshold_qp_input(&mut data, size);
                if model == "hubbard_chain_real" {
                    threshold_transfer_input(&mut data, size);
                }
                data.modpara.nvmc_warmup = 10;
                data.modpara.nvmc_sample = 200;
                data.modpara.nvmc_interval = 1;
                data.modpara.nsr_opt_itr_step = 2;
                data.modpara.nsr_opt_itr_smp = 2;
                data.modpara.nstore_o = store;
                data.modpara.nsrcg = cg;
                if real_fsz {
                    data.complex_flags = vec![0];
                }
                let variant = if real_fsz { "real-fsz" } else { model };
                let label = format!("synthetic-{variant}-{size}-store{store}-cg{cg}");
                let output = output_root.join(&label);
                std::fs::create_dir(&output).unwrap();
                let mut rng = Sfmt19937Rng::new(1);
                init_parameter(&mut data, &mut rng);
                if real_fsz {
                    for value in &mut data.slater_params {
                        value.im = 0.0;
                    }
                }
                mvmc_core::sync::sync_modified_parameter(&mut data, &SingleProcessReducer);
                init_qp_weight(&mut data);
                discrete(&format!("{label}-initial-rng"), &rng);
                let mut state = VmcOptimizationState::zeros(
                    data.modpara.nsite as usize,
                    data.modpara.nelec as usize,
                    data.projection_layout().n_proj,
                    data.count_variational_parameters(),
                    size,
                    200,
                    mvmc_core::get_all_complex_flag(&data),
                    data.i_flg_orbital_general != 0,
                );
                let mut callback = |step, data: &mut ExpertModeData, energy: C, status| {
                    discrete(&format!("{label}-step{step}-status"), status);
                    complex(&format!("{label}-step{step}-parameters"), &parameters(data));
                    complex(&format!("{label}-step{step}-energy"), &[energy]);
                    Ok(())
                };
                mvmc_core::vmc_para_opt(
                    &mut data,
                    &mut state,
                    &mut rng,
                    Some(&output),
                    &SingleProcessReducer,
                    OptimizationOptions {
                        callback: Some(&mut callback),
                        skip_sr: false,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: output {}: {e}", output.display()));
                discrete(&format!("{label}-final-rng"), &rng);
                emit_state(&label, &data, &state);
            }
            let fixture = root.join(format!("test/integration/reference/{model}/physcal_ref"));
            let mode = if model.ends_with("real") {
                "real"
            } else if model.ends_with("fsz") {
                "fsz"
            } else {
                "cmp"
            };
            let mut prepared = mvmc_core::prepare_phys_cal_from_namelist(
                fixture.join("inputs/namelist.def"),
                fixture.join("zqp_opt.dat"),
                mode,
                Some(1),
            )
            .unwrap();
            threshold_qp_input(&mut prepared.data, size);
            if model == "hubbard_chain_real" {
                threshold_transfer_input(&mut prepared.data, size);
            }
            prepared.data.modpara.nvmc_warmup = 2;
            prepared.data.modpara.nvmc_sample = 4;
            prepared.data.modpara.n_data_qty_smp = 2;
            if real_fsz {
                prepared.data.complex_flags = vec![0];
                for value in &mut prepared.data.slater_params {
                    value.im = 0.0;
                }
            }
            let variant = if real_fsz { "real-fsz" } else { model };
            let label = format!("synthetic-{variant}-{size}-physcal");
            discrete(&format!("{label}-initial-rng"), &prepared.rng);
            let output = output_root.join(&label);
            let result = mvmc_core::vmc_phys_cal_to_dir(prepared, &output)
                .unwrap_or_else(|e| panic!("{label}: output {}: {e}", output.display()));
            discrete(&format!("{label}-final-rng"), &result.final_rng);
            discrete(&format!("{label}-iterations"), result.iterations);
            emit_state(&label, &result.data, &result.state);
        }
    }
}

fn threshold_qp_input(data: &mut ExpertModeData, size: usize) {
    if data.i_flg_orbital_general != 0 {
        // Exercise FSZ QP workers without replacing the reference's disabled
        // spin quadrature with a different physical projection. Repeated
        // translation maps with split weights represent the same single sector.
        assert_eq!(data.modpara.nmp_trans.unsigned_abs(), 1);
        let mut entry = data.qp_trans_entries[0].clone();
        let weight = data.para_qp_trans[0] / size as f64;
        entry.weight = weight;
        data.qp_trans_entries = vec![entry; size];
        data.para_qp_trans = vec![weight; size];
        data.n_qp_trans = size as i64;
        data.modpara.nmp_trans = data.modpara.nmp_trans.signum() * size as i64;
        data.modpara.nsp_gauss_leg = 1;
    } else {
        data.modpara.nmp_trans = data.modpara.nmp_trans.signum();
        data.modpara.nsp_gauss_leg = size as i64;
    }
}

fn threshold_transfer_input(data: &mut ExpertModeData, size: usize) {
    // Split only the last hopping coefficient into repeated equal entries.
    // This retains the Hamiltonian while exercising work lengths 31/32/33;
    // the energy reduction still visits entries in their declared order.
    assert!(!data.transfer_terms.is_empty() && data.transfer_terms.len() <= size);
    let mut last = data.transfer_terms.pop().unwrap();
    let copies = size - data.transfer_terms.len();
    last.value /= copies as f64;
    data.transfer_terms
        .extend(std::iter::repeat_n(last, copies));
    assert_eq!(data.transfer_terms.len(), size);
}

fn transfer_site(output_root: &Path) {
    // Synthetic kernel threshold operands, not a new public-input oracle:
    // split the final coefficient into repeated terms preserving Hamiltonian.
    let input = runner_fixture_root().join(
        "extern/Julia-mVMC/test/integration/reference/hubbard_chain_real/inputs/namelist.def",
    );
    for size in SIZES {
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(&input).unwrap();
        assert!(!mvmc_core::get_all_complex_flag(&data));
        threshold_transfer_input(&mut data, size);
        data.modpara.nvmc_sample = 4;
        data.modpara.nsr_opt_itr_step = 1;
        data.modpara.nsr_opt_itr_smp = 1;
        data.modpara.nstore_o = 0;
        data.modpara.nsrcg = 0;
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng);
        mvmc_core::sync::sync_modified_parameter(&mut data, &SingleProcessReducer);
        init_qp_weight(&mut data);
        let nqp = data.qp_weights.as_ref().unwrap().qp_full_weight.len();
        let mut state = VmcOptimizationState::zeros(
            data.modpara.nsite as usize,
            data.modpara.nelec as usize,
            data.projection_layout().n_proj,
            data.count_variational_parameters(),
            nqp,
            4,
            false,
            false,
        );
        let output = output_root.join(format!("synthetic-transfer-{size}"));
        std::fs::create_dir(&output).unwrap();
        let observer = start_observation();
        mvmc_core::vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: None,
                skip_sr: true,
            },
        )
        .unwrap();
        let snapshot = observer.finish();
        assert_eq!(
            snapshot.executed_term_items,
            4 * size,
            "transfer actual measurement jobs"
        );
        if inner_parallel_enabled(size) {
            assert_eq!(snapshot.parallel_term_items, 4 * size);
            assert_eq!(snapshot.serial_term_items, 0);
            assert!(!snapshot.worker_ids.is_empty());
        } else {
            assert_eq!(snapshot.serial_term_items, 4 * size);
            assert_eq!(snapshot.parallel_term_items, 0);
        }
        assert!(snapshot
            .worker_ids
            .iter()
            .all(|&id| id < inner_thread_config().threads));
        eprintln!("actual site vmc_main_cal.jl:1707 real transfer terms={size}: {snapshot:?}");
        discrete(&format!("synthetic-transfer-{size}-rng"), &rng);
        emit_state(&format!("synthetic-transfer-{size}"), &data, &state);
    }
}

fn independent_runner_prefixes(output_root: &Path, verify_stage: bool) {
    let repo = runner_fixture_root();
    let references = repo.join("tests/fixtures/ctest_model_prefixes");
    let provenance = std::fs::read_to_string(references.join("provenance.txt")).unwrap();
    assert!(provenance.contains("Julia=1.13.1"));
    assert!(provenance.contains("actual serial native C FSZ local energy"));
    for (model, input_model) in INDEPENDENT_PREFIX_CASES {
        let expected = references.join(model).join("step-1");
        let reviewed = repo.join("tests/fixtures/reviewed_cg_62b/canonical_general_rbm");
        let final_reference = |name: &str| {
            if model == "general_rbm_cmp_cg" {
                reviewed.join(format!("step-1-{name}"))
            } else {
                expected.join(name)
            }
        };
        if model == "general_rbm_cmp_cg" {
            let metadata = std::fs::read_to_string(reviewed.join("step-1/provenance.txt")).unwrap();
            assert!(metadata.contains("62b0f97f076fb55c71c3ab0caa041a9adff94e04"));
            assert!(metadata.contains("Julia=1.13.1"));
            assert!(metadata.contains("C-faithful CG recurrence"));
            assert!(metadata.contains("stochastic_opt.jl sha256=b11d75d9b2baaef31abc59c110c09fedc86a7705b1a3c2ce2bb6c75b8b8a17b3"));
            // The independent pre-SR arrays remain valid only for this same
            // public input/overlay identity, before the changed CG recurrence.
            for line in std::fs::read_to_string(expected.join("inputs.sha256"))
                .unwrap()
                .lines()
            {
                let mut fields = line.split_whitespace();
                let hash = fields.next().unwrap();
                let name = fields.next().unwrap();
                assert!(
                    metadata.contains(&format!("inputs/{name} sha256={hash}")),
                    "reviewed CG input identity {name}"
                );
            }
        }
        assert!(!expected.join("UNVERIFIED.txt").exists());
        assert_eq!(
            std::fs::read_to_string(final_reference("status.txt"))
                .unwrap()
                .trim(),
            if model == "general_rbm_cmp_cg" {
                "0 -1"
            } else {
                "0"
            }
        );
        let input = repo.join(format!(
            "extern/Julia-mVMC/test/integration/reference/{input_model}/inputs/namelist.def"
        ));
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false).unwrap();
        if model == "general_rbm_cmp_cg" {
            assert_eq!((data.modpara.nsrcg, data.modpara.nstore_o), (0, 1));
            data.modpara.nsrcg = 1;
            data.modpara.nstore_o = 0;
        }
        let settings = std::fs::read_to_string(expected.join("model-settings.txt")).unwrap();
        let setting = |name: &str| {
            settings
                .split_whitespace()
                .filter_map(|v| v.split_once('='))
                .find_map(|(key, value)| (key == name).then_some(value))
        };
        for (name, actual) in [
            ("NSRCG", data.modpara.nsrcg),
            ("NStore", data.modpara.nstore_o),
        ] {
            assert_eq!(
                setting(name),
                Some(actual.to_string().as_str()),
                "{model} reference settings {name}"
            );
        }
        assert_eq!(
            setting("seed").or_else(|| setting("RndSeed")),
            Some(data.modpara.rnd_seed.to_string().as_str()),
            "{model} reference seed"
        );
        let mut rng = Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
        init_parameter(&mut data, &mut rng);
        let initial = input.parent().unwrap().join("initial.def");
        if initial.is_file() {
            assert!(mvmc_core::read_initial_def(&mut data, initial).unwrap());
        }
        mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(&mut data, &input)
            .unwrap();
        mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
        mvmc_core::qp::init_qp_weight(&mut data);
        data.modpara.nsr_opt_itr_step = 1;
        data.modpara.nsr_opt_itr_smp = 1;
        let nqp = data.modpara.nsp_gauss_leg.max(1) as usize
            * data.modpara.nmp_trans.unsigned_abs() as usize
            * data.n_qp_opt_trans.max(1) as usize;
        let create_state = |data: &ExpertModeData| {
            VmcOptimizationState::zeros(
                data.modpara.nsite as usize,
                data.modpara.nelec as usize,
                data.projection_layout().n_proj,
                data.count_variational_parameters(),
                nqp,
                data.modpara.nvmc_sample as usize,
                mvmc_core::get_all_complex_flag(data),
                data.i_flg_orbital_general != 0,
            )
        };
        let mut state = create_state(&data);
        // The oracle's PRE_SR capture is after weighted normalization, before
        // the solver. Direct SR owns its S/g; CG takes an immutable state and
        // clones OO/HO. Prove the boundary separately without altering this run.
        let pre_sr = verify_stage.then(|| {
            let mut before = data.clone();
            let mut before_rng = rng.clone();
            let mut before_state = create_state(&before);
            let output = output_root.join(format!("independent-{model}-pre-sr"));
            std::fs::create_dir(&output).unwrap();
            mvmc_core::vmc_para_opt(
                &mut before,
                &mut before_state,
                &mut before_rng,
                Some(&output),
                &SingleProcessReducer,
                OptimizationOptions {
                    callback: None,
                    skip_sr: true,
                },
            )
            .unwrap();
            (before_state, before_rng)
        });
        let output = output_root.join(format!("independent-{model}"));
        std::fs::create_dir(&output).unwrap();
        mvmc_core::vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap();
        let all_complex = mvmc_core::get_all_complex_flag(&data);
        let (oo, ho) = normalized_sr_buffers(&state, all_complex);
        for (name, actual) in [("sr_oo.txt", &oo), ("sr_ho.txt", &ho)] {
            let expected_values: Vec<f64> = std::fs::read_to_string(expected.join(name))
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(
                expected_values.len(),
                2 * actual.len(),
                "{model} {name}: full allocated shape"
            );
            // Existing canonical OO/HO policy from ctest_model_prefixes, not
            // inferred from worker agreement and not the looser parameter bound.
            numerical_comparison::assert_values_close(
                actual.iter().flat_map(|z| [z.re, z.im]),
                expected_values,
                1e-12,
                1e-12,
                format!("independent {model} normalized pre-SR {name}"),
            );
        }
        if let Some((before, before_rng)) = pre_sr {
            let (before_oo, before_ho) = normalized_sr_buffers(&before, all_complex);
            for (name, actual, expected) in [("OO", &oo, &before_oo), ("HO", &ho, &before_ho)] {
                numerical_comparison::assert_values_close(
                    actual.iter().flat_map(|z| [z.re, z.im]),
                    expected.iter().flat_map(|z| [z.re, z.im]),
                    1e-12,
                    1e-12,
                    format!("{model} {name}: solver must retain normalized pre-SR arrays"),
                );
            }
            assert_eq!(
                format!("{rng:?}"),
                format!("{before_rng:?}"),
                "{model}: solver must not draw RNG"
            );
            eprintln!(
                "independent SR {model}: normalized pre-solver full OO={} HO={} verified",
                oo.len(),
                ho.len()
            );
        }
        let cfg = &state.electron_config;
        let mut rows = vec![
            cfg.ele_idx.as_slice(),
            cfg.ele_cfg.as_slice(),
            cfg.ele_num.as_slice(),
            cfg.ele_proj_cnt.as_slice(),
        ];
        if data.i_flg_orbital_general != 0 {
            rows.push(cfg.ele_spn.as_slice());
        }
        rows.push(cfg.burn_ele_idx.as_slice());
        rows.push(cfg.counter.as_slice());
        let configs = std::fs::read_to_string(final_reference("configs.txt")).unwrap();
        assert_eq!(configs.lines().count(), rows.len());
        for (line, actual) in configs.lines().zip(rows) {
            let reference: Vec<i64> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(actual, reference, "independent {model} configuration");
        }
        let words: Vec<u32> = std::fs::read_to_string(final_reference("rng.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(words.len(), 624);
        let mut probe = rng.clone();
        assert_eq!(
            (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
            words,
            "independent {model} RNG"
        );
        for (file, actual) in [
            ("parameters.txt", parameters(&data)),
            ("energy.txt", vec![state.energy.etot]),
        ] {
            let reference: Vec<f64> = if model == "general_rbm_cmp_cg" && file == "parameters.txt" {
                // The semantic term-flatten file repeats mappings. Read the
                // independently captured C-declared pack, never truncate it.
                let text =
                    std::fs::read_to_string(reviewed.join("step-1/c-window-input.txt")).unwrap();
                let mut lines = text.lines();
                let header: Vec<usize> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(header.len(), 17);
                assert_eq!(header[0], 1, "effective parameter window");
                assert_eq!(header[1], data.count_variational_parameters());
                assert_eq!(
                    header[2]
                        + header[3]
                        + 6 * header[4]
                        + 10 * header[5]
                        + header[6..].iter().sum::<usize>(),
                    header[1]
                );
                let row: Vec<f64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert!(lines.next().is_none());
                assert_eq!(
                    row.len(),
                    2 * (header[1] + 2),
                    "complete declared E/E2/Para row"
                );
                assert_eq!(actual.len(), header[1], "actual declared parameter pack");
                row[4..].to_vec()
            } else if model == "general_rbm_cmp_cg" {
                numerical_comparison::hex_values(
                    &std::fs::read_to_string(final_reference(file)).unwrap(),
                )
            } else {
                std::fs::read_to_string(final_reference(file))
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect()
            };
            // Reuse the established canonical prefix policy (1e-11): small
            // deterministic one-step mixed C/Julia oracle, not a new tolerance
            // inferred from worker agreement or Monte Carlo noise.
            numerical_comparison::assert_values_close(
                actual.iter().flat_map(|z| [z.re, z.im]),
                reference,
                1e-11,
                1e-11,
                format!("independent {model} {file}"),
            );
        }
        if model == "general_rbm_cmp_cg" {
            let mapped = mapped_parameters(&data);
            let values = numerical_comparison::hex_values(
                &std::fs::read_to_string(final_reference("parameters.txt")).unwrap(),
            );
            assert_eq!(
                values.len(),
                2 * mapped.len(),
                "complete mapped structured RBM schema"
            );
            numerical_comparison::assert_values_close(
                mapped.iter().flat_map(|value| [value.re, value.im]),
                values,
                1e-11,
                1e-11,
                "reviewed CG full mapped structured parameters",
            );
            eprintln!("reviewed CG declared parameters={} mapped parameters={}; independent schemas both checked", parameters(&data).len(), mapped.len());
        }
        let label = format!("independent-{model}");
        discrete(&format!("{label}-final-rng"), &rng);
        emit_state(&label, &data, &state);
    }
}

fn reviewed_cg_long20(output_root: &Path) {
    let repo = runner_fixture_root();
    let root = repo.join("tests/fixtures/reviewed_cg_62b/canonical_general_rbm");
    let metadata = std::fs::read_to_string(root.join("step-20/provenance.txt")).unwrap();
    assert!(metadata.contains("62b0f97f076fb55c71c3ab0caa041a9adff94e04"));
    assert!(metadata.contains("Julia=1.13.1"));
    assert!(metadata.contains(
        "stochastic_opt.jl sha256=b11d75d9b2baaef31abc59c110c09fedc86a7705b1a3c2ce2bb6c75b8b8a17b3"
    ));
    assert!(metadata.contains("successful_steps=20 effective_window=20 selected_rows=20"));
    let input = repo
        .join("extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def");
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(&input).unwrap();
    data.modpara.nsr_opt_itr_step = 20;
    data.modpara.nsr_opt_itr_smp = 20;
    data.modpara.nsrcg = 1;
    data.modpara.nstore_o = 0;
    let mut rng = Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
    init_parameter(&mut data, &mut rng);
    assert!(
        mvmc_core::read_initial_def(&mut data, input.parent().unwrap().join("initial.def"))
            .unwrap()
    );
    mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(&mut data, &input)
        .unwrap();
    mvmc_core::sync::sync_modified_parameter(&mut data, &SingleProcessReducer);
    init_qp_weight(&mut data);
    let nqp = data.qp_weights.as_ref().unwrap().qp_full_weight.len();
    let mut state = VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        nqp,
        data.modpara.nvmc_sample as usize,
        true,
        false,
    );
    let output = output_root.join("reviewed-cg-long20");
    std::fs::create_dir(&output).unwrap();
    let observer = start_observation();
    mvmc_core::vmc_para_opt(
        &mut data,
        &mut state,
        &mut rng,
        Some(&output),
        &SingleProcessReducer,
        OptimizationOptions::default(),
    )
    .unwrap();
    let snapshot = observer.finish();
    if inner_parallel_enabled(2 * (data.count_variational_parameters() + 1)) {
        assert!(snapshot.parallel_entry_items > 0);
        assert!(!snapshot.worker_ids.is_empty());
    } else {
        assert_eq!(snapshot.worker_entries, 0);
    }
    assert!(snapshot
        .worker_ids
        .iter()
        .all(|&id| id < inner_thread_config().threads));
    eprintln!("actual canonical CG 20 steps/window20: {snapshot:?}");
    assert_eq!(
        std::fs::read_to_string(root.join("step-20-status.txt"))
            .unwrap()
            .trim(),
        "0 -1"
    );
    let compare_hex = |name: &str, values: Vec<C>| {
        let expected = numerical_comparison::hex_values(
            &std::fs::read_to_string(root.join(format!("step-20-{name}.txt"))).unwrap(),
        );
        assert_eq!(expected.len(), 2 * values.len(), "long20 {name} schema");
        numerical_comparison::assert_values_close(
            values.iter().flat_map(|z| [z.re, z.im]),
            expected,
            1e-11,
            1e-11,
            format!("reviewed long20 {name}"),
        );
    };
    compare_hex("parameters", mapped_parameters(&data));
    compare_hex("energy", vec![state.energy.etot]);
    let window = std::fs::read_to_string(root.join("step-20/c-window-input.txt")).unwrap();
    let mut lines = window.lines();
    let header: Vec<usize> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(header.len(), 17);
    assert_eq!(header[0], 20);
    assert_eq!(header[1], data.count_variational_parameters());
    assert_eq!(
        header[2] + header[3] + 6 * header[4] + 10 * header[5] + header[6..].iter().sum::<usize>(),
        header[1]
    );
    let rows: Vec<Vec<f64>> = lines
        .map(|line| {
            line.split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(rows.len(), 20);
    assert!(rows.iter().all(|row| row.len() == 2 * (header[1] + 2)));
    numerical_comparison::assert_values_close(
        parameters(&data).iter().flat_map(|z| [z.re, z.im]),
        rows.last().unwrap()[4..].iter().copied(),
        1e-11,
        1e-11,
        "long20 complete final declared pack",
    );
    let cfg = &state.electron_config;
    let actual_rows = [
        cfg.ele_idx.as_slice(),
        cfg.ele_cfg.as_slice(),
        cfg.ele_num.as_slice(),
        cfg.ele_proj_cnt.as_slice(),
        cfg.burn_ele_idx.as_slice(),
        cfg.counter.as_slice(),
    ];
    let configs = std::fs::read_to_string(root.join("step-20-configs.txt")).unwrap();
    assert_eq!(configs.lines().count(), actual_rows.len());
    for (line, actual) in configs.lines().zip(actual_rows) {
        let expected: Vec<i64> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(actual, expected, "long20 exact saved configurations");
    }
    let words: Vec<u32> = std::fs::read_to_string(root.join("step-20-rng.txt"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(words.len(), 624);
    let mut probe = rng.clone();
    assert_eq!(
        (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
        words
    );
    discrete("reviewed-cg-long20-rng", &rng);
    emit_state("reviewed-cg-long20", &data, &state);
}

fn normalized_sr_buffers(state: &VmcOptimizationState, all_complex: bool) -> (Vec<C>, Vec<C>) {
    let size = state.sr_opt.sr_opt_size * if all_complex { 2 } else { 1 };
    let (oo, ho) = if all_complex {
        (
            state.sr_opt.sr_opt_oo.clone(),
            state.sr_opt.sr_opt_ho.clone(),
        )
    } else {
        (
            state
                .sr_opt
                .sr_opt_oo_real
                .iter()
                .map(|&v| C::new(v, 0.0))
                .collect(),
            state
                .sr_opt
                .sr_opt_ho_real
                .iter()
                .map(|&v| C::new(v, 0.0))
                .collect(),
        )
    };
    assert_eq!(
        oo.len(),
        size * (size + 2),
        "full OO including reserved trailing blocks"
    );
    assert_eq!(ho.len(), size, "full HO");
    (oo, ho)
}

fn independent_physcal_root() -> std::path::PathBuf {
    runner_fixture_root().join("tests/fixtures/physcal_181/two-samples")
}

fn require_independent_physcal_fixtures() {
    for (model, _) in INDEPENDENT_PHYSCAL_CASES {
        let root = independent_physcal_root().join(model);
        for name in [
            "inputs/namelist.def",
            "zqp_opt.dat",
            "provenance.txt",
            "fixed-parameters.txt",
            "seeded/next624.txt",
            "seeded/draw-count.txt",
            "initialized/next624.txt",
            "initialized/draw-count.txt",
            "sample-1/next624.txt",
            "sample-1/draw-count.txt",
            "sample-1/ele_idx.txt",
            "sample-1/ele_cfg.txt",
            "sample-1/ele_num.txt",
            "sample-1/ele_proj_cnt.txt",
            "sample-1/ele_spn.txt",
            "sample-1/counter.txt",
            "expected/zvo_out_007.dat",
            "expected/zvo_var_007.dat",
            "expected/zvo_cisajs_007.dat",
            "expected/zvo_cisajscktalt_007.dat",
            "expected/zvo_cisajscktaltex_007.dat",
            "expected/zvo_out_008.dat",
            "expected/zvo_var_008.dat",
            "expected/zvo_cisajs_008.dat",
            "expected/zvo_cisajscktalt_008.dat",
            "expected/zvo_cisajscktaltex_008.dat",
        ] {
            let path = root.join(name);
            if !path.is_file() {
                support::missing_fixture("threaded-issue182-physcal", path.display().to_string());
            }
        }
    }
}

fn assert_independent_rng(stage: &Path, rng: &Sfmt19937Rng) {
    let expected: Vec<u32> = std::fs::read_to_string(stage.join("next624.txt"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(expected.len(), 624, "{} RNG shape", stage.display());
    let count: u128 = std::fs::read_to_string(stage.join("draw-count.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(
        rng.words_consumed(),
        count,
        "{} actual primitive draw count",
        stage.display()
    );
    let mut peek = rng.clone();
    assert_eq!(
        (0..624).map(|_| peek.gen_rand32()).collect::<Vec<_>>(),
        expected,
        "{} independent RNG",
        stage.display()
    );
}

fn assert_independent_physcal_output(actual: &Path, expected: &Path) {
    let name = expected.file_name().unwrap().to_str().unwrap();
    // Existing #181 independent output contracts; keep max(atol,rtol*scale)
    // rather than loosening them to a summed tolerance or worker agreement.
    let (indices, width, absolute, relative) = if name.starts_with("zvo_ls_out_") {
        (0, Some(3), 1e-8, 0.0)
    } else if name.starts_with("zvo_ls_qqqq_") {
        (0, Some(16), 1e-10, 0.0)
    } else if name.starts_with("zvo_ls_cisajscktaltex_") {
        (0, None, 1e-8, 0.0)
    } else if name.starts_with("zvo_ls_cisajscktalt_") {
        (8, Some(10), 1e-8, 0.0)
    } else if name.starts_with("zvo_ls_cisajs_") {
        (4, Some(6), 1e-8, 0.0)
    } else if name.starts_with("zvo_cisajscktaltex_") {
        (0, None, 1e-12, 1e-9)
    } else if name.starts_with("zvo_cisajscktalt_") {
        (8, Some(10), 1e-12, 1e-9)
    } else if name.starts_with("zvo_cisajs_") {
        (4, Some(6), 1e-12, 1e-10)
    } else if name.starts_with("zvo_out_") {
        (0, Some(6), 1e-12, 1e-10)
    } else if name.starts_with("zvo_var_") {
        (0, None, 1e-12, 1e-10)
    } else {
        panic!("unsupported independent output contract: {name}")
    };
    let rows = |text: &str| {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let actual_rows = rows(&std::fs::read_to_string(actual).unwrap());
    let expected_rows = rows(&std::fs::read_to_string(expected).unwrap());
    assert!(
        !expected_rows.is_empty(),
        "{name} empty reference is not evidence"
    );
    assert_eq!(
        actual_rows.len(),
        expected_rows.len(),
        "{name} ordered row count"
    );
    if indices == 0 {
        assert_eq!(expected_rows.len(), 1, "{name} value-only row contract");
    }
    for (row, (a, e)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        assert_eq!(
            e.len(),
            width.unwrap_or(e.len()),
            "{name} expected row width"
        );
        assert_eq!(a.len(), e.len(), "{name} actual row width");
        if name.contains("cisajscktaltex_") {
            assert_eq!(e.len() % 2, 0, "ordered GEx pairs");
        }
        for (column, (a, e)) in a.iter().zip(e).enumerate() {
            if column < indices {
                assert_eq!(
                    a.parse::<i64>().unwrap(),
                    e.parse::<i64>().unwrap(),
                    "{name} discrete index"
                );
                assert_eq!(a, e, "{name} ordered index token");
            } else {
                let a: f64 = a.parse().unwrap();
                let e: f64 = e.parse().unwrap();
                let scale = a.abs().max(e.abs());
                let bound = f64::max(absolute, relative * scale).min(1e-10 + 1e-8 * scale);
                assert!(a.is_finite() && e.is_finite() && (a-e).abs() <= bound,
                    "{name} row {row} column {column}: actual={a:.17e} expected={e:.17e} bound={bound:.3e}");
            }
        }
    }
}

fn independent_physcal(output_root: &Path) {
    for (model, mode) in INDEPENDENT_PHYSCAL_CASES {
        let root = independent_physcal_root().join(model);
        let provenance = std::fs::read_to_string(root.join("provenance.txt")).unwrap();
        assert!(provenance.contains("julia=1.13.1") && provenance.contains("seed=1"));
        let prepared = mvmc_core::prepare_phys_cal_from_namelist(
            root.join("inputs/namelist.def"),
            root.join("zqp_opt.dat"),
            mode,
            Some(1),
        )
        .unwrap();
        assert!(prepared.data.inter_all_terms.is_empty());
        // Keep the actual supported fixture. Hubbard's existing reference is
        // non-InterAll Lanczos mode 2; do not replace it with an easier mode 0.
        assert_eq!(
            prepared.data.modpara.lanczos_mode,
            if model == "hubbard_chain_real" { 2 } else { 0 },
            "supported independent measurement mode"
        );
        assert_eq!(
            (
                prepared.data.modpara.n_data_qty_smp,
                prepared.data.modpara.n_data_idx_start
            ),
            (2, 7)
        );
        // Preparation deliberately returns BEFORE the driver's single
        // InitParameter call; compare its actual seeded stage, not initialized.
        assert_independent_rng(&root.join("seeded"), &prepared.rng);
        let mut init_probe = prepared.rng.clone();
        let mut init_data = prepared.data.clone();
        init_parameter(&mut init_data, &mut init_probe);
        assert_independent_rng(&root.join("initialized"), &init_probe);
        let fixed = parameters(&prepared.data);
        let fixed_reference: Vec<f64> = std::fs::read_to_string(root.join("fixed-parameters.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        numerical_comparison::assert_values_close(
            fixed.iter().flat_map(|z| [z.re, z.im]),
            fixed_reference,
            1e-12,
            1e-12,
            format!("{model} independent fixed records"),
        );
        let flags = prepared.data.optimization_flags.clone();
        let output = output_root.join(format!("independent-physcal-{model}"));
        let result = mvmc_core::vmc_phys_cal_to_dir(prepared, &output).unwrap();
        assert_eq!(result.iterations, 2);
        assert_eq!(result.data.optimization_flags, flags);
        numerical_comparison::assert_values_close(
            parameters(&result.data).iter().flat_map(|z| [z.re, z.im]),
            fixed.iter().flat_map(|z| [z.re, z.im]),
            1e-12,
            1e-12,
            format!("{model} PhysCal fixed preservation"),
        );
        let stage = root.join("sample-1");
        let cfg = &result.state.electron_config;
        for (name, actual) in [
            ("ele_idx", cfg.ele_idx.as_slice()),
            ("ele_cfg", cfg.ele_cfg.as_slice()),
            ("ele_num", cfg.ele_num.as_slice()),
            ("ele_proj_cnt", cfg.ele_proj_cnt.as_slice()),
            ("ele_spn", cfg.ele_spn.as_slice()),
            ("counter", cfg.counter.as_slice()),
        ] {
            let expected: Vec<i64> = std::fs::read_to_string(stage.join(format!("{name}.txt")))
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(actual, expected, "{model} independent saved {name}");
        }
        assert_independent_rng(&stage, &result.final_rng);
        let names = |path: &Path| {
            std::fs::read_dir(path)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(
            names(&output),
            names(&root.join("expected")),
            "{model} exact output file set"
        );
        for name in names(&root.join("expected")) {
            assert_independent_physcal_output(
                &output.join(&name),
                &root.join("expected").join(name),
            );
        }
        let label = format!("independent-physcal-{model}");
        discrete(&format!("{label}-final-rng"), &result.final_rng);
        emit_state(&label, &result.data, &result.state);
        eprintln!("independent PhysCal {model}: two frames, exact saved/config/draw count, ordered output reference verified");
    }
}

fn independent_real_fsz_root() -> std::path::PathBuf {
    std::env::var_os("MVMC_RS_THREADED_REAL_FSZ_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| runner_fixture_root().join("tests/fixtures/threaded_182_real_fsz"))
}

fn require_independent_real_fsz_fixtures() {
    let root = independent_real_fsz_root();
    for name in [
        "inputs/namelist.def",
        "provenance.txt",
        "status.txt",
        "optimization-flags.txt",
        "initialized/qp_weights.txt",
        "pre-sr/sr_oo.txt",
        "pre-sr/sr_ho.txt",
    ] {
        let path = root.join(name);
        if !path.is_file() {
            support::missing_fixture("threaded-issue182-real-fsz", path.display().to_string());
        }
    }
    for stage in ["seeded", "initialized", "pre-sr", "final"] {
        for name in ["draw-count.txt", "next624.txt", "parameters.txt"] {
            let path = root.join(stage).join(name);
            if !path.is_file() {
                support::missing_fixture("threaded-issue182-real-fsz", path.display().to_string());
            }
        }
        if stage != "seeded" {
            for name in [
                "julia-raw-flags.txt",
                "c-written-mask.txt",
                "defined-flags.txt",
            ] {
                let path = root.join(stage).join(name);
                if !path.is_file() {
                    support::missing_fixture(
                        "threaded-issue182-real-fsz",
                        path.display().to_string(),
                    );
                }
            }
        }
    }
    for stage in ["pre-sr", "final"] {
        for name in [
            "ele_idx.txt",
            "ele_cfg.txt",
            "ele_num.txt",
            "ele_proj_cnt.txt",
            "ele_spn.txt",
            "burn_ele_idx.txt",
            "counter.txt",
        ] {
            let path = root.join(stage).join(name);
            if !path.is_file() {
                support::missing_fixture("threaded-issue182-real-fsz", path.display().to_string());
            }
        }
    }
}

fn independent_real_fsz(output_root: &Path) {
    require_independent_real_fsz_fixtures();
    let root = independent_real_fsz_root();
    let provenance = std::fs::read_to_string(root.join("provenance.txt")).unwrap();
    assert!(provenance.contains("ADAPTED C-compatible real-FSZ"));
    assert!(provenance.contains("Julia=1.13.1"));
    assert!(provenance.contains("direct/no-store"));
    assert!(!root.join("UNVERIFIED.txt").exists());
    assert_eq!(
        std::fs::read_to_string(root.join("status.txt"))
            .unwrap()
            .trim(),
        "0"
    );
    let input = root.join("inputs/namelist.def");
    assert!(!root.join("inputs/initial.def").exists());
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false).unwrap();
    assert!(!mvmc_core::get_all_complex_flag(&data));
    assert_ne!(data.i_flg_orbital_general, 0);
    assert_eq!((data.modpara.nsrcg, data.modpara.nstore_o), (0, 0));
    let mut rng = Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
    assert_independent_rng(&root.join("seeded"), &rng);
    init_parameter(&mut data, &mut rng);
    mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(&mut data, &input)
        .unwrap();
    mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
    mvmc_core::qp::init_qp_weight(&mut data);
    assert_independent_rng(&root.join("initialized"), &rng);
    let check_flags = |stage: &str, actual: &[i64]| {
        let read = |name: &str| -> Vec<i64> {
            std::fs::read_to_string(root.join(stage).join(name))
                .unwrap()
                .split_whitespace()
                .map(|value| value.parse().unwrap())
                .collect()
        };
        let raw = read("julia-raw-flags.txt");
        if stage == "initialized" {
            let legacy_raw: Vec<i64> = std::fs::read_to_string(root.join("optimization-flags.txt"))
                .unwrap()
                .split_whitespace()
                .map(|value| i64::from(value.parse::<bool>().unwrap()))
                .collect();
            assert_eq!(raw, legacy_raw, "initialized raw diagnostic flag identity");
        }
        let mask = read("c-written-mask.txt");
        let defined = read("defined-flags.txt");
        assert_eq!(raw.len(), actual.len(), "{stage} raw flag shape");
        assert_eq!(mask.len(), actual.len(), "{stage} written mask shape");
        assert_eq!(defined.len(), actual.len(), "{stage} defined flag shape");
        for (index, &flag) in actual.iter().enumerate() {
            assert!(matches!(mask[index], 0 | 1), "{stage} invalid written mask");
            if index % 2 == 0 {
                assert_eq!(mask[index], 1, "{stage} active real flag must be written");
                assert_eq!(
                    raw[index], defined[index],
                    "{stage} Julia active flag {index}"
                );
            } else {
                assert_eq!(flag, 0, "{stage} explicitly defined inactive flag {index}");
                assert_eq!(
                    defined[index], 0,
                    "{stage} inactive reference policy {index}"
                );
            }
            assert_eq!(flag, defined[index], "{stage} exact defined flag {index}");
        }
        eprintln!(
            "{stage} raw Julia flags={raw:?}; C written mask={mask:?}; defined flags={defined:?}"
        );
    };
    check_flags("initialized", &data.optimization_flags);
    // Initialization uses scalar mathematical functions, not an SR solve or
    // MPI reduction. Allow a small rounding budget for provider differences
    // (the reviewed initial first difference was 8.7e-19), while all flags
    // and RNG state/draw contracts above remain exact. Do not propagate this
    // explanation to downstream solver conditioning or sampling drift.
    const INITIAL_MATH_ABS_REL: f64 = 8.0 * f64::EPSILON;
    let check_numeric = |path: &Path, actual: Vec<C>, tolerance: f64| {
        let reference: Vec<f64> = std::fs::read_to_string(path)
            .unwrap()
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(
            reference.len(),
            2 * actual.len(),
            "{} full shape",
            path.display()
        );
        numerical_comparison::assert_values_close(
            actual.iter().flat_map(|value| [value.re, value.im]),
            reference,
            tolerance,
            tolerance,
            format!("independent adapted real FSZ {}", path.display()),
        );
    };
    check_numeric(
        &root.join("initialized/parameters.txt"),
        parameters(&data),
        INITIAL_MATH_ABS_REL,
    );
    check_numeric(
        &root.join("initialized/qp_weights.txt"),
        data.qp_weights.as_ref().unwrap().qp_full_weight.clone(),
        INITIAL_MATH_ABS_REL,
    );
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
    let nqp = data.modpara.nsp_gauss_leg.max(1) as usize
        * data.modpara.nmp_trans.unsigned_abs() as usize
        * data.n_qp_opt_trans.max(1) as usize;
    let create_state = |data: &ExpertModeData| {
        VmcOptimizationState::zeros(
            data.modpara.nsite as usize,
            data.modpara.nelec as usize,
            data.projection_layout().n_proj,
            data.count_variational_parameters(),
            nqp,
            data.modpara.nvmc_sample as usize,
            false,
            true,
        )
    };
    for (stage, skip_sr) in [("pre-sr", true), ("final", false)] {
        let mut run_data = data.clone();
        let mut run_rng = rng.clone();
        let mut state = create_state(&run_data);
        let output = output_root.join(format!("independent-real-fsz-{stage}"));
        std::fs::create_dir(&output).unwrap();
        let observer = start_observation();
        mvmc_core::vmc_para_opt(
            &mut run_data,
            &mut state,
            &mut run_rng,
            Some(&output),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: None,
                skip_sr,
            },
        )
        .unwrap();
        let observed = observer.finish();
        check_flags(stage, &run_data.optimization_flags);
        // Public input selects one QP plane: capacity is not QP activation.
        assert!(observed.serial_qp_items > 0);
        assert!(observed
            .worker_ids
            .iter()
            .all(|&id| id < inner_thread_config().threads));
        eprintln!("actual independent real-FSZ {stage}: {observed:?}");
        let (oo, ho) = normalized_sr_buffers(&state, false);
        check_numeric(&root.join("pre-sr/sr_oo.txt"), oo, 1e-12);
        check_numeric(&root.join("pre-sr/sr_ho.txt"), ho, 1e-12);
        assert_independent_rng(&root.join(stage), &run_rng);
        let cfg = &state.electron_config;
        for (name, actual) in [
            ("ele_idx", cfg.ele_idx.as_slice()),
            ("ele_cfg", cfg.ele_cfg.as_slice()),
            ("ele_num", cfg.ele_num.as_slice()),
            ("ele_proj_cnt", cfg.ele_proj_cnt.as_slice()),
            ("ele_spn", cfg.ele_spn.as_slice()),
            ("burn_ele_idx", cfg.burn_ele_idx.as_slice()),
            ("counter", cfg.counter.as_slice()),
        ] {
            let reference: Vec<i64> =
                std::fs::read_to_string(root.join(stage).join(format!("{name}.txt")))
                    .unwrap()
                    .split_whitespace()
                    .map(|value| value.parse().unwrap())
                    .collect();
            assert_eq!(actual, reference, "independent real FSZ {stage} {name}");
        }
        check_numeric(
            &root.join(stage).join("parameters.txt"),
            parameters(&run_data),
            1e-11,
        );
        discrete(&format!("independent-real-fsz-{stage}-rng"), &run_rng);
        emit_state(&format!("independent-real-fsz-{stage}"), &run_data, &state);
    }
}
