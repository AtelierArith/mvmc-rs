//! Process-scoped worker matrix: no shared environment mutation between tests.
//! Kernel expectations are analytic or archived independent Julia values.
//! Runner comparisons are Rust worker invariance, not fresh C/Julia parity.

#[path = "support/case_observation.rs"]
mod case_observation;
#[path = "support/ctest_provenance.rs"]
mod ctest_provenance;
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
    let mut command = Command::new(std::env::current_exe().unwrap());
    let invocation = case_observation::configure_child(&mut command, job, workers, threshold);
    let result = command
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
    case_observation::retain_child(invocation, job, workers, threshold, &result);
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
    case_observation::comparison(expected, actual);
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
#[ignore = "opt-in synthetic long20 supported success/rejection outcomes; MVMC_RS_THREADED_182=1"]
fn runner_twenty_step_workers_repeat_supported_outcomes_and_physcal() {
    support::require_gate("threaded-issue182-matrix-long20", "MVMC_RS_THREADED_182");
    require_runner_fixtures();
    let mut serial = None;
    for workers in [1, 2, 4] {
        let first = child("runners-long20-observed", workers, 32);
        compare(&first, &child("runners-long20-observed", workers, 32));
        if let Some(reference) = &serial {
            compare(reference, &first);
        } else {
            serial = Some(first);
        }
    }
}

#[test]
#[ignore = "opt-in actual non-SPD direct SR boundary; MVMC_RS_THREADED_182=1"]
fn direct_sr_non_spd_boundary_repeats_workers_one_two_four() {
    support::require_gate(
        "threaded-issue182-direct-sr-boundary",
        "MVMC_RS_THREADED_182",
    );
    require_runner_fixtures();
    let mut serial = None;
    for workers in [1, 2, 4] {
        let first = child("direct-sr-failure-boundary", workers, 32);
        compare(&first, &child("direct-sr-failure-boundary", workers, 32));
        if let Some(reference) = &serial {
            compare(reference, &first);
        } else {
            serial = Some(first);
        }
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

fn require_runner_fixtures() {
    let repo = runner_fixture_root();
    let mut required = Vec::new();
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
    }
    for path in required {
        if !path.is_file() {
            support::missing_fixture("threaded-issue182", path.display().to_string());
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
    let _recorder_job = case_observation::enter_job();
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
    } else if job == "transfer-site" {
        transfer_site(&fresh_output_root());
    } else if job == "direct-sr-failure-boundary" {
        runner_matrix(20, true, false);
    } else if job == "runners-long20-diagnostic" {
        support::require_gate(
            "threaded-issue182-diagnostic",
            "ISSUE182_DIAGNOSTIC_COLLECT_ALL",
        );
        require_runner_fixtures();
        runner_matrix(20, false, true);
    } else {
        assert!(matches!(
            job.as_str(),
            "runners" | "runners-observed" | "runners-long20-observed"
        ));
        let observer = job.ends_with("-observed").then(start_observation);
        runner_matrix(
            if job == "runners-long20-observed" {
                20
            } else {
                2
            },
            false,
            false,
        );
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
    if std::env::var("MVMC_EXPERIMENT_ASSERT_SHARED").as_deref() == Ok("1")
        && config.threads > 1
        && matches!(job.as_str(), "runners" | "runners-observed")
    {
        let (attempts, successes, busy, unavailable, participants) =
            mvmc_core::stage_backend::probe_cpu_snapshot();
        assert_eq!(attempts, successes + busy + unavailable);
        assert!(
            attempts > 0 && successes > 0 && participants >= 2,
            "shared team was not exercised: {attempts}/{successes}/{participants}"
        );
        eprintln!("shared team probe attempts={attempts}, successes={successes}, busy={busy}, unavailable={unavailable}, participants={participants}");
    }
    case_observation::finish_child();
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
    // C's real mode takes creal(SlaterElm); complex-valued Slater input is outside its
    // contract, so the real FSZ kernel is checked on the real-valued table.
    for (dst, &src) in fsz
        .slater_elm
        .as_mut_slice()
        .iter_mut()
        .zip(real.as_slice())
    {
        *dst = C::new(src, 0.0);
    }
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
        ri.as_slice().iter().flat_map(|&v| [v, 0.0]),
        BOUND,
        BOUND,
        "analytic FSZ inverse",
    );
    numerical_comparison::assert_values_close(
        fsz.pf_m.iter().flat_map(|z| [z.re, z.im]),
        rp.iter().flat_map(|&v| [v, 0.0]),
        BOUND,
        BOUND,
        "analytic FSZ pf",
    );
    for (qp, expected_pf) in rp.iter().enumerate().skip(1).take(size) {
        numerical_comparison::assert_close(
            fsz.pf_m_real[qp],
            *expected_pf,
            BOUND,
            BOUND,
            "FSZ real shadow",
        );
        numerical_comparison::assert_values_close(
            fsz.inv_m_real.qp_matrix_slice(qp).iter().copied(),
            ri.qp_matrix_slice(qp).iter().copied(),
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

fn parameters(data: &ExpertModeData) -> Vec<C> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_params.iter().copied())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn emit_state(label: &str, data: &ExpertModeData, state: &VmcOptimizationState) {
    case_observation::retain_layout(label, data, state);
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

fn runner_matrix(steps: i64, failure_boundary_only: bool, diagnostic_collect_all: bool) {
    let output_root = fresh_output_root();
    let mut diagnostic_failures = Vec::new();
    let mut executed_optimization_cases = 0;
    let mut verified_rejections = 0;
    for (model, real_fsz) in [
        ("heisenberg_chain_real", false),
        ("heisenberg_chain_cmp", false),
        ("heisenberg_chain_fsz", false),
        ("heisenberg_chain_fsz", true),
        ("hubbard_chain_real", false),
    ] {
        let root = runner_fixture_root().join("extern/Julia-mVMC");
        for size in SIZES {
            // Long mode couples sample and QP lengths at the 31/32/33
            // threshold and has 5*3*3 optimization cases per child. The
            // historical two-step mode retains all four settings and 200
            // samples. This is new coverage, not the external 200-sample proof.
            let samples = if steps == 20 { size } else { 200 };
            let settings: &[(i64, i64)] = if steps == 20 {
                &[(0, 0), (1, 0), (0, 1)]
            } else {
                &[(0, 0), (1, 0), (0, 1), (1, 1)]
            };
            for &(store, cg) in settings {
                // A separately named failure-contract test, not a skipped
                // case in the full success matrix (which still propagates it).
                if failure_boundary_only
                    && !(model == "hubbard_chain_real" && size == 32 && store == 0 && cg == 0)
                {
                    continue;
                }
                let recorded_case = (case_observation::enabled() && steps == 20).then(|| {
                    if failure_boundary_only {
                        "failure/hubbard_chain_real/qp32/store0/cg0".to_owned()
                    } else {
                        let variant = if real_fsz { "real-fsz" } else { model };
                        format!("long20/{variant}/qp{size}/store{store}/cg{cg}")
                    }
                });
                if let Some(case) = &recorded_case {
                    case_observation::begin(case, "runtime-outcome");
                }
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
                data.modpara.nvmc_sample = samples as i64;
                data.modpara.nvmc_interval = 1;
                data.modpara.nsr_opt_itr_step = steps;
                data.modpara.nsr_opt_itr_smp = steps;
                data.modpara.nstore_o = store;
                data.modpara.nsrcg = cg;
                data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
                if real_fsz {
                    data.complex_flags = vec![0];
                }
                let variant = if real_fsz { "real-fsz" } else { model };
                let label =
                    format!("synthetic-{variant}-qp{size}-samples{samples}-store{store}-cg{cg}");
                let output = output_root.join(&label);
                std::fs::create_dir(&output).unwrap();
                let mut rng = Sfmt19937Rng::new(1);
                case_observation::retain_launch_seed(&label, 1);
                init_parameter(&mut data, &mut rng).unwrap();
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
                    samples,
                    mvmc_core::get_all_complex_flag(&data).unwrap(),
                    data.i_flg_orbital_general != 0,
                );
                let mut completed_steps = Vec::new();
                let mut last_parameters = parameters(&data);
                let mut callback = |step, data: &mut ExpertModeData, energy: C, status| {
                    completed_steps.push(step);
                    last_parameters = parameters(data);
                    discrete(&format!("{label}-step{step}-status"), status);
                    complex(&format!("{label}-step{step}-parameters"), &parameters(data));
                    complex(&format!("{label}-step{step}-energy"), &[energy]);
                    Ok(())
                };
                // Developer-only observation of the ORIGINAL solve, including
                // its failure buffers; never reconstruct a system from outputs.
                // Retained original operands were independently replayed by
                // native C DPOSV on all six captured worker/repeat operands.
                // Only these case/step/INFO tuples are verified rejections.
                let rejection = if steps == 20 && model == "hubbard_chain_real" && cg == 0 {
                    match (size, store) {
                        (32, 0) => Some((10, 2)),
                        (32, 1) => Some((10, 4)),
                        (33, 1) => Some((16, 2)),
                        _ => None,
                    }
                } else {
                    None
                };
                let verified_non_spd = rejection.is_some();
                let capture = (diagnostic_collect_all && cg == 0
                    || verified_non_spd
                    || std::env::var_os("ISSUE182_SR_DIAGNOSTICS").is_some()
                        && model == "hubbard_chain_real"
                        && size == 32
                        && cg == 0)
                    .then(|| mvmc_core::sr::observer::capture().unwrap());
                executed_optimization_cases += 1;
                let trace_moves = std::env::var_os("MVMC_EXPERIMENT_TRACE_MOVES").is_some()
                    && size == 32
                    && store == 0
                    && cg == 0;
                if trace_moves {
                    mvmc_core::sampling::driver::trace::start();
                }
                let result = mvmc_core::vmc_para_opt(
                    &mut data,
                    &mut state,
                    &mut rng,
                    Some(&output),
                    &SingleProcessReducer,
                    OptimizationOptions {
                        callback: Some(&mut callback),
                        skip_sr: false,
                        file_flush_interval: None,
                        binary_output: false,
                    },
                );
                if trace_moves {
                    let events = mvmc_core::sampling::driver::trace::finish();
                    assert!(
                        events.iter().any(|event| event[0] == 7),
                        "no Metropolis decisions captured"
                    );
                    discrete(&format!("{label}-sampling-events"), events);
                }
                if diagnostic_collect_all {
                    let mut next = rng.clone();
                    std::fs::write(
                        output.join("diagnostic-outcome.txt"),
                        format!(
                            "label={label}\nworkers={:?}\nresult={result:?}\ncompleted_steps={completed_steps:?}\nlast_post_sync_parameters={last_parameters:?}\nparameters={:?}\nrng={rng:?}\nwords={}\nnext624={:?}\nconfig={:?}\n",
                            inner_thread_config(), parameters(&data), rng.words_consumed(),
                            (0..624).map(|_| next.gen_rand32()).collect::<Vec<_>>(), state.electron_config,
                        ),
                    ).unwrap();
                    eprintln!(
                        "DIAGNOSTIC CASE {label}: {result:?}; artifacts {}",
                        output.display()
                    );
                    if let Err(error) = &result {
                        diagnostic_failures.push(format!("{label}: {error}"));
                    }
                }
                if let Some(capture) = capture {
                    let records = capture.finish();
                    if verified_non_spd && !diagnostic_collect_all {
                        let (failed_step, factor_info) = rejection.unwrap();
                        assert_eq!(result.as_ref().unwrap_err(), &format!("vmc_para_opt: direct SR failed at step {failed_step} (local status 1); parameters were not updated"));
                        assert_eq!(completed_steps, (0..failed_step).collect::<Vec<_>>());
                        let bits = |values: &[C]| {
                            values
                                .iter()
                                .map(|z| (z.re.to_bits(), z.im.to_bits()))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(
                            bits(&parameters(&data)),
                            bits(&last_parameters),
                            "failed SR must not update last successful post-sync parameters"
                        );
                        assert_eq!(records.len(), failed_step + 1);
                        assert!(records[..failed_step].iter().all(|r| r.status == Some(0)));
                        let failed = &records[failed_step];
                        case_observation::retain_sr_dimension(&label, failed.dimension);
                        assert_eq!(failed.dimension, 5);
                        assert_eq!(failed.triangle, 'U');
                        assert_eq!(failed.status, Some(1));
                        assert_eq!(failed.factor_info, Some(factor_info));
                        assert_eq!(
                            failed.solve_info, None,
                            "POTRS must not run after failed POTRF"
                        );
                        assert_eq!(
                            failed.rhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            failed
                                .increment
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "failed POTRF must leave RHS untouched, including signed zero"
                        );
                        if factor_info == 2 {
                            assert!(
                                failed.matrix[0] * failed.matrix[6]
                                    - failed.matrix[5] * failed.matrix[5]
                                    < 0.0
                            );
                        }
                        verified_rejections += 1;
                        discrete(
                            &format!("{label}-outcome"),
                            format!("verified-non-spd-step{failed_step}-info{factor_info}"),
                        );
                        // A/RHS are computed values: compare across workers
                        // using the existing numerical policy, not bitwise.
                        numerical(
                            &format!("{label}-failed-matrix"),
                            failed.matrix.iter().copied(),
                        );
                        numerical(&format!("{label}-failed-rhs"), failed.rhs.iter().copied());
                        discrete(
                            &format!("{label}-failed-shape"),
                            (failed.dimension, failed.triangle),
                        );
                        discrete(&format!("{label}-failed-flags"), &failed.flags);
                        discrete(
                            &format!("{label}-failed-active-indices"),
                            &failed.active_indices,
                        );
                        discrete(
                            &format!("{label}-failed-status"),
                            (failed.status, failed.factor_info, failed.solve_info),
                        );
                    }
                    // Preserve actual failure-boundary discrete state before
                    // propagating the error. This does not turn failure green.
                    std::fs::write(
                        output.join("direct-sr-boundary-state.txt"),
                        format!(
                            "result={result:?}\nrng={rng:?}\nwords={}\nnext624={:?}\nconfig={:?}\nparameters={:?}\ncompleted_steps={completed_steps:?}\nlast_post_sync_parameters={last_parameters:?}\n",
                            rng.words_consumed(),
                            (0..624).map({ let mut next = rng.clone(); move |_| next.gen_rand32() }).collect::<Vec<_>>(),
                            state.electron_config,
                            parameters(&data),
                        ),
                    ).unwrap();
                    for (index, record) in records.iter().enumerate() {
                        let prefix = output.join(format!("direct-sr-observed-{index}"));
                        std::fs::write(
                            prefix.with_extension("metadata.txt"),
                            format!("{record:#?}"),
                        )
                        .unwrap();
                        for (name, values) in [
                            ("matrix", &record.matrix),
                            ("rhs", &record.rhs),
                            ("increment", &record.increment),
                        ] {
                            let text: String =
                                values.iter().map(|v| format!("{v:.17e}\n")).collect();
                            std::fs::write(prefix.with_extension(format!("{name}.txt")), text)
                                .unwrap();
                        }
                    }
                    eprintln!(
                        "original direct SR observations: {} records in {}",
                        records.len(),
                        output.display()
                    );
                }
                if failure_boundary_only {
                    case_observation::retain_layout(&label, &data, &state);
                    discrete("direct-sr-failure-rng", &rng);
                    discrete("direct-sr-failure-words", rng.words_consumed());
                    let mut next = rng.clone();
                    discrete(
                        "direct-sr-failure-next624",
                        (0..624).map(|_| next.gen_rand32()).collect::<Vec<_>>(),
                    );
                    discrete("direct-sr-failure-config", &state.electron_config);
                    complex("direct-sr-failure-parameters", &parameters(&data));
                    if let Some(case) = &recorded_case {
                        case_observation::complete(case, "runtime-outcome");
                    }
                    return;
                }
                if diagnostic_collect_all && result.is_err() {
                    // Only this explicit diagnostic mode continues. The
                    // normal gate still rejects every unverified error.
                    emit_state(&label, &data, &state);
                    continue;
                }
                if !verified_non_spd {
                    result.unwrap_or_else(|e| panic!("{label}: output {}: {e}", output.display()));
                    discrete(&format!("{label}-outcome"), "success");
                }
                discrete(&format!("{label}-final-rng"), &rng);
                discrete(&format!("{label}-draw-count"), rng.words_consumed());
                let mut peek = rng.clone();
                discrete(
                    &format!("{label}-next624"),
                    (0..624).map(|_| peek.gen_rand32()).collect::<Vec<_>>(),
                );
                emit_state(&label, &data, &state);
                if let Some(case) = &recorded_case {
                    case_observation::complete(case, "runtime-outcome");
                }
            }
            let recorded_physcal = (case_observation::enabled() && steps == 20).then(|| {
                let variant = if real_fsz { "real-fsz" } else { model };
                format!("long20/{variant}/qp{size}/physcal")
            });
            if let Some(case) = &recorded_physcal {
                case_observation::begin(case, "runtime-outcome");
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
            case_observation::retain_launch_seed(&label, 1);
            discrete(&format!("{label}-initial-rng"), &prepared.rng);
            let output = output_root.join(&label);
            let result = mvmc_core::vmc_phys_cal_to_dir(prepared, &output);
            if diagnostic_collect_all {
                if let Err(error) = &result {
                    diagnostic_failures.push(format!("{label}: {error}"));
                    eprintln!("DIAGNOSTIC PHYSCAL {label}: {error}");
                    continue;
                }
            }
            let result =
                result.unwrap_or_else(|e| panic!("{label}: output {}: {e}", output.display()));
            discrete(&format!("{label}-final-rng"), &result.final_rng);
            discrete(&format!("{label}-iterations"), result.iterations);
            emit_state(&label, &result.data, &result.state);
            if let Some(case) = &recorded_physcal {
                case_observation::complete(case, "runtime-outcome");
            }
        }
    }
    if steps == 20 && !failure_boundary_only {
        assert_eq!(
            executed_optimization_cases, 45,
            "no missing or dropped long20 case"
        );
        if !diagnostic_collect_all {
            assert_eq!(
                verified_rejections, 3,
                "all independently verified rejections must execute"
            );
        }
    }
    if diagnostic_collect_all {
        std::fs::write(
            output_root.join("diagnostic-failures.txt"),
            diagnostic_failures.join("\n"),
        )
        .unwrap();
        assert!(
            diagnostic_failures.is_empty(),
            "diagnostic collection completed, NOT acceptance: {} actual failures; see {}",
            diagnostic_failures.len(),
            output_root.display()
        );
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
        let recorded_case = if case_observation::enabled() {
            format!("transfer/hubbard_chain_real/terms{size}")
        } else {
            String::new()
        };
        case_observation::begin(&recorded_case, "measured-activation");
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(&input).unwrap();
        assert!(!mvmc_core::get_all_complex_flag(&data).unwrap());
        threshold_transfer_input(&mut data, size);
        data.modpara.nvmc_sample = 4;
        data.modpara.nsr_opt_itr_step = 1;
        data.modpara.nsr_opt_itr_smp = 1;
        data.modpara.nstore_o = 0;
        data.modpara.nsrcg = 0;
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng).unwrap();
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
        if case_observation::enabled() {
            case_observation::retain_launch_seed(&format!("synthetic-transfer-{size}"), 1);
        }
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
                file_flush_interval: None,
                binary_output: false,
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
        case_observation::complete(&recorded_case, "measured-activation");
        eprintln!("actual site vmc_main_cal.jl:1707 real transfer terms={size}: {snapshot:?}");
        discrete(&format!("synthetic-transfer-{size}-rng"), &rng);
        emit_state(&format!("synthetic-transfer-{size}"), &data, &state);
    }
}
