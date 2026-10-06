//! Optional CUDA gate (issue #420). Ignored by default.
//!
//! * `MVMC_RS_CUDA_GATE=1` requests the gate: no device is a hard failure.
//! * Unset (or any other value) with no device: explicit "skipped, no device", never a pass
//!   claim. With a device the gate always runs.
//!
//! Optional environment: `MVMC_RS_CUDA_GATE_SIZES` (comma list, default `256,1024`),
//! `MVMC_RS_CUDA_GATE_REPS` (default 5), `MVMC_RS_CUDA_GATE_OUT` (write the report there).

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, device_report, BackendKind, CudaGateDecision,
    CUDA_GATE_VARIABLE,
};
use mvmc_gpu_cuda::{bench, sr_bench};

#[test]
#[ignore = "optional CUDA gate: run with --ignored (MVMC_RS_CUDA_GATE=1 to require a device)"]
fn cuda_gate_dot_general_and_cholesky_match_cpu() {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("cuda-gate: ExplicitSkip: skipped, no device ({why})");
            return;
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("cuda-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}");
        }
        CudaGateDecision::Run => {}
    }

    let report = device_report(BackendKind::Cuda(0)).expect("device report");
    let sizes: Vec<usize> = std::env::var("MVMC_RS_CUDA_GATE_SIZES")
        .unwrap_or_else(|_| "256,1024".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("size"))
        .collect();
    let reps = std::env::var("MVMC_RS_CUDA_GATE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let rows = bench::run_all(0, &sizes, reps).expect("bench");
    let text = format!(
        "## CUDA gate report\n\n```\n{}```\n\nmedian of {reps} runs after 1 warm-up, milliseconds; \
         tolerances: dot {:.0e}, cholesky {:.0e} (relative max-norm vs tenferro CPU)\n\n{}",
        report.render(),
        bench::TOL_DOT,
        bench::TOL_CHOLESKY,
        bench::render(&rows)
    );
    println!("{text}");
    if let Ok(path) = std::env::var("MVMC_RS_CUDA_GATE_OUT") {
        std::fs::write(path, &text).expect("write report");
    }
    for r in &rows {
        assert!(
            r.ok(),
            "{:?} {:?} n={} rel_err {:.3e} > {:.0e}",
            r.op,
            r.dtype,
            r.n,
            r.rel_err,
            r.tol
        );
    }
}

/// Validation harness (issue #424) on the CUDA backend: teacher-forced replay against the
/// C-order CPU oracle, decision recording, 20-step repeatability and the metadata block.
#[test]
#[ignore = "optional CUDA gate: run with --ignored (MVMC_RS_CUDA_GATE=1 to require a device)"]
fn cuda_gate_validation_harness_matches_c_order_oracle() {
    use mvmc_core::accel_validation::{
        bench_stages, repeatability, replay, AcceleratedStages, BenchMetadata, ReplayConfig,
        DEFAULT_ITERATIONS, DEFAULT_WARMUPS,
    };
    use mvmc_gpu_cuda::stages::EagerStages;

    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("cuda-gate: ExplicitSkip: skipped, no device ({why})");
            return;
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("cuda-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}");
        }
        CudaGateDecision::Run => {}
    }
    let device = device_report(BackendKind::Cuda(0)).expect("device report");
    let cfg = ReplayConfig::default();
    let mut cuda = EagerStages::cuda(0).expect("cuda stages");
    let rep = replay(&mut cuda, &cfg).expect("replay");
    let (pf_ms, sr_ms) = bench_stages(&mut cuda, &cfg, DEFAULT_WARMUPS, DEFAULT_ITERATIONS);
    let meta = BenchMetadata::collect(
        &cuda.provider(),
        "f64",
        1,
        DEFAULT_WARMUPS,
        DEFAULT_ITERATIONS,
        true,
        Some(&device),
    );
    let text = format!(
        "## CUDA validation harness\n\n```\n{}```\n\n{}\nSR stage median {}; Pfaffian stage {}\n",
        meta.render(),
        rep.render(),
        sr_ms.map_or("unsupported".to_string(), |v| format!("{v:.3} ms")),
        pf_ms.map_or("unsupported".to_string(), |v| format!("{v:.3} ms")),
    );
    println!("{text}");
    if let Ok(path) = std::env::var("MVMC_RS_CUDA_GATE_VALIDATION_OUT") {
        std::fs::write(path, &text).expect("write report");
    }
    assert!(rep.violations().is_empty(), "{:?}", rep.violations());
    assert!(rep.s.compared > 0 && rep.g.compared > 0);
    let short = ReplayConfig { steps: 20, ..cfg };
    repeatability(|| EagerStages::cuda(0).expect("cuda stages"), &short).expect("repeatability");
}

/// Device round-trip floor (issue #425): launch/sync latency and pageable transfer bandwidth
/// per buffer size. This is the cost an accelerated stage must amortize over many walkers.
#[test]
#[ignore = "optional CUDA gate: run with --ignored (MVMC_RS_CUDA_GATE=1 to require a device)"]
fn cuda_gate_roundtrip_floor() {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("cuda-gate: ExplicitSkip: skipped, no device ({why})");
            return;
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("cuda-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}");
        }
        CudaGateDecision::Run => {}
    }
    let device = device_report(BackendKind::Cuda(0)).expect("device report");
    let ctx = bench::cuda_runtime(0).expect("cuda runtime");
    let sizes = [1usize, 1 << 10, 1 << 13, 1 << 16, 1 << 19, 1 << 22, 1 << 24];
    let rows: Vec<_> = sizes
        .iter()
        .map(|&n| bench::roundtrip(&ctx, n, 5, 30).expect("roundtrip"))
        .collect();
    let text = format!(
        "## CUDA round-trip floor\n\n```\n{}```\n\nmedian of 30 after 5 warm-ups; pageable host \
         memory; upload and download each synchronized\n\n{}",
        device.render(),
        bench::render_roundtrip(&rows)
    );
    println!("{text}");
    if let Ok(path) = std::env::var("MVMC_RS_CUDA_GATE_ROUNDTRIP_OUT") {
        std::fs::write(path, &text).expect("write report");
    }
    assert!(rows.iter().all(|r| r.total_ms() > 0.0));
}

#[test]
#[ignore = "optional CUDA gate: run with --ignored (MVMC_RS_CUDA_GATE=1 to require a device)"]
fn cuda_gate_sr_stages_match_c_order() {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("cuda-gate: ExplicitSkip: skipped, no device ({why})");
            return;
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("cuda-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}");
        }
        CudaGateDecision::Run => {}
    }
    let report = device_report(BackendKind::Cuda(0)).expect("device report");
    let sizes: Vec<usize> = std::env::var("MVMC_RS_CUDA_GATE_SR_SIZES")
        .unwrap_or_else(|_| "388,1000".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("size"))
        .collect();
    let reps = std::env::var("MVMC_RS_CUDA_GATE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let mut rows = Vec::new();
    for n in sizes {
        // Two sample counts: the benchmark default 300 and the large-sample regime 2 n.
        for samples in [300, 2 * n] {
            rows.push(sr_bench::run_case(0, n, samples, reps).expect("SR stages"));
        }
    }
    let text = format!(
        "## CUDA gate: SR stages (issue #421)\n\n```\n{}```\n\nmedian of {reps} runs after 1 warm-up, \
         milliseconds per stage including upload, synchronized compute and download; \
         relative max-norm difference against the C-order BLAS/LAPACK result.\n\n{}",
        report.render(),
        sr_bench::render(&rows)
    );
    println!("{text}");
    if let Ok(path) = std::env::var("MVMC_RS_CUDA_GATE_OUT") {
        let sr_path = std::path::Path::new(&path).with_file_name("cuda-gate-sr.md");
        std::fs::write(sr_path, &text).expect("write SR report");
    }
    for r in &rows {
        assert!(
            r.ok(),
            "SR stages out of bound at n={} samples={}",
            r.n,
            r.samples
        );
    }
}
