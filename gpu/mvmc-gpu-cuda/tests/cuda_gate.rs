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
use mvmc_gpu_cuda::bench;

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
