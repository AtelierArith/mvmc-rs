//! Host-device transfer gate (issue #432). Ignored by default; same selector semantics as
//! `cuda_gate.rs`: `MVMC_RS_CUDA_GATE=1` requires a device (hard failure without one), any other
//! value reports an explicit "skipped, no device".
//!
//! `MVMC_RS_CUDA_GATE_TRANSFER_OUT` writes the report.

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, device_report, BackendKind, CudaGateDecision,
    CUDA_GATE_VARIABLE,
};
use mvmc_gpu_cuda::transfer::{PinnedKind, PinnedPool, TransferStream};
use mvmc_gpu_cuda::transfer_bench as tb;

/// `true` when the gate should run; `false` for an explicit skip; panics when required.
fn gate() -> bool {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("cuda-gate: ExplicitSkip: skipped, no device ({why})");
            false
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("cuda-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}");
        }
        CudaGateDecision::Run => true,
    }
}

#[test]
#[ignore = "optional CUDA gate: run with --ignored (MVMC_RS_CUDA_GATE=1 to require a device)"]
fn transfer_helper_is_correct_and_orders_streams() {
    if !gate() {
        return;
    }
    let ctx = cudarc::driver::CudaContext::new(0).expect("context");
    // SAFETY: the helper orders every cross-stream dependency with explicit events.
    unsafe { mvmc_gpu_cuda::transfer::disable_event_tracking(&ctx) };
    let pool = PinnedPool::new(&ctx);
    let ts = TransferStream::new(&ctx).expect("stream");
    let n = 4096;
    let mut up = pool
        .take(n * 8, PinnedKind::WriteCombined)
        .expect("pinned up");
    for (i, v) in up.as_f64_mut().iter_mut().enumerate() {
        *v = i as f64 * 0.5 - 3.0;
    }
    let mut dev = ts.stream().alloc_zeros::<f64>(n).expect("device buffer");
    ts.upload_async(&up, &mut dev)
        .expect("upload")
        .wait()
        .expect("wait");
    let mut down = pool.take(n * 8, PinnedKind::Cached).expect("pinned down");
    ts.download_async(&dev, &mut down)
        .expect("download")
        .wait()
        .expect("wait");
    for (i, v) in down.as_f64()[..n].iter().enumerate() {
        assert_eq!(*v, i as f64 * 0.5 - 3.0, "element {i}");
    }
    // pool reuse: no new allocation for the same size class
    pool.put(up);
    pool.put(down);
    assert_eq!(pool.idle(), 2);
    let again = pool.take(n * 8, PinnedKind::WriteCombined).expect("reuse");
    assert_eq!(pool.idle(), 1);
    pool.put(again);
    // a second stream waits on an event of the first (device-side dependency, no host sync)
    let other = TransferStream::new(&ctx).expect("second stream");
    let mut down2 = pool.take(n * 8, PinnedKind::Cached).expect("pinned");
    let ev = ts.record().expect("event");
    other.wait_event(&ev).expect("wait on event");
    other
        .download_async(&dev, &mut down2)
        .expect("download")
        .wait()
        .expect("wait");
    assert_eq!(down2.as_f64()[17], 17.0 * 0.5 - 3.0);
    ev.wait_host().expect("event");
    assert!(ev.is_complete());
}

#[test]
#[ignore = "optional CUDA gate: run with --ignored (MVMC_RS_CUDA_GATE=1 to require a device)"]
fn transfer_microbenchmark_report() {
    if !gate() {
        return;
    }
    let device = device_report(BackendKind::Cuda(0)).expect("device report");
    let ctx = cudarc::driver::CudaContext::new(0).expect("context");
    // SAFETY: the benchmark orders cross-stream work with explicit events and synchronization.
    unsafe { mvmc_gpu_cuda::transfer::disable_event_tracking(&ctx) };
    let info = tb::engine_info(&ctx).expect("engine info");
    let mut backend = tenferro_gpu::cuda::CudaBackend::new(
        tenferro_gpu::cuda::cuda_devices()
            .expect("devices")
            .into_iter()
            .next()
            .expect("device 0")
            .id(),
    )
    .expect("tenferro backend");

    let sizes = [
        8usize,
        512,
        4 << 10,
        32 << 10,
        128 << 10,
        1 << 20,
        4 << 20,
        16 << 20,
        64 << 20,
        256 << 20,
    ];
    let sweep = tb::sweep(&ctx, &mut backend, &sizes).expect("sweep");

    let overlap_bytes = 16 << 20;
    let overlap = tb::overlap(&ctx, overlap_bytes).expect("overlap");

    let mut pp = Vec::new();
    for cfg in [
        // latency-bound: tiny work per step
        (4 << 10, 4 << 10, 0.05, 20u64, 400usize),
        // small walker batch: configurations in, pf/energies out
        (4 << 10, 64 << 10, 0.3, 100, 200),
        (4 << 10, 64 << 10, 1.0, 300, 200),
        // larger O-vector output
        (64 << 10, 1 << 20, 1.0, 200, 100),
    ] {
        let cfg = tb::PingPongConfig {
            in_bytes: cfg.0,
            out_bytes: cfg.1,
            kernel_ms: cfg.2,
            host_us: cfg.3,
            steps: cfg.4,
        };
        pp.push(tb::pingpong(&ctx, cfg).expect("pingpong"));
    }

    let kernel_ms = overlap.first().map_or(0.0, |r| r.kernel_ms);
    let text = format!(
        "## Transfer micro-benchmark (issue #432)\n\n```\n{}```\n\n\
         asyncEngineCount={} concurrentKernels={} multiprocessors={} pci=({})\n\n\
         ### Per-call time and bandwidth vs size (median)\n\n{}\n\
         ### Copy/kernel overlap\n\n{}\n\
         ### Ping-pong of two walker groups\n\n{}\n",
        device.render(),
        info.async_engine_count,
        info.concurrent_kernels,
        info.multiprocessors,
        info.pci,
        tb::render_sweep(&sweep),
        tb::render_overlap(&overlap, overlap_bytes, kernel_ms),
        tb::render_pingpong(&pp),
    );
    println!("{text}");
    if let Ok(path) = std::env::var("MVMC_RS_CUDA_GATE_TRANSFER_OUT") {
        std::fs::write(path, &text).expect("write report");
    }
    assert!(info.async_engine_count >= 1);
    assert!(sweep
        .iter()
        .all(|r| r.pinned_up > 0.0 && r.tenferro_up > 0.0));
}
