//! Family 4: host-device transfers (issue #432).
//!
//! Numerical verdict: data movement must be exact. Every path (cudarc pageable, cudarc pinned
//! through the `TransferStream` helper, tenferro `upload_tensor`/`download_tensor`) round-trips
//! a patterned buffer and the number of differing bits must be 0 (bound 0: this is a copy, not
//! arithmetic, so bitwise comparison is the right test here). Timings: per-call time versus
//! size, copy/kernel overlap and the two-group ping-pong, all as reference rows.

use cudarc::driver::CudaContext;
use mvmc_gpu_cuda::transfer::{PinnedBuf, PinnedKind, TransferStream};
use mvmc_gpu_cuda::transfer_bench as tb;
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::Tensor;

use crate::csv::{Csv, Row, Stat, Verdict};
use crate::Cfg;

const FAMILY: &str = "transfers";

fn pattern(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64) * 0.37 - 5.0 + f64::from_bits(0x3cb0_0000_0000_0000 ^ i as u64))
        .collect()
}

fn mismatches(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count() as f64
        + a.len().abs_diff(b.len()) as f64
}

fn integrity(
    csv: &mut Csv,
    ctx: &std::sync::Arc<CudaContext>,
    backend: &mut CudaBackend,
    bytes: usize,
) {
    let n = bytes / 8;
    let params = format!("bytes={bytes}");
    let src = pattern(n);
    let row = |v: &str| Row::new(FAMILY, "roundtrip_integrity", v, "f64", &params);
    let report = |csv: &mut Csv, v: &str, r: Result<Vec<f64>, String>| match r {
        Ok(back) => csv.push(row(v).dev("differing_elements", mismatches(&back, &src), 0.0)),
        Err(m) => csv.push(row(v).error(&m)),
    };
    // pageable
    let pageable = (|| -> Result<Vec<f64>, String> {
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let mut dev = stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
        stream
            .memcpy_htod(&src, &mut dev)
            .map_err(|e| e.to_string())?;
        let mut back = vec![0.0; n];
        stream
            .memcpy_dtoh(&dev, &mut back)
            .map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;
        Ok(back)
    })();
    report(csv, "cudarc-pageable", pageable);
    // pinned
    let pinned = (|| -> Result<Vec<f64>, String> {
        let ts = TransferStream::new(ctx).map_err(|e| e.to_string())?;
        let mut up =
            PinnedBuf::new(ctx, bytes, PinnedKind::WriteCombined).map_err(|e| e.to_string())?;
        up.as_f64_mut()[..n].copy_from_slice(&src);
        let mut dev = ts
            .stream()
            .alloc_zeros::<f64>(n)
            .map_err(|e| e.to_string())?;
        ts.upload_async(&up, &mut dev)
            .map_err(|e| e.to_string())?
            .wait()
            .map_err(|e| e.to_string())?;
        let mut down = PinnedBuf::new(ctx, bytes, PinnedKind::Cached).map_err(|e| e.to_string())?;
        down.as_f64_mut().fill(0.0);
        ts.download_async(&dev, &mut down)
            .map_err(|e| e.to_string())?
            .wait()
            .map_err(|e| e.to_string())?;
        Ok(down.as_f64()[..n].to_vec())
    })();
    report(csv, "cudarc-pinned", pinned);
    // tenferro
    let tf = (|| -> Result<Vec<f64>, String> {
        let t = Tensor::from_vec_col_major(vec![n], src.clone()).map_err(|e| e.to_string())?;
        let up = upload_tensor(backend.runtime(), &t).map_err(|e| e.to_string())?;
        backend.runtime().synchronize().map_err(|e| e.to_string())?;
        let back = download_tensor(backend.runtime(), &up).map_err(|e| e.to_string())?;
        Ok(back.as_slice::<f64>().map_err(|e| e.to_string())?.to_vec())
    })();
    report(csv, "tenferro", tf);
}

fn ms(v: f64) -> Stat {
    Stat {
        reps: 1,
        median: v * 1e-3,
        min: f64::NAN,
        max: f64::NAN,
    }
}

pub fn run(cfg: &Cfg, csv: &mut Csv) {
    if cfg.cpu_only {
        csv.push(Row::new(FAMILY, "all", "-", "-", "").verdict(Verdict::Skipped, "cpu-only pass"));
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            csv.push(Row::new(FAMILY, "all", "-", "-", "").error(&format!("CUDA context: {e}")));
            return;
        }
    };
    // SAFETY: the benchmark orders cross-stream work with explicit events and synchronization.
    unsafe { mvmc_gpu_cuda::transfer::disable_event_tracking(&ctx) };
    let devs = match tenferro_gpu::cuda::cuda_devices() {
        Ok(d) => d,
        Err(e) => {
            csv.push(Row::new(FAMILY, "all", "-", "-", "").error(&e.to_string()));
            return;
        }
    };
    let mut backend = match CudaBackend::new(devs[0].id()) {
        Ok(b) => b,
        Err(e) => {
            csv.push(Row::new(FAMILY, "all", "-", "-", "").error(&e.to_string()));
            return;
        }
    };
    let sizes: &[usize] = if cfg.full {
        &[
            8,
            512,
            4 << 10,
            32 << 10,
            128 << 10,
            1 << 20,
            4 << 20,
            16 << 20,
            64 << 20,
            256 << 20,
        ]
    } else {
        &[512, 32 << 10, 1 << 20, 16 << 20]
    };
    for &b in sizes {
        integrity(csv, &ctx, &mut backend, b);
    }
    if let Ok(info) = tb::engine_info(&ctx) {
        csv.push(
            Row::new(FAMILY, "copy_engines", "device", "-", "").note(&format!(
                "asyncEngineCount={} concurrentKernels={} multiprocessors={} pci=({})",
                info.async_engine_count, info.concurrent_kernels, info.multiprocessors, info.pci
            )),
        );
    }
    match tb::sweep(&ctx, &mut backend, sizes) {
        Ok(rows) => {
            for r in rows {
                let params = format!("bytes={}", r.bytes);
                for (name, v) in [
                    ("pageable_up", r.pageable_up),
                    ("pageable_down", r.pageable_down),
                    ("pinned_wc_up", r.pinned_up),
                    ("pinned_cached_up", r.pinned_up_cached),
                    ("pinned_down", r.pinned_down),
                    ("pinned_up_enqueue", r.pinned_up_enqueue),
                    ("tenferro_up", r.tenferro_up),
                    ("tenferro_down", r.tenferro_down),
                ] {
                    csv.push(
                        Row::new(FAMILY, "transfer_time", name, "f64", &params)
                            .stat(ms(v))
                            .note(&format!(
                                "{:.2} GB/s; median only",
                                r.bytes as f64 / (v * 1e-3) / 1e9
                            )),
                    );
                }
            }
        }
        Err(m) => csv.push(Row::new(FAMILY, "transfer_time", "sweep", "-", "").error(&m)),
    }
    let ob = if cfg.full { 16 << 20 } else { 4 << 20 };
    match tb::overlap(&ctx, ob) {
        Ok(rows) => {
            for r in rows {
                let params = format!("bytes={ob};scenario={}", r.label);
                csv.push(
                    Row::new(FAMILY, "copy_kernel_overlap", "concurrent", "-", &params)
                        .stat(ms(r.both_ms))
                        .note(&format!(
                            "copy_ms={:.3} kernel_ms={:.3} overlap_fraction={:.2}",
                            r.copy_ms,
                            r.kernel_ms,
                            r.overlap()
                        )),
                );
            }
        }
        Err(m) => csv.push(Row::new(FAMILY, "copy_kernel_overlap", "-", "-", "").error(&m)),
    }
    let cfgs: &[(usize, usize, f64, u64, usize)] = if cfg.full {
        &[
            (4 << 10, 4 << 10, 0.05, 20, 400),
            (4 << 10, 64 << 10, 0.3, 100, 200),
            (4 << 10, 64 << 10, 1.0, 300, 200),
            (64 << 10, 1 << 20, 1.0, 200, 100),
        ]
    } else {
        &[(4 << 10, 64 << 10, 0.3, 100, 100)]
    };
    for &(i, o, k, h, steps) in cfgs {
        let c = tb::PingPongConfig {
            in_bytes: i,
            out_bytes: o,
            kernel_ms: k,
            host_us: h,
            steps,
        };
        let params = format!("in={i};out={o};kernel_ms={k};host_us={h};steps={steps}");
        match tb::pingpong(&ctx, c) {
            Ok(r) => {
                for (name, v) in [
                    ("serial-pageable", r.serial_pageable_ms),
                    ("serial-pinned", r.serial_pinned_ms),
                    ("pingpong-two-streams", r.pingpong_ms),
                ] {
                    csv.push(
                        Row::new(FAMILY, "pingpong", name, "-", &params)
                            .stat(ms(v))
                            .note("total ms for all steps; median only"),
                    );
                }
            }
            Err(m) => csv.push(Row::new(FAMILY, "pingpong", "-", "-", &params).error(&m)),
        }
    }
}
