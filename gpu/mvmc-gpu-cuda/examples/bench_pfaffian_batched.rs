//! Benchmark of the batched Pfaffian + inverse backends (issue #423).
//!
//! ```text
//! cd gpu/mvmc-gpu-cuda
//! cargo run --release --example bench_pfaffian_batched -- \
//!     [--quick] [--out results.csv] [--dtype f64|c64|both] [--max-bytes N]
//! ```
//!
//! Method: for every `(dtype, n, B)` with `NQP = 8`, the same skew-symmetric planes are fed to
//! every backend. Per backend: one warm-up, then up to 7 repetitions (stopping after a 12 s
//! budget, at least 3), reporting the median wall time of the whole call. CUDA timings include
//! host-to-device upload, kernel and device-to-host download, each phase ended by a stream
//! synchronization; kernel-only, upload and download parts are reported separately (one raw session and
//! one module load for the whole benchmark; NVRTC compile excluded, reported once). CPU rows use
//! `pfapack` with default (scalar) features: one thread and rayon over all cores. The
//! tensor-native and `ExtensionOp` rows run on tenferro `CpuBackend` (faer provider). A backend
//! whose single run would be unreasonably slow for a configuration is skipped (`-`).

use std::io::Write;
use std::time::Instant;

use mvmc_gpu::{pfaffian_inverse_batched, Backend, PfScalar};
use mvmc_gpu_cuda::pfaffian::{with_session, PfSession};
use num_complex::Complex64;

struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_planes<T: PfScalar>(n: usize, planes: usize, seed: u64) -> Vec<T> {
    let mut rng = Lcg(seed);
    let mut out = vec![T::zero_value(); n * n * planes];
    let w = T::lanes();
    let lanes = T::as_f64_lanes_mut(&mut out);
    for p in 0..planes {
        let base = p * n * n;
        for j in 0..n {
            for i in 0..j {
                for l in 0..w {
                    let v = rng.uniform();
                    lanes[(base + i + j * n) * w + l] = v;
                    lanes[(base + j + i * n) * w + l] = -v;
                }
            }
        }
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Warm up once, then repeat until 7 runs or a 12 s budget (min 3 runs); per-run closure
/// returns a vector of measured values (e.g. total, kernel, transfer); medians per column.
fn time_it<const K: usize>(mut f: impl FnMut() -> [f64; K]) -> [f64; K] {
    f();
    let mut samples: Vec<[f64; K]> = Vec::new();
    let start = Instant::now();
    while samples.len() < 7 && (samples.len() < 3 || start.elapsed().as_secs_f64() < 12.0) {
        samples.push(f());
    }
    let mut out = [0.0; K];
    for (k, o) in out.iter_mut().enumerate() {
        *o = median(samples.iter().map(|s| s[k]).collect());
    }
    out
}

fn wall(mut f: impl FnMut()) -> [f64; 1] {
    let t = Instant::now();
    f();
    [t.elapsed().as_secs_f64()]
}

fn fmt(v: Option<f64>) -> String {
    v.map_or("-".into(), |x| format!("{x:.4e}"))
}

#[allow(clippy::too_many_arguments)]
fn bench_one<T: PfScalar>(
    session: &PfSession<'_, '_>,
    n: usize,
    nqp: usize,
    batch: usize,
    max_bytes: usize,
) -> Option<String> {
    let count = nqp * batch;
    let complex = T::lanes() == 2;
    if n * n * count * std::mem::size_of::<T>() > max_bytes {
        return None;
    }
    let planes = random_planes::<T>(n, count, 1234 + n as u64);
    let work = (n as f64).powi(3) * count as f64;
    let run = |backend: &Backend<'_>| {
        wall(|| {
            let out = pfaffian_inverse_batched(backend, &planes, n, nqp, batch).unwrap();
            std::hint::black_box(&out);
        })[0]
    };
    let serial = time_it(|| [run(&Backend::CpuPfapack)])[0];
    let rayon = time_it(|| [run(&Backend::CpuPfapackRayon)])[0];
    let ext = (work <= 2.0e9).then(|| time_it(|| [run(&Backend::TenferroExtension)])[0]);
    let native = (work <= 3.5e7).then(|| time_it(|| [run(&Backend::TenferroNative)])[0]);

    let [cuda_total, cuda_kernel, cuda_up, cuda_alloc, cuda_down] = time_it(|| {
        let t = Instant::now();
        let (out, tm) = session.run_timed(&planes, n, count).unwrap();
        std::hint::black_box(&out);
        [
            t.elapsed().as_secs_f64(),
            tm.kernel_s,
            tm.upload_s,
            tm.alloc_s,
            tm.download_s,
        ]
    });
    // FP32 variant, timing only (the physics path is f64).
    let lanes32: Vec<f32> = T::as_f64_lanes(&planes).iter().map(|&x| x as f32).collect();
    let [f32_total, f32_kernel] = time_it(|| {
        let t = Instant::now();
        let (_, _, _, tm) = session.run_f32_timed(complex, &lanes32, n, count).unwrap();
        [t.elapsed().as_secs_f64(), tm.kernel_s]
    });
    Some(format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        if complex { "c64" } else { "f64" },
        n,
        batch,
        count,
        fmt(Some(serial)),
        fmt(Some(rayon)),
        fmt(native),
        fmt(ext),
        fmt(Some(cuda_total)),
        fmt(Some(cuda_kernel)),
        fmt(Some(cuda_up)),
        fmt(Some(cuda_alloc)),
        fmt(Some(cuda_down)),
        fmt(Some(f32_total)),
        fmt(Some(f32_kernel)),
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name);
    let quick = flag("--quick").is_some();
    let out_path = flag("--out").and_then(|i| args.get(i + 1)).cloned();
    let dtype = flag("--dtype")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or("both".into());
    let max_bytes: usize = flag("--max-bytes")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4 << 30);
    let ns: Vec<usize> = if quick {
        vec![16, 32]
    } else {
        vec![16, 32, 64, 128]
    };
    let batches: Vec<usize> = if quick {
        vec![1, 8, 64]
    } else {
        vec![1, 8, 64, 512, 4096]
    };
    let nqp = 8;

    mvmc_gpu_cuda::install();
    match mvmc_core::backend::device_report(mvmc_core::backend::BackendKind::Cuda(0)) {
        Ok(report) => println!("# {}", report.render().trim_end().replace('\n', "\n# ")),
        Err(e) => println!("# device report unavailable: {e}"),
    }
    println!(
        "# cores: {:?}, RAYON_NUM_THREADS={:?}",
        std::thread::available_parallelism().map(|n| n.get()),
        std::env::var("RAYON_NUM_THREADS").ok()
    );
    println!("# NQP = {nqp}; times are medians in seconds");
    let header = "dtype,n,B,planes,pfapack_1thread,pfapack_rayon,tenferro_native,tenferro_extop,cuda_total,cuda_kernel,cuda_upload,cuda_alloc,cuda_download,cuda_f32_total,cuda_f32_kernel";
    println!("{header}");
    let mut csv = out_path.map(|p| std::fs::File::create(p).unwrap());
    if let Some(f) = csv.as_mut() {
        writeln!(f, "{header}").unwrap();
    }
    with_session(0, |session| {
        let t0 = Instant::now();
        // Touch the module once so JIT/driver warm-up is not charged to the first row.
        let probe = random_planes::<f64>(16, 8, 1);
        session.run_timed(&probe, 16, 8)?;
        println!(
            "# first launch (JIT/driver warm-up included): {:.3} s",
            t0.elapsed().as_secs_f64()
        );
        for complex in [false, true] {
            if (dtype == "f64" && complex) || (dtype == "c64" && !complex) {
                continue;
            }
            for &n in &ns {
                for &b in &batches {
                    let row = if complex {
                        bench_one::<Complex64>(session, n, nqp, b, max_bytes)
                    } else {
                        bench_one::<f64>(session, n, nqp, b, max_bytes)
                    };
                    let line = row.unwrap_or_else(|| {
                        format!(
                            "{},{},{},{},skipped (exceeds --max-bytes),,,,,,,,,,",
                            if complex { "c64" } else { "f64" },
                            n,
                            b,
                            nqp * b
                        )
                    });
                    println!("{line}");
                    if let Some(f) = csv.as_mut() {
                        writeln!(f, "{line}").unwrap();
                    }
                }
            }
        }
        Ok(())
    })
    .expect("CUDA benchmark session");
}
