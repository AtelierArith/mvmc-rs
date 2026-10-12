//! Kernel-only roofline of the batched Pfaffian recompute (issue #517).
//!
//! Times `pfinv_f64` (the recompute hot kernel of `pfaffian_batched.cu`, same math as
//! `crates/pfapack`) alone, with CUDA events, for a range of plane counts at `n = 256` and
//! `n = 448` (`NQP = 8`, so `planes = 8 * recompute items`). Reports the achieved FP64
//! TFLOP/s against the analytic operation count of the kernel (the same count as
//! `bench-out/issue517-roofline-20261011/flops.py`; `fma = 2` flops) and the A100 FP64 ALU
//! peak 9.7 TFLOP/s.
//!
//! ```text
//! cd gpu/mvmc-gpu-cuda
//! cargo run --release --example bench_recompute_roofline
//! ```

use std::sync::Arc;

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const SRC_PFAFFIAN: &str = include_str!("../src/pfaffian_batched.cu");

/// FP64 operations of one `pfinv_plane` (mul/add/div/negate = 1, fma = 2).
fn plane_flops(n: usize) -> u64 {
    let mut f: u64 = 0;
    for k0 in (1..n).rev() {
        let kk = k0 - 1;
        if kk >= 1 {
            f += (kk * (kk - 1) / 2) as u64 * 6;
            f += kk as u64;
            f += 1;
        }
    }
    f += (n / 2) as u64;
    let mut back: u64 = 0;
    for c in 0..n.saturating_sub(1) {
        back += (c * (c + 1) / 2) as u64;
    }
    f += 2 * back;
    let mut fwd = 0u64;
    let mut i = 2usize;
    while i < n {
        fwd += 1;
        i += 2;
    }
    let mut bwd = 0u64;
    if n >= 3 {
        let mut i = (n - 3) as isize;
        while i >= 1 {
            bwd += 1;
            i -= 2;
        }
    }
    f += (fwd + bwd) * 3 * n as u64;
    f += (n as u64) * (n * (n + 1) / 2) as u64 * 2;
    f
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Deterministic LCG (same values as `bench_pfaffian_batched`).
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

/// Random skew-symmetric planes, column major, dense (no zero pivot expected).
fn random_skew(n: usize, planes: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut out = vec![0.0f64; n * n * planes];
    for p in 0..planes {
        let base = p * n * n;
        for j in 0..n {
            for i in 0..j {
                let v = rng.uniform();
                out[base + i + j * n] = v;
                out[base + j + i * n] = -v;
            }
        }
    }
    out
}

struct Engine {
    ctx: Arc<CudaContext>,
    pfinv: CudaFunction,
    pfinv_wide: CudaFunction,
}

impl Engine {
    fn new() -> Self {
        let ctx = CudaContext::new(0).expect("device");
        let ptx = compile_ptx_with_opts(
            SRC_PFAFFIAN,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .expect("NVRTC");
        let module = ctx.load_module(ptx).expect("module");
        // Wide-block variant: the fast path is latency-bound when only a few planes are in
        // flight, so the block-size cap of the production entry point is relaxed here.
        let wide_src =
            SRC_PFAFFIAN.replace("__launch_bounds__(256, 2)", "__launch_bounds__(1024, 1)");
        let wptx = compile_ptx_with_opts(
            &wide_src,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .expect("NVRTC wide");
        let wmodule = ctx.load_module(wptx).expect("wide module");
        Self {
            ctx,
            pfinv: module.load_function("pfinv_f64").expect("pfinv_f64"),
            pfinv_wide: wmodule.load_function("pfinv_f64").expect("pfinv_f64 wide"),
        }
    }

    /// Kernel-only seconds for `planes` planes of side `n`; median of `reps`. `wide` selects the
    /// relaxed-launch-bounds entry point.
    fn time_with(&self, n: usize, planes: usize, reps: usize, wide: bool) -> (f64, usize) {
        let pfinv = if wide { &self.pfinv_wide } else { &self.pfinv };
        let force_nt: Option<u32> = std::env::var("MVMC_PFINV_NT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let s = self.ctx.default_stream();
        let nn = n * n;
        let host_a = random_skew(n, planes, 0x5170_0000 + n as u64);
        let mut a = s.alloc_zeros::<f64>(planes * nn).expect("A");
        let mut w = s.alloc_zeros::<f64>(planes * nn).expect("W");
        let mut out = s.alloc_zeros::<f64>(planes * nn).expect("OUT");
        let mut pf = s.alloc_zeros::<f64>(planes).expect("PF");
        let mut st = s.alloc_zeros::<i32>(planes).expect("ST");
        let n_i = n as i32;
        let (nt, shared) = mvmc_gpu_cuda::pfaffian::launch_geometry(n, 8, 1);
        let nt = force_nt.unwrap_or(nt);
        macro_rules! refresh_and_run {
            ($f:expr) => {{
                {
                    let mut view = a.slice_mut(..host_a.len());
                    s.memcpy_htod(&host_a, &mut view).expect("htod");
                }
                let mut b = s.launch_builder($f);
                b.arg(&mut a)
                    .arg(&mut w)
                    .arg(&mut out)
                    .arg(&mut pf)
                    .arg(&mut st)
                    .arg(&n_i);
                // SAFETY: kernel ABI (double* x4, int*, int); buffers sized for `planes` planes.
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (planes as u32, 1, 1),
                        block_dim: (nt, 1, 1),
                        shared_mem_bytes: shared,
                    })
                }
                .expect("launch");
            }};
        }
        let start = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");
        let end = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");
        refresh_and_run!(pfinv);
        s.synchronize().expect("sync");
        let status = s.clone_dtoh(&st).expect("status");
        let failed = status.iter().filter(|&&v| v != 0).count();
        let mut times = Vec::new();
        for _ in 0..reps {
            {
                let mut view = a.slice_mut(..host_a.len());
                s.memcpy_htod(&host_a, &mut view).expect("htod");
            }
            start.record(&s).expect("record");
            let mut b = s.launch_builder(pfinv);
            b.arg(&mut a)
                .arg(&mut w)
                .arg(&mut out)
                .arg(&mut pf)
                .arg(&mut st)
                .arg(&n_i);
            // SAFETY: see `refresh_and_run!`.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (planes as u32, 1, 1),
                    block_dim: (nt, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .expect("launch");
            end.record(&s).expect("record");
            s.synchronize().expect("sync");
            times.push(f64::from(start.elapsed_ms(&end).unwrap_or(0.0)) * 1e-3);
        }
        (median(times), failed)
    }
}

fn main() {
    mvmc_gpu_cuda::install();
    let engine = Engine::new();
    println!("# pfinv_f64 kernel-only roofline (issue #517); A100 FP64 ALU peak 9.7 TFLOP/s");
    println!("n,planes,items,ms,tflops,fraction_of_peak,us_per_item,failed_planes");
    let sizes: Vec<usize> = match std::env::var("MVMC_PFINV_N") {
        Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![256, 448],
    };
    let plan_list: Vec<usize> = match std::env::var("MVMC_PFINV_PLANES") {
        Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![8, 64, 512, 2048],
    };
    let wide = std::env::var("MVMC_PFINV_WIDE").is_ok();
    for n in sizes {
        let flops = plane_flops(n);
        for planes in plan_list.iter().copied() {
            let _ = wide;
            let (secs, failed) = engine.time_with(n, planes, 5, wide);
            let items = planes / 8;
            let tflops = planes as f64 * flops as f64 / secs / 1e12;
            println!(
                "{n},{planes},{items},{:.3},{tflops:.3},{:.3},{:.1},{failed}",
                secs * 1e3,
                tflops / 9.7,
                secs * 1e6 / items as f64
            );
        }
    }
}
