//! Host-device transfer micro-benchmarks (issue #432).
//!
//! 1. cudarc pinned memcpy (sync, and async enqueue cost),
//! 2. cudarc pageable memcpy,
//! 3. tenferro `upload_tensor` / `download_tensor`,
//!
//! each as per-call fixed cost and bandwidth versus size; plus copy/kernel overlap (pinned and
//! pageable), the copy-engine count, and a two-stream ping-pong of walker groups.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{
    sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DriverError, LaunchConfig,
    PushKernelArg,
};
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::Tensor;

use crate::transfer::{PinnedBuf, PinnedKind, TransferStream};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn drv(e: DriverError) -> String {
    e.to_string()
}

/// Copy-engine and device facts recorded with the results.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// `CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT` (`cudaDeviceProp::asyncEngineCount`).
    pub async_engine_count: i32,
    /// `CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`.
    pub concurrent_kernels: i32,
    /// `CU_DEVICE_ATTRIBUTE_PCI_BUS_ID` and device ids.
    pub pci: String,
    /// Multiprocessor count.
    pub multiprocessors: i32,
}

/// Query [`EngineInfo`].
pub fn engine_info(ctx: &Arc<CudaContext>) -> Result<EngineInfo, String> {
    use sys::CUdevice_attribute_enum as A;
    let a = |attr| ctx.attribute(attr).map_err(drv);
    Ok(EngineInfo {
        async_engine_count: a(A::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?,
        concurrent_kernels: a(A::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)?,
        pci: format!(
            "domain {} bus {} device {}",
            a(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
            a(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
            a(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?
        ),
        multiprocessors: a(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
    })
}

/// One size point of the transfer sweep (medians, milliseconds).
#[derive(Debug, Clone, Copy, Default)]
pub struct SweepRow {
    /// Bytes per transfer.
    pub bytes: usize,
    /// cudarc pageable host-to-device, synchronized.
    pub pageable_up: f64,
    /// cudarc pageable device-to-host, synchronized.
    pub pageable_down: f64,
    /// cudarc pinned (write-combined) host-to-device, synchronized.
    pub pinned_up: f64,
    /// cudarc pinned (cached) device-to-host, synchronized.
    pub pinned_down: f64,
    /// cudarc pinned (cached, not write-combined) host-to-device, synchronized.
    pub pinned_up_cached: f64,
    /// Host time to enqueue one pinned host-to-device copy (does not wait).
    pub pinned_up_enqueue: f64,
    /// tenferro `upload_tensor`.
    pub tenferro_up: f64,
    /// tenferro `download_tensor`.
    pub tenferro_down: f64,
}

fn gbps(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}

fn reps_for(bytes: usize) -> usize {
    (256 << 20) / bytes.max(1) / 4 + 5
}

/// Run the sweep for the given byte sizes (multiples of 8).
pub fn sweep(
    ctx: &Arc<CudaContext>,
    backend: &mut CudaBackend,
    sizes: &[usize],
) -> Result<Vec<SweepRow>, String> {
    let stream = ctx.new_stream().map_err(drv)?;
    let ts = TransferStream::new(ctx).map_err(drv)?;
    let mut rows = Vec::new();
    for &bytes in sizes {
        let n = bytes / 8;
        let reps = reps_for(bytes).min(60);
        let warm = 3;
        let mut dev: CudaSlice<f64> = unsafe { stream.alloc(n) }.map_err(drv)?;
        let mut host = vec![1.0_f64; n];
        let mut pin_up = PinnedBuf::new(ctx, bytes, PinnedKind::WriteCombined).map_err(drv)?;
        pin_up.as_f64_mut().fill(1.0);
        let mut pin_down = PinnedBuf::new(ctx, bytes, PinnedKind::Cached).map_err(drv)?;
        pin_down.as_f64_mut().fill(1.0);
        let tensor =
            Tensor::from_vec_col_major(vec![n], vec![1.0_f64; n]).map_err(|e| e.to_string())?;
        let mut t = [const { Vec::new() }; 8];
        let mut gdev = None;
        for i in 0..warm + reps {
            let rec = i >= warm;
            let start = Instant::now();
            stream.memcpy_htod(&host, &mut dev).map_err(drv)?;
            stream.synchronize().map_err(drv)?;
            if rec {
                t[0].push(ms(start));
            }
            let start = Instant::now();
            stream.memcpy_dtoh(&dev, &mut host).map_err(drv)?;
            stream.synchronize().map_err(drv)?;
            if rec {
                t[1].push(ms(start));
            }
            let start = Instant::now();
            ts.upload_async(&pin_up, &mut dev)
                .map_err(drv)?
                .wait()
                .map_err(drv)?;
            if rec {
                t[2].push(ms(start));
            }
            let start = Instant::now();
            ts.download_async(&dev, &mut pin_down)
                .map_err(drv)?
                .wait()
                .map_err(drv)?;
            if rec {
                t[3].push(ms(start));
            }
            let start = Instant::now();
            let pending = ts.upload_async(&pin_up, &mut dev).map_err(drv)?;
            if rec {
                t[4].push(ms(start));
            }
            pending.wait().map_err(drv)?;
            let start = Instant::now();
            ts.upload_async(&pin_down, &mut dev)
                .map_err(drv)?
                .wait()
                .map_err(drv)?;
            if rec {
                t[7].push(ms(start));
            }
            let start = Instant::now();
            let up = upload_tensor(backend.runtime(), &tensor).map_err(|e| e.to_string())?;
            backend.runtime().synchronize().map_err(|e| e.to_string())?;
            if rec {
                t[5].push(ms(start));
            }
            let start = Instant::now();
            let _back = download_tensor(backend.runtime(), &up).map_err(|e| e.to_string())?;
            if rec {
                t[6].push(ms(start));
            }
            gdev = Some(up);
        }
        drop(gdev);
        let mut m = |i: usize| median(&mut t[i]);
        rows.push(SweepRow {
            bytes,
            pageable_up: m(0),
            pageable_down: m(1),
            pinned_up: m(2),
            pinned_down: m(3),
            pinned_up_cached: m(7),
            pinned_up_enqueue: m(4),
            tenferro_up: m(5),
            tenferro_down: m(6),
        });
    }
    Ok(rows)
}

/// Markdown table of a sweep: time per call and bandwidth.
pub fn render_sweep(rows: &[SweepRow]) -> String {
    let mut s = String::from(
        "| bytes | pageable up ms (GB/s) | pageable down ms (GB/s) | pinned WC up ms (GB/s) | pinned cached up ms (GB/s) | pinned down ms (GB/s) | pinned up enqueue ms | tenferro up ms (GB/s) | tenferro down ms (GB/s) |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        let c = |t: f64| format!("{:.4} ({:.2})", t, gbps(r.bytes, t));
        s += &format!(
            "| {} | {} | {} | {} | {} | {} | {:.4} | {} | {} |\n",
            r.bytes,
            c(r.pageable_up),
            c(r.pageable_down),
            c(r.pinned_up),
            c(r.pinned_up_cached),
            c(r.pinned_down),
            r.pinned_up_enqueue,
            c(r.tenferro_up),
            c(r.tenferro_down)
        );
    }
    s
}

const SPIN_SRC: &str = r#"
extern "C" __global__ void spin(double* x, long iters) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    double v = x[i % 1024] + (double)i;
    for (long k = 0; k < iters; ++k) { v = v * 1.0000001 + 1e-9; }
    x[i % 1024] = v;
}
"#;

/// A calibrated compute kernel.
pub struct SpinKernel {
    func: cudarc::driver::CudaFunction,
    scratch: CudaSlice<f64>,
    grid: u32,
    /// Iterations that make one launch take about `target_ms`.
    pub iters: i64,
    /// Measured time of one launch with `iters` (ms).
    pub measured_ms: f64,
}

impl SpinKernel {
    /// Compile the kernel and calibrate its run time to about `target_ms`.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        target_ms: f64,
    ) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx(SPIN_SRC).map_err(|e| e.to_string())?;
        let module = ctx.load_module(ptx).map_err(drv)?;
        let func = module.load_function("spin").map_err(drv)?;
        let scratch = stream.alloc_zeros::<f64>(1024).map_err(drv)?;
        let sm = ctx
            .attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(drv)? as u32;
        let mut k = Self {
            func,
            scratch,
            grid: sm * 4,
            iters: 1000,
            measured_ms: 0.0,
        };
        let probe = k.time(stream, 20_000)?;
        k.iters = ((20_000.0 * target_ms / probe) as i64).max(1);
        k.measured_ms = k.time(stream, k.iters)?;
        Ok(k)
    }

    fn launch(&mut self, stream: &Arc<CudaStream>, iters: i64) -> Result<(), String> {
        let cfg = LaunchConfig {
            grid_dim: (self.grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.func);
        b.arg(&mut self.scratch);
        b.arg(&iters);
        // SAFETY: argument types match the kernel signature `(double*, long)`.
        unsafe { b.launch(cfg) }.map_err(drv)?;
        Ok(())
    }

    fn time(&mut self, stream: &Arc<CudaStream>, iters: i64) -> Result<f64, String> {
        self.launch(stream, iters)?; // warm-up (module load)
        stream.synchronize().map_err(drv)?;
        let mut t = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            self.launch(stream, iters)?;
            stream.synchronize().map_err(drv)?;
            t.push(ms(start));
        }
        Ok(median(&mut t))
    }

    /// Enqueue one launch with the calibrated iteration count.
    pub fn enqueue(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        let iters = self.iters;
        self.launch(stream, iters)
    }
}

/// Overlap of one copy kind with a concurrent kernel.
#[derive(Debug, Clone)]
pub struct OverlapRow {
    /// Scenario label.
    pub label: String,
    /// Copy alone (ms).
    pub copy_ms: f64,
    /// Kernel alone (ms).
    pub kernel_ms: f64,
    /// Both concurrently on two streams (ms).
    pub both_ms: f64,
}

impl OverlapRow {
    /// Overlap fraction `(copy + kernel - both) / min(copy, kernel)`: 1 is perfect hiding of
    /// the shorter, 0 is none (fully serialized), negative is worse than serial.
    pub fn overlap(&self) -> f64 {
        (self.copy_ms + self.kernel_ms - self.both_ms) / self.copy_ms.min(self.kernel_ms)
    }
}

/// Measure copy/kernel overlap for a `bytes` copy with a kernel calibrated to the pinned copy
/// time.
pub fn overlap(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Vec<OverlapRow>, String> {
    let n = bytes / 8;
    let sk = ctx.new_stream().map_err(drv)?;
    let ts = TransferStream::new(ctx).map_err(drv)?;
    let ts2 = TransferStream::new(ctx).map_err(drv)?;
    let mut up_dev: CudaSlice<f64> = unsafe { sk.alloc(n) }.map_err(drv)?;
    let mut down_dev: CudaSlice<f64> = unsafe { sk.alloc(n) }.map_err(drv)?;
    sk.memset_zeros(&mut down_dev).map_err(drv)?;
    let mut pin_up = PinnedBuf::new(ctx, bytes, PinnedKind::WriteCombined).map_err(drv)?;
    pin_up.as_f64_mut().fill(1.0);
    let mut pin_down = PinnedBuf::new(ctx, bytes, PinnedKind::Cached).map_err(drv)?;
    let mut page = vec![1.0_f64; n];

    // calibrate the kernel to the pinned upload time
    let t0 = {
        let mut t = Vec::new();
        for i in 0..8 {
            let s = Instant::now();
            ts.upload_async(&pin_up, &mut up_dev)
                .map_err(drv)?
                .wait()
                .map_err(drv)?;
            if i >= 2 {
                t.push(ms(s));
            }
        }
        median(&mut t)
    };
    let mut kern = SpinKernel::new(ctx, &sk, t0)?;
    let mut rows = Vec::new();
    let reps = 9;

    macro_rules! scenario {
        ($label:expr, $alone:expr, $both:expr) => {{
            let mut a = Vec::new();
            let mut b = Vec::new();
            let mut kk = Vec::new();
            for i in 0..3 + reps {
                let s = Instant::now();
                $alone;
                let alone_ms = ms(s);
                let s = Instant::now();
                kern.enqueue(&sk)?;
                sk.synchronize().map_err(drv)?;
                let k_ms = ms(s);
                let s = Instant::now();
                $both;
                let both_ms = ms(s);
                if i >= 3 {
                    a.push(alone_ms);
                    kk.push(k_ms);
                    b.push(both_ms);
                }
            }
            rows.push(OverlapRow {
                label: $label.to_string(),
                copy_ms: median(&mut a),
                kernel_ms: median(&mut kk),
                both_ms: median(&mut b),
            });
        }};
    }

    scenario!(
        "pinned host-to-device + kernel",
        {
            ts.upload_async(&pin_up, &mut up_dev)
                .map_err(drv)?
                .wait()
                .map_err(drv)?;
        },
        {
            kern.enqueue(&sk)?;
            let p = ts.upload_async(&pin_up, &mut up_dev).map_err(drv)?;
            sk.synchronize().map_err(drv)?;
            p.wait().map_err(drv)?;
        }
    );
    scenario!(
        "pinned device-to-host + kernel",
        {
            ts.download_async(&down_dev, &mut pin_down)
                .map_err(drv)?
                .wait()
                .map_err(drv)?;
        },
        {
            kern.enqueue(&sk)?;
            let p = ts.download_async(&down_dev, &mut pin_down).map_err(drv)?;
            sk.synchronize().map_err(drv)?;
            p.wait().map_err(drv)?;
        }
    );
    scenario!(
        "pinned both directions + kernel",
        {
            let p = ts.upload_async(&pin_up, &mut up_dev).map_err(drv)?;
            let q = ts2.download_async(&down_dev, &mut pin_down).map_err(drv)?;
            p.wait().map_err(drv)?;
            q.wait().map_err(drv)?;
        },
        {
            kern.enqueue(&sk)?;
            let p = ts.upload_async(&pin_up, &mut up_dev).map_err(drv)?;
            let q = ts2.download_async(&down_dev, &mut pin_down).map_err(drv)?;
            sk.synchronize().map_err(drv)?;
            p.wait().map_err(drv)?;
            q.wait().map_err(drv)?;
        }
    );
    scenario!(
        "pageable host-to-device (on a second stream) + kernel",
        {
            let s2 = ts.stream();
            s2.memcpy_htod(&page, &mut up_dev).map_err(drv)?;
            s2.synchronize().map_err(drv)?;
        },
        {
            kern.enqueue(&sk)?;
            let s2 = ts.stream();
            s2.memcpy_htod(&page, &mut up_dev).map_err(drv)?;
            s2.synchronize().map_err(drv)?;
            sk.synchronize().map_err(drv)?;
        }
    );
    scenario!(
        "pageable device-to-host (on a second stream) + kernel",
        {
            let s2 = ts.stream();
            s2.memcpy_dtoh(&down_dev, &mut page).map_err(drv)?;
            s2.synchronize().map_err(drv)?;
        },
        {
            kern.enqueue(&sk)?;
            let s2 = ts.stream();
            s2.memcpy_dtoh(&down_dev, &mut page).map_err(drv)?;
            s2.synchronize().map_err(drv)?;
            sk.synchronize().map_err(drv)?;
        }
    );
    let _ = (&mut up_dev, &mut down_dev);
    let _ = (up_dev.device_ptr(&sk), down_dev.device_ptr_mut(&sk));
    Ok(rows)
}

/// Markdown table of overlap rows.
pub fn render_overlap(rows: &[OverlapRow], bytes: usize, kernel_ms: f64) -> String {
    let mut s = format!(
        "copy size {bytes} bytes, kernel calibrated to {kernel_ms:.3} ms\n\n| scenario | copy ms | kernel ms | both ms | overlap |\n|---|---|---|---|---|\n"
    );
    for r in rows {
        s += &format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.2} |\n",
            r.label,
            r.copy_ms,
            r.kernel_ms,
            r.both_ms,
            r.overlap()
        );
    }
    s
}

/// Ping-pong configuration of two walker groups.
#[derive(Debug, Clone, Copy)]
pub struct PingPongConfig {
    /// Bytes uploaded per group-step (walker configurations).
    pub in_bytes: usize,
    /// Bytes downloaded per group-step (pf, energies, O vectors).
    pub out_bytes: usize,
    /// Device kernel time per group-step (ms, calibrated).
    pub kernel_ms: f64,
    /// Host decision time per group-step (microseconds, busy wait).
    pub host_us: u64,
    /// Group-steps per group.
    pub steps: usize,
}

/// Result of one ping-pong comparison.
#[derive(Debug, Clone, Copy)]
pub struct PingPongRow {
    /// Configuration.
    pub cfg: PingPongConfig,
    /// One stream, pageable synchronous copies (the current tenferro-like practice) (ms total).
    pub serial_pageable_ms: f64,
    /// One stream, pinned asynchronous copies, still waiting every step (ms total).
    pub serial_pinned_ms: f64,
    /// Two streams, two groups pipelined (ms total for both groups).
    pub pingpong_ms: f64,
}

fn host_work(us: u64) {
    let end = Instant::now() + Duration::from_micros(us);
    while Instant::now() < end {
        std::hint::spin_loop();
    }
}

/// Run one ping-pong comparison.
pub fn pingpong(ctx: &Arc<CudaContext>, cfg: PingPongConfig) -> Result<PingPongRow, String> {
    let k = cfg.steps;
    let in_n = cfg.in_bytes / 8;
    let out_n = cfg.out_bytes / 8;
    let sa = TransferStream::new(ctx).map_err(drv)?;
    let sb = TransferStream::new(ctx).map_err(drv)?;
    let mut kern = SpinKernel::new(ctx, sa.stream(), cfg.kernel_ms)?;
    let mk = |s: &TransferStream| -> Result<_, String> {
        let din: CudaSlice<f64> = unsafe { s.stream().alloc(in_n) }.map_err(drv)?;
        let dout: CudaSlice<f64> = s.stream().alloc_zeros(out_n).map_err(drv)?;
        let mut pin =
            PinnedBuf::new(ctx, cfg.in_bytes, PinnedKind::for_upload(cfg.in_bytes)).map_err(drv)?;
        pin.as_f64_mut().fill(1.0);
        let pout = PinnedBuf::new(ctx, cfg.out_bytes, PinnedKind::Cached).map_err(drv)?;
        Ok((din, dout, pin, pout))
    };
    let (mut ain, mut aout, apin, mut apout) = mk(&sa)?;
    let (mut bin, mut bout, bpin, mut bpout) = mk(&sb)?;

    // serial with pageable synchronous copies (2 groups x k steps, one after another)
    let hin = vec![1.0_f64; in_n];
    let mut hout = vec![0.0_f64; out_n];
    let s = sa.stream().clone();
    let mut run_serial_pageable = |kern: &mut SpinKernel| -> Result<f64, String> {
        let start = Instant::now();
        for _ in 0..2 * k {
            s.memcpy_htod(&hin, &mut ain).map_err(drv)?;
            kern.enqueue(&s)?;
            s.memcpy_dtoh(&aout, &mut hout).map_err(drv)?;
            s.synchronize().map_err(drv)?;
            host_work(cfg.host_us);
        }
        Ok(ms(start))
    };
    run_serial_pageable(&mut kern)?;
    let serial_pageable_ms = run_serial_pageable(&mut kern)?;

    // serial with pinned async copies, waiting each step
    let mut run_serial_pinned = |kern: &mut SpinKernel| -> Result<f64, String> {
        let start = Instant::now();
        for _ in 0..2 * k {
            let _p = sa.upload_async(&apin, &mut ain).map_err(drv)?;
            kern.enqueue(sa.stream())?;
            let q = sa.download_async(&aout, &mut apout).map_err(drv)?;
            q.wait().map_err(drv)?;
            drop(_p);
            host_work(cfg.host_us);
        }
        Ok(ms(start))
    };
    run_serial_pinned(&mut kern)?;
    let serial_pinned_ms = run_serial_pinned(&mut kern)?;

    // ping-pong: groups A and B on separate streams, pipelined
    let mut run_pp = |kern: &mut SpinKernel| -> Result<f64, String> {
        let start = Instant::now();
        // enqueue helper cannot capture both mutable bufs; inline for clarity
        let mut pa = Some(enqueue_cycle(
            &sa, &apin, &mut ain, &aout, &mut apout, kern,
        )?);
        let mut pb = Some(enqueue_cycle(
            &sb, &bpin, &mut bin, &bout, &mut bpout, kern,
        )?);
        for step in 0..k {
            pa.take().expect("pending A").wait().map_err(drv)?;
            host_work(cfg.host_us);
            if step + 1 < k {
                pa = Some(enqueue_cycle(
                    &sa, &apin, &mut ain, &aout, &mut apout, kern,
                )?);
            }
            pb.take().expect("pending B").wait().map_err(drv)?;
            host_work(cfg.host_us);
            if step + 1 < k {
                pb = Some(enqueue_cycle(
                    &sb, &bpin, &mut bin, &bout, &mut bpout, kern,
                )?);
            }
        }
        Ok(ms(start))
    };
    run_pp(&mut kern)?;
    let pingpong_ms = run_pp(&mut kern)?;
    let _ = (&mut bout, &mut aout);
    Ok(PingPongRow {
        cfg,
        serial_pageable_ms,
        serial_pinned_ms,
        pingpong_ms,
    })
}

/// Enqueue `upload; kernel; download` on `s` and return an owned completion token.
fn enqueue_cycle(
    s: &TransferStream,
    pin_in: &PinnedBuf,
    din: &mut CudaSlice<f64>,
    dout: &CudaSlice<f64>,
    pin_out: &mut PinnedBuf,
    kern: &mut SpinKernel,
) -> Result<CycleToken, String> {
    // The borrow-checked `Pending` ties the host buffers to this scope; the ping-pong keeps
    // the buffers alive for the whole loop and waits before reuse, so the token only carries
    // the final event.
    let up = s.upload_async(pin_in, din).map_err(drv)?;
    kern.enqueue(s.stream())?;
    let down = s.download_async(dout, pin_out).map_err(drv)?;
    // SAFETY: the ping-pong loop owns the buffers for its whole duration and waits on the
    // returned token before touching or reusing them.
    let event = unsafe {
        drop(up.detach());
        down.detach()
    };
    Ok(CycleToken(event))
}

/// Completion of an enqueued cycle.
pub struct CycleToken(crate::transfer::TransferEvent);

impl CycleToken {
    fn wait(self) -> Result<(), DriverError> {
        self.0.wait_host()
    }
}

/// Markdown table of ping-pong rows.
pub fn render_pingpong(rows: &[PingPongRow]) -> String {
    let mut s = String::from(
        "| in bytes | out bytes | kernel ms | host us | steps/group | serial pageable ms | serial pinned ms | ping-pong ms | speedup vs pageable | speedup vs pinned serial |\n|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        s += &format!(
            "| {} | {} | {:.2} | {} | {} | {:.1} | {:.1} | {:.1} | {:.2}x | {:.2}x |\n",
            r.cfg.in_bytes,
            r.cfg.out_bytes,
            r.cfg.kernel_ms,
            r.cfg.host_us,
            r.cfg.steps,
            r.serial_pageable_ms,
            r.serial_pinned_ms,
            r.pingpong_ms,
            r.serial_pageable_ms / r.pingpong_ms,
            r.serial_pinned_ms / r.pingpong_ms
        );
    }
    s
}
