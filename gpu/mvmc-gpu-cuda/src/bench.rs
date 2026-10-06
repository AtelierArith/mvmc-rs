//! FP64 / C64 `dot_general` and Cholesky micro-benchmark, CPU versus CUDA.
//!
//! Timing follows the tenferro-decision-rs survey: upload, compute (with an explicit
//! `synchronize` inside the timed region) and download are timed separately; the median of
//! `reps` runs after one warm-up is reported. Correctness is checked against the tenferro CPU
//! result with explicit tolerances (see [`TOL_DOT`], [`TOL_CHOLESKY`]). An unsupported
//! operation or dtype surfaces as an error; there is no silent CPU fallback.

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_linalg::EagerTensorLinalgExt;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};

/// Relative max-norm tolerance for `dot_general` CUDA vs CPU: both accumulate `n` products of
/// O(1) entries in different order, so the error scales with `n * eps` (eps = 2.2e-16);
/// 1e-11 leaves a >1e3 margin at n = 1024 while still catching a wrong-layout or dtype defect.
pub const TOL_DOT: f64 = 1e-11;
/// Relative max-norm tolerance on the lower triangle of Cholesky factors (diagonally dominant
/// well-conditioned input; backward-stable in both implementations).
pub const TOL_CHOLESKY: f64 = 1e-10;

/// Operation under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `C = A B` through `dot_general`.
    Dot,
    /// Lower Cholesky factor of a Hermitian positive-definite matrix.
    Cholesky,
}

/// Scalar type under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// `f64`.
    F64,
    /// `Complex64` (two f64).
    C64,
}

/// Wall times in milliseconds (median).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stages {
    /// Host to device.
    pub upload_ms: f64,
    /// Operation plus synchronize.
    pub compute_ms: f64,
    /// Device to host.
    pub download_ms: f64,
}

/// One measured case.
#[derive(Debug, Clone)]
pub struct Row {
    /// Operation.
    pub op: Op,
    /// Dtype.
    pub dtype: Dtype,
    /// Matrix dimension.
    pub n: usize,
    /// CPU timing.
    pub cpu: Stages,
    /// CUDA timing.
    pub cuda: Stages,
    /// Relative max-norm difference CUDA vs CPU result.
    pub rel_err: f64,
    /// Tolerance applied.
    pub tol: f64,
}

impl Row {
    /// Whether the correctness check passed.
    pub fn ok(&self) -> bool {
        self.rel_err <= self.tol
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

fn random_real(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n * n).map(|_| lcg(&mut s)).collect()
}

fn random_complex(n: usize, seed: u64) -> Vec<Complex64> {
    let mut s = seed;
    (0..n * n)
        .map(|_| {
            let re = lcg(&mut s);
            Complex64::new(re, lcg(&mut s))
        })
        .collect()
}

/// Hermitian, diagonally dominant (hence positive definite) real matrix, column-major.
fn spd_real(n: usize) -> Vec<f64> {
    let r = random_real(n, 7);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] = 0.5 * (r[i + j * n] + r[j + i * n]);
        }
        a[j + j * n] += n as f64;
    }
    a
}

fn spd_complex(n: usize) -> Vec<Complex64> {
    let z = random_complex(n, 11);
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] = (z[i + j * n] + z[j + i * n].conj()) * 0.5;
        }
        a[j + j * n] = Complex64::new(a[j + j * n].re + n as f64, 0.0);
    }
    a
}

fn host(dtype: Dtype, n: usize, data: Data) -> Result<Tensor, String> {
    let shape = vec![n, n];
    match (dtype, data) {
        (Dtype::F64, Data::R(v)) => Tensor::from_vec_col_major(shape, v),
        (Dtype::C64, Data::C(v)) => Tensor::from_vec_col_major(shape, v),
        _ => unreachable!("dtype/data mismatch"),
    }
    .map_err(|e| e.to_string())
}

enum Data {
    R(Vec<f64>),
    C(Vec<Complex64>),
}

fn inputs(op: Op, dtype: Dtype, n: usize) -> Result<Vec<Tensor>, String> {
    match (op, dtype) {
        (Op::Dot, Dtype::F64) => Ok(vec![
            host(dtype, n, Data::R(random_real(n, 1)))?,
            host(dtype, n, Data::R(random_real(n, 2)))?,
        ]),
        (Op::Dot, Dtype::C64) => Ok(vec![
            host(dtype, n, Data::C(random_complex(n, 1)))?,
            host(dtype, n, Data::C(random_complex(n, 2)))?,
        ]),
        (Op::Cholesky, Dtype::F64) => Ok(vec![host(dtype, n, Data::R(spd_real(n)))?]),
        (Op::Cholesky, Dtype::C64) => Ok(vec![host(dtype, n, Data::C(spd_complex(n)))?]),
    }
}

/// Flatten to f64 (re, im interleaved for complex); `lower_only` keeps `i >= j`.
fn flatten(t: &Tensor, dtype: Dtype, n: usize, lower_only: bool) -> Result<Vec<f64>, String> {
    let keep = |idx: usize| !lower_only || idx % n >= idx / n;
    match dtype {
        Dtype::F64 => {
            let s = t.as_slice::<f64>().map_err(|e| e.to_string())?;
            Ok(s.iter()
                .enumerate()
                .filter(|(i, _)| keep(*i))
                .map(|(_, v)| *v)
                .collect())
        }
        Dtype::C64 => {
            let s = t.as_slice::<Complex64>().map_err(|e| e.to_string())?;
            Ok(s.iter()
                .enumerate()
                .filter(|(i, _)| keep(*i))
                .flat_map(|(_, v)| [v.re, v.im])
                .collect())
        }
    }
}

fn run_once(ctx: &Arc<EagerRuntime>, op: Op, hosts: &[Tensor]) -> Result<(Tensor, Stages), String> {
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut eager = Vec::new();
    for h in hosts {
        let dev = ctx
            .with_execution_session(|s| s.upload_host_tensor(TensorRead::from_tensor(h)))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        eager.push(EagerTensor::from_tensor_in(dev, ctx.clone()).map_err(|e| e.to_string())?);
    }
    ctx.synchronize().map_err(|e| e.to_string())?;
    let upload_ms = ms(t);

    let t = Instant::now();
    let out = match op {
        Op::Dot => {
            let cfg = DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            };
            eager[0].dot_general(&eager[1], cfg)
        }
        Op::Cholesky => eager[0].cholesky(),
    }
    .map_err(|e| e.to_string())?;
    ctx.synchronize().map_err(|e| e.to_string())?;
    let compute_ms = ms(t);

    let t = Instant::now();
    let dev_out = out.to_tensor().map_err(|e| e.to_string())?;
    let host_out = ctx
        .with_execution_session(|s| s.download_to_host(TensorRead::from_tensor(&dev_out)))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let download_ms = ms(t);
    Ok((
        host_out,
        Stages {
            upload_ms,
            compute_ms,
            download_ms,
        },
    ))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn measure(
    ctx: &Arc<EagerRuntime>,
    op: Op,
    hosts: &[Tensor],
    reps: usize,
) -> Result<(Tensor, Stages), String> {
    run_once(ctx, op, hosts)?; // warm-up (plan/kernels/handles)
    let (mut u, mut c, mut d) = (vec![], vec![], vec![]);
    let mut last = None;
    for _ in 0..reps.max(1) {
        let (out, st) = run_once(ctx, op, hosts)?;
        u.push(st.upload_ms);
        c.push(st.compute_ms);
        d.push(st.download_ms);
        last = Some(out);
    }
    Ok((
        last.expect("at least one repetition"),
        Stages {
            upload_ms: median(&mut u),
            compute_ms: median(&mut c),
            download_ms: median(&mut d),
        },
    ))
}

/// CPU runtime.
pub fn cpu_runtime() -> Result<Arc<EagerRuntime>, String> {
    EagerRuntime::with_cpu_backend(CpuBackend::new()).map_err(|e| e.to_string())
}

/// CUDA runtime on the given device ordinal.
pub fn cuda_runtime(ordinal: usize) -> Result<Arc<EagerRuntime>, String> {
    let devices = cuda_devices().map_err(|e| e.to_string())?;
    let device = devices
        .get(ordinal)
        .ok_or_else(|| format!("no CUDA device with ordinal {ordinal}"))?;
    let backend = CudaBackend::new(device.id()).map_err(|e| e.to_string())?;
    EagerRuntime::with_cuda_backend(backend).map_err(|e| e.to_string())
}

/// Run one case on both backends and compare.
pub fn run_case(
    cpu: &Arc<EagerRuntime>,
    cuda: &Arc<EagerRuntime>,
    op: Op,
    dtype: Dtype,
    n: usize,
    reps: usize,
) -> Result<Row, String> {
    let hosts = inputs(op, dtype, n)?;
    let (cpu_out, cpu_t) = measure(cpu, op, &hosts, reps)?;
    let (cuda_out, cuda_t) = measure(cuda, op, &hosts, reps)?;
    let lower = op == Op::Cholesky;
    let a = flatten(&cpu_out, dtype, n, lower)?;
    let b = flatten(&cuda_out, dtype, n, lower)?;
    let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = a
        .iter()
        .zip(&b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    Ok(Row {
        op,
        dtype,
        n,
        cpu: cpu_t,
        cuda: cuda_t,
        rel_err: diff / scale.max(f64::MIN_POSITIVE),
        tol: if op == Op::Dot { TOL_DOT } else { TOL_CHOLESKY },
    })
}

/// Run the standard case matrix (`dot_general` and Cholesky, f64 and c64) for each size.
pub fn run_all(ordinal: usize, sizes: &[usize], reps: usize) -> Result<Vec<Row>, String> {
    let cpu = cpu_runtime()?;
    let cuda = cuda_runtime(ordinal)?;
    let mut rows = Vec::new();
    for &n in sizes {
        for op in [Op::Dot, Op::Cholesky] {
            for dtype in [Dtype::F64, Dtype::C64] {
                rows.push(run_case(&cpu, &cuda, op, dtype, n, reps)?);
            }
        }
    }
    Ok(rows)
}

/// Markdown table of rows (milliseconds, medians).
pub fn render(rows: &[Row]) -> String {
    let mut s = String::from(
        "| op | dtype | n | CPU compute | CUDA upload | CUDA compute | CUDA download | \
         CPU/CUDA compute | rel err | tol | check |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        s += &format!(
            "| {:?} | {:?} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2}x | {:.2e} | {:.0e} | {} |\n",
            r.op,
            r.dtype,
            r.n,
            r.cpu.compute_ms,
            r.cuda.upload_ms,
            r.cuda.compute_ms,
            r.cuda.download_ms,
            r.cpu.compute_ms / r.cuda.compute_ms,
            r.rel_err,
            r.tol,
            if r.ok() { "ok" } else { "FAIL" }
        );
    }
    s
}

/// Host-to-device-to-host round-trip measurement for one buffer size.
#[derive(Debug, Clone, Copy)]
pub struct RoundTrip {
    /// Buffer size in bytes (one f64 vector).
    pub bytes: usize,
    /// Median pageable upload (ms), synchronized.
    pub upload_ms: f64,
    /// Median elementwise-add launch plus synchronize (ms): the launch/sync floor.
    pub op_ms: f64,
    /// Median pageable download (ms).
    pub download_ms: f64,
}

impl RoundTrip {
    /// Whole round trip in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.upload_ms + self.op_ms + self.download_ms
    }
    /// Upload bandwidth in GB/s.
    pub fn upload_gbps(&self) -> f64 {
        self.bytes as f64 / (self.upload_ms * 1e-3) / 1e9
    }
    /// Download bandwidth in GB/s.
    pub fn download_gbps(&self) -> f64 {
        self.bytes as f64 / (self.download_ms * 1e-3) / 1e9
    }
}

/// Measure the device round trip for a vector of `n` f64 values (median of `reps` after
/// `warmups`): the data cost a stage must amortize before any compute benefit.
pub fn roundtrip(
    ctx: &Arc<EagerRuntime>,
    n: usize,
    warmups: usize,
    reps: usize,
) -> Result<RoundTrip, String> {
    let host = Tensor::from_vec_col_major(vec![n], vec![1.0_f64; n]).map_err(|e| e.to_string())?;
    let upload = |ctx: &Arc<EagerRuntime>| -> Result<EagerTensor, String> {
        let dev = ctx
            .with_execution_session(|s| s.upload_host_tensor(TensorRead::from_tensor(&host)))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let t = EagerTensor::from_tensor_in(dev, ctx.clone()).map_err(|e| e.to_string())?;
        ctx.synchronize().map_err(|e| e.to_string())?;
        Ok(t)
    };
    let (mut up, mut op, mut down) = (vec![], vec![], vec![]);
    let a = upload(ctx)?;
    let b = upload(ctx)?;
    for i in 0..warmups + reps.max(1) {
        let t = Instant::now();
        let _ = upload(ctx)?;
        let u = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let c = a.add(&b).map_err(|e| e.to_string())?;
        ctx.synchronize().map_err(|e| e.to_string())?;
        let o = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let dev = c.to_tensor().map_err(|e| e.to_string())?;
        let _ = ctx
            .with_execution_session(|s| s.download_to_host(TensorRead::from_tensor(&dev)))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let d = t.elapsed().as_secs_f64() * 1e3;
        if i >= warmups {
            up.push(u);
            op.push(o);
            down.push(d);
        }
    }
    Ok(RoundTrip {
        bytes: n * 8,
        upload_ms: median(&mut up),
        op_ms: median(&mut op),
        download_ms: median(&mut down),
    })
}

/// Markdown table of round-trip rows.
pub fn render_roundtrip(rows: &[RoundTrip]) -> String {
    let mut s = String::from(
        "| bytes | upload ms | upload GB/s | add+sync ms | download ms | download GB/s | total ms |\n\
         |---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        s += &format!(
            "| {} | {:.4} | {:.2} | {:.4} | {:.4} | {:.2} | {:.4} |\n",
            r.bytes,
            r.upload_ms,
            r.upload_gbps(),
            r.op_ms,
            r.download_ms,
            r.download_gbps(),
            r.total_ms()
        );
    }
    s
}
