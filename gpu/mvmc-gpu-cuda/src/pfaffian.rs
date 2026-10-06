//! Batched Pfaffian and inverse on CUDA (issue #423).
//!
//! One thread block per `[n, n]` plane; the kernel (`pfaffian_batched.cu`) is compiled with
//! NVRTC at run time and launched through tenferro's raw CUDA session
//! (`CudaExecSession::with_raw`, `raw::Session::compile_nvrtc` / `launch`). It implements the
//! same math as `pfapack` (`dsktf2`/`zsktf2`, `utu2pfa`, `utu2inv`); see the file header of the
//! kernel for the algorithm and the status encoding.
//!
//! Module caching: `raw::Module` is `!Send`, so it cannot live in the Send-only
//! `Session::resource` store; it is therefore compiled/loaded once per [`with_session`] scope
//! (NVRTC takes tens of milliseconds, the driver caches the PTX JIT). Use
//! [`with_session`] for repeated runs (benchmarks); the [`CudaEngine`] convenience wrapper pays
//! the compile for every call.
//!
//! No silent CPU fallback: every failure is a typed [`mvmc_gpu::Error`]. All timings bracket an
//! explicit `Session::synchronize`.

use std::time::Instant;

use mvmc_gpu::{BatchOutput, BatchedEngine, Error, PfScalar, PlaneStatus};
use num_complex::Complex64;
use tenferro_gpu::cuda::raw::{Function, KernelArg, LaunchConfig, NvrtcOptions, Session};
use tenferro_gpu::cuda::{cuda_devices, with_cuda_exec_session, CudaBackend};
use tenferro_tensor::BackendSessionHost;

/// CUDA source of the batched kernel.
pub const KERNEL_SRC: &str = include_str!("pfaffian_batched.cu");
/// Largest supported matrix side (dynamic shared memory holds `O(n)` per block).
pub const MAX_N: usize = 1024;
/// Device bytes (work copy + workspace + output) allowed per chunk of planes.
const CHUNK_BYTES: usize = 3 << 30;

/// Wall-clock phases of one run, each ended by a stream synchronization.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Host to device copy of the input planes.
    pub upload_s: f64,
    /// Allocation of the workspace and output buffers.
    pub alloc_s: f64,
    /// Kernel launches of all chunks.
    pub kernel_s: f64,
    /// Device to host copies.
    pub download_s: f64,
}

impl Timings {
    /// Sum of the phases.
    pub fn total_s(&self) -> f64 {
        self.upload_s + self.alloc_s + self.kernel_s + self.download_s
    }
}

fn cuerr(err: impl std::fmt::Display) -> Error {
    Error::Backend(err.to_string())
}

/// Result of a lane-level run: `(pf lanes, inv lanes, status codes, timings)`.
pub type LaneOutput<E> = (Vec<E>, Vec<E>, Vec<i32>, Timings);

/// A loaded kernel module inside one raw CUDA session.
pub struct PfSession<'a, 's> {
    sess: &'a mut Session<'s>,
    f64_real: Function,
    f64_complex: Function,
    f32_real: Function,
    f32_complex: Function,
}

macro_rules! lane_runner {
    ($name:ident, $elem:ty) => {
        fn $name(
            &self,
            kernel: &Function,
            complex: bool,
            planes: &[$elem],
            n: usize,
            count: usize,
        ) -> Result<LaneOutput<$elem>, Error> {
            if n > MAX_N {
                return Err(Error::InvalidShape(format!(
                    "CUDA kernel supports n <= {MAX_N}, got {n}"
                )));
            }
            let lanes = if complex { 2 } else { 1 };
            let esz = std::mem::size_of::<$elem>();
            let plane_elems = n * n * lanes;
            let chunk = (CHUNK_BYTES / (3 * plane_elems * esz)).clamp(1, count.max(1));
            let nt = n.next_power_of_two().clamp(32, 256);
            let shared = (n * esz * lanes + nt * esz + nt * 4 + n * 8) as u32;

            let mut pf = vec![<$elem>::default(); count * lanes];
            let mut inv = vec![<$elem>::default(); count * plane_elems];
            let mut status = vec![0i32; count];
            let mut t = Timings::default();
            let mut done = 0usize;
            while done < count {
                let m = chunk.min(count - done);
                let host = &planes[done * plane_elems..(done + m) * plane_elems];
                let sess = &*self.sess;

                let t0 = Instant::now();
                // SAFETY: f32/f64 have no padding or invalid byte patterns; the byte view lives
                // only for the upload call.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        host.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(host),
                    )
                };
                let d_a = sess.upload_bytes(bytes, "pfaffian.upload").map_err(cuerr)?;
                sess.synchronize().map_err(cuerr)?;
                t.upload_s += t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let d_w = sess
                    .alloc_bytes(m * plane_elems * esz, "pfaffian.workspace")
                    .map_err(cuerr)?;
                let mut d_out = sess
                    .alloc_output::<$elem>(&[m * plane_elems])
                    .map_err(cuerr)?;
                let mut d_pf = sess.alloc_output::<$elem>(&[m * lanes]).map_err(cuerr)?;
                let mut d_st = sess.alloc_output::<i32>(&[m]).map_err(cuerr)?;
                sess.synchronize().map_err(cuerr)?;
                t.alloc_s += t0.elapsed().as_secs_f64();

                let t1 = Instant::now();
                {
                    let out_ref = sess.tensor_mut(&mut d_out).map_err(cuerr)?;
                    let pf_ref = sess.tensor_mut(&mut d_pf).map_err(cuerr)?;
                    let st_ref = sess.tensor_mut(&mut d_st).map_err(cuerr)?;
                    let config = LaunchConfig {
                        grid: [m as u32, 1, 1],
                        block: [nt as u32, 1, 1],
                        shared_mem_bytes: shared,
                    };
                    // SAFETY: kernel ABI is (T* A, T* W, T* OUT, T* PF, int* ST, int n). A and W
                    // are device byte buffers of `m` planes, OUT/PF/ST are freshly allocated
                    // tensors of exactly `m` planes / `m` scalars / `m` ints, none alias, the
                    // grid has one block per plane, the dynamic shared size matches the
                    // kernel layout, and every buffer outlives the synchronize below.
                    unsafe {
                        sess.launch(
                            kernel,
                            config,
                            &[
                                KernelArg::workspace(&d_a),
                                KernelArg::workspace(&d_w),
                                KernelArg::output(&out_ref),
                                KernelArg::output(&pf_ref),
                                KernelArg::output(&st_ref),
                                KernelArg::i32(n as i32),
                            ],
                        )
                    }
                    .map_err(cuerr)?;
                }
                sess.synchronize().map_err(cuerr)?;
                t.kernel_s += t1.elapsed().as_secs_f64();

                let t2 = Instant::now();
                let h_out = sess
                    .download_tensor(&d_out, "pfaffian.download")
                    .map_err(cuerr)?;
                let h_pf = sess
                    .download_tensor(&d_pf, "pfaffian.download")
                    .map_err(cuerr)?;
                let h_st = sess
                    .download_tensor(&d_st, "pfaffian.download")
                    .map_err(cuerr)?;
                inv[done * plane_elems..(done + m) * plane_elems]
                    .copy_from_slice(h_out.as_slice().map_err(cuerr)?);
                pf[done * lanes..(done + m) * lanes]
                    .copy_from_slice(h_pf.as_slice().map_err(cuerr)?);
                status[done..done + m].copy_from_slice(h_st.as_slice().map_err(cuerr)?);
                t.download_s += t2.elapsed().as_secs_f64();
                done += m;
            }
            Ok((pf, inv, status, t))
        }
    };
}

impl PfSession<'_, '_> {
    lane_runner!(run_lanes_f64, f64);
    lane_runner!(run_lanes_f32, f32);

    /// Batched Pfaffian and inverse of `count` planes with phase timings.
    pub fn run_timed<T: PfScalar>(
        &self,
        planes: &[T],
        n: usize,
        count: usize,
    ) -> Result<(BatchOutput<T>, Timings), Error> {
        let complex = T::lanes() == 2;
        let kernel = if complex {
            &self.f64_complex
        } else {
            &self.f64_real
        };
        let (pf, inv, status, t) =
            self.run_lanes_f64(kernel, complex, T::as_f64_lanes(planes), n, count)?;
        let mut pf_t = vec![T::zero_value(); count];
        T::as_f64_lanes_mut(&mut pf_t).copy_from_slice(&pf);
        let mut inv_t = vec![T::zero_value(); planes.len()];
        T::as_f64_lanes_mut(&mut inv_t).copy_from_slice(&inv);
        let status = status.into_iter().map(PlaneStatus::from_code).collect();
        Ok(((pf_t, inv_t, status), t))
    }

    /// Single-precision variant on interleaved lanes. Timing comparison only; never a physics
    /// path.
    pub fn run_f32_timed(
        &self,
        complex: bool,
        planes: &[f32],
        n: usize,
        count: usize,
    ) -> Result<LaneOutput<f32>, Error> {
        let kernel = if complex {
            &self.f32_complex
        } else {
            &self.f32_real
        };
        self.run_lanes_f32(kernel, complex, planes, n, count)
    }
}

/// Number of CUDA devices visible to tenferro.
pub fn device_count() -> Result<usize, Error> {
    cuda_devices().map(|d| d.len()).map_err(cuerr)
}

/// Compile the kernels and run `f` inside one raw CUDA session on `device`.
///
/// # Errors
///
/// [`Error::BackendUnavailable`] when the device or NVRTC is unusable, or the errors of `f`.
pub fn with_session<R: Send>(
    device: usize,
    f: impl FnOnce(&mut PfSession<'_, '_>) -> Result<R, Error> + Send,
) -> Result<R, Error> {
    let devices = cuda_devices().map_err(|e| Error::BackendUnavailable(e.to_string()))?;
    let device = devices.get(device).ok_or_else(|| {
        Error::BackendUnavailable(format!("no CUDA device with ordinal {device}"))
    })?;
    let mut backend = CudaBackend::new(device.id())
        .map_err(|e| Error::BackendUnavailable(format!("CUDA backend: {e}")))?;
    let outcome = backend.with_backend_session(|session| {
        with_cuda_exec_session(session, |exec| {
            exec.with_raw("mvmc-gpu.pfaffian", |sess| {
                // --fmad=false keeps pfapack's operation order in the rank-2 update.
                let opts = NvrtcOptions {
                    arch: Some("compute_80".into()),
                    std: None,
                    extra: vec!["--fmad=false".into()],
                };
                let module = sess.compile_nvrtc(KERNEL_SRC, &opts)?;
                let load = |name: &str| module.function(name);
                let mut ps = PfSession {
                    f64_real: load("pfinv_f64")?,
                    f64_complex: load("pfinv_c64")?,
                    f32_real: load("pfinv_f32")?,
                    f32_complex: load("pfinv_c32")?,
                    sess,
                };
                Ok(f(&mut ps))
            })
        })
    });
    match outcome {
        Some(Ok(inner)) => inner,
        Some(Err(e)) => Err(Error::BackendUnavailable(format!("CUDA session: {e}"))),
        None => Err(Error::BackendUnavailable(
            "backend session is not a CUDA execution session".into(),
        )),
    }
}

/// Convenience engine for [`mvmc_gpu::Backend::Engine`]: compiles and loads the module for
/// every call (use [`with_session`] when timing or looping).
#[derive(Debug, Clone)]
pub struct CudaEngine {
    /// CUDA device ordinal.
    pub device: usize,
}

impl BatchedEngine for CudaEngine {
    fn name(&self) -> String {
        format!("cuda:{}", self.device)
    }
    fn run_f64(&self, planes: &[f64], n: usize, count: usize) -> Result<BatchOutput<f64>, Error> {
        with_session(self.device, |s| Ok(s.run_timed(planes, n, count)?.0))
    }
    fn run_c64(
        &self,
        planes: &[Complex64],
        n: usize,
        count: usize,
    ) -> Result<BatchOutput<Complex64>, Error> {
        with_session(self.device, |s| Ok(s.run_timed(planes, n, count)?.0))
    }
}
