//! Device-resident SR pipeline (issue #447): the saved-sample matrix, the Gram product, the S/g
//! assembly, the diagonal shift, the Cholesky factorization and solve, or the whole CG loop,
//! stay on the GPU between stages. Only the per-step inputs (the sample matrix once per step,
//! `HO`/gradient/mean/diagonal vectors) and the small outputs (the solution `x`, scalars) cross
//! PCIe, through the pinned, chunked, double-buffered upload of the #432 helper
//! ([`crate::transfer`]).
//!
//! Stages (real parameters; the CG product also takes an imaginary sample matrix):
//!
//! * **Direct SR** ([`DeviceSr::upload_store`], [`DeviceSr::solve_direct`]): cuBLAS `dsyrk`
//!   `G = O O^T` on the resident store `[n, samples]`; kernel `k_assemble_s`/`k_assemble_g`
//!   builds the shifted `S` and the force `g` from `G` (the formulas of
//!   `COrderSr::assemble_s_g`, with the active-component map); cuSOLVER `dpotrf('U')` +
//!   `dpotrs('U')`, i.e. the `DPOSV` of C `stcopt_dposv.c`. A nonzero `info` or a nonfinite
//!   solution is [`SrDeviceError::SolveFailed`], like the host's `Err(())`.
//! * **CG** ([`DeviceSr::set_cg_operand`], [`DeviceSr::solve_cg`]): the loop of
//!   `SampledSrOperator::solve` ported statement for statement (threshold, 20-iteration
//!   residual refresh, `beta * delta` recurrence, breakdown semantics); matvecs are two cuBLAS
//!   `dgemv` per sample matrix, vector updates are `daxpy`/`dscal`, the three scalars per
//!   iteration (`mean.x`, `d.q`, `r.r`) are returned by `ddot` and are the only host syncs.
//!
//! Numerics: cuBLAS reductions are parallel trees, so the sums differ from the C-order
//! sequential sums by `O(k eps)` per reduction; see `sr_device_bounds` below and
//! `docs/design/gpu-readiness.md` section 14 for the bounds and the CG amplification policy
//! (#358). The elementwise kernels are bit-compatible with the host formulas (`--fmad=false`).
//!
//! There is no CPU fallback: every failure is a typed [`SrDeviceError`].

use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{sys as cublas_sys, CudaBlas};
use cudarc::cusolver::{sys as cusolver_sys, DnHandle};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::transfer::{PinnedBuf, PinnedKind, PinnedPool, TransferStream};

const SRC: &str = include_str!("sr_kernels.cu");
/// Upload chunk: 32 MiB per pinned staging buffer, two buffers (host copy overlaps PCIe).
const CHUNK_DOUBLES: usize = 4 << 20;

/// Typed failure of the device SR pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum SrDeviceError {
    /// The device, a library or a kernel is unusable.
    Unavailable(String),
    /// A CUDA/cuBLAS/cuSOLVER call failed.
    Backend(String),
    /// Inconsistent sizes or a missing resident operand.
    Shape(String),
    /// `dpotrf`/`dpotrs` reported `info != 0` or the solution is nonfinite (host `Err(())`).
    SolveFailed {
        /// LAPACK `info` (0 for a nonfinite solution).
        info: i32,
    },
}

impl std::fmt::Display for SrDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(s) => write!(f, "device unavailable: {s}"),
            Self::Backend(s) => write!(f, "device backend error: {s}"),
            Self::Shape(s) => write!(f, "shape error: {s}"),
            Self::SolveFailed { info } => write!(f, "Cholesky solve failed (info {info})"),
        }
    }
}

impl std::error::Error for SrDeviceError {}

fn be(e: impl std::fmt::Debug) -> SrDeviceError {
    SrDeviceError::Backend(format!("{e:?}"))
}

fn blas_ok(s: cublas_sys::cublasStatus_t, what: &str) -> Result<(), SrDeviceError> {
    if s == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(SrDeviceError::Backend(format!("cuBLAS {what}: {s:?}")))
    }
}

fn solver_ok(s: cusolver_sys::cusolverStatus_t, what: &str) -> Result<(), SrDeviceError> {
    if s == cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(SrDeviceError::Backend(format!("cuSOLVER {what}: {s:?}")))
    }
}

/// Wall time of the phases of the last pipeline call (seconds), each ended by a synchronization.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepTimings {
    /// Host to device copy of the sample matrix (pinned, chunked).
    pub upload_store_s: f64,
    /// Host to device copy of the small vectors.
    pub upload_vec_s: f64,
    /// Gram product.
    pub gram_s: f64,
    /// S/g assembly kernels.
    pub assemble_s: f64,
    /// Cholesky factorization and solve (direct) or the CG loop.
    pub solve_s: f64,
    /// Device to host copy of the result.
    pub download_s: f64,
}

impl StepTimings {
    /// Sum of the phases.
    pub fn total_s(&self) -> f64 {
        self.upload_store_s
            + self.upload_vec_s
            + self.gram_s
            + self.assemble_s
            + self.solve_s
            + self.download_s
    }
}

struct Store {
    data: CudaSlice<f64>,
    n: usize,
    samples: usize,
}

struct Cg {
    real: CudaSlice<f64>,
    imag: Option<CudaSlice<f64>>,
    comp: usize,
    samples: usize,
}

/// Result of [`DeviceSr::solve_cg`].
#[derive(Debug, Clone)]
pub struct CgOutcome {
    /// Solution increment.
    pub solution: Vec<f64>,
    /// Iteration count in the convention of `CgSolution::iterations`.
    pub iterations: usize,
    /// Last recurrence value of `r.r` (`delta`).
    pub delta: f64,
}

/// The device-resident SR pipeline.
pub struct DeviceSr {
    ctx: Arc<CudaContext>,
    ts: Arc<TransferStream>,
    blas: CudaBlas,
    solver: DnHandle,
    k_assemble_s: CudaFunction,
    k_assemble_g: CudaFunction,
    k_combine: CudaFunction,
    k_nonfinite: CudaFunction,
    pool: PinnedPool,
    stage: [Option<PinnedBuf>; 2],
    store: Option<Store>,
    cg: Option<Cg>,
    gram: Option<CudaSlice<f64>>,
    s: Option<CudaSlice<f64>>,
    vecs: Vec<CudaSlice<f64>>,
    map_dev: Option<CudaSlice<i64>>,
    ints: Option<CudaSlice<i32>>,
    /// Timings of the last call.
    pub timings: StepTimings,
}

fn dp(s: &CudaSlice<f64>, st: &Arc<CudaStream>) -> u64 {
    let (p, _g) = s.device_ptr(st);
    p
}

impl DeviceSr {
    /// Initialise device `device`: compile the kernels, create the cuBLAS/cuSOLVER handles on
    /// the pipeline's stream.
    ///
    /// # Errors
    ///
    /// [`SrDeviceError::Unavailable`] when the device, NVRTC or a library cannot be loaded.
    pub fn new(device: usize) -> Result<Self, SrDeviceError> {
        let ctx = CudaContext::new(device)
            .map_err(|e| SrDeviceError::Unavailable(format!("CUDA device {device}: {e:?}")))?;
        // SAFETY: every cross-stream dependency is ordered by this module (one stream).
        unsafe { crate::transfer::disable_event_tracking(&ctx) };
        let ts = Arc::new(
            TransferStream::new(&ctx).map_err(|e| SrDeviceError::Unavailable(format!("{e:?}")))?,
        );
        let ptx = compile_ptx_with_opts(
            SRC,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .map_err(|e| SrDeviceError::Unavailable(format!("NVRTC: {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SrDeviceError::Unavailable(format!("module: {e:?}")))?;
        let f = |name: &str| module.load_function(name).map_err(be);
        let blas = CudaBlas::new(ts.stream().clone())
            .map_err(|e| SrDeviceError::Unavailable(format!("cuBLAS: {e:?}")))?;
        let solver = DnHandle::new(ts.stream().clone())
            .map_err(|e| SrDeviceError::Unavailable(format!("cuSOLVER: {e:?}")))?;
        Ok(Self {
            k_assemble_s: f("k_assemble_s")?,
            k_assemble_g: f("k_assemble_g")?,
            k_combine: f("k_cg_combine")?,
            k_nonfinite: f("k_nonfinite")?,
            pool: PinnedPool::new(&ctx),
            stage: [None, None],
            ctx,
            ts,
            blas,
            solver,
            store: None,
            cg: None,
            gram: None,
            s: None,
            vecs: Vec::new(),
            map_dev: None,
            ints: None,
            timings: StepTimings::default(),
        })
    }

    /// The cudarc context (device report, benchmark metadata).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Device memory in bytes: `(free, total)`.
    pub fn mem_info(&self) -> Result<(usize, usize), SrDeviceError> {
        self.ctx.mem_get_info().map_err(be)
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.ts.stream().clone()
    }

    fn sync(&self) -> Result<(), SrDeviceError> {
        self.ts.stream().synchronize().map_err(be)
    }

    /// Copy `src` into `dst` through two pinned write-combined staging buffers: while chunk `k`
    /// crosses PCIe the host fills chunk `k+1`.
    fn upload_chunked(
        &mut self,
        dst: &mut CudaSlice<f64>,
        src: &[f64],
    ) -> Result<(), SrDeviceError> {
        assert!(dst.len() >= src.len());
        let base = dp(dst, self.ts.stream());
        let chunk = CHUNK_DOUBLES.min(src.len().max(1));
        for slot in &mut self.stage {
            if slot.as_ref().is_none_or(|b| b.len_bytes() < chunk * 8) {
                if let Some(old) = slot.take() {
                    self.pool.put(old);
                }
                *slot = Some(
                    self.pool
                        .take(chunk * 8, PinnedKind::WriteCombined)
                        .map_err(be)?,
                );
            }
        }
        let mut inflight: [Option<crate::transfer::TransferEvent>; 2] = [None, None];
        let mut off = 0usize;
        let mut k = 0usize;
        while off < src.len() {
            let len = chunk.min(src.len() - off);
            let slot = k % 2;
            if let Some(ev) = inflight[slot].take() {
                ev.wait_host().map_err(be)?;
            }
            let buf = self.stage[slot].as_mut().expect("staging buffer");
            buf.as_f64_mut()[..len].copy_from_slice(&src[off..off + len]);
            // SAFETY: `buf` is pinned; it is not touched again until the event of this copy
            // completed (waited above, or in the final loop); `dst` holds `off + len` doubles.
            let pending = unsafe {
                self.ts.upload_raw(
                    buf.as_f64().as_ptr().cast::<u8>(),
                    base + (off * 8) as u64,
                    len * 8,
                )
            }
            .map_err(be)?;
            // SAFETY: the event is stored and waited for before `buf` is reused or dropped.
            inflight[slot] = Some(unsafe { pending.detach() });
            off += len;
            k += 1;
        }
        for ev in inflight.into_iter().flatten() {
            ev.wait_host().map_err(be)?;
        }
        Ok(())
    }

    /// Upload a small vector (pageable copy; the data is `O(n)`).
    fn upload_small(&self, dst: &mut CudaSlice<f64>, src: &[f64]) -> Result<(), SrDeviceError> {
        let mut view = dst.slice_mut(..src.len());
        self.ts.stream().memcpy_htod(src, &mut view).map_err(be)
    }

    fn ensure(&self, slot: &mut Option<CudaSlice<f64>>, len: usize) -> Result<(), SrDeviceError> {
        if slot.as_ref().is_none_or(|b| b.len() < len) {
            *slot = None;
            *slot = Some(self.ts.stream().alloc_zeros(len.max(1)).map_err(be)?);
        }
        Ok(())
    }

    fn vec(&mut self, i: usize, len: usize) -> Result<(), SrDeviceError> {
        while self.vecs.len() <= i {
            self.vecs.push(self.ts.stream().alloc_zeros(1).map_err(be)?);
        }
        if self.vecs[i].len() < len {
            self.vecs[i] = self.ts.stream().alloc_zeros(len).map_err(be)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------
    // Direct SR
    // ------------------------------------------------------------------------------------

    /// Upload the saved-sample store `[n, samples]` (column-major, `O[i + s * n]`), the
    /// once-per-step input of the direct pipeline.
    pub fn upload_store(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
    ) -> Result<(), SrDeviceError> {
        if store.len() != n * samples || n == 0 || samples == 0 {
            return Err(SrDeviceError::Shape(format!(
                "store length {} != {n} x {samples}",
                store.len()
            )));
        }
        let t = Instant::now();
        let mut data = match self.store.take() {
            Some(s) if s.data.len() >= store.len() => s.data,
            _ => self.ts.stream().alloc_zeros(store.len()).map_err(be)?,
        };
        self.upload_chunked(&mut data, store)?;
        self.store = Some(Store { data, n, samples });
        self.timings.upload_store_s = t.elapsed().as_secs_f64();
        Ok(())
    }

    /// Gram, S/g assembly (device-resident `S` and `g`).
    ///
    /// `map` lists the active components (S order), `offset` is the index of the first
    /// parameter in the store (1 for real parameters), `ho` the `HO` vector of length `n`.
    pub fn assemble(
        &mut self,
        ho: &[f64],
        map: &[usize],
        offset: usize,
        sta_del: f64,
        step_dt: f64,
    ) -> Result<(), SrDeviceError> {
        let (n, samples) = {
            let st = self
                .store
                .as_ref()
                .ok_or_else(|| SrDeviceError::Shape("no store uploaded".into()))?;
            (st.n, st.samples)
        };
        let nmap = map.len();
        if ho.len() != n || nmap == 0 || map.iter().any(|&p| p + offset >= n) {
            return Err(SrDeviceError::Shape(
                "ho/map inconsistent with the store".into(),
            ));
        }
        let stream = self.stream();
        // vectors: 0 = ho, 1 = g (rhs / solution)
        let t = Instant::now();
        self.vec(0, n)?;
        self.vec(1, nmap)?;
        let mut gram = self.gram.take();
        self.ensure(&mut gram, n * n)?;
        self.gram = gram;
        let mut s = self.s.take();
        self.ensure(&mut s, nmap * nmap)?;
        self.s = s;
        {
            let mut v0 = std::mem::replace(&mut self.vecs[0], stream.alloc_zeros(1).map_err(be)?);
            self.upload_small(&mut v0, ho)?;
            self.vecs[0] = v0;
        }
        let map_i64: Vec<i64> = map.iter().map(|&p| p as i64).collect();
        if self.map_dev.as_ref().is_none_or(|m| m.len() < nmap) {
            self.map_dev = Some(stream.alloc_zeros(nmap).map_err(be)?);
        }
        stream
            .memcpy_htod(
                &map_i64,
                &mut self.map_dev.as_mut().expect("map").slice_mut(..nmap),
            )
            .map_err(be)?;
        self.sync()?;
        self.timings.upload_vec_s = t.elapsed().as_secs_f64();

        // G = O O^T (upper triangle)
        let t = Instant::now();
        let alpha = 1.0f64;
        let beta = 0.0f64;
        let st = self.store.as_ref().expect("store");
        let gram = self.gram.as_mut().expect("gram");
        // SAFETY: O is [n, samples] with ld n, G has n*n doubles; handle is bound to `stream`.
        blas_ok(
            unsafe {
                cublas_sys::cublasDsyrk_v2(
                    *self.blas.handle(),
                    cublas_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    samples as i32,
                    &alpha,
                    dp(&st.data, &stream) as *const f64,
                    n as i32,
                    &beta,
                    dp(gram, &stream) as *mut f64,
                    n as i32,
                )
            },
            "dsyrk",
        )?;
        self.sync()?;
        self.timings.gram_s = t.elapsed().as_secs_f64();

        // S and g
        let t = Instant::now();
        let ratio_diag = 1.0 + sta_del;
        let gptr = dp(self.gram.as_ref().expect("gram"), &stream);
        let mptr = dp_i64(self.map_dev.as_ref().expect("map"), &stream);
        let (ld, nm, off) = (n as i64, nmap as i64, offset as i64);
        let sptr = dp(self.s.as_ref().expect("s"), &stream);
        let mut b = stream.launch_builder(&self.k_assemble_s);
        b.arg(&gptr)
            .arg(&ld)
            .arg(&mptr)
            .arg(&nm)
            .arg(&off)
            .arg(&ratio_diag)
            .arg(&sptr);
        // SAFETY: kernel ABI (const double*, long, const long*, long, long, double, double*);
        // one thread per S entry, buffers sized above.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((nmap * nmap).div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(be)?;
        let hoptr = dp(&self.vecs[0], &stream);
        let goptr = dp(&self.vecs[1], &stream);
        let mut b = stream.launch_builder(&self.k_assemble_g);
        b.arg(&hoptr)
            .arg(&gptr)
            .arg(&ld)
            .arg(&mptr)
            .arg(&nm)
            .arg(&off)
            .arg(&step_dt)
            .arg(&goptr);
        // SAFETY: kernel ABI (const double*, const double*, long, const long*, long, long,
        // double, double*); one thread per active component.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nmap.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(be)?;
        self.sync()?;
        self.timings.assemble_s = t.elapsed().as_secs_f64();
        Ok(())
    }

    /// Download the assembled `S` (full, symmetric `nmap x nmap`) and `g` (test and diagnostics
    /// path; the production step never copies `S`).
    pub fn download_s_g(&mut self, nmap: usize) -> Result<(Vec<f64>, Vec<f64>), SrDeviceError> {
        self.sync()?;
        let s = self
            .ts
            .stream()
            .clone_dtoh(
                &self
                    .s
                    .as_ref()
                    .ok_or_else(|| SrDeviceError::Shape("no S".into()))?
                    .slice(..nmap * nmap),
            )
            .map_err(be)?;
        let g = self
            .ts
            .stream()
            .clone_dtoh(&self.vecs[1].slice(..nmap))
            .map_err(be)?;
        Ok((s, g))
    }

    /// Download the Gram matrix (upper triangle valid; lower triangle is whatever the library
    /// left): returned symmetrized.
    pub fn download_gram(&mut self) -> Result<Vec<f64>, SrDeviceError> {
        self.sync()?;
        let n = self
            .store
            .as_ref()
            .ok_or_else(|| SrDeviceError::Shape("no store".into()))?
            .n;
        let mut g = self
            .ts
            .stream()
            .clone_dtoh(
                &self
                    .gram
                    .as_ref()
                    .ok_or_else(|| SrDeviceError::Shape("no Gram".into()))?
                    .slice(..n * n),
            )
            .map_err(be)?;
        for j in 0..n {
            for i in j + 1..n {
                g[i + j * n] = g[j + i * n];
            }
        }
        Ok(g)
    }

    /// Cholesky factorization and solve of the resident `S x = g` (`DPOSV`, upper); `S` is
    /// overwritten by the factor. Returns `x`.
    pub fn factor_solve(&mut self, nmap: usize) -> Result<Vec<f64>, SrDeviceError> {
        let stream = self.stream();
        let t = Instant::now();
        let n = nmap as i32;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
        let sptr = dp(
            self.s
                .as_ref()
                .ok_or_else(|| SrDeviceError::Shape("no S".into()))?,
            &stream,
        );
        let mut lwork = 0i32;
        // SAFETY: valid handle; `lwork` receives the workspace size for this S.
        solver_ok(
            unsafe {
                cusolver_sys::cusolverDnDpotrf_bufferSize(
                    self.solver.cu(),
                    uplo,
                    n,
                    sptr as *mut f64,
                    n,
                    &mut lwork,
                )
            },
            "potrf_bufferSize",
        )?;
        let mut work: CudaSlice<f64> = stream.alloc_zeros((lwork as usize).max(1)).map_err(be)?;
        let info: CudaSlice<i32> = stream.alloc_zeros(2).map_err(be)?;
        let wptr = dp(&work, &stream);
        let iptr = {
            let (p, _g) = info.device_ptr(&stream);
            p
        };
        // SAFETY: S is nmap x nmap with ld nmap, workspace sized by bufferSize, devInfo is a
        // device int; the solver handle is bound to `stream`.
        solver_ok(
            unsafe {
                cusolver_sys::cusolverDnDpotrf(
                    self.solver.cu(),
                    uplo,
                    n,
                    sptr as *mut f64,
                    n,
                    wptr as *mut f64,
                    lwork,
                    iptr as *mut i32,
                )
            },
            "potrf",
        )?;
        let gptr = dp(&self.vecs[1], &stream);
        // SAFETY: B is the nmap-vector g (ldb nmap), factor from potrf above, devInfo+1.
        solver_ok(
            unsafe {
                cusolver_sys::cusolverDnDpotrs(
                    self.solver.cu(),
                    uplo,
                    n,
                    1,
                    sptr as *const f64,
                    n,
                    gptr as *mut f64,
                    n,
                    (iptr + 4) as *mut i32,
                )
            },
            "potrs",
        )?;
        self.sync()?;
        self.timings.solve_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let infos = stream.clone_dtoh(&info).map_err(be)?;
        let x = stream.clone_dtoh(&self.vecs[1].slice(..nmap)).map_err(be)?;
        self.timings.download_s = t.elapsed().as_secs_f64();
        drop(work.slice(..0));
        let _ = &mut work;
        if infos[0] != 0 || infos[1] != 0 {
            return Err(SrDeviceError::SolveFailed {
                info: if infos[0] != 0 { infos[0] } else { infos[1] },
            });
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(SrDeviceError::SolveFailed { info: 0 });
        }
        Ok(x)
    }

    /// Direct SR step on the resident store: Gram, S/g assembly, Cholesky solve, download of
    /// `x` (the production semantics of `cholesky_solve`).
    pub fn solve_direct(
        &mut self,
        ho: &[f64],
        map: &[usize],
        offset: usize,
        sta_del: f64,
        step_dt: f64,
    ) -> Result<Vec<f64>, SrDeviceError> {
        self.assemble(ho, map, offset, sta_del, step_dt)?;
        self.factor_solve(map.len())
    }

    // ------------------------------------------------------------------------------------
    // CG
    // ------------------------------------------------------------------------------------

    /// Upload the CG sample matrices `[components, samples]` (real, and imaginary for complex
    /// parameters), the once-per-step input of the CG pipeline.
    pub fn set_cg_operand(
        &mut self,
        real: &[f64],
        imag: Option<&[f64]>,
        components: usize,
        samples: usize,
    ) -> Result<(), SrDeviceError> {
        if real.len() != components * samples || imag.is_some_and(|i| i.len() != real.len()) {
            return Err(SrDeviceError::Shape("CG operand length mismatch".into()));
        }
        let t = Instant::now();
        let stream = self.stream();
        let mut re = match self.cg.take() {
            Some(c) if c.real.len() >= real.len() => c.real,
            _ => stream.alloc_zeros(real.len().max(1)).map_err(be)?,
        };
        self.upload_chunked(&mut re, real)?;
        let im = match imag {
            Some(i) => {
                let mut d = stream.alloc_zeros(i.len().max(1)).map_err(be)?;
                self.upload_chunked(&mut d, i)?;
                Some(d)
            }
            None => None,
        };
        self.cg = Some(Cg {
            real: re,
            imag: im,
            comp: components,
            samples,
        });
        self.timings.upload_store_s = t.elapsed().as_secs_f64();
        Ok(())
    }

    /// `y = A^T x`, `z = A y (+ A_i A_i^T x)` for the resident operand (device vectors).
    #[allow(clippy::too_many_arguments)]
    fn local_product(&self, x: u64, y: u64, yi: u64, z: u64) -> Result<(), SrDeviceError> {
        let cg = self.cg.as_ref().expect("CG operand");
        let stream = self.ts.stream();
        let (m, k) = (cg.comp as i32, cg.samples as i32);
        let (one, zero) = (1.0f64, 0.0f64);
        let h = *self.blas.handle();
        let pairs: [(&CudaSlice<f64>, u64, f64); 2] = [
            (&cg.real, y, zero),
            (cg.imag.as_ref().unwrap_or(&cg.real), yi, zero),
        ];
        let nmat = if cg.imag.is_some() { 2 } else { 1 };
        for (idx, (mat, yv, _)) in pairs.iter().enumerate().take(nmat) {
            let a = dp(mat, stream);
            // SAFETY: A is [m, k] with ld m; x has m entries, y has k entries.
            blas_ok(
                unsafe {
                    cublas_sys::cublasDgemv_v2(
                        h,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        m,
                        k,
                        &one,
                        a as *const f64,
                        m,
                        x as *const f64,
                        1,
                        &zero,
                        *yv as *mut f64,
                        1,
                    )
                },
                "dgemv T",
            )?;
            let beta = if idx == 0 { &zero } else { &one };
            // SAFETY: z has m entries; y has k entries.
            blas_ok(
                unsafe {
                    cublas_sys::cublasDgemv_v2(
                        h,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        m,
                        k,
                        &one,
                        a as *const f64,
                        m,
                        *yv as *const f64,
                        1,
                        beta,
                        z as *mut f64,
                        1,
                    )
                },
                "dgemv N",
            )?;
        }
        Ok(())
    }

    fn ddot(&self, n: usize, x: u64, y: u64) -> Result<f64, SrDeviceError> {
        let mut r = 0.0f64;
        // SAFETY: both vectors hold n doubles; host pointer mode returns r synchronously.
        blas_ok(
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.blas.handle(),
                    n as i32,
                    x as *const f64,
                    1,
                    y as *const f64,
                    1,
                    &mut r,
                )
            },
            "ddot",
        )?;
        Ok(r)
    }

    fn axpy(&self, n: usize, alpha: f64, x: u64, y: u64) -> Result<(), SrDeviceError> {
        // SAFETY: both vectors hold n doubles.
        blas_ok(
            unsafe {
                cublas_sys::cublasDaxpy_v2(
                    *self.blas.handle(),
                    n as i32,
                    &alpha,
                    x as *const f64,
                    1,
                    y as *mut f64,
                    1,
                )
            },
            "daxpy",
        )
    }

    fn scal(&self, n: usize, alpha: f64, x: u64) -> Result<(), SrDeviceError> {
        // SAFETY: x holds n doubles.
        blas_ok(
            unsafe {
                cublas_sys::cublasDscal_v2(*self.blas.handle(), n as i32, &alpha, x as *mut f64, 1)
            },
            "dscal",
        )
    }

    /// `z = inv_weight * (A A^T x) - (mean.x) mean + shift * diag * x` on device vectors
    /// (`SampledSrOperator::apply`).
    fn apply(
        &self,
        n: usize,
        x: u64,
        z: u64,
        inv_weight: f64,
        shift: f64,
    ) -> Result<(), SrDeviceError> {
        let stream = self.ts.stream();
        let mean = dp(&self.vecs[0], stream);
        let diag = dp(&self.vecs[1], stream);
        let y = dp(&self.vecs[2], stream);
        let yi = dp(&self.vecs[3], stream);
        let tmp = dp(&self.vecs[4], stream);
        self.local_product(x, y, yi, tmp)?;
        let coef = self.ddot(n, mean, x)?;
        let nn = n as i64;
        let mut b = stream.launch_builder(&self.k_combine);
        b.arg(&tmp)
            .arg(&x)
            .arg(&mean)
            .arg(&diag)
            .arg(&inv_weight)
            .arg(&coef)
            .arg(&shift)
            .arg(&nn)
            .arg(&z);
        // SAFETY: kernel ABI (const double* x3, const double*, double x3, long, double*);
        // all vectors hold n doubles.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (n.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(be)?;
        Ok(())
    }

    fn cg_vectors(&mut self, n: usize, samples: usize) -> Result<(), SrDeviceError> {
        // 0 mean, 1 diagonal, 2 y (samples), 3 y_imag (samples), 4 product, 5 solution,
        // 6 residual, 7 direction, 8 gradient, 9 scratch
        for i in 0..10 {
            let len = if i == 2 || i == 3 { samples } else { n };
            self.vec(i, len.max(1))?;
        }
        Ok(())
    }

    /// Sampled operator product `z = inv_weight*Re(O O^H) x - (mean.x) mean + shift*diag*x` for
    /// the resident operand (operand test path; downloads `z`).
    pub fn apply_operator(
        &mut self,
        x: &[f64],
        mean: &[f64],
        diag: &[f64],
        inv_weight: f64,
        shift: f64,
    ) -> Result<Vec<f64>, SrDeviceError> {
        let (comp, samples) = {
            let cg = self
                .cg
                .as_ref()
                .ok_or_else(|| SrDeviceError::Shape("no CG operand".into()))?;
            (cg.comp, cg.samples)
        };
        if x.len() != comp || mean.len() != comp || diag.len() != comp {
            return Err(SrDeviceError::Shape("vector length != components".into()));
        }
        self.cg_vectors(comp, samples)?;
        let stream = self.stream();
        for (i, v) in [(0usize, mean), (1, diag), (7, x)] {
            let mut d = std::mem::replace(&mut self.vecs[i], stream.alloc_zeros(1).map_err(be)?);
            self.upload_small(&mut d, v)?;
            self.vecs[i] = d;
        }
        let (xp, zp) = (dp(&self.vecs[7], &stream), dp(&self.vecs[9], &stream));
        self.apply(comp, xp, zp, inv_weight, shift)?;
        self.sync()?;
        stream.clone_dtoh(&self.vecs[9].slice(..comp)).map_err(be)
    }

    /// The CG loop of `SampledSrOperator::solve` on the resident operand.
    ///
    /// `gradient`, `mean`, `diagonal` are the per-step small inputs (length `components`);
    /// `inv_weight`, `shift`, `tolerance`, `max_iterations` as in the host solver. The threshold
    /// is `tolerance^2 * n * n` evaluated left to right as in `stcopt_cg_impl.c:265`.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_cg(
        &mut self,
        gradient: &[f64],
        mean: &[f64],
        diagonal: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<CgOutcome, SrDeviceError> {
        let (n, samples) = {
            let cg = self
                .cg
                .as_ref()
                .ok_or_else(|| SrDeviceError::Shape("no CG operand".into()))?;
            (cg.comp, cg.samples)
        };
        if gradient.len() != n || mean.len() != n || diagonal.len() != n {
            return Err(SrDeviceError::Shape("vector length != components".into()));
        }
        self.cg_vectors(n, samples)?;
        let stream = self.stream();
        let t = Instant::now();
        for (i, v) in [
            (0usize, mean),
            (1, diagonal),
            (8, gradient),
            (7, gradient),
            (6, gradient),
        ] {
            let mut d = std::mem::replace(&mut self.vecs[i], stream.alloc_zeros(1).map_err(be)?);
            self.upload_small(&mut d, v)?;
            self.vecs[i] = d;
        }
        {
            let zeros = vec![0.0f64; n];
            let mut d = std::mem::replace(&mut self.vecs[5], stream.alloc_zeros(1).map_err(be)?);
            self.upload_small(&mut d, &zeros)?;
            self.vecs[5] = d;
        }
        self.sync()?;
        self.timings.upload_vec_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let p = |i: usize| dp(&self.vecs[i], &stream);
        let (sol, res, dir, grad, prod) = (p(5), p(6), p(7), p(8), p(4));
        let threshold = tolerance * tolerance * n as f64 * n as f64;
        let mut delta = self.ddot(n, res, res)?;
        let mut iterations = 0usize;
        for iteration in 1..=max_iterations {
            iterations = iteration;
            if delta < threshold {
                iterations = iteration - 1;
                break;
            }
            self.apply(n, dir, prod, inv_weight, shift)?;
            let dq = self.ddot(n, dir, prod)?;
            let alpha = delta / dq;
            self.axpy(n, alpha, dir, sol)?;
            if iteration % 20 == 0 {
                self.apply(n, sol, res, inv_weight, shift)?;
                // residual = gradient - residual (negation is exact)
                self.scal(n, -1.0, res)?;
                self.axpy(n, 1.0, grad, res)?;
            } else {
                self.axpy(n, -alpha, prod, res)?;
            }
            let delta_new = self.ddot(n, res, res)?;
            let beta = delta_new / delta;
            // C:336 rounds the quotient and multiplies it by the old norm.
            delta = beta * delta;
            self.scal(n, beta, dir)?;
            self.axpy(n, 1.0, res, dir)?;
        }
        self.sync()?;
        self.timings.solve_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let solution = stream.clone_dtoh(&self.vecs[5].slice(..n)).map_err(be)?;
        self.timings.download_s = t.elapsed().as_secs_f64();
        Ok(CgOutcome {
            solution,
            iterations,
            delta,
        })
    }

    /// Whether any element of the device vector `i` is nonfinite (diagnostic).
    pub fn any_nonfinite(&mut self, x: &[f64]) -> Result<bool, SrDeviceError> {
        let stream = self.stream();
        let mut d = stream.clone_htod(x).map_err(be)?;
        if self.ints.is_none() {
            self.ints = Some(stream.alloc_zeros(1).map_err(be)?);
        }
        let flag = self.ints.as_mut().expect("flag");
        stream.memset_zeros(flag).map_err(be)?;
        let xp = dp(&d, &stream);
        let nn = x.len() as i64;
        let fp = {
            let (p, _g) = flag.device_ptr(&stream);
            p
        };
        let mut b = stream.launch_builder(&self.k_nonfinite);
        b.arg(&xp).arg(&nn).arg(&fp);
        // SAFETY: kernel ABI (const double*, long, int*).
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (x.len().div_ceil(256).max(1) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(be)?;
        let out = stream.clone_dtoh(flag).map_err(be)?;
        drop(d.slice(..0));
        let _ = &mut d;
        Ok(out[0] != 0)
    }
}

fn dp_i64(s: &CudaSlice<i64>, st: &Arc<CudaStream>) -> u64 {
    let (p, _g) = s.device_ptr(st);
    p
}
