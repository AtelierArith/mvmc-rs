//! Single-stream raw CUDA execution for the engines' fast device forward
//! (`cuda` feature).
//!
//! The tenferro-native device forward issues ~700 eager ops per call, each
//! with host-side provider bookkeeping, and every raw-kernel scope flushes
//! CubeCL with a host barrier. [`with_raw_exec`] instead runs a whole forward
//! inside one `CudaExecSession::with_raw` scope: raw NVRTC kernels and cuBLAS
//! GEMMs are enqueued on tenferro's own captured stream, inputs are copied in
//! stream order, and the only host barrier is the final download.
//!
//! Engines keep weights and scratch resident in [`DeviceBuffer`]s: plain
//! driver allocations on tenferro's primary context (retained by the
//! buffer), freed when the buffer drops. They bypass CubeCL's memory pool on
//! purpose: that pool is process-global per device and never returns memory
//! to the driver, so gigabytes of engine weights parked in it would stay
//! reserved after the engine is gone. Addresses are passed to kernels as
//! 64-bit scalars, which is ABI-identical to a pointer parameter.
//!
//! Compiled modules and cuBLAS handles are `!Send` / context-bound, so they
//! live in per-thread caches keyed by runtime identity (CUDA eager sessions
//! run their callback on the calling thread). cuBLAS handles are created on
//! the tenferro primary context and intentionally never destroyed (one per
//! thread and runtime; destroying them at thread exit could race context
//! teardown).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use cudarc::cublas::{result as cublas, sys as cublas_sys};
use cudarc::driver::{result as driver, sys as driver_sys};
use tenferro_gpu::cuda::raw::{with_cuda_engine, Session};
use tenferro_gpu::cuda::{CudaRuntime, CudaRuntimeIdentity};
use tenferro_tensor::Engine;

mod rng;
pub use rng::DeviceRng;
mod f64_linalg;

/// Traffic issued through `RawExec` on the calling host thread. These counters do
/// not include transfers or implicit barriers inside CUDA libraries or the provider.
/// Take snapshots around a workload; no process-global reset or contended atomics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawExecCounters {
    pub upload_calls: u64,
    pub upload_bytes: u64,
    pub download_calls: u64,
    pub download_bytes: u64,
    pub synchronizations: u64,
    pub scopes: u64,
}

impl std::ops::Sub for RawExecCounters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            upload_calls: self.upload_calls - rhs.upload_calls,
            upload_bytes: self.upload_bytes - rhs.upload_bytes,
            download_calls: self.download_calls - rhs.download_calls,
            download_bytes: self.download_bytes - rhs.download_bytes,
            synchronizations: self.synchronizations - rhs.synchronizations,
            scopes: self.scopes - rhs.scopes,
        }
    }
}

thread_local! {
    static COUNTERS: Cell<RawExecCounters> = Cell::new(RawExecCounters::default());
}

/// Snapshot of explicit raw-execution traffic on this thread.
pub fn raw_exec_counters() -> RawExecCounters {
    COUNTERS.with(Cell::get)
}

fn count(f: impl FnOnce(&mut RawExecCounters)) {
    COUNTERS.with(|cell| {
        let mut counters = cell.get();
        f(&mut counters);
        cell.set(counters);
    });
}

/// A device address (bytes).
pub type DevPtr = u64;

/// Byte address of float `index` after `base`.
#[inline]
pub fn at(base: DevPtr, index: usize) -> DevPtr {
    base + 4 * index as u64
}

/// Byte address of `f64` element `index` after `base`.
#[inline]
pub fn at8(base: DevPtr, index: usize) -> DevPtr {
    base + 8 * index as u64
}

fn backend(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported(op, message.into())
}

/// A dense f32 device allocation on tenferro's primary context (see the
/// module docs). Dropping it frees the memory; the owner must not drop it
/// while enqueued work still uses it (every [`with_raw_exec`] scope ends
/// synchronized).
pub struct DeviceBuffer {
    ptr: driver_sys::CUdeviceptr,
    ctx: driver_sys::CUcontext,
    device: driver_sys::CUdevice,
    floats: usize,
    bytes: usize,
    runtime: CudaRuntimeIdentity,
}

// SAFETY: the buffer is a device address plus a retained primary context;
// freeing pushes that context on whichever thread drops it.
unsafe impl Send for DeviceBuffer {}
// SAFETY: shared references expose only the length.
unsafe impl Sync for DeviceBuffer {}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: `ctx` is the primary context retained at allocation, so it
        // is alive; the allocation belongs to it and is no longer in use.
        unsafe {
            if driver_sys::cuCtxPushCurrent_v2(self.ctx).result().is_ok() {
                let _ = driver::free_sync(self.ptr);
                let mut previous = std::ptr::null_mut();
                let _ = driver_sys::cuCtxPopCurrent_v2(&mut previous);
            }
            let _ = driver::primary_ctx::release(self.device);
        }
    }
}

impl DeviceBuffer {
    /// Number of f32 elements.
    pub fn len(&self) -> usize {
        self.floats
    }

    /// Number of bytes allocated.
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("floats", &self.floats)
            .finish()
    }
}

/// A compiled NVRTC module with its functions by name.
pub struct RawModule {
    runtime: CudaRuntimeIdentity,
    source: usize,
    disable_fmad: bool,
    kernels: Vec<&'static str>,
    functions: HashMap<&'static str, Function>,
    module: driver_sys::CUmodule,
    context: driver_sys::CUcontext,
    _runtime: CudaRuntime,
}

/// A function whose owning module is retained by the caller's `RawModule`.
pub struct Function(driver_sys::CUfunction);

impl Drop for RawModule {
    fn drop(&mut self) {
        // The runtime retains the context. Modules are thread-local and every raw
        // scope synchronizes before its caller may release the last reference.
        unsafe {
            if driver_sys::cuCtxPushCurrent_v2(self.context)
                .result()
                .is_ok()
            {
                let _ = driver::module::unload(self.module);
                let mut previous = std::ptr::null_mut();
                let _ = driver_sys::cuCtxPopCurrent_v2(&mut previous);
            }
        }
    }
}

impl RawModule {
    /// A function of the module.
    pub fn function(&self, name: &str) -> tenferro_tensor::Result<&Function> {
        self.functions
            .get(name)
            .ok_or_else(|| backend("tenferro-ext::raw_exec", format!("no kernel `{name}`")))
    }
}

thread_local! {
    static MODULES: RefCell<Vec<Rc<RawModule>>> = const { RefCell::new(Vec::new()) };
    static BLAS: RefCell<Vec<(CudaRuntimeIdentity, usize)>> = const { RefCell::new(Vec::new()) };
}

const MAX_CACHED_MODULES: usize = 16;

/// cuBLAS transpose flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Use the matrix as stored (column-major).
    N,
    /// Use its transpose.
    T,
}

impl Op {
    fn sys(self) -> cublas_sys::cublasOperation_t {
        match self {
            Op::N => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            Op::T => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        }
    }
}

/// A kernel argument.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// A device address.
    P(DevPtr),
    /// A 32-bit signed integer.
    I(i32),
    /// A 64-bit signed integer.
    L(i64),
    /// A 32-bit float.
    F(f32),
    /// A 64-bit float.
    D(f64),
}

/// One strided-batched GEMM operand: base address, leading dimension and
/// batch stride (floats).
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    /// Base device address.
    pub ptr: DevPtr,
    /// Leading dimension (floats).
    pub ld: usize,
    /// Batch stride (floats).
    pub stride: usize,
}

impl Mat {
    /// An operand with leading dimension `ld` and batch stride `stride`.
    pub fn new(ptr: DevPtr, ld: usize, stride: usize) -> Self {
        Self { ptr, ld, stride }
    }
}

/// The execution context of one raw scope (see the module docs).
pub struct RawExec<'r, 's> {
    _raw: &'r Session<'s>,
    runtime: CudaRuntime,
    identity: CudaRuntimeIdentity,
    context: driver_sys::CUcontext,
    stream: u64,
    blas: cublas_sys::cublasHandle_t,
    arch: &'r str,
}

impl<'r, 's> RawExec<'r, 's> {
    /// The NVRTC architecture of this device (`compute_XY`).
    pub fn arch(&self) -> &str {
        self.arch
    }

    /// CUDA stream handle, valid only inside this raw scope.
    pub fn stream_handle(&self) -> u64 {
        self.stream
    }

    /// Borrow the active engine's scope for runtime-owned scratch allocations.
    pub fn raw_session(&self) -> &Session<'s> {
        self._raw
    }

    /// Allocate a dense f32 buffer of `floats` elements (uninitialized) on
    /// the scope's primary context.
    pub fn alloc(&self, floats: usize) -> tenferro_tensor::Result<DeviceBuffer> {
        self.alloc_bytes(4 * floats)
    }

    /// Allocate a dense `f64` buffer of `n` elements (uninitialized).
    pub fn alloc_f64(&self, n: usize) -> tenferro_tensor::Result<DeviceBuffer> {
        self.alloc_bytes(8 * n)
    }

    /// Allocate `nbytes` bytes (uninitialized) on the scope's primary context.
    pub fn alloc_bytes(&self, nbytes: usize) -> tenferro_tensor::Result<DeviceBuffer> {
        let fail = |what: &str, error: driver::DriverError| {
            backend("tenferro-ext::raw_exec::alloc", format!("{what}: {error}"))
        };
        // SAFETY: the raw scope keeps tenferro's primary context current.
        unsafe {
            let mut device = 0;
            driver_sys::cuCtxGetDevice(&mut device)
                .result()
                .map_err(|e| fail("cuCtxGetDevice", e))?;
            let ctx = driver::primary_ctx::retain(device).map_err(|e| fail("retain", e))?;
            match driver::malloc_sync(nbytes.max(1)) {
                Ok(ptr) => Ok(DeviceBuffer {
                    ptr,
                    ctx,
                    device,
                    floats: nbytes / 4,
                    bytes: nbytes,
                    runtime: self.identity.clone(),
                }),
                Err(error) => {
                    let _ = driver::primary_ctx::release(device);
                    Err(fail(&format!("cuMemAlloc of {nbytes} bytes"), error))
                }
            }
        }
    }

    /// The device address of `buffer`.
    pub fn ptr(&self, buffer: &DeviceBuffer) -> tenferro_tensor::Result<DevPtr> {
        if buffer.ctx != self.context || buffer.runtime != self.identity {
            return Err(backend(
                "tenferro-ext::raw_exec::ptr",
                "buffer belongs to a different CUDA runtime",
            ));
        }
        Ok(buffer.ptr as DevPtr)
    }

    /// Compile (once per thread and runtime) and return the module of
    /// `source` with the listed kernels. `source` must be `'static` (its
    /// address keys the cache).
    pub fn module(
        &self,
        source: &'static str,
        kernels: &[&'static str],
    ) -> tenferro_tensor::Result<Rc<RawModule>> {
        self.module_with_options(source, kernels, false)
    }

    /// Compile a module with fused multiply-add disabled for C-order numerical kernels.
    pub fn module_without_fmad(
        &self,
        source: &'static str,
        kernels: &[&'static str],
    ) -> tenferro_tensor::Result<Rc<RawModule>> {
        self.module_with_options(source, kernels, true)
    }

    fn module_with_options(
        &self,
        source: &'static str,
        kernels: &[&'static str],
        disable_fmad: bool,
    ) -> tenferro_tensor::Result<Rc<RawModule>> {
        let runtime = self.identity.clone();
        let key = source.as_ptr() as usize;
        if let Some(module) = MODULES.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|m| {
                    m.runtime == runtime
                        && m.source == key
                        && m.disable_fmad == disable_fmad
                        && m.kernels == kernels
                })
                .cloned()
        }) {
            return Ok(module);
        }
        let mut options = cudarc::nvrtc::CompileOptions::default();
        options
            .options
            .push(format!("--gpu-architecture={}", self.arch));
        options.options.push("--std=c++14".into());
        if disable_fmad {
            options.options.push("--fmad=false".into());
        }
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
            .map_err(|e| backend("raw_exec::module", e.to_string()))?;
        let ptx = std::ffi::CString::new(ptx.to_src())
            .map_err(|e| backend("raw_exec::module", e.to_string()))?;
        // SAFETY: the raw scope has made this runtime's CUDA context current.
        let module = unsafe { driver::module::load_data(ptx.as_ptr().cast()) }
            .map_err(|e| backend("raw_exec::module", e.to_string()))?;
        let mut functions = HashMap::new();
        for name in kernels {
            let c_name = std::ffi::CString::new(*name)
                .map_err(|e| backend("raw_exec::module", e.to_string()))?;
            let function = unsafe { driver::module::get_function(module, c_name) };
            match function {
                Ok(function) => {
                    functions.insert(*name, Function(function));
                }
                Err(e) => {
                    unsafe {
                        let _ = driver::module::unload(module);
                    }
                    return Err(backend("raw_exec::module", e.to_string()));
                }
            }
        }
        let module = Rc::new(RawModule {
            runtime,
            source: key,
            disable_fmad,
            kernels: kernels.to_vec(),
            functions,
            module,
            context: self.context,
            _runtime: self.runtime.clone(),
        });
        MODULES.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= MAX_CACHED_MODULES {
                cache.remove(0);
            }
            cache.push(module.clone());
        });
        Ok(module)
    }

    /// Enqueue `function` with grid/block geometry and dynamic shared memory.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's parameter list, and every address must
    /// stay valid (and in range for the launch) until the scope synchronizes.
    pub unsafe fn launch(
        &self,
        function: &Function,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        args: &[Arg],
    ) -> tenferro_tensor::Result<()> {
        if block == 0 || block > 1024 || grid.0 > i32::MAX as u32 {
            return Err(backend("raw_exec::launch", "invalid CUDA launch geometry"));
        }
        let mut params: Vec<*mut std::ffi::c_void> = args
            .iter()
            .map(|arg| match arg {
                Arg::P(v) => (v as *const u64).cast_mut().cast(),
                Arg::I(v) => (v as *const i32).cast_mut().cast(),
                Arg::L(v) => (v as *const i64).cast_mut().cast(),
                Arg::F(v) => (v as *const f32).cast_mut().cast(),
                Arg::D(v) => (v as *const f64).cast_mut().cast(),
            })
            .collect();
        // SAFETY: forwarded ABI and lifetime contract; CUDA copies parameters on enqueue.
        unsafe {
            driver::launch_kernel(
                function.0,
                (grid.0.max(1), grid.1.max(1), grid.2.max(1)),
                (block, 1, 1),
                shared,
                self.stream as driver_sys::CUstream,
                &mut params,
            )
        }
        .map_err(|e| backend("raw_exec::launch", e.to_string()))
    }

    /// `C = alpha op(A) op(B) + beta C` for column-major `C` `(m, n)`.
    ///
    /// # Safety
    ///
    /// Operand addresses and leading dimensions must describe in-range device
    /// memory that stays valid until the scope synchronizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: DevPtr,
        lda: usize,
        b: DevPtr,
        ldb: usize,
        beta: f32,
        c: DevPtr,
        ldc: usize,
    ) -> tenferro_tensor::Result<()> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        // SAFETY: forwarded caller contract; alpha/beta are host scalars
        // (pointer mode host, the handle default).
        unsafe {
            cublas::sgemm(
                self.blas,
                op_a.sys(),
                op_b.sys(),
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a as *const f32,
                lda.max(1) as i32,
                b as *const f32,
                ldb.max(1) as i32,
                &beta,
                c as *mut f32,
                ldc.max(1) as i32,
            )
        }
        .map_err(|error| backend("tenferro-ext::raw_exec::gemm", error.to_string()))
    }

    /// Strided-batched [`Self::gemm`].
    ///
    /// # Safety
    ///
    /// As [`Self::gemm`], for every batch member.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_batched(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: Mat,
        b: Mat,
        beta: f32,
        c: Mat,
        batch: usize,
    ) -> tenferro_tensor::Result<()> {
        if m == 0 || n == 0 || batch == 0 {
            return Ok(());
        }
        // SAFETY: forwarded caller contract.
        unsafe {
            cublas::sgemm_strided_batched(
                self.blas,
                op_a.sys(),
                op_b.sys(),
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a.ptr as *const f32,
                a.ld.max(1) as i32,
                a.stride as i64,
                b.ptr as *const f32,
                b.ld.max(1) as i32,
                b.stride as i64,
                &beta,
                c.ptr as *mut f32,
                c.ld.max(1) as i32,
                c.stride as i64,
                batch as i32,
            )
        }
        .map_err(|error| backend("tenferro-ext::raw_exec::gemm_batched", error.to_string()))
    }

    /// Copy host data to `dst` in stream order. Pageable sources are staged
    /// by the driver before this returns, so `data` may be dropped after.
    ///
    /// # Safety
    ///
    /// `dst` must have room for `data`.
    pub unsafe fn upload<T: Copy>(&self, dst: DevPtr, data: &[T]) -> tenferro_tensor::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: forwarded caller contract.
        unsafe { driver::memcpy_htod_async(dst, data, self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::upload", error.to_string()))?;
        count(|c| {
            c.upload_calls += 1;
            c.upload_bytes += std::mem::size_of_val(data) as u64;
        });
        Ok(())
    }

    /// Copy `src` into `out` after all enqueued work and wait for it.
    ///
    /// # Safety
    ///
    /// `src` must hold at least `out.len()` elements.
    pub unsafe fn download<T: Copy>(
        &self,
        out: &mut [T],
        src: DevPtr,
    ) -> tenferro_tensor::Result<()> {
        if !out.is_empty() {
            // SAFETY: forwarded caller contract.
            unsafe { driver::memcpy_dtoh_async(out, src, self.stream as driver_sys::CUstream) }
                .map_err(|error| backend("tenferro-ext::raw_exec::download", error.to_string()))?;
            count(|c| {
                c.download_calls += 1;
                c.download_bytes += std::mem::size_of_val(out) as u64;
            });
        }
        self.synchronize()
    }

    /// Fill `floats` floats at `dst` with zero bytes, in stream order.
    ///
    /// # Safety
    ///
    /// `dst` must have room for `floats` floats.
    pub unsafe fn zero(&self, dst: DevPtr, floats: usize) -> tenferro_tensor::Result<()> {
        // SAFETY: forwarded caller contract.
        unsafe { self.zero_bytes(dst, 4 * floats) }
    }

    /// Fill `nbytes` bytes at `dst` with zero, in stream order.
    ///
    /// # Safety
    ///
    /// `dst` must have room for `nbytes` bytes.
    pub unsafe fn zero_bytes(&self, dst: DevPtr, nbytes: usize) -> tenferro_tensor::Result<()> {
        if nbytes == 0 {
            return Ok(());
        }
        // SAFETY: forwarded caller contract.
        unsafe { driver::memset_d8_async(dst, 0, nbytes, self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::zero", error.to_string()))
    }

    /// Enqueue a device-to-device copy on the engine's stream.
    ///
    /// # Safety
    /// Both non-overlapping ranges must belong to this context, cover `nbytes`,
    /// and remain alive until the scope synchronizes.
    pub unsafe fn copy_device(
        &self,
        dst: DevPtr,
        src: DevPtr,
        nbytes: usize,
    ) -> tenferro_tensor::Result<()> {
        if nbytes == 0 {
            return Ok(());
        }
        unsafe { driver::memcpy_dtod_async(dst, src, nbytes, self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::copy_device", error.to_string()))
    }

    /// Block until every enqueued operation has completed.
    pub fn synchronize(&self) -> tenferro_tensor::Result<()> {
        count(|c| c.synchronizations += 1);
        // SAFETY: the stream is the scope's captured tenferro stream.
        unsafe { driver::stream::synchronize(self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::synchronize", error.to_string()))
    }
}

/// The NVRTC virtual architecture: `TENFERRO_CUDA_ARCH` when set, else the
/// session device's compute capability.
fn cuda_arch<B: Engine + ?Sized>(session: &mut B) -> tenferro_tensor::Result<String> {
    if let Ok(arch) = std::env::var("TENFERRO_CUDA_ARCH") {
        if !arch.is_empty() {
            return Ok(arch);
        }
    }
    thread_local! {
        static ARCH: RefCell<Vec<(CudaRuntimeIdentity, String)>> = const { RefCell::new(Vec::new()) };
    }
    let (device, identity) = with_cuda_engine(session, |cuda| {
        (cuda.runtime().device_id(), cuda.runtime_identity())
    })
    .ok_or_else(|| backend("tenferro-ext::raw_exec", "CUDA execution session required"))?;
    if let Some(arch) = ARCH.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(id, _)| *id == identity)
            .map(|(_, arch)| arch.clone())
    }) {
        return Ok(arch);
    }
    let devices = tenferro_gpu::cuda::cuda_devices().map_err(|error| {
        backend(
            "tenferro-ext::raw_exec",
            format!("device discovery: {error}"),
        )
    })?;
    let info = devices
        .iter()
        .find(|info| info.id() == device)
        .ok_or_else(|| backend("tenferro-ext::raw_exec", "session device not discovered"))?;
    let capability = info.compute_capability();
    let arch = format!("compute_{}{}", capability.major, capability.minor);
    ARCH.with(|cache| cache.borrow_mut().push((identity, arch.clone())));
    Ok(arch)
}

fn blas_handle(
    identity: CudaRuntimeIdentity,
) -> tenferro_tensor::Result<cublas_sys::cublasHandle_t> {
    if let Some(handle) = BLAS.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(id, _)| *id == identity)
            .map(|(_, handle)| *handle)
    }) {
        return Ok(handle as cublas_sys::cublasHandle_t);
    }
    let handle = cublas::create_handle()
        .map_err(|error| backend("tenferro-ext::raw_exec", format!("cublasCreate: {error}")))?;
    BLAS.with(|cache| cache.borrow_mut().push((identity, handle as usize)));
    Ok(handle)
}

/// Run `f` in one raw scope on the session's CUDA stream (see the module
/// docs). Errors when the session is not a CUDA session.
///
/// `f` must leave no work in flight that references memory it does not own
/// past its return: end with [`RawExec::download`] or
/// [`RawExec::synchronize`] (this function synchronizes again on success and
/// on error, so retained buffers may be dropped by the caller afterwards).
pub fn with_raw_exec<B, R>(
    session: &mut B,
    label: &'static str,
    f: impl FnOnce(&mut RawExec<'_, '_>) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R>
where
    B: Engine + ?Sized,
{
    let arch = cuda_arch(session)?;
    let result = with_cuda_engine(session, |cuda| {
        let runtime = cuda.runtime().clone();
        let identity = cuda.runtime_identity();
        cuda.with_raw(label, |raw| {
            Ok((|| -> tenferro_tensor::Result<R> {
                let blas = blas_handle(identity.clone())?;
                // SAFETY: the stream handle is used only inside this scope.
                let stream = raw.stream().raw_handle();
                // SAFETY: valid handle and the scope's captured stream.
                unsafe { cublas::set_stream(blas, stream as cublas_sys::cudaStream_t) }
                    .map_err(|error| backend(label, format!("cublasSetStream: {error}")))?;
                let mut context = std::ptr::null_mut();
                // SAFETY: the provider has entered the captured stream's context.
                unsafe { driver_sys::cuCtxGetCurrent(&mut context).result() }
                    .map_err(|error| backend(label, format!("cuCtxGetCurrent: {error}")))?;
                let mut exec = RawExec {
                    _raw: raw,
                    runtime,
                    identity,
                    context,
                    stream,
                    blas,
                    arch: &arch,
                };
                count(|c| c.scopes += 1);
                let result = f(&mut exec);
                let synced = exec.synchronize();
                match result {
                    Ok(value) => synced.map(|()| value),
                    Err(error) => Err(error),
                }
            })())
        })
        .map_err(|error| backend(label, error.to_string()))?
    })
    .ok_or_else(|| backend(label, "CUDA execution session required"))?;
    result
}

/// Alignment of arena entries (floats; 256 bytes, enough for vector loads
/// and cuBLAS's preferred operand alignment).
pub const ARENA_ALIGN: usize = 64;

/// Round `floats` up to [`ARENA_ALIGN`].
#[inline]
pub fn align(floats: usize) -> usize {
    floats.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// Host staging of one weight arena: tensors are appended at aligned
/// offsets, then uploaded as one device allocation ([`Self::upload`]).
#[derive(Default)]
pub struct ArenaBuilder {
    data: Vec<f32>,
}

impl ArenaBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `values` and return their float offset.
    pub fn push(&mut self, values: &[f32]) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(values);
        self.data.resize(align(self.data.len()), 0.0);
        offset
    }

    /// Append the transpose of column-major `(rows, cols)` `values` (i.e.
    /// store it column-major as `(cols, rows)`) and return its offset.
    pub fn push_transposed(&mut self, values: &[f32], rows: usize, cols: usize) -> usize {
        debug_assert_eq!(values.len(), rows * cols);
        use rayon::prelude::*;
        let offset = self.data.len();
        self.data.resize(offset + rows * cols, 0.0);
        const BLOCK: usize = 64;
        // Each task fills BLOCK destination columns (source rows r0..).
        self.data[offset..]
            .par_chunks_mut(cols * BLOCK)
            .enumerate()
            .for_each(|(block, dst)| {
                let r0 = block * BLOCK;
                let r1 = (r0 + BLOCK).min(rows);
                for c0 in (0..cols).step_by(BLOCK) {
                    for c in c0..(c0 + BLOCK).min(cols) {
                        for r in r0..r1 {
                            dst[c + cols * (r - r0)] = values[r + rows * c];
                        }
                    }
                }
            });
        self.data.resize(align(self.data.len()), 0.0);
        offset
    }

    /// Append parts given row-major `(inp, out_k)` (i.e. column-major
    /// `(out_k, inp)`), stacked along the output axis into one column-major
    /// `(sum out_k, inp)` matrix, and return its offset.
    pub fn push_stacked(&mut self, parts: &[(&[f32], usize)], inp: usize) -> usize {
        let offset = self.data.len();
        let rows: usize = parts.iter().map(|(_, out)| out).sum();
        self.data.reserve(rows * inp);
        for i in 0..inp {
            for (values, out) in parts {
                debug_assert_eq!(values.len(), out * inp);
                self.data.extend_from_slice(&values[i * out..(i + 1) * out]);
            }
        }
        self.data.resize(align(self.data.len()), 0.0);
        offset
    }

    /// The staged floats (aligned padding included).
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Floats staged so far.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether nothing is staged.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Allocate a device buffer and copy the staged data into it.
    pub fn upload(self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<DeviceBuffer> {
        let buffer = exec.alloc(self.data.len())?;
        let ptr = exec.ptr(&buffer)?;
        // SAFETY: the buffer holds exactly `data.len()` floats; pageable
        // uploads are staged before returning, so `data` may be dropped.
        unsafe { exec.upload(ptr, &self.data)? };
        exec.synchronize()?;
        Ok(buffer)
    }
}
