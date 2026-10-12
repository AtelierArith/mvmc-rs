//! cuRAND library stream used by Julia's `CUDA.rand(UInt32, ...)`.
//! This is the library's interleaved XORWOW sequence, not `curand_init` per walker.

use super::{backend, driver, driver_sys, DeviceBuffer, RawExec};
use cudarc::curand::{result as curand, sys};

/// Independently owned cuRAND generator, retaining its device's primary context.
/// Keep it alive across raw scopes to preserve the sequence across Opt steps.
/// Generation requires exclusive access and runs on the current scope's stream.
pub struct DeviceRng {
    handle: sys::curandGenerator_t,
    ctx: driver_sys::CUcontext,
    device: driver_sys::CUdevice,
}

// SAFETY: use is exclusive, scopes synchronize before returning, and each operation
// checks the current context. Drop binds the retained context on the dropping thread.
unsafe impl Send for DeviceRng {}

impl DeviceRng {
    /// Julia CUDA.jl's default library generator, seed and offset-zero initialization.
    pub fn new(exec: &RawExec<'_, '_>, seed: u64) -> tenferro_tensor::Result<Self> {
        let mut device = 0;
        // SAFETY: the raw scope has a current primary context.
        let ctx = unsafe {
            driver_sys::cuCtxGetDevice(&mut device)
                .result()
                .map_err(|e| backend("DeviceRng", e.to_string()))?;
            driver::primary_ctx::retain(device).map_err(|e| backend("DeviceRng", e.to_string()))?
        };
        let handle = match curand::create_generator() {
            Ok(handle) => handle,
            Err(e) => {
                // SAFETY: balances the retain above, no live generator.
                unsafe {
                    let _ = driver::primary_ctx::release(device);
                }
                return Err(backend(
                    "DeviceRng",
                    format!("curandCreateGenerator: {e:?}"),
                ));
            }
        };
        let mut rng = Self {
            handle,
            ctx,
            device,
        };
        rng.reseed(exec, seed)?;
        Ok(rng)
    }

    fn bind(&mut self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<()> {
        if exec.context != self.ctx {
            return Err(backend(
                "DeviceRng",
                "generator belongs to a different CUDA context",
            ));
        }
        // SAFETY: a live exclusively borrowed generator on its retained context.
        unsafe {
            curand::set_stream(self.handle, exec.stream as sys::cudaStream_t)
                .map_err(|e| backend("DeviceRng", format!("curandSetStream: {e:?}")))
        }
    }

    /// Restart at offset zero. Normal sampling calls only `fill_u32`, never reseeds.
    pub fn reseed(&mut self, exec: &RawExec<'_, '_>, seed: u64) -> tenferro_tensor::Result<()> {
        self.bind(exec)?;
        // SAFETY: the handle is live, exclusive, and bound to the scope's stream/context.
        unsafe {
            curand::set_seed(self.handle, seed)
                .and_then(|()| curand::set_offset(self.handle, 0))
                .and_then(|()| sys::curandGenerateSeeds(self.handle).result())
                .map_err(|e| backend("DeviceRng", format!("seed cuRAND: {e:?}")))
        }
    }

    /// Generate into the first `words` UInt32 slots of a resident buffer without a
    /// host tape or synchronization. Subsequent kernels use stream ordering.
    pub fn fill_u32(
        &mut self,
        exec: &RawExec<'_, '_>,
        buffer: &mut DeviceBuffer,
        words: usize,
    ) -> tenferro_tensor::Result<()> {
        if words > buffer.bytes / 4 || buffer.ctx != self.ctx {
            return Err(backend("DeviceRng", "output size or CUDA context mismatch"));
        }
        self.bind(exec)?;
        if words == 0 {
            return Ok(());
        }
        // SAFETY: checked capacity and context; buffer outlives the synchronized scope.
        unsafe { curand::generate::uniform_u32(self.handle, buffer.ptr as *mut u32, words) }
            .map_err(|e| backend("DeviceRng", format!("curandGenerate: {e:?}")))
    }
}

impl Drop for DeviceRng {
    fn drop(&mut self) {
        // SAFETY: the retained primary context outlives the generator. Raw scopes
        // finish their work before owners can drop this handle on another thread.
        unsafe {
            if driver_sys::cuCtxPushCurrent_v2(self.ctx).result().is_ok() {
                let _ = curand::destroy_generator(self.handle);
                let mut previous = std::ptr::null_mut();
                let _ = driver_sys::cuCtxPopCurrent_v2(&mut previous);
            }
            let _ = driver::primary_ctx::release(self.device);
        }
    }
}
