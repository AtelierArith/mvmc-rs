//! CUDA executor for the shared batched Pfaffian tenferro extension.
//!
//! Input and all outputs stay on the caller's CUDA engine. The executor accepts
//! compact owned tensors in `[n,n,NQP,B]` column-major order; unsupported views
//! are rejected explicitly. No host staging is performed.

use mvmc_gpu::tenferro_ext::{PfaffianInverseOp, FAMILY};
use num_complex::Complex64;
use std::sync::Arc;
use tenferro_ext::raw_exec::{with_raw_exec, Arg, DeviceBuffer, RawExec};
use tenferro_runtime::extension::{
    define_extension_runtime, ExtensionCacheKey, ExtensionCacheStore, ExtensionExecutionContext,
};
use tenferro_runtime::{DType, EngineId, ExtensionModule, RuntimeConfigError, Tensor};
use tenferro_tensor::{Engine, TensorRead, TensorScalar};

mod implementation {
    use super::*;
    define_extension_runtime! {
        runtime = CudaPfaffianInverseRuntime,
        family_id = FAMILY,
        op_type = PfaffianInverseOp,
        execute_reads = execute_reads,
    }
}

/// Build the shared operation's CUDA executor for an engine registration.
///
/// # Errors
/// Returns a runtime configuration error if the extension identity is invalid.
pub fn cuda_extension_module(
    engine_id: EngineId,
) -> Result<Arc<dyn ExtensionModule>, RuntimeConfigError> {
    implementation::extension_module::<tenferro_gpu::cuda::CudaEngine>(engine_id)
}

fn error(message: impl std::fmt::Display) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_source(FAMILY, std::io::Error::other(message.to_string()))
}

fn execute_reads<B: Engine + ?Sized + 'static>(
    _op: &PfaffianInverseOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let input = match &inputs[0] {
        TensorRead::Tensor(input) => *input,
        TensorRead::View(_) => {
            return Err(tenferro_tensor::Error::invalid_argument(
                FAMILY,
                "input",
                "CUDA executor requires an owned compact tensor",
            ))
        }
    };
    let shape = input.shape();
    if shape.len() != 4
        || shape[0] != shape[1]
        || shape[0] < 2
        || !shape[0].is_multiple_of(2)
        || shape[0] > crate::pfaffian::MAX_N
    {
        return Err(tenferro_tensor::Error::invalid_argument(
            FAMILY,
            "shape",
            "expected an even square [n,n,NQP,B] with n <= 1024",
        ));
    }
    let (engine, caches) = ctx.parts_mut();
    match input.dtype() {
        DType::F64 => with_raw_exec(engine, FAMILY, |exec| {
            run::<f64>(exec, input, "pfinv_f64", 1, caches)
        }),
        DType::C64 => with_raw_exec(engine, FAMILY, |exec| {
            run::<Complex64>(exec, input, "pfinv_c64", 2, caches)
        }),
        _ => Err(tenferro_tensor::Error::invalid_argument(
            FAMILY,
            "dtype",
            "expected f64 or complex f64",
        )),
    }
}

struct Scratch {
    a: DeviceBuffer,
    work: DeviceBuffer,
    bytes: usize,
}

fn scratch(exec: &RawExec<'_, '_>, bytes: usize) -> tenferro_tensor::Result<Scratch> {
    Ok(Scratch {
        a: exec.alloc_bytes(bytes)?,
        work: exec.alloc_bytes(bytes)?,
        bytes,
    })
}

fn run<T: TensorScalar>(
    exec: &RawExec<'_, '_>,
    input: &Tensor,
    kernel: &'static str,
    lanes: usize,
    caches: &mut ExtensionCacheStore,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let shape = input.shape();
    let n = shape[0];
    let count = shape[2].checked_mul(shape[3]).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(FAMILY, "shape", "plane count overflows")
    })?;
    let bytes = n
        .checked_mul(n)
        .and_then(|x| x.checked_mul(count))
        .and_then(|x| x.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(FAMILY, "shape", "input byte count overflows")
        })?;
    let grid = u32::try_from(count).map_err(|_| {
        tenferro_tensor::Error::invalid_argument(FAMILY, "shape", "too many planes")
    })?;
    let session = exec.raw_session();
    let input = input.as_typed::<T>().ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(FAMILY, "dtype", "input type mismatch")
    })?;
    let src = session.tensor(input).map_err(error)?;
    if src.byte_len() < bytes {
        return Err(tenferro_tensor::Error::invalid_argument(
            FAMILY,
            "input",
            "input span is too small",
        ));
    }
    let inv = session.alloc_output::<T>(shape).map_err(error)?;
    let pf = session.alloc_output::<T>(&shape[2..]).map_err(error)?;
    let status = session.alloc_output::<i32>(&shape[2..]).map_err(error)?;
    if count != 0 {
        // Keep at most 64 MiB of scratch per engine; oversized work is temporary.
        // The engine-owned extension cache may impose a smaller retention limit.
        let key = ExtensionCacheKey::new(FAMILY, "cuda-scratch", 0);
        let retained = bytes.saturating_mul(2);
        if retained <= 64 * 1024 * 1024
            && caches
                .get_mut::<Scratch>(&key)
                .is_none_or(|s| s.bytes < bytes)
        {
            caches.put(key, scratch(exec, bytes)?, retained);
        }
        let (ptr_a, ptr_work) =
            if let Some(cached) = caches.get_mut::<Scratch>(&key).filter(|s| s.bytes >= bytes) {
                (exec.ptr(&cached.a)?, exec.ptr(&cached.work)?)
            } else {
                // Runtime scope owns oversized allocations through synchronization,
                // including when the kernel launch returns an error.
                let a = session.alloc_bytes(bytes, FAMILY).map_err(error)?;
                let work = session.alloc_bytes(bytes, FAMILY).map_err(error)?;
                let mut ptr_a = 0;
                let mut ptr_work = 0;
                a.with_ptr(|p| ptr_a = p as u64);
                work.with_ptr(|p| ptr_work = p as u64);
                (ptr_a, ptr_work)
            };
        // SAFETY: the input and scratch belong to this engine, are disjoint and
        // span `bytes`; the scope synchronizes before scratch can be evicted.
        unsafe { exec.copy_device(ptr_a, src.raw_ptr() as u64, bytes) }?;
        let out = session.tensor_mut(&inv).map_err(error)?;
        let pfout = session.tensor_mut(&pf).map_err(error)?;
        let st = session.tensor_mut(&status).map_err(error)?;
        let module =
            exec.module_without_fmad(crate::pfaffian::KERNEL_SRC, &["pfinv_f64", "pfinv_c64"])?;
        let function = module.function(kernel)?;
        let (threads, shared) = crate::pfaffian::launch_geometry(n, 8, lanes);
        // SAFETY: each allocation spans the batch, outputs are disjoint and the
        // one-block-per-plane kernel ABI matches the supplied arguments. The raw
        // scope synchronizes before these buffers or the module can be released.
        unsafe {
            exec.launch(
                function,
                (grid, 1, 1),
                threads,
                shared,
                &[
                    Arg::P(ptr_a),
                    Arg::P(ptr_work),
                    Arg::P(out.raw_ptr() as u64),
                    Arg::P(pfout.raw_ptr() as u64),
                    Arg::P(st.raw_ptr() as u64),
                    Arg::I(n as i32),
                ],
            )?;
        }
    }
    Ok(vec![
        Tensor::from_typed(pf),
        Tensor::from_typed(inv),
        Tensor::from_typed(status),
    ])
}
