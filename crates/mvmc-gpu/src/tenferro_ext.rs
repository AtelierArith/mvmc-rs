//! pfapack per-plane kernel as a tenferro `ExtensionOp` run in a tenferro runtime on
//! `CpuEngine`.
//!
//! One op consumes a `[n, n, NQP, B]` tensor and produces `pf [NQP, B]`,
//! `inv [n, n, NQP, B]` and i32 status `[NQP, B]` ([`PlaneStatus::to_code`]). The input is read zero-copy with
//! `TensorRead::as_slice`; the per-plane workspace lives in the runtime's
//! `ExtensionCacheStore` (keyed by `n` and dtype), so it is reused across planes and calls
//! that share one runtime.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::extension::{
    apply, define_extension_runtime, ExtensionAliasDeclaration, ExtensionCacheKey,
    ExtensionEffectDeclaration, ExtensionExecutionContext, ExtensionOp, ExtensionShapeContext,
};
use tenferro_runtime::{
    DType, EngineId, ExecutionEngine, GraphCompiler, Runtime, SymDim, Tensor, TracedTensor,
};
use tenferro_tensor::{Engine, TensorRead};

use crate::cpu::PlaneWorkspace;
use crate::{BatchOutput, Error, PfScalar, PlaneStatus};

/// Shared operation family used by the CPU and CUDA executors.
pub const FAMILY: &str = "mvmc-gpu.pfaffian-inverse.v1";

/// Batched Pfaffian and true inverse with per-plane i32 status.
#[derive(Clone, Debug)]
pub struct PfaffianInverseOp;

impl ExtensionOp for PfaffianInverseOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(0);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        3
    }
    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }
    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }
    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        let shape = ctx.input_shape(0)?.to_vec();
        if shape.len() != 4 {
            return Err(tenferro_tensor::Error::invalid_argument(
                FAMILY,
                "shape",
                "expected [n,n,NQP,B]",
            ));
        }
        if let (Some(rows), Some(cols)) = (shape[0].constant_value(), shape[1].constant_value()) {
            if rows != cols || rows < 2 || !rows.is_multiple_of(2) {
                return Err(tenferro_tensor::Error::invalid_argument(
                    FAMILY,
                    "shape",
                    "expected an even square matrix side >= 2",
                ));
            }
        }
        if !matches!(dtype, DType::F64 | DType::C64) {
            return Err(tenferro_tensor::Error::invalid_argument(
                FAMILY,
                "dtype",
                "expected f64 or complex f64",
            ));
        }
        let planes = shape[2..].to_vec();
        Ok(vec![
            (dtype, planes.clone()),
            (dtype, shape),
            (DType::I32, planes),
        ])
    }
}

/// Install the CPU executor for this operation on an Engine registration.
pub fn cpu_extension_module(
    engine_id: EngineId,
) -> std::result::Result<
    Arc<dyn tenferro_runtime::ExtensionModule>,
    tenferro_runtime::RuntimeConfigError,
> {
    runtime_impl::extension_module::<CpuEngine>(engine_id)
}

mod runtime_impl {
    use super::*;

    define_extension_runtime! {
        runtime = PfaffianInverseRuntime,
        family_id = FAMILY,
        op_type = PfaffianInverseOp,
        execute_reads = execute_reads,
    }
}

fn execute_reads<B: Engine + ?Sized + 'static>(
    _op: &PfaffianInverseOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let input = &inputs[0];
    let shape = input.shape().to_vec();
    match input.dtype() {
        DType::F64 => run_plane_loop::<f64, B>(input, &shape, ctx),
        DType::C64 => run_plane_loop::<Complex64, B>(input, &shape, ctx),
        other => Err(tenferro_tensor::Error::backend_source(
            "mvmc-gpu.pfaffian-inverse",
            std::io::Error::other(format!("unsupported dtype {other:?}")),
        )),
    }
}

fn run_plane_loop<T: PfScalar, B: Engine + ?Sized + 'static>(
    input: &TensorRead<'_>,
    shape: &[usize],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if shape.len() != 4 || shape[0] != shape[1] || shape[0] < 2 || !shape[0].is_multiple_of(2) {
        return Err(tenferro_tensor::Error::invalid_argument(
            FAMILY,
            "shape",
            "expected even square [n,n,NQP,B]",
        ));
    }
    let n = shape[0];
    let nn = n.checked_mul(n).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(FAMILY, "shape", "matrix size overflows")
    })?;
    let planes = shape[2].checked_mul(shape[3]).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(FAMILY, "shape", "plane count overflows")
    })?;
    // Zero-copy host view of the input planes.
    let src = input.as_slice::<T>()?;
    // The workspace is cached per (dtype, n) in the runtime's extension cache store.
    let key = ExtensionCacheKey::new(
        FAMILY,
        "plane-workspace",
        ((n as u64) << 1) | T::IS_COMPLEX as u64,
    );
    let caches = ctx.caches_mut();
    if caches.get_mut::<PlaneWorkspace<T>>(&key).is_none() {
        let bytes = (n * n + n) * std::mem::size_of::<T>() + n * 4;
        caches.put(key, PlaneWorkspace::<T>::new(n), bytes);
    }
    let ws = caches
        .get_mut::<PlaneWorkspace<T>>(&key)
        .expect("workspace was just inserted");

    let mut inv = src.to_vec();
    let mut pf = vec![T::ZERO; planes];
    let mut status = vec![0i32; planes];
    for ((plane, pf), status) in inv.chunks_mut(nn).zip(pf.iter_mut()).zip(status.iter_mut()) {
        let (value, st) = T::pfapack_plane(plane, n, ws);
        *pf = value;
        *status = st.to_code();
    }
    Ok(vec![
        T::make_tensor(shape[2..].to_vec(), pf)?,
        T::make_tensor(shape.to_vec(), inv)?,
        Tensor::from_vec_col_major(shape[2..].to_vec(), status)?,
    ])
}

fn backend_err(err: impl std::fmt::Display) -> Error {
    Error::Backend(err.to_string())
}

pub(crate) fn run<T: PfScalar>(
    planes: &[T],
    n: usize,
    count: usize,
) -> crate::Result<BatchOutput<T>> {
    let mut backend = CpuEngine::serial();
    let mut builder = Runtime::builder();
    let engine_id: EngineId = tenferro_cpu::runtime_engine_id().map_err(backend_err)?;
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration_with_id(&backend, engine_id.clone())
                .map_err(backend_err)?,
        )
        .map_err(backend_err)?;
    builder
        .install_extension_module(cpu_extension_module(engine_id).map_err(backend_err)?)
        .map_err(backend_err)?;
    let runtime = builder.build().map_err(backend_err)?;

    // `compile_with_input_specs` takes one output only (no multi-output variant), so the
    // plane tensor is attached to the graph as a concrete leaf (one extra input copy).
    let tensor = T::make_tensor(vec![n, n, count, 1], planes.to_vec()).map_err(backend_err)?;
    let input = TracedTensor::from_tensor_concrete_shape(tensor).map_err(backend_err)?;
    let outputs = apply(Arc::new(PfaffianInverseOp), &[&input]).map_err(backend_err)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&outputs[0], &outputs[1], &outputs[2]])
        .map_err(backend_err)?;
    let prepared = runtime
        .prepare(&program, &[], &[backend.target()])
        .map_err(backend_err)?;
    let results = runtime
        .run(&mut [&mut backend], &prepared, &[])
        .map_err(backend_err)?
        .into_iter()
        .map(|output| output.into_tensor(&mut backend).map_err(backend_err))
        .collect::<crate::Result<Vec<_>>>()?;
    let [pf, inv, status] = <[Tensor; 3]>::try_from(results)
        .map_err(|r| Error::Backend(format!("expected 3 outputs, got {}", r.len())))?;
    let pf = T::read_vec(&pf).map_err(backend_err)?;
    let inv = T::read_vec(&inv).map_err(backend_err)?;
    let status = status
        .as_slice::<i32>()
        .map_err(backend_err)?
        .iter()
        .copied()
        .map(PlaneStatus::from_code)
        .collect();
    Ok((pf, inv, status))
}
