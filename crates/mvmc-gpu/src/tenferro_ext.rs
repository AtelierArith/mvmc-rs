//! pfapack per-plane kernel as a tenferro `ExtensionOp` run in a tenferro runtime on
//! `CpuBackend`.
//!
//! One op consumes a `[n, n, P]` tensor and produces `pf [P]`, `inv [n, n, P]` and a status
//! code tensor `[P]` (f64-encoded [`PlaneStatus::to_code`]). The input is read zero-copy with
//! `TensorRead::as_slice`; the per-plane workspace lives in the runtime's
//! `ExtensionCacheStore` (keyed by `n` and dtype), so it is reused across planes and calls
//! that share one runtime.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{
    apply, define_extension_runtime, ExtensionAliasDeclaration, ExtensionCacheKey,
    ExtensionEffectDeclaration, ExtensionExecutionContext, ExtensionOp, ExtensionShapeContext,
};
use tenferro_runtime::{DType, EngineId, GraphCompiler, Runtime, SymDim, Tensor, TracedTensor};
use tenferro_tensor::{TensorBackend, TensorRead};

use crate::cpu::PlaneWorkspace;
use crate::scalar::private::Impl;
use crate::{BatchOutput, Error, PfScalar, PlaneStatus};

const FAMILY: &str = "mvmc-gpu.pfaffian-inverse.v1";

/// Batched Pfaffian + inverse over the trailing plane axis of an `[n, n, P]` tensor.
#[derive(Clone, Debug)]
struct PfaffianInverseOp;

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
        let planes = shape[2..].to_vec();
        Ok(vec![
            (dtype, planes.clone()),
            (dtype, shape),
            (DType::F64, planes),
        ])
    }
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

fn execute_reads<B: TensorBackend + 'static>(
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

fn run_plane_loop<T: PfScalar, B: TensorBackend + 'static>(
    input: &TensorRead<'_>,
    shape: &[usize],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let n = shape[0];
    let nn = n * n;
    let planes = shape[2];
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
    let mut status = vec![0.0f64; planes];
    for ((plane, pf), status) in inv.chunks_mut(nn).zip(pf.iter_mut()).zip(status.iter_mut()) {
        let (value, st) = T::pfapack_plane(plane, n, ws);
        *pf = value;
        *status = f64::from(st.to_code());
    }
    Ok(vec![
        T::make_tensor(vec![planes], pf)?,
        T::make_tensor(shape.to_vec(), inv)?,
        <f64 as Impl>::make_tensor(vec![planes], status)?,
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
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    let engine_id: EngineId = tenferro_cpu::runtime_engine_id().map_err(backend_err)?;
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration_with_id(&backend, engine_id.clone())
                .map_err(backend_err)?,
        )
        .map_err(backend_err)?;
    builder
        .install_extension_module(
            runtime_impl::extension_module::<CpuBackend>(engine_id).map_err(backend_err)?,
        )
        .map_err(backend_err)?;
    let runtime = builder.build().map_err(backend_err)?;

    // `compile_with_input_specs` takes one output only (no multi-output variant), so the
    // plane tensor is attached to the graph as a concrete leaf (one extra input copy).
    let tensor = T::make_tensor(vec![n, n, count], planes.to_vec()).map_err(backend_err)?;
    let input = TracedTensor::from_tensor_concrete_shape(tensor).map_err(backend_err)?;
    let outputs = apply(Arc::new(PfaffianInverseOp), &[&input]).map_err(backend_err)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&outputs[0], &outputs[1], &outputs[2]])
        .map_err(backend_err)?;
    let results = runtime.run_compiled(&program, &[]).map_err(backend_err)?;
    let [pf, inv, status] = <[Tensor; 3]>::try_from(results)
        .map_err(|r| Error::Backend(format!("expected 3 outputs, got {}", r.len())))?;
    let pf = T::read_vec(&pf).map_err(backend_err)?;
    let inv = T::read_vec(&inv).map_err(backend_err)?;
    let status = <f64 as Impl>::read_vec(&status)
        .map_err(backend_err)?
        .into_iter()
        .map(|c| PlaneStatus::from_code(c as i32))
        .collect();
    Ok((pf, inv, status))
}
