//! The shared PfaPack extension operation run in a tenferro runtime on `CpuEngine`.
//!
//! The operation itself ([`PfaffianInverseOp`], its family [`FAMILY`] and the CPU executor
//! [`cpu_extension_module`]) is defined in `mvmc_core::tenferro_pfaffian`, where it also backs
//! the production measurement stage `MVMC_RS_MEASURE_PF_BACKEND=tenferro` (issue #613). One op
//! consumes a `[n, n, NQP, B]` tensor and produces `pf [NQP, B]`, `inv [n, n, NQP, B]` and i32
//! status `[NQP, B]` ([`PlaneStatus::to_code`]).

use std::sync::Arc;

use tenferro_cpu::CpuEngine;
use tenferro_runtime::extension::apply;
use tenferro_runtime::{EngineId, ExecutionEngine, GraphCompiler, Runtime, Tensor, TracedTensor};

pub use mvmc_core::tenferro_pfaffian::{cpu_extension_module, PfaffianInverseOp, FAMILY};

use crate::{BatchOutput, Error, PfScalar, PlaneStatus};

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
