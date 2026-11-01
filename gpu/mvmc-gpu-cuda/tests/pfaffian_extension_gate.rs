//! Shared CPU/CUDA tenferro extension: shape, numerical status, and device residency.
use mvmc_gpu::tenferro_ext::{cpu_extension_module, PfaffianInverseOp};
use mvmc_gpu::{pfaffian_inverse_batched, Backend};
use mvmc_gpu_cuda::pfaffian_extension::cuda_extension_module;
use std::sync::Arc;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Runtime, Tensor, TensorRead, TraceContext,
};
use tenferro_tensor::TensorDeviceTransfer;

fn exercise(engine: &mut dyn ExecutionEngine, runtime: &Runtime, device: bool) {
    let shape = [2, 2, 2, 2];
    let matrices = vec![
        0.0, -1.0, 1.0, 0.0, 0.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0,
    ];
    let expected = pfaffian_inverse_batched(&Backend::CpuPfapack, &matrices, 2, 2, 2).unwrap();
    let host = Tensor::from_vec_col_major(shape.to_vec(), matrices).unwrap();
    let input = if device {
        engine
            .upload_host_tensor(TensorRead::from_tensor(&host))
            .unwrap()
    } else {
        host
    };
    let mut trace = TraceContext::new();
    let x = trace
        .input(ProgramInputSpec::new(DType::F64, shape.map(DimExpr::Const)))
        .unwrap();
    let y = trace
        .add_extension(Arc::new(PfaffianInverseOp), &[x])
        .unwrap();
    let graph = trace.finish(&y).unwrap();
    let program = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let reads = [TensorRead::from_tensor(&input)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .unwrap();
    let before = tenferro_ext::raw_exec::raw_exec_counters();
    let result = runtime.run(&mut [engine], &prepared, &reads).unwrap();
    let mut tensors = Vec::new();
    for output in result {
        tensors.push(output.into_tensor(engine).unwrap());
    }
    let traffic = tenferro_ext::raw_exec::raw_exec_counters() - before;
    assert_eq!(
        (
            traffic.upload_calls,
            traffic.download_calls,
            traffic.upload_bytes,
            traffic.download_bytes
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(tensors[0].shape(), &[2, 2]);
    assert_eq!(tensors[1].shape(), &shape);
    assert_eq!(tensors[2].shape(), &[2, 2]);
    if device {
        let domain = input.as_typed::<f64>().unwrap().allocation_domain();
        assert!(domain.is_some());
        assert_eq!(
            tensors[1].as_typed::<f64>().unwrap().allocation_domain(),
            domain
        );
        tensors = tensors
            .into_iter()
            .map(|value| {
                engine
                    .download_to_host(TensorRead::from_tensor(&value))
                    .unwrap()
            })
            .collect();
    }
    for (a, b) in tensors[0]
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(expected.pf)
    {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in tensors[1]
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(expected.inv)
    {
        assert!((a - b).abs() < 1e-12);
    }
    let status: Vec<_> = expected.status.into_iter().map(|v| v.to_code()).collect();
    assert_eq!(tensors[2].as_slice::<i32>().unwrap(), status);
}

#[test]
fn shared_extension_cpu_preserves_layout_and_status() {
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            cpu_extension_module(tenferro_cpu::runtime_engine_id().unwrap()).unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();
    exercise(&mut engine, &runtime, false);
    exercise_complex(&mut engine, &runtime, false);
}

#[test]
#[ignore = "optional CUDA gate: requires a device"]
fn shared_extension_cuda_preserves_layout_status_and_residency() {
    let devices = tenferro_gpu::cuda::cuda_devices().unwrap();
    assert!(!devices.is_empty(), "CUDA device required");
    let mut engine = tenferro_gpu::cuda::CudaEngine::open(devices[0].id()).unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(&engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            cuda_extension_module(engine.runtime_engine_id().clone()).unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();
    exercise(&mut engine, &runtime, true);
    exercise_complex(&mut engine, &runtime, true);
    let caches = engine.take_extension_caches();
    let stats = caches.stats(tenferro_runtime::ExtensionCacheSelector::Cache {
        family_id: mvmc_gpu::tenferro_ext::FAMILY,
        cache_name: "cuda-scratch",
    });
    assert_eq!(stats.entries, 1);
    assert!(stats.hits > 0 && stats.retained_bytes <= 64 * 1024 * 1024);
    let modules = caches.stats(tenferro_runtime::ExtensionCacheSelector::Cache {
        family_id: mvmc_gpu::tenferro_ext::FAMILY,
        cache_name: "cuda-module",
    });
    assert_eq!(modules.entries, 1);
    assert!(modules.hits > 0 && modules.retained_bytes > 0);
    engine.restore_extension_caches(caches);
    // A deliberately tiny retention budget forces scoped scratch and module
    // release; both must remain live until the GPU launch completes.
    let mut caches = engine.take_extension_caches();
    caches.set_limits(
        tenferro_runtime::ExtensionCacheLimits::default()
            .with_max_retained_bytes(std::num::NonZeroUsize::new(1).unwrap()),
    );
    engine.restore_extension_caches(caches);
    exercise(&mut engine, &runtime, true);
}

fn exercise_complex(engine: &mut dyn ExecutionEngine, runtime: &Runtime, device: bool) {
    use num_complex::Complex64 as C;
    let shape = [2, 2, 2, 1];
    let z = C::new(0.0, 0.0);
    let a = C::new(2.0, -1.0);
    let nan = C::new(f64::NAN, 0.0);
    let values = vec![z, -a, a, z, z, -nan, nan, z];
    let expected = pfaffian_inverse_batched(&Backend::CpuPfapack, &values, 2, 2, 1).unwrap();
    let host = Tensor::from_vec_col_major(shape.to_vec(), values).unwrap();
    let input = if device {
        engine
            .upload_host_tensor(TensorRead::from_tensor(&host))
            .unwrap()
    } else {
        host
    };
    let mut trace = TraceContext::new();
    let x = trace
        .input(ProgramInputSpec::new(DType::C64, shape.map(DimExpr::Const)))
        .unwrap();
    let y = trace
        .add_extension(Arc::new(PfaffianInverseOp), &[x])
        .unwrap();
    let graph = trace.finish(&y).unwrap();
    let program = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let reads = [TensorRead::from_tensor(&input)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .unwrap();
    for _ in 0..2 {
        let before = tenferro_ext::raw_exec::raw_exec_counters();
        let result = runtime.run(&mut [engine], &prepared, &reads).unwrap();
        let mut tensors: Vec<_> = result
            .into_iter()
            .map(|value| value.into_tensor(engine).unwrap())
            .collect();
        let traffic = tenferro_ext::raw_exec::raw_exec_counters() - before;
        assert_eq!((traffic.upload_bytes, traffic.download_bytes), (0, 0));
        if device {
            tensors = tensors
                .into_iter()
                .map(|value| {
                    engine
                        .download_to_host(TensorRead::from_tensor(&value))
                        .unwrap()
                })
                .collect();
        }
        let status: Vec<_> = expected.status.iter().map(|v| v.to_code()).collect();
        assert_eq!(tensors[2].as_slice::<i32>().unwrap(), status);
        let pf = tensors[0].as_slice::<C>().unwrap();
        assert!((pf[0] - expected.pf[0]).norm() < 1e-12);
        let inv = tensors[1].as_slice::<C>().unwrap();
        for (a, b) in inv.iter().zip(&expected.inv) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }
}

#[test]
fn shared_extension_rejects_invalid_shape_and_dtype() {
    for (dtype, shape) in [
        (DType::F64, vec![2, 2, 1]),
        (DType::F64, vec![2, 4, 1, 1]),
        (DType::F64, vec![3, 3, 1, 1]),
        (DType::F32, vec![2, 2, 1, 1]),
    ] {
        let mut trace = TraceContext::new();
        let x = trace
            .input(ProgramInputSpec::new(
                dtype,
                shape.into_iter().map(DimExpr::Const),
            ))
            .unwrap();
        assert!(trace
            .add_extension(Arc::new(PfaffianInverseOp), &[x])
            .is_err());
    }
}

#[test]
#[ignore = "optional CUDA gate: requires a device"]
fn shared_extension_cuda_supports_an_empty_batch() {
    let devices = tenferro_gpu::cuda::cuda_devices().unwrap();
    assert!(!devices.is_empty(), "CUDA device required");
    let mut engine = tenferro_gpu::cuda::CudaEngine::open(devices[0].id()).unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(&engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            cuda_extension_module(engine.runtime_engine_id().clone()).unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();
    let shape = [2, 2, 0, 1];
    let host = Tensor::from_vec_col_major(shape.to_vec(), Vec::<f64>::new()).unwrap();
    let input = engine
        .upload_host_tensor(TensorRead::from_tensor(&host))
        .unwrap();
    let mut trace = TraceContext::new();
    let x = trace
        .input(ProgramInputSpec::new(DType::F64, shape.map(DimExpr::Const)))
        .unwrap();
    let y = trace
        .add_extension(Arc::new(PfaffianInverseOp), &[x])
        .unwrap();
    let graph = trace.finish(&y).unwrap();
    let program = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let reads = [TensorRead::from_tensor(&input)];
    let prepared = runtime
        .prepare(&program, &reads, &[engine.target()])
        .unwrap();
    let outputs = runtime.run(&mut [&mut engine], &prepared, &reads).unwrap();
    let outputs: Vec<_> = outputs
        .into_iter()
        .map(|v| v.into_tensor(&mut engine).unwrap())
        .collect();
    assert_eq!(outputs[0].shape(), &[0, 1]);
    assert_eq!(outputs[1].shape(), &shape);
    assert_eq!(outputs[2].shape(), &[0, 1]);
}
