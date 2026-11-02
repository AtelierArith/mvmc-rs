//! Compare the shared ExtensionOp with the existing CPU and CUDA Pfaffian entry points.
//! The CUDA resident ExtensionOp and the host-buffer legacy entry point have different scopes.
use std::sync::Arc;
use std::time::Instant;

use mvmc_gpu::tenferro_ext::{cpu_extension_module, PfaffianInverseOp};
use mvmc_gpu::{pfaffian_inverse_batched, Backend};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Runtime, Tensor, TensorRead, TraceContext,
};

fn median(mut f: impl FnMut()) -> f64 {
    f();
    let mut elapsed = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        f();
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    elapsed.sort_by(f64::total_cmp);
    elapsed[3]
}

fn run_extension(
    engine: &mut dyn ExecutionEngine,
    runtime: &Runtime,
    host: Tensor,
    device: bool,
) -> f64 {
    let input = if device {
        engine
            .upload_host_tensor(TensorRead::from_tensor(&host))
            .unwrap()
    } else {
        host
    };
    let mut trace = TraceContext::new();
    let x = trace
        .input(ProgramInputSpec::new(
            DType::F64,
            input
                .shape()
                .iter()
                .copied()
                .map(tenferro_ops::dim_expr::DimExpr::Const),
        ))
        .unwrap();
    let outputs = trace
        .add_extension(Arc::new(PfaffianInverseOp), &[x])
        .unwrap();
    let graph = trace.finish(&outputs).unwrap();
    let program = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let inputs = [TensorRead::from_tensor(&input)];
    let prepared = runtime
        .prepare(&program, &inputs, &[engine.target()])
        .unwrap();
    median(|| {
        let results = runtime.run(&mut [engine], &prepared, &inputs).unwrap();
        for value in results {
            std::hint::black_box(value.into_tensor(engine).unwrap());
        }
    })
}

fn main() {
    let n = 32;
    let nqp = 8;
    let batch = 8;
    let mut planes = vec![0.0; n * n * nqp * batch];
    for p in 0..nqp * batch {
        for j in 0..n {
            for i in 0..j {
                let value = if j == i + 1 && i % 2 == 0 {
                    2.0
                } else {
                    ((p + i * 7 + j * 11) % 17) as f64 * 0.001
                };
                planes[p * n * n + i + j * n] = value;
                planes[p * n * n + j + i * n] = -value;
            }
        }
    }
    let host = Tensor::from_vec_col_major(vec![n, n, nqp, batch], planes.clone()).unwrap();
    let legacy_cpu = median(|| {
        std::hint::black_box(
            pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nqp, batch).unwrap(),
        );
    });
    let mut cpu = tenferro_cpu::CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&cpu).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            cpu_extension_module(tenferro_cpu::runtime_engine_id().unwrap()).unwrap(),
        )
        .unwrap();
    let cpu_runtime = builder.build().unwrap();
    let extension_cpu = run_extension(&mut cpu, &cpu_runtime, host, false);
    let devices = tenferro_gpu::cuda::cuda_devices().unwrap();
    let mut cuda = tenferro_gpu::cuda::CudaEngine::open(devices[0].id()).unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(&cuda).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            mvmc_gpu_cuda::pfaffian_extension::cuda_extension_module(
                cuda.runtime_engine_id().clone(),
            )
            .unwrap(),
        )
        .unwrap();
    let gpu_runtime = builder.build().unwrap();
    let before = tenferro_ext::raw_exec::raw_exec_counters();
    let host = Tensor::from_vec_col_major(vec![n, n, nqp, batch], planes.clone()).unwrap();
    let extension_gpu = run_extension(&mut cuda, &gpu_runtime, host, true);
    let traffic = tenferro_ext::raw_exec::raw_exec_counters() - before;
    let (legacy_gpu, legacy_kernel) = mvmc_gpu_cuda::pfaffian::with_session(0, |session| {
        let mut kernels = Vec::new();
        let total = median(|| {
            let (output, timings) = session.run_timed(&planes, n, nqp * batch).unwrap();
            std::hint::black_box(output);
            kernels.push(timings.kernel_s * 1000.0);
        });
        kernels.remove(0); // warmup
        kernels.sort_by(f64::total_cmp);
        Ok((total, kernels[3]))
    })
    .unwrap();
    println!("f64 n={n} NQP={nqp} B={batch}, milliseconds, median of 7 after warmup");
    println!("CPU PfaPack host: {legacy_cpu:.6}; CPU extension host: {extension_cpu:.6}");
    println!("CUDA legacy host including uploads/downloads: {legacy_gpu:.6}; CUDA extension device-resident: {extension_gpu:.6}");
    println!("CUDA legacy synchronized kernel phase: {legacy_kernel:.6}");
    println!(
        "Extension explicit raw transfers: upload={} download={} bytes",
        traffic.upload_bytes, traffic.download_bytes
    );
}
