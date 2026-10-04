//! One entered CPU scope for ordinary eager, eager AD and prepared trace work.
//! Run: cargo run -p tenferro-ad --example cpu_execution_scope

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let owner = CpuBackend::with_threads(1)?;
    let eager = EagerRuntime::with_cpu_backend(owner.clone())?;
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?,
        Arc::clone(&eager),
    )?;
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2])?,
        Arc::clone(&eager),
    )?;

    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&owner)?)?;
    let runtime = builder.build()?;
    let traced = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?;
    let graph = GraphCompiler::new().compile(&traced.mul(&traced)?)?;
    let prepared = runtime.prepare_compiled(&graph, &[])?;

    // For benchmarking, start the steady-state clock INSIDE this callback,
    // after warm-up. Scope entry and graph/input preparation are not timed.
    let (primal, jvp, vjp, replay) =
        owner.with_execution_scope(|| -> tenferro_ad::Result<_> {
            let primal = x.mul(&x)?;
            let jvp = eager.jvp(&primal, &x, &seed)?;
            let vjp = eager.vjp(&primal, &x, &seed)?;
            let replay = runtime.run_prepared(&prepared, &[])?;
            Ok((primal, jvp, vjp, replay))
        })??;

    // Returned values remain usable after the execution scope releases its permit.
    assert_eq!(primal.to_tensor()?.as_slice::<f64>()?, &[4.0, 9.0]);
    assert_eq!(jvp.to_tensor()?.as_slice::<f64>()?, &[4.0, 6.0]);
    assert_eq!(vjp.to_tensor()?.as_slice::<f64>()?, &[4.0, 6.0]);
    assert_eq!(replay[0].as_slice::<f64>()?, &[4.0, 9.0]);
    println!("CPU shared scope: eager primal/JVP/VJP and prepared replay passed");
    Ok(())
}
