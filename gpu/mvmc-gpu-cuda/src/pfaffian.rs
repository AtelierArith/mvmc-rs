//! Launch geometry of the batched Pfaffian kernels of `pfaffian_batched.cu` (issue #423).
//!
//! The kernels are compiled into the resident interval module (see [`crate::interval`]); this
//! module only exposes the matrix-size limit and the per-plane block geometry shared by
//! [`crate::interval`] and [`crate::opt`].

/// Largest supported matrix side (dynamic shared memory holds `O(n)` per block).
pub const MAX_N: usize = 1024;

/// Dynamic shared bytes and threads per block of one plane for element type
/// `elem_bytes`/`lanes` (2 for `Complex64`, 1 for the real kernels).
///
/// `pfinv_f64` (real double, even `n >= 64`) uses the specialized fast path of
/// `pfaffian_batched.cu`: `xs, ys, t1s, t2s` (4n doubles) plus a 32-entry reduction scratch, a
/// 32-entry value scratch, `piv` and `idx` (n ints each) = `40 n + 384` bytes, and it is
/// measured fastest with 256 threads for `n >= 128`. All other kernels keep the original
/// layout (`n * elem * lanes + nt * elem + nt * 4 + n * 8` bytes) and thread rule. Both are
/// returned for the f64 case and the larger is used, so the launch stays valid for either
/// kernel.
pub fn launch_geometry(n: usize, elem_bytes: usize, lanes: usize) -> (u32, u32) {
    let fast = elem_bytes == 8 && lanes == 1 && n >= 64 && n.is_multiple_of(2);
    let nt = if fast && n >= 128 {
        256
    } else {
        n.next_power_of_two().clamp(32, 256)
    };
    let generic = (n * elem_bytes * lanes + nt * elem_bytes + nt * 4 + n * 8) as u32;
    let fast_shared = if fast { (40 * n + 384) as u32 } else { 0 };
    (nt as u32, generic.max(fast_shared))
}

#[cfg(test)]
mod tests {
    use mvmc_gpu::{pfaffian_inverse_batched, testkit::random_planes, Backend, PfScalar};
    use num_complex::Complex64;
    use tenferro_ext::raw_exec::{with_raw_exec, Arg};
    use tenferro_gpu::cuda::{cuda_devices, CudaEngine};

    const PROBE: &str = concat!(
        include_str!("pfaffian_panel.cuh"),
        "\n",
        include_str!("resident_dense.cuh"),
        "\n",
        include_str!("resident_complex.cuh"),
        r#"
extern "C" __global__ void pfaffian_panel_probe(const double* input, double* work,
    double* output, int n, int lanes) {
  extern __shared__ char scratch[];
  const size_t base = (size_t)blockIdx.x*n*n*lanes;
  for (int i=threadIdx.x;i<n*n*lanes;i+=blockDim.x) work[base+i]=input[base+i];
  __syncthreads();
  if (lanes==1) {
    double* tau=reinterpret_cast<double*>(scratch);
    const double value=resident_panel_pfaffian<ResidentRealPfOps>(work+base,n,tau,tau+n);
    if (threadIdx.x==0) output[blockIdx.x]=value;
  } else {
    rcx* tau=reinterpret_cast<rcx*>(scratch);
    const rcx value=resident_panel_pfaffian<ResidentComplexPfOps>(reinterpret_cast<rcx*>(work+base),n,tau,tau+n);
    if (threadIdx.x==0) reinterpret_cast<rcx*>(output)[blockIdx.x]=value;
  }
}
"#
    );

    fn check<T: PfScalar + mvmc_gpu::testkit::TestScalar>(engine: &mut CudaEngine) {
        for n in [2, 6, 24, 26, 64, 128, 130, 256] {
            let count = 4;
            let input = random_planes::<T>(n, count, 731 + n as u64);
            let expected =
                pfaffian_inverse_batched(&Backend::CpuPfapack, &input, n, count, 1).unwrap();
            let lanes = T::lanes();
            let input = T::as_f64_lanes(&input);
            let actual = with_raw_exec(engine, "pfaffian.panel.probe", |exec| {
                let module = exec.module(PROBE, &["pfaffian_panel_probe"])?;
                let data = exec.alloc_f64(input.len())?;
                let work = exec.alloc_f64(input.len())?;
                let out = exec.alloc_f64(count * lanes)?;
                let mut actual = vec![0f64; count * lanes];
                unsafe {
                    exec.upload(exec.ptr(&data)?, input)?;
                    exec.launch(
                        module.function("pfaffian_panel_probe")?,
                        (count as u32, 1, 1),
                        128,
                        (2 * n * lanes * 8) as u32,
                        &[
                            Arg::P(exec.ptr(&data)?),
                            Arg::P(exec.ptr(&work)?),
                            Arg::P(exec.ptr(&out)?),
                            Arg::I(n as i32),
                            Arg::I(lanes as i32),
                        ],
                    )?;
                    exec.download(&mut actual, exec.ptr(&out)?)?;
                }
                Ok(actual)
            })
            .unwrap();
            for (plane, (a, e)) in actual
                .chunks_exact(lanes)
                .zip(T::as_f64_lanes(&expected.pf).chunks_exact(lanes))
                .enumerate()
            {
                let scale = e.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
                for (&a, &e) in a.iter().zip(e) {
                    // Independent upper-LTL reference versus lower pair panels;
                    // includes pivot/rounding differences over up to 256 rows.
                    assert!(
                        a.is_finite() && (a - e).abs() <= 1e-10 * scale,
                        "n={n} lanes={lanes} plane={plane}: {a} vs {e}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA; panelized real/complex Pfaffians against CPU reference"]
    fn panel_pfaffians_match_cpu_reference() {
        let mut engine = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
        check::<f64>(&mut engine);
        check::<Complex64>(&mut engine);
    }
}
