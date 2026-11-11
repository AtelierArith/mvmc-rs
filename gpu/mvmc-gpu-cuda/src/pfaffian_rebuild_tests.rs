use mvmc_gpu::{
    pfaffian_inverse_batched,
    testkit::{random_planes, TestScalar},
    Backend, PfScalar, PlaneStatus,
};
use num_complex::Complex64;
use tenferro_ext::raw_exec::{with_raw_exec, Arg};
use tenferro_gpu::cuda::{cuda_devices, CudaEngine};

const SOURCE: &str = concat!(
    include_str!("pfaffian_panel.cuh"),
    "\n",
    include_str!("resident_dense.cuh"),
    "\n",
    include_str!("resident_complex.cuh"),
    "\n",
    include_str!("pfaffian_inverse_panel.cuh"),
    "\n",
    include_str!("pfaffian_batched.cu"),
    r#"
extern "C" __global__ void pfinv_reference_f64(double* a,double* w,double* out,double* pf,int* st,int n) {
  PfinvRealPanelOps::reference(a,w,out,pf,st,n);
}
extern "C" __global__ void pfinv_reference_c64(rcx* a,rcx* w,rcx* out,rcx* pf,int* st,int n) {
  PfinvComplexPanelOps::reference(a,w,out,pf,st,n);
}
"#
);

#[test]
fn full_rebuild_geometry_keeps_panel_and_reference_within_shared_limit() {
    for n in (2..=super::MAX_N).step_by(2) {
        for lanes in [1, 2] {
            let (nt, shared) = super::launch_geometry(n, 8, lanes);
            assert_eq!(nt % 32, 0);
            assert!(shared <= 48 * 1024, "n={n} lanes={lanes} shared={shared}");
            if n >= 64 {
                assert!(shared as usize >= 20 * n * lanes + 4 * n);
            }
        }
    }
}

fn check<T: PfScalar + TestScalar>(engine: &mut CudaEngine, kernel: &str) {
    for n in [24, 26, 64, 128, 130, 256] {
        let count = 5;
        let mut input = random_planes::<T>(n, count, 731 + n as u64);
        input[4 * n * n..].fill(T::zero_value());
        let expected = pfaffian_inverse_batched(&Backend::CpuPfapack, &input, n, count, 1).unwrap();
        let lanes = T::lanes();
        let (actual, pf, status) = with_raw_exec(engine, "pfaffian.rebuild.probe", |exec| {
            let module = exec.module(SOURCE, &["pfinv_f64", "pfinv_c64"])?;
            let a = exec.alloc_f64(input.len() * lanes)?;
            let w = exec.alloc_f64(input.len() * lanes)?;
            let out = exec.alloc_f64(input.len() * lanes)?;
            let pf = exec.alloc_f64(count * lanes)?;
            let status = exec.alloc_bytes(count * 4)?;
            let mut host = (
                vec![0.0; input.len() * lanes],
                vec![0.0; count * lanes],
                vec![0i32; count],
            );
            let (nt, shared) = super::launch_geometry(n, 8, lanes);
            unsafe {
                exec.upload(exec.ptr(&a)?, T::as_f64_lanes(&input))?;
                exec.launch(
                    module.function(kernel)?,
                    (count as u32, 1, 1),
                    nt,
                    shared,
                    &[
                        Arg::P(exec.ptr(&a)?),
                        Arg::P(exec.ptr(&w)?),
                        Arg::P(exec.ptr(&out)?),
                        Arg::P(exec.ptr(&pf)?),
                        Arg::P(exec.ptr(&status)?),
                        Arg::I(n as i32),
                    ],
                )?;
                exec.download(&mut host.0, exec.ptr(&out)?)?;
                exec.download(&mut host.1, exec.ptr(&pf)?)?;
                exec.download(&mut host.2, exec.ptr(&status)?)?;
            }
            Ok(host)
        })
        .unwrap();
        let to_complex =
            |data: &[f64]| Complex64::new(data[0], if lanes == 2 { data[1] } else { 0.0 });
        let expected_pf = T::as_f64_lanes(&expected.pf);
        let expected_inv = T::as_f64_lanes(&expected.inv);
        let input = T::as_f64_lanes(&input);
        for plane in 0..count {
            let code = match expected.status[plane] {
                PlaneStatus::Ok => 0,
                PlaneStatus::ZeroPivot { row } => row as i32,
                PlaneStatus::NonFinite => -1,
            };
            assert_eq!(status[plane], code, "n={n} lanes={lanes} plane={plane}");
            let p = to_complex(&pf[plane * lanes..]);
            let e = to_complex(&expected_pf[plane * lanes..]);
            assert!(
                (p - e).norm() <= 1e-10 * e.norm().max(1.0),
                "n={n} plane={plane} Pf {p} vs {e}"
            );
            let base = plane * n * n * lanes;
            let entry =
                |data: &[f64], i: usize, j: usize| to_complex(&data[base + (i + j * n) * lanes..]);
            if code != 0 {
                assert!(actual[base..base + n * n * lanes].iter().all(|&x| x == 0.0));
                continue;
            }
            let mut norm_a = 0.0f64;
            let mut norm_inv = 0.0f64;
            let mut residual = 0.0f64;
            let mut difference = 0.0f64;
            for i in 0..n {
                norm_a = norm_a.max((0..n).map(|j| entry(input, i, j).norm()).sum());
                norm_inv = norm_inv.max((0..n).map(|j| entry(expected_inv, i, j).norm()).sum());
                let mut row = 0.0;
                for j in 0..n {
                    let mut r = -Complex64::new(f64::from(i == j), 0.0);
                    for k in 0..n {
                        r += entry(input, i, k) * entry(&actual, k, j);
                    }
                    row += r.norm();
                    difference =
                        difference.max((entry(&actual, i, j) - entry(expected_inv, i, j)).norm());
                }
                residual = residual.max(row);
            }
            // Backward residual and conditioning-scaled forward comparison
            // against independent C-order upper LTL, for n-term dot products.
            let bound = 64.0 * f64::EPSILON * n as f64;
            assert!(
                residual / (norm_a * norm_inv) <= bound,
                "n={n} plane={plane} residual={residual:e}"
            );
            assert!(
                difference <= bound * (norm_a * norm_inv).max(1.0) * norm_inv,
                "n={n} plane={plane} inverse difference={difference:e}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA; restore original input after paired inverse overflow"]
fn paired_inverse_overflow_restores_reference_input() {
    let mut engine = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let n = 64;
    // A tridiagonal prefix with a=1e-308, b=2, d=1e308:
    // Pf=a*d ~= 1; inv01=-1/a, inv03=-b/(a*d), inv23=-1/d.
    // Every exact inverse entry is finite, but the lower paired L entry
    // -b/a overflows. The reference also has arithmetic-generated NaNs at
    // this scale; preserve its INFO/Pf and output classification rather than
    // claiming stronger extreme-scale accuracy than the retained contract.
    // Unit connecting edges retain the full C tridiagonal INFO contract;
    // disconnected 2x2 blocks can report zero intermediate pivot columns.
    for lanes in [1, 2] {
        let mut input = vec![0.0f64; n * n * lanes];
        let mut expected = vec![0.0f64; n * n * lanes];
        for i in (0..n).step_by(2) {
            let value = if i == 0 {
                1e-308
            } else if i == 2 {
                1e308
            } else {
                1.0
            };
            input[(i + (i + 1) * n) * lanes] = value;
            input[(i + 1 + i * n) * lanes] = -value;
        }
        for i in (1..n - 1).step_by(2) {
            let edge = if i == 1 { 2.0 } else { 1.0 };
            input[(i + (i + 1) * n) * lanes] = edge;
            input[(i + 1 + i * n) * lanes] = -edge;
        }
        // Closed-form tridiagonal inverse: only even-row/odd-column
        // upper entries are nonzero. Form ratios before products to avoid
        // overflow in the independent expected-value calculation itself.
        for i in (0..n).step_by(2) {
            let mut value = -1.0 / input[(i + (i + 1) * n) * lanes];
            for j in ((i + 1)..n).step_by(2) {
                if j > i + 1 {
                    value *= input[(j - 2 + (j - 1) * n) * lanes] / input[(j - 1 + j * n) * lanes];
                }
                expected[(i + j * n) * lanes] = value;
                expected[(j + i * n) * lanes] = -value;
            }
        }
        let mut reference_output = None;
        for reference in [true, false] {
            let kernel = match (lanes, reference) {
                (1, true) => "pfinv_reference_f64",
                (1, false) => "pfinv_f64",
                (_, true) => "pfinv_reference_c64",
                (_, false) => "pfinv_c64",
            };
            let (actual, pf, status) =
                with_raw_exec(&mut engine, "pfaffian.scaled.probe", |exec| {
                    let module = exec.module(
                        SOURCE,
                        &[
                            "pfinv_f64",
                            "pfinv_c64",
                            "pfinv_reference_f64",
                            "pfinv_reference_c64",
                        ],
                    )?;
                    let a = exec.alloc_f64(input.len())?;
                    let w = exec.alloc_f64(input.len())?;
                    let out = exec.alloc_f64(input.len())?;
                    let pf = exec.alloc_f64(lanes)?;
                    let status = exec.alloc_bytes(4)?;
                    let mut host = (vec![0.0f64; input.len()], vec![0.0f64; lanes], vec![0i32]);
                    let (nt, shared) = super::launch_geometry(n, 8, lanes);
                    unsafe {
                        exec.upload(exec.ptr(&a)?, &input)?;
                        exec.launch(
                            module.function(kernel)?,
                            (1, 1, 1),
                            nt,
                            shared,
                            &[
                                Arg::P(exec.ptr(&a)?),
                                Arg::P(exec.ptr(&w)?),
                                Arg::P(exec.ptr(&out)?),
                                Arg::P(exec.ptr(&pf)?),
                                Arg::P(exec.ptr(&status)?),
                                Arg::I(n as i32),
                            ],
                        )?;
                        exec.download(&mut host.0, exec.ptr(&out)?)?;
                        exec.download(&mut host.1, exec.ptr(&pf)?)?;
                        exec.download(&mut host.2, exec.ptr(&status)?)?;
                    }
                    Ok(host)
                })
                .unwrap();
            assert_eq!(
                status,
                [0],
                "lanes={lanes} reference={reference}: preserve C-order INFO"
            );
            assert!((pf[0] - 1.0).abs() <= 1e-13);
            if reference {
                reference_output = Some(actual);
                continue;
            }
            let baseline = reference_output.as_ref().unwrap();
            for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                if baseline[i].is_nan() {
                    assert!(a.is_nan(), "lanes={lanes} NaN class {i}");
                    continue;
                }
                if baseline[i].is_infinite() {
                    assert_eq!(a, baseline[i], "infinity sign {i}");
                    continue;
                }
                assert!(
                    a.is_finite() && (a - e).abs() <= 1e-13 * e.abs().max(1.0),
                    "lanes={lanes} inverse[{i}] {a} vs {e}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; production real/complex full rebuilds against CPU and inverse identities"]
fn full_panel_rebuilds_match_cpu_and_inverse_residuals() {
    let mut engine = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    check::<f64>(&mut engine, "pfinv_f64");
    check::<Complex64>(&mut engine, "pfinv_c64");
}
