//! Optional CUDA gate for the batched Pfaffian + inverse kernel (issue #423). Ignored by
//! default; run with `MVMC_RS_CUDA_GATE=1 cargo test --test pfaffian_gate -- --ignored
//! --nocapture` (see `scripts/run_cuda_gate.sh`).
//!
//! * `MVMC_RS_CUDA_GATE=1`: requested; no usable device is a hard failure.
//! * unset: the tests return early with an explicit "skipped" line, never a pass claim.
//!
//! Tolerances are tied to conditioning, not tuned: the forward error of a backward-stable
//! inverse is `c * n * eps * cond(A)` with `cond` taken in the Frobenius norm of the actual
//! plane; the kernel uses pfapack's pivot rule and (with `--fmad=false`) its real operation
//! order, so `c = 16` is generous and the observed errors are printed.

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, CudaGateDecision, CUDA_GATE_VARIABLE,
};
use mvmc_gpu::testkit::*;
use mvmc_gpu::{pfaffian_inverse_batched, Backend, PfScalar, PlaneStatus};
use mvmc_gpu_cuda::pfaffian::CudaEngine;
use num_complex::Complex64;

/// `None` on an explicit skip (gate not requested, no device); panics when the gate is
/// requested without a usable device. Reuses the #420 gate decision machinery.
fn engine() -> Option<CudaEngine> {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("pfaffian-gate: ExplicitSkip: skipped, no device ({why})");
            None
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("pfaffian-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}")
        }
        CudaGateDecision::Run => Some(CudaEngine { device: 0 }),
    }
}

fn fro_lanes<T: PfScalar>(a: &[T]) -> f64 {
    T::as_f64_lanes(a).iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn check<T: TestScalar>(engine: &CudaEngine, n: usize, nqp: usize, batch: usize, seed: u64) {
    let planes = random_planes::<T>(n, nqp * batch, seed);
    let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nqp, batch).unwrap();
    let gpu = pfaffian_inverse_batched(&Backend::Engine(engine), &planes, n, nqp, batch).unwrap();
    assert_eq!(gpu.status, cpu.status, "status n={n}");
    let nn = n * n;
    let (mut worst_inv, mut worst_pf, mut worst_ratio, mut max_cond) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for p in 0..nqp * batch {
        let a = &planes[p * nn..(p + 1) * nn];
        let ci = &cpu.inv[p * nn..(p + 1) * nn];
        let gi = &gpu.inv[p * nn..(p + 1) * nn];
        let cond = fro_lanes(a) * fro_lanes(ci);
        max_cond = max_cond.max(cond);
        let allowed = 16.0 * n as f64 * f64::EPSILON * cond;
        let diff: Vec<f64> = T::as_f64_lanes(gi)
            .iter()
            .zip(T::as_f64_lanes(ci))
            .map(|(x, y)| x - y)
            .collect();
        let inv_err = diff.iter().map(|x| x * x).sum::<f64>().sqrt() / fro_lanes(ci);
        let pf_err = (gpu.pf[p] - cpu.pf[p]).modulus() / cpu.pf[p].modulus();
        worst_inv = worst_inv.max(inv_err);
        worst_pf = worst_pf.max(pf_err);
        worst_ratio = worst_ratio.max(inv_err / allowed).max(pf_err / allowed);
    }
    eprintln!(
        "n={n:3} {}: max inv rel {worst_inv:.2e}, max pf rel {worst_pf:.2e}, max cond {max_cond:.1e}, worst observed/allowed {worst_ratio:.3}",
        if T::COMPLEX { "c64" } else { "f64" }
    );
    assert!(worst_ratio < 1.0, "n={n}: observed/allowed = {worst_ratio}");
    // Independent invariants of the GPU result (not relative to the CPU result).
    let inv = invariants(&planes, &gpu);
    let scale = 16.0 * n as f64 * f64::EPSILON * max_cond;
    assert!(
        inv.max_identity_residual <= scale,
        "A*inv residual {}",
        inv.max_identity_residual
    );
    assert!(
        inv.max_skew_defect <= scale,
        "skew defect {}",
        inv.max_skew_defect
    );
    assert!(
        inv.max_pf2_det_rel <= 4.0 * scale,
        "Pf^2/det {}",
        inv.max_pf2_det_rel
    );
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_matches_pfapack_real() {
    let Some(e) = engine() else { return };
    for n in [2usize, 4, 6, 16, 32, 64, 128] {
        check::<f64>(&e, n, 8, 4, 100 + n as u64);
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_matches_pfapack_complex() {
    let Some(e) = engine() else { return };
    for n in [2usize, 4, 6, 16, 32, 64, 128] {
        check::<Complex64>(&e, n, 8, 4, 200 + n as u64);
    }
}

fn degenerate_planes<T: TestScalar>(n: usize) -> Vec<T> {
    let nn = n * n;
    let mut p = random_planes::<T>(n, 4, 3);
    p[nn..2 * nn].fill(T::zero()); // plane 1: all zero
    p[2 * nn + 4 * n + 1] = T::from_pair(f64::NAN, f64::NAN); // plane 2: NaN
    for k in 0..n {
        // plane 3: zero row/column 3 (singular)
        p[3 * nn + 3 + k * n] = T::zero();
        p[3 * nn + k + 3 * n] = T::zero();
    }
    p
}

fn degenerate_case<T: TestScalar>(engine: &CudaEngine) {
    let n = 8;
    let planes = degenerate_planes::<T>(n);
    let g = pfaffian_inverse_batched(&Backend::Engine(engine), &planes, n, 2, 2).unwrap();
    let c = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, 2, 2).unwrap();
    assert_eq!(g.status, c.status);
    assert_eq!(g.status[0], PlaneStatus::Ok);
    assert!(matches!(g.status[1], PlaneStatus::ZeroPivot { .. }));
    assert_ne!(g.status[2], PlaneStatus::Ok, "NaN plane must not be Ok");
    assert!(matches!(g.status[3], PlaneStatus::ZeroPivot { .. }));
    assert_eq!(g.pf[1].modulus(), 0.0);
    assert!(T::as_f64_lanes(&g.inv[n * n..2 * n * n])
        .iter()
        .all(|&x| x == 0.0));
    assert!(g.pf[0].modulus().is_finite());
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_reports_zero_pivot_and_non_finite_planes() {
    let Some(e) = engine() else { return };
    degenerate_case::<f64>(&e);
    degenerate_case::<Complex64>(&e);
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_large_batch_agrees_with_rayon_reference() {
    let Some(e) = engine() else { return };
    let n = 16;
    let planes = random_planes::<f64>(n, 3000, 9);
    let cpu = pfaffian_inverse_batched(&Backend::CpuPfapackRayon, &planes, n, 8, 375).unwrap();
    let gpu = pfaffian_inverse_batched(&Backend::Engine(&e), &planes, n, 8, 375).unwrap();
    assert_eq!(gpu.status, cpu.status);
    let worst = gpu
        .pf
        .iter()
        .zip(&cpu.pf)
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-10, "{worst}");
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_tensor_native_and_extension_backends_agree_with_the_kernel() {
    let Some(e) = engine() else { return };
    let n = 16;
    let planes = random_planes::<f64>(n, 16, 21);
    let gpu = pfaffian_inverse_batched(&Backend::Engine(&e), &planes, n, 8, 2).unwrap();
    for backend in [Backend::TenferroNative, Backend::TenferroExtension] {
        let cpu = pfaffian_inverse_batched(&backend, &planes, n, 8, 2).unwrap();
        assert_eq!(cpu.status, gpu.status);
        for (a, b) in gpu.pf.iter().zip(&cpu.pf) {
            assert!(((a - b) / b).abs() < 1e-12, "{backend:?}: {a} vs {b}");
        }
        let num: f64 = gpu
            .inv
            .iter()
            .zip(&cpu.inv)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f64 = cpu.inv.iter().map(|a| a * a).sum();
        assert!((num / den).sqrt() < 1e-11, "{backend:?}");
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_pfaffian_passes_the_validation_harness_replay() {
    use mvmc_core::accel_validation::{replay, ReplayConfig};
    use mvmc_gpu::stages::BatchedStages;

    let Some(_) = engine() else { return };
    // the persistent engine compiles the kernel once; the replay issues hundreds of calls
    let e = mvmc_gpu_cuda::pfaffian::PersistentCudaEngine::new(0);
    let mut stages =
        BatchedStages::new(Backend::Engine(&e), "batched-cuda-pfaffian").into_stage_backend();
    let rep = replay(&mut stages, &ReplayConfig::default()).expect("replay");
    println!("{}", rep.render());
    // the SR slot is the C-order oracle here; everything Pfaffian-related is compared
    let bad: Vec<_> = rep
        .violations()
        .into_iter()
        .filter(|v| !v.starts_with("S:") && !v.starts_with("g:"))
        .collect();
    assert!(bad.is_empty(), "{bad:?}");
    assert!(rep.pf.compared > 0 && rep.inv.compared > 0 && rep.o.compared > 0);
    assert!(!rep
        .unsupported
        .iter()
        .any(|u| u.starts_with("pfaffian_inverse")));
    assert_eq!(rep.defects, 0);
}
