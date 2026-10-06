//! The batched Pfaffian backends under the #424 validation harness: teacher-forced replay
//! against the C-order oracle with the decision-flip detector, plus the inverse-sign mapping.

use mvmc_core::accel_validation::{replay, ReplayConfig};
use mvmc_core::stage_backend::{COrderPfaffian, PfaffianStages};
use mvmc_gpu::stages::BatchedStages;
use mvmc_gpu::Backend;

fn run(backend: Backend<'_>, label: &str) {
    let mut stages = BatchedStages::new(backend, label).into_stage_backend();
    let rep = replay(&mut stages, &ReplayConfig::default()).expect("replay");
    assert!(
        rep.violations()
            .iter()
            .all(|v| v.starts_with("S:") || v.starts_with("g:")),
        "{label}: {:?}",
        rep.violations()
    );
    // the Pfaffian stage is really exercised: values were compared, nothing was skipped
    assert!(
        rep.pf.compared > 0 && rep.inv.compared > 0 && rep.o.compared > 0,
        "{label}"
    );
    assert!(
        !rep.unsupported
            .iter()
            .any(|u| u.starts_with("pfaffian_inverse")),
        "{label}: {:?}",
        rep.unsupported
    );
    assert_eq!(rep.defects, 0, "{label}");
    eprintln!(
        "{label}: pf max rel {:.2e}, inv max abs {:.2e}, flips {}, defects {}, min margin {:.2e}",
        rep.pf.max_rel, rep.inv.max_abs, rep.flips, rep.defects, rep.min_margin
    );
}

#[test]
fn cpu_pfapack_batched_passes_the_validation_replay() {
    run(Backend::CpuPfapack, "batched-cpu-pfapack");
    run(Backend::CpuPfapackRayon, "batched-cpu-pfapack-rayon");
}

#[test]
fn tenferro_extension_passes_the_validation_replay() {
    run(Backend::TenferroExtension, "batched-tenferro-extop");
}

#[test]
fn tenferro_native_passes_the_validation_replay() {
    run(Backend::TenferroNative, "batched-tenferro-native");
}

/// `inv` is X^-1 (the harness oracle's convention); mVMC's invM is its negative.
#[test]
fn inv_convention_matches_the_oracle() {
    let n = 6;
    let planes = mvmc_gpu::testkit::random_planes::<f64>(n, 1, 3);
    let got = mvmc_gpu::pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, 1, 1).unwrap();
    let want = COrderPfaffian.pfaffian_inverse(&planes, n).unwrap();
    assert_eq!(got.pf[0], want.pf);
    assert_eq!(got.inv, want.inv);
    for i in 0..n {
        for j in 0..n {
            let acc: f64 = (0..n).map(|k| planes[i + k * n] * got.inv[k + j * n]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((acc - want).abs() < 1e-10);
        }
    }
}
