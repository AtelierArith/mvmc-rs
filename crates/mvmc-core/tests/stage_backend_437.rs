//! Issue #437: one layered backend object for production and the validation harness.

use mvmc_core::accel_validation::{replay, ReplayConfig};
use mvmc_core::stage_backend::{
    acquire, open_stage_backend, parse_stage_backend, set_stage_backend_override, COrderPfaffian,
    PfaffianStages, PlaneOutcome, StageBackend, StageBackendKind, StageError,
};

#[test]
fn selector_values_parse_and_invalid_values_are_errors() {
    assert_eq!(parse_stage_backend(""), Ok(StageBackendKind::COrder));
    assert_eq!(parse_stage_backend("c-order"), Ok(StageBackendKind::COrder));
    assert_eq!(
        parse_stage_backend("tenferro"),
        Ok(StageBackendKind::TenferroCpu)
    );
    assert_eq!(parse_stage_backend("cuda"), Ok(StageBackendKind::Cuda(0)));
    assert_eq!(parse_stage_backend("CUDA:3"), Ok(StageBackendKind::Cuda(3)));
    assert!(parse_stage_backend("cuda:x").is_err());
    assert!(parse_stage_backend("gpu").is_err());
}

#[test]
fn cuda_is_an_error_without_a_registered_provider_never_a_fallback() {
    let err = open_stage_backend(StageBackendKind::Cuda(0)).unwrap_err();
    assert!(matches!(err, StageError::Unsupported(_)), "{err}");
}

#[test]
fn c_order_composite_provides_every_stage() {
    let mut b = StageBackend::c_order();
    assert_eq!(b.label(), "c-order-cpu");
    assert!(b.provider().contains("pfapack"));
    // 4x4 skew-symmetric: pf = x01 x23 - x02 x13 + x03 x12 = 1*6 - 2*5 + 3*4 = 8
    let (x01, x02, x03, x12, x13, x23) = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let mut x = vec![0.0; 16];
    for (i, j, v) in [
        (0, 1, x01),
        (0, 2, x02),
        (0, 3, x03),
        (1, 2, x12),
        (1, 3, x13),
        (2, 3, x23),
    ] {
        x[i + j * 4] = v;
        x[j + i * 4] = -v;
    }
    let two = [x.clone(), x].concat();
    let out = b.pfaffian().pfaffian_inverse_batch(&two, 4, 2).unwrap();
    assert_eq!(out.outcome, vec![PlaneOutcome::Ok; 2]);
    for pf in &out.pf {
        assert!((pf - 8.0).abs() < 1e-12);
    }
    // the SR slot answers too (S and g of a one-parameter system)
    let sg = b
        .sr()
        .sr_s_g(&[1.0, -1.0], 2, 1, &[2.0, 4.0], &[0.5, 0.5])
        .unwrap();
    assert!((sg.s[0] - 1.0).abs() < 1e-15 && (sg.g[0] + 1.0).abs() < 1e-15);
}

#[test]
fn zero_pivot_plane_is_reported_per_plane() {
    let mut p = COrderPfaffian;
    let out = p.pfaffian_inverse_batch(&[0.0; 16], 4, 1).unwrap();
    assert!(matches!(out.outcome[0], PlaneOutcome::ZeroPivot { .. }));
    assert!(matches!(
        p.pfaffian_inverse(&[0.0; 16], 4),
        Err(StageError::Failed(_))
    ));
}

#[test]
fn tenferro_cpu_composite_reports_the_pfaffian_gap_as_unsupported() {
    let mut b = StageBackend::tenferro_cpu().unwrap();
    assert!(matches!(
        b.pfaffian().pfaffian_inverse(&[0.0; 4], 2),
        Err(StageError::Unsupported(_))
    ));
}

/// Validated means deployed: the object production acquires for the selected kind is the
/// object the harness replays, and a replay through it matches the oracle.
#[test]
fn the_harness_drives_the_object_production_acquires() {
    set_stage_backend_override(Some(StageBackendKind::TenferroCpu));
    let label = {
        let mut handle = acquire();
        let label = handle.backend().label();
        let rep = replay(handle.backend(), &ReplayConfig::default()).unwrap();
        assert!(rep.violations().is_empty(), "{:?}", rep.violations());
        assert!(rep.s.compared > 0 && rep.g.compared > 0);
        // the stats counter proves the very instance used by the harness is the shared one
        assert!(handle.sr().stats().is_some());
        label
    };
    assert_eq!(label, "tenferro-cpu");
    assert!(mvmc_core::stage_backend::tenferro_stats().is_some());
    set_stage_backend_override(None);
    assert_eq!(acquire().backend().label(), "c-order-cpu");
}
