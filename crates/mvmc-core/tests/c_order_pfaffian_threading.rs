//! #586: independent batched planes retain scalar arithmetic and status ordering.
use mvmc_core::stage_backend::{COrderPfaffian, PfaffianStages, PlaneOutcome};
use mvmc_core::threading;
use std::process::Command;

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical;

#[test]
fn mixed_c_order_planes_preserve_layout_status_and_actual_parallel_execution() {
    for workers in [1, 4, 16] {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "c_order_plane_child", "--nocapture"])
            .env("MVMC_RS_INNER_THREADS", workers.to_string())
            .env("MVMC_RS_INNER_THRESHOLD", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .status()
            .unwrap();
        assert!(status.success(), "{workers}-worker C-order stage failed");
    }
}

#[test]
#[ignore = "process-scoped inner-pool helper"]
fn c_order_plane_child() {
    const N: usize = 4;
    const PLANES: usize = 7;
    let mut input = vec![0.0; N * N * PLANES];
    let mut expected_inverse = vec![0.0; input.len()];
    let mut expected_pf = [0.0; PLANES];
    for plane in [0, 2, 4, 6] {
        let scale = (plane + 1) as f64;
        let sign = if plane % 4 == 0 { -1.0 } else { 1.0 };
        // For a 4x4 skew matrix, Pf(A)=a01*a23-a02*a13+a03*a12.
        // The six base upper entries (1,2,3,4,5,6) give Pf=8;
        // the analytic inverse upper entries are (-6,5,-4,-3,2,-1)/8.
        // A signed diagonal congruence flips row/column zero, and scaling
        // multiplies the Pfaffian by scale^2 and the inverse by 1/scale.
        expected_pf[plane] = sign * 8.0 * scale * scale;
        for (i, j, coefficient, inverse) in [
            (0, 1, 1.0, -0.75),
            (0, 2, 2.0, 0.625),
            (0, 3, 3.0, -0.5),
            (1, 2, 4.0, -0.375),
            (1, 3, 5.0, 0.25),
            (2, 3, 6.0, -0.125),
        ] {
            let congruence = if i == 0 { sign } else { 1.0 };
            let upper = plane * N * N + i + j * N;
            let lower = plane * N * N + j + i * N;
            input[upper] = congruence * scale * coefficient;
            input[lower] = -input[upper];
            expected_inverse[upper] = congruence * inverse / scale;
            expected_inverse[lower] = -expected_inverse[upper];
        }
    }

    input[3 * N * N..4 * N * N].fill(f64::NAN);
    threading::install(|| {
        let observer = threading::start_observation();
        for _ in 0..2 {
            let output = COrderPfaffian
                .pfaffian_inverse_batch(&input, N, PLANES)
                .unwrap();
            assert_eq!(output.pf.len(), PLANES);
            assert_eq!(output.inv.len(), input.len());
            assert_eq!(output.outcome.len(), PLANES);
            for (plane, &pf) in expected_pf.iter().enumerate() {
                if plane == 3 {
                    assert_eq!(output.outcome[plane], PlaneOutcome::NonFinite);
                    assert!(!output.pf[plane].is_finite());
                    continue;
                }
                assert_eq!(
                    output.outcome[plane],
                    if plane == 1 || plane == 5 {
                        PlaneOutcome::ZeroPivot { row: N - 1 }
                    } else {
                        PlaneOutcome::Ok
                    },
                    "plane {plane}"
                );
                // Four-dimensional dense input: a small, explicit
                // roundoff envelope for products, divisions and permutations.
                numerical::assert_values_close(
                    [output.pf[plane]],
                    [pf],
                    128.0 * f64::EPSILON,
                    128.0 * f64::EPSILON,
                    "analytic dense Pfaffian",
                );
                let range = plane * N * N..(plane + 1) * N * N;
                numerical::assert_values_close(
                    output.inv[range.clone()].iter().copied(),
                    expected_inverse[range].iter().copied(),
                    128.0 * f64::EPSILON,
                    128.0 * f64::EPSILON,
                    "analytic dense inverse / failed-plane zero fill",
                );
            }
        }
        let observed = observer.finish();
        if threading::inner_thread_config().threads > 1 {
            assert!(observed.parallel_calls >= 2, "{observed:?}");
            assert_eq!(observed.parallel_qp_items, 2 * PLANES);
            assert!(observed.distinct_workers >= 2, "{observed:?}");
        } else {
            assert_eq!(observed.parallel_calls, 0);
            assert_eq!(observed.serial_qp_items, 2 * PLANES);
        }
        check_plane_preparation();
        let mut stage = COrderPfaffian;
        let empty = stage.pfaffian_inverse_batch(&[], N, 0).unwrap();
        assert!(empty.pf.is_empty() && empty.inv.is_empty() && empty.outcome.is_empty());
        assert!(stage
            .pfaffian_inverse_batch(&input[..input.len() - 1], N, PLANES)
            .is_err());
    });
}

#[derive(Default)]
struct RecordingPlanes {
    input: Vec<f64>,
}
impl PfaffianStages for RecordingPlanes {
    fn label(&self) -> String {
        "record-input-only".into()
    }
    fn provider(&self) -> String {
        "layout test double, not a numerical oracle".into()
    }
    fn pfaffian_inverse_batch(
        &mut self,
        input: &[f64],
        _n: usize,
        planes: usize,
    ) -> Result<mvmc_core::stage_backend::PfInvBatch, mvmc_core::stage_backend::StageError> {
        self.input = input.to_vec();
        Ok(mvmc_core::stage_backend::PfInvBatch {
            pf: (1..=planes).map(|value| value as f64).collect(),
            inv: vec![1.0; input.len()],
            outcome: vec![PlaneOutcome::Ok; planes],
        })
    }
}

fn check_plane_preparation() {
    use mvmc_core::measurement_batch::{stage_a_pfaffian, MeasurementBatchWorkspace, SlotStatus};
    use mvmc_core::SlaterElmFlat;
    let mut slater = SlaterElmFlat::<f64>::zeros(3, 4);
    for qp in 0..3 {
        for row in 0..8 {
            for col in row + 1..8 {
                let value = (100 * (qp + 1) + 10 * row + col + 1) as f64;
                slater.set(qp, row, col, value);
                slater.set(qp, col, row, -value);
            }
        }
    }
    let mut batch = MeasurementBatchWorkspace::default();
    batch.configure(4, 3, 4, false);
    batch.ele_idx.copy_from_slice(&[
        0, 2, 1, 3, // slot 0 -> source indices 0,2,5,7
        0, 1, 2, 3, // inactive slot 1
        -1, 1, 0, 2, // invalid slot 2
        3, 1, 0, 2, // slot 3 -> source indices 3,1,4,6
    ]);
    batch.status.fill(SlotStatus::Skipped);
    batch.real[1].pf.fill(77.0);
    batch.real[2].pf.fill(99.0);
    batch.real[1].inv.as_mut_slice().fill(77.0);
    batch.real[2].inv.as_mut_slice().fill(99.0);
    let mut recorder = RecordingPlanes::default();
    let observer = threading::start_observation();
    // Deliberately non-contiguous and reversed active slots.
    stage_a_pfaffian(
        &mut recorder,
        &slater,
        &mut batch,
        &[3, 2, 0],
        4,
        2,
        3,
        true,
    )
    .unwrap();
    let observed = observer.finish();
    // Independently enumerated column-major source submatrices, before the
    // adapter's inverse sign flip. Each subsequent QP adds 100 in magnitude.
    let first: [f64; 16] = [
        0.0, 114.0, -135.0, -137.0, -114.0, 0.0, -115.0, -117.0, 135.0, 115.0, 0.0, -147.0, 137.0,
        117.0, 147.0, 0.0,
    ];
    let last: [f64; 16] = [
        0.0, -103.0, -106.0, -108.0, 103.0, 0.0, -126.0, -128.0, 106.0, 126.0, 0.0, -158.0, 108.0,
        128.0, 158.0, 0.0,
    ];
    let mut expected = Vec::new();
    for base in [Some(first), None, Some(last)] {
        for qp in 0..3 {
            expected.extend(base.unwrap_or([0.0; 16]).map(|value| {
                value + value.signum() * if value == 0.0 { 0.0 } else { 100.0 * qp as f64 }
            }));
        }
    }
    numerical::assert_values_close(
        recorder.input,
        expected,
        128.0 * f64::EPSILON,
        128.0 * f64::EPSILON,
        "sample/QP order and invalid-plane zero fill",
    );
    assert_eq!(
        batch.status,
        vec![
            SlotStatus::Ready,
            SlotStatus::Skipped,
            SlotStatus::Failed,
            SlotStatus::Ready
        ]
    );
    for (slot, expected) in [
        (0, [7.0, 8.0, 9.0]),
        (1, [77.0; 3]),
        (2, [99.0; 3]),
        (3, [1.0, 2.0, 3.0]),
    ] {
        numerical::assert_values_close(
            batch.real[slot].pf.iter().copied(),
            expected,
            128.0 * f64::EPSILON,
            128.0 * f64::EPSILON,
            "slot publication",
        );
    }
    for (slot, value) in [(1, 77.0), (2, 99.0)] {
        let actual = batch.real[slot].inv.as_slice();
        numerical::assert_values_close(
            actual.iter().copied(),
            std::iter::repeat_n(value, actual.len()),
            128.0 * f64::EPSILON,
            128.0 * f64::EPSILON,
            "inactive/invalid inverse tables remain untouched",
        );
    }
    if threading::inner_thread_config().threads > 1 {
        assert_eq!(observed.parallel_entry_items, 9, "{observed:?}");
        assert!(observed.distinct_workers >= 2, "{observed:?}");
    } else {
        assert_eq!(observed.serial_entry_items, 9, "{observed:?}");
    }
}
