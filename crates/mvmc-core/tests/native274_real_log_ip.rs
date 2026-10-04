//! Literal analytic real-overlap branch checks, with no C/toolbox runtime.
use mvmc_core::observables::calculate_log_ip_real;
use mvmc_core::reducer::{Reducer, SingleProcessReducer};
use mvmc_core::sampling::driver::sampling_log_ip_real;
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
use num_complex::Complex64;
use std::cell::Cell;
use std::ops::Range;

fn data(n: usize) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    let mut weights = QuantumProjectionWeights::new();
    weights.qp_full_weight = vec![Complex64::new(1.0, 0.0); n];
    data.qp_weights = Some(weights);
    data
}

fn public_logs(ip: f64) -> [f64; 2] {
    let data = data(1);
    [
        sampling_log_ip_real(&[ip], &data, &SingleProcessReducer),
        calculate_log_ip_real(&[ip], 0, 1, &data),
    ]
}

#[test]
fn real_zero_ip_preserves_c_nonfinite_recovery_predicate() {
    for ip in [0.0, -0.0] {
        for actual in public_logs(ip) {
            assert_eq!(actual, f64::NEG_INFINITY);
            assert!(!actual.is_finite());
        }
    }
}

#[test]
fn real_negative_ip_discards_complex_log_phase_not_magnitude() {
    // ln|+1| = ln|-1| = 0 exactly; no computed-float bitwise oracle.
    for ip in [1.0, -1.0] {
        for actual in public_logs(ip) {
            assert_eq!(actual, 0.0);
        }
    }
    // 0 < ln(2) < 1 is an analytic inequality, not a math-function tolerance.
    for ip in [2.0, -2.0] {
        for actual in public_logs(ip) {
            assert!(actual > 0.0 && actual < 1.0);
        }
    }
}

#[test]
fn real_smallest_subnormal_is_not_replaced_by_epsilon_floor() {
    // ln(2^-1074)=-1074*ln(2) lies strictly between -745 and -744.
    // The old +1e-100 path is approximately -230, so cannot satisfy this.
    for ip in [f64::from_bits(1), -f64::from_bits(1)] {
        for actual in public_logs(ip) {
            assert!(actual.is_finite() && actual > -745.0 && actual < -744.0);
        }
    }
}

struct Comm1Literal {
    result: f64,
    calls: Cell<usize>,
}
impl Reducer for Comm1Literal {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {
        panic!("must use comm1, not global reduction");
    }
    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {
        panic!("not complex reduction");
    }
    fn allreduce_sum_i64(&self, _: &mut [i64]) {
        panic!("not integer reduction");
    }
    fn sampling_qp_range(&self, length: usize) -> Range<usize> {
        assert_eq!(length, 2);
        1..2
    }
    fn sampling_sum_f64(&self, values: &mut [f64]) {
        assert_eq!(values, &[-2.0]); // local owned-QP contribution before log
        self.calls.set(self.calls.get() + 1);
        values[0] = self.result;
    }
}

#[test]
fn real_log_is_taken_after_comm1_owned_ip_sum() {
    for reduced in [0.0, -1.0, f64::from_bits(1)] {
        let reducer = Comm1Literal {
            result: reduced,
            calls: Cell::new(0),
        };
        let actual = sampling_log_ip_real(&[99.0, -2.0], &data(2), &reducer);
        assert_eq!(reducer.calls.get(), 1);
        if reduced == 0.0 {
            assert_eq!(actual, f64::NEG_INFINITY);
        } else if reduced == -1.0 {
            assert_eq!(actual, 0.0);
        } else {
            assert!(actual.is_finite() && actual > -745.0 && actual < -744.0);
        }
    }
}
