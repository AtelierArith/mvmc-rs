//! Pure-Rust default Reducer contracts, not a native MPI proof.
//! No oracle or reference runtime is used by this target.
use mvmc_core::Reducer;
use num_complex::Complex64;
use std::cell::RefCell;

struct Observe {
    rank: usize,
    calls: RefCell<Vec<Vec<i64>>>,
}

impl Observe {
    fn new(rank: usize) -> Self {
        Self {
            rank,
            calls: RefCell::new(Vec::new()),
        }
    }
}

impl Reducer for Observe {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {
        panic!("unexpected FP reduction");
    }

    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {
        panic!("unexpected complex reduction");
    }

    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        self.calls.borrow_mut().push(values.to_vec());
        for value in values {
            *value += 100;
        }
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        2
    }
}

#[test]
fn counter_default_reduces_exactly_six_and_keeps_logical_slots_on_root() {
    let reducer = Observe::new(0);
    let mut counters = [1, 2, 3, 4, 5, 6, -7, 8, 1, -10];
    reducer.reduce_counters(&mut counters);
    assert_eq!(reducer.calls.borrow().as_slice(), &[vec![1, 2, 3, 4, 5, 6]]);
    assert_eq!(counters, [101, 102, 103, 104, 105, 106, -7, 8, 1, -10]);
}

#[test]
fn counter_default_nonroot_participates_once_without_writeback() {
    let reducer = Observe::new(1);
    let mut counters = [1, 2, 3, 4, 5, 6, 7, 8, 1, 10];
    let before = counters;
    reducer.reduce_counters(&mut counters);
    assert_eq!(reducer.calls.borrow().len(), 1);
    assert_eq!(reducer.calls.borrow()[0], [1, 2, 3, 4, 5, 6]);
    assert_eq!(counters, before);
}

#[test]
fn counter_default_short_buffer_never_reduces_outside_slice() {
    let reducer = Observe::new(0);
    let mut counters = [3, 5];
    reducer.reduce_counters(&mut counters);
    assert_eq!(reducer.calls.borrow().as_slice(), &[vec![3, 5]]);
    assert_eq!(counters, [103, 105]);
}

#[test]
fn default_multirank_broadcasts_fail_without_mutating_payload() {
    let reducer = Observe::new(0);
    let mut integers = [7, -9];
    let mut real = [0.5, -1.0];
    assert!(reducer.broadcast_i64(0, &mut integers).is_err());
    assert!(reducer.broadcast_f64(0, &mut real).is_err());
    assert_eq!(integers, [7, -9]);
    assert_eq!(real, [0.5, -1.0]);
    assert!(reducer.calls.borrow().is_empty());
}

#[test]
fn counter_default_empty_root_keeps_one_empty_participation() {
    let reducer = Observe::new(0);
    let mut counters: [i64; 0] = [];
    reducer.reduce_counters(&mut counters);
    assert_eq!(counters, [0_i64; 0]);
    assert_eq!(reducer.calls.borrow().as_slice(), &[Vec::<i64>::new()]);
}

#[test]
fn counter_default_empty_nonroot_keeps_one_empty_participation() {
    let reducer = Observe::new(1);
    let mut counters: [i64; 0] = [];
    reducer.reduce_counters(&mut counters);
    assert_eq!(counters, [0_i64; 0]);
    assert_eq!(reducer.calls.borrow().as_slice(), &[Vec::<i64>::new()]);
}
