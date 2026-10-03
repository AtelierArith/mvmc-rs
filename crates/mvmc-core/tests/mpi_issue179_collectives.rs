//! Optional live MPI protocol checks, independent of the numerical runner.
#[cfg(feature = "mpi")]
#[test]
#[ignore = "launch this exact test with mpirun -n 2 or -n 4"]
fn issue179_collectives() {
    use mvmc_core::{mpi::MpiContext, reducer::Reducer};
    use num_complex::Complex64;
    let world = MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let group = world.split_groups(2).unwrap();
    for reducer in [&world as &dyn Reducer, &group as &dyn Reducer] {
        // All ranks enter the same empty collectives before nonempty work.
        reducer.allreduce_sum_f64(&mut []);
        reducer.allreduce_sum_c64(&mut []);
        reducer.allreduce_sum_i64(&mut []);
        let rank = world.rank();
        let n = world.world_size();
        let mut counts = [if rank < 3 { 1 } else { 0 }, rank as i64];
        reducer.allreduce_sum_i64(&mut counts);
        assert_eq!(counts, [n.min(3) as i64, (n * (n - 1) / 2) as i64]);
        // Dyadic/integer synthetic values have no rounding ambiguity here.
        // Numerical kernels require separately justified #190 bounds.
        let mut values = [rank as f64 + 0.5];
        reducer.allreduce_sum_f64(&mut values);
        assert!((values[0] - (n * n) as f64 / 2.0).abs() <= f64::EPSILON);
        let mut complex = [Complex64::new(rank as f64, 0.5)];
        reducer.allreduce_sum_c64(&mut complex);
        assert!((complex[0].re - (n * (n - 1) / 2) as f64).abs() <= f64::EPSILON);
        assert!((complex[0].im - n as f64 / 2.0).abs() <= f64::EPSILON);
        assert!(reducer.any_failure(rank == n - 1));
        assert!(!reducer.any_failure(false));
        // Fresh contributions catch accidental extra group multiplicity.
        let mut once = [1_i64];
        reducer.allreduce_sum_i64(&mut once);
        assert_eq!(once, [n as i64]);
    }
}
