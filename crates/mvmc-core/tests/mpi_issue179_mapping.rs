//! Exact communicator/seed endpoint contracts from C and Julia.
#![cfg(feature = "mpi")]

#[test]
#[ignore = "launch with MPI -n 2 or -n 4"]
fn issue179_group_width_endpoints() {
    use mvmc_core::{mpi::MpiContext, Reducer};
    let world = MpiContext::initialize().unwrap();
    let n = world.world_size();
    assert!(matches!(n, 2 | 4));
    if let Ok(expected) = std::env::var("MPI179_EXPECT_RANKS") {
        assert_eq!(n, expected.parse::<usize>().unwrap());
    }
    for width in [1, 2, 3, n] {
        let group = world.split_groups(width).unwrap();
        assert_eq!(group.seed_offset(), world.rank() / width);
        assert_eq!(group.rank(), world.rank() % width);
        let local_size = width.min(n - world.rank() / width * width);
        assert_eq!(group.world_size(), local_size);
        assert_eq!(group.reduction_size(), n);
        assert_eq!(group.is_output_root(), world.rank() == 0);
        assert!(group.sampling_any_failure(group.rank() == local_size - 1));
        assert_eq!(
            group.sampling_any_failure(world.rank() == n - 1),
            group.group() == (n - 1) / width,
            "sampling status must not leak between independent chains"
        );
        // Preserve #177: integer seed payload comes from global root, not
        // each group's local root. The assertion observes the real context.
        let mut seed = [if world.rank() == 0 { 179 } else { -1 }];
        Reducer::broadcast_i64(&group, 0, &mut seed).unwrap();
        assert_eq!(seed, [179]);
        // Nonzero global roots must also work with an uneven final group.
        let mut payload = [if world.rank() == n - 1 { 196 } else { -1 }];
        Reducer::broadcast_i64(&group, n - 1, &mut payload).unwrap();
        assert_eq!(payload, [196]);
        let mut cg_vector = [if world.rank() == n - 1 { 179.5 } else { -1.0 }];
        Reducer::broadcast_f64(&group, n - 1, &mut cg_vector).unwrap();
        Reducer::barrier(&group);
        assert_eq!(cg_vector, [179.5]);
        let mut total = [world.rank() as i64 + 1];
        group.allreduce_sum_i64(&mut total);
        assert_eq!(total, [(n * (n + 1) / 2) as i64]);
        let range = group.sampling_qp_range(5);
        let mut qp_count = [(range.end - range.start) as f64];
        group.sampling_sum_f64(&mut qp_count);
        assert_eq!(qp_count, [5.0]);
        // Exercise the actual sampling IP interface, not just MPI helpers.
        // NaNs model stale MainCal values outside this rank's owned range.
        // NQP=1 is smaller than grouped widths; empty ranks contribute zero.
        for qp_len in [1, 5] {
            use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
            use num_complex::Complex64;
            let mut data = ExpertModeData::default();
            data.qp_weights = Some(QuantumProjectionWeights {
                qp_full_weight: vec![Complex64::new(1.0, 0.0); qp_len],
                ..Default::default()
            });
            let owned = group.sampling_qp_range(qp_len);
            let mut real_pf = vec![f64::NAN; qp_len];
            let mut complex_pf = vec![Complex64::new(f64::NAN, f64::NAN); qp_len];
            for qp in owned {
                real_pf[qp] = if qp == 0 { 2.0 } else { 0.0 };
                complex_pf[qp] = Complex64::new(real_pf[qp], 0.0);
            }
            let real = mvmc_core::sampling::driver::sampling_log_ip_real(&real_pf, &data, &group);
            let complex =
                mvmc_core::sampling::driver::sampling_log_ip_complex(&complex_pf, &data, &group);
            // Well-conditioned log(2), allowing only elementary-function roundoff.
            assert!((real - 2.0_f64.ln()).abs() <= 4.0 * f64::EPSILON);
            assert!((complex.re - 2.0_f64.ln()).abs() <= 4.0 * f64::EPSILON);
            assert!(complex.im.abs() <= 4.0 * f64::EPSILON);
        }
        let mut counters = [1_i64; 10];
        group.reduce_counters(&mut counters);
        let chains_at_this_local_rank = (n - 1 - group.rank()) / width + 1;
        assert_eq!(
            &counters[..6],
            &[if group.group() == 0 {
                chains_at_this_local_rank as i64
            } else {
                1
            }; 6]
        );
        assert_eq!(&counters[6..], &[1; 4]);
        // The runner's counter entry point must preserve the same comm2/root
        // domain and logical slots, rather than accidentally summing all ten.
        let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, false);
        state.electron_config.counter.copy_from_slice(&[1; 10]);
        mvmc_core::counter::reduce_counter(&mut state, &group);
        assert_eq!(
            state.electron_config.counter.as_slice(),
            counters.as_slice()
        );
        assert!(group.any_failure(world.rank() == n - 1));
    }
}
