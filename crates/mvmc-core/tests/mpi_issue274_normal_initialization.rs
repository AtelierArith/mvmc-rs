//! Optional live signed comm1 MAX and real/complex typed preflight protocol.
//! Synthetic MPI operands are not native C kernel/RNG-model parity evidence.

#[test]
#[ignore = "explicit MPI 2/4-rank gate; use bounded external timeout"]
fn normal_initializer_signed_max_and_peer_preflight() {
    #[cfg(not(feature = "mpi"))]
    panic!("Unsupported: issue274 live protocol requires feature mpi");

    #[cfg(feature = "mpi")]
    {
        use mvmc_core::{
            c_timer::CTimer, mpi::MpiContext, sampling::driver, ExpertModeData, Reducer,
            VmcOptimizationState,
        };
        use sfmt19937::Sfmt19937Rng;
        let world = MpiContext::initialize().unwrap();
        let initializing_thread = std::thread::current().id();
        assert!(rayon::current_thread_index().is_none());
        let ranks = world.world_size();
        assert!(
            matches!(ranks, 2 | 4),
            "Unsupported: actual MPI world must be 2/4"
        );
        let rank = world.rank();
        assert_eq!(
            world
                .sampling_max_info(if rank == 0 { -3 } else { -7 })
                .unwrap(),
            -3
        );
        assert_eq!(
            world
                .sampling_max_info(if rank == ranks - 1 { 4 } else { 0 })
                .unwrap(),
            4
        );
        for width in [1, 2] {
            let group = world.split_groups(width).unwrap();
            assert_eq!(group.world_size(), width);
            assert_eq!(group.reduction_size(), ranks); // Global SR domain, not comm1.
            {
                use mpi::traits::Communicator;
                assert_eq!(group.communicator().size() as usize, width);
            }
            let local_rank = rank % width;
            assert_eq!(
                group
                    .sampling_max_info(if local_rank == 0 { -3 } else { -7 })
                    .unwrap(),
                -3
            );
            assert_eq!(
                group
                    .sampling_max_info(if local_rank == width - 1 { 4 } else { 0 })
                    .unwrap(),
                4
            );
            for complex in [false, true] {
                for failure in [
                    "fresh-shape",
                    "burn-shape",
                    "burn-kernel",
                    "ne-domain",
                    "site-domain",
                ] {
                    assert_eq!(std::thread::current().id(), initializing_thread);
                    assert!(rayon::current_thread_index().is_none());
                    println!("\nISSUE274 participant rank={rank} world={ranks} width={width} complex={complex} failure={failure} phase=begin");
                    let burn = failure.starts_with("burn-");
                    let mut data = ExpertModeData::new();
                    data.modpara.nsite = 2;
                    data.modpara.nelec = 1;
                    let mut state = VmcOptimizationState::zeros(
                        2,
                        1,
                        data.projection_layout().n_proj,
                        0,
                        2,
                        1,
                        complex,
                        false,
                    );
                    if burn {
                        state.electron_config.counter[9] = 1;
                    }
                    if local_rank == width - 1 {
                        if failure == "ne-domain" {
                            data.modpara.nelec = i64::MAX;
                        } else if failure == "site-domain" {
                            data.modpara.nsite = i64::MAX;
                        } else if failure == "burn-kernel" {
                            state.electron_config.burn_ele_idx[0] = 9;
                        } else if burn {
                            state.electron_config.burn_ele_idx.clear();
                        } else {
                            state.electron_config.tmp_ele_cfg.pop();
                        }
                    }
                    let mut rng = Sfmt19937Rng::new(12395);
                    let before = rng.state_snapshot();
                    let count = rng.words_consumed();
                    let mut future = [0; 624];
                    rng.dump_rand32(&mut future);
                    let mut timer = CTimer::<false>::new();
                    let result = if complex {
                        driver::vmc_make_sample_with_reducer_timed(
                            &data, &mut state, &mut rng, &mut timer, &group,
                        )
                    } else {
                        driver::vmc_make_sample_real_with_reducer_timed(
                            &data, &mut state, &mut rng, &mut timer, &group,
                        )
                    };
                    let error = result.unwrap_err();
                    if local_rank == width - 1 {
                        if failure == "burn-kernel" {
                            assert!(matches!(error, mvmc_core::sampling::normal_initial::NormalInitializationError::Kernel(mvmc_core::CalcMAllError::SiteOutOfRange { .. })));
                        } else {
                            assert!(matches!(error, mvmc_core::sampling::normal_initial::NormalInitializationError::Precondition(_)));
                        }
                    } else {
                        assert!(matches!(error, mvmc_core::sampling::normal_initial::NormalInitializationError::PeerPrecondition));
                    }
                    assert_eq!(rng.state_snapshot(), before);
                    assert_eq!(rng.words_consumed(), count);
                    let mut after_future = [0; 624];
                    rng.dump_rand32(&mut after_future);
                    assert_eq!(after_future, future);
                    group.barrier(); // Every rank returned, not just the bad owner.
                    assert_eq!(std::thread::current().id(), initializing_thread);
                    println!("\nISSUE274 participant rank={rank} world={ranks} width={width} complex={complex} failure={failure} phase=returned raw_rng=unchanged");
                }
            }
        }
        world.barrier();
    }
}
