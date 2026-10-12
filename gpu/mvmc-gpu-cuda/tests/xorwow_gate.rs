//! Explicit GPU1 RNG and sampler replay gate; never runs in ordinary CPU tests.
use mvmc_core::sampling::rng::{SamplingRng, XorwowRng};
use mvmc_gpu_cuda::xorwow::{DeviceXorwow, GpuSamplingRng};

#[test]
#[ignore = "requires CUDA GPU; set MVMC_RS_XORWOW_DEVICE (default 1)"]
fn device_words_states_and_refill_match_cpu_xorwow() {
    let device = std::env::var("MVMC_RS_XORWOW_DEVICE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    for seed in [0, 1, 11272, u64::MAX] {
        let mut gpu = DeviceXorwow::new(device, seed, 7, 4096).unwrap();
        let mut cpu: Vec<_> = (0..7)
            .map(|i| XorwowRng::new(seed.wrapping_add(i)))
            .collect();
        for n in [1, 17, 4096, 3] {
            gpu.generate(n).unwrap();
            let words = gpu.download().unwrap();
            for chunk in words.as_chunks::<7>().0 {
                for (got, rng) in chunk.iter().zip(&mut cpu) {
                    assert_eq!(*got, rng.gen_rand32());
                }
            }
            assert_eq!(
                gpu.states().unwrap(),
                cpu.iter().flat_map(|r| r.state()).collect::<Vec<_>>()
            );
        }
        let mut gpu = GpuSamplingRng::new(device, seed, 17).unwrap();
        let mut cpu = XorwowRng::new(seed);
        for _ in 0..1000 {
            let mut a = [0; 624];
            let mut b = [0; 624];
            gpu.dump_rand32(&mut a);
            cpu.dump_rand32(&mut b);
            assert_eq!(a, b);
            assert_eq!(gpu.gen_rand32(), cpu.gen_rand32());
            assert_eq!(gpu.words_consumed(), cpu.words_consumed());
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU; set MVMC_RS_XORWOW_DEVICE (default 1)"]
fn sampling_replays_cpu_xorwow_proposals_and_decisions() {
    use mvmc_core::device_sampler::{run_lockstep_real_one, LockstepOptions, Teacher};
    use mvmc_core::reducer::SingleProcessReducer;
    use mvmc_core::sampling::driver::{trace, vmc_make_sample_real};
    let device = std::env::var("MVMC_RS_XORWOW_DEVICE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let prepare = || {
        let mut prep =
            mvmc_core::prepare_phys_cal_with_seed_offset(&path, None, "real", Some(7), true, 0)
                .unwrap();
        prep.data.modpara.nvmc_sample = 20;
        mvmc_core::run::prepare_sampling_walker(prep).unwrap()
    };
    let mut reference = prepare();
    let mut cpu = XorwowRng::new(11272);
    trace::start();
    trace::start_decisions();
    vmc_make_sample_real(&reference.data, &mut reference.state, &mut cpu).unwrap();
    let decisions = trace::finish_decisions();
    let events = trace::finish();
    let mut candidate = prepare();
    let mut gpu = GpuSamplingRng::new(device, 11272, 17).unwrap();
    let mut service =
        mvmc_gpu_cuda::device_sampler::CudaSamplerService::new_pinned(device).unwrap();
    trace::start();
    let (run, stats) = run_lockstep_real_one(
        &candidate.data,
        &mut candidate.state,
        &mut gpu,
        &SingleProcessReducer,
        &mut service,
        LockstepOptions {
            teachers: vec![Teacher {
                reference: decisions.clone(),
                weight_abs: 1e-12,
                weight_rel: 1e-10,
            }],
            ..Default::default()
        },
    )
    .unwrap();
    run.stats.unwrap();
    let got = trace::finish();
    let report = run.teacher.unwrap();
    assert_eq!(report.draw_mismatches, 0);
    assert_eq!(report.defects, 0);
    assert_eq!(report.overrun, 0);
    assert_eq!(report.compared, decisions.len());
    assert!(stats.proposals > 0);
    // Event 7 contains floating weights; compare all integer proposal/draw/config events exactly.
    assert_eq!(
        events.iter().filter(|e| e[0] != 7).collect::<Vec<_>>(),
        got.iter().filter(|e| e[0] != 7).collect::<Vec<_>>()
    );
    assert_eq!(gpu.words_consumed(), cpu.words_consumed());
    assert_eq!(
        candidate.state.electron_config.tmp_ele_idx,
        reference.state.electron_config.tmp_ele_idx
    );
    let mut a = [0; 624];
    let mut b = [0; 624];
    gpu.dump_rand32(&mut a);
    cpu.dump_rand32(&mut b);
    assert_eq!(a, b);
    eprintln!(
        "XORWOW sampler: {} proposals, {} decisions, {:?}",
        stats.proposals, report.compared, report
    );
}

#[test]
#[ignore = "requires CUDA GPU; set MVMC_RS_XORWOW_DEVICE (default 1)"]
fn production_rng_selection_retains_state_across_sample_series() {
    use mvmc_core::c_timer::CTimer;
    use mvmc_core::reducer::SingleProcessReducer;
    use mvmc_core::sampler_backend::SamplerStages;
    use mvmc_core::sampling::driver::{trace, vmc_make_sample_real};
    let device = std::env::var("MVMC_RS_XORWOW_DEVICE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let prepare = || {
        let mut prep =
            mvmc_core::prepare_phys_cal_with_seed_offset(&path, None, "real", Some(7), true, 0)
                .unwrap();
        prep.data.modpara.nvmc_sample = 20;
        mvmc_core::run::prepare_sampling_walker(prep).unwrap()
    };
    let mut cpu = prepare();
    let mut gpu = prepare();
    // Reproduce the implicit seed derivation from the run's initialized SFMT state.
    let mut seed_rng = cpu.rng.clone();
    let seed = u64::from(seed_rng.gen_rand32()) | (u64::from(seed_rng.gen_rand32()) << 32);
    let mut replay = XorwowRng::new(seed);
    let mut stage =
        mvmc_gpu_cuda::sampler_stages::CudaSamplerStages::with_rng(device, true, None).unwrap();
    assert!(stage.label().ends_with("rng=xorwow"));
    let mut timed = CTimer::<true>::new();
    let mut untimed = CTimer::<false>::new();
    for series in 0..2 {
        trace::start();
        let expected = vmc_make_sample_real(&cpu.data, &mut cpu.state, &mut replay).unwrap();
        let events = trace::finish();
        trace::start();
        let timing: &mut dyn mvmc_core::sampler_backend::SamplerTiming = if series == 0 {
            &mut untimed
        } else {
            &mut timed
        };
        let actual = stage
            .sample_real(
                &gpu.data,
                &mut gpu.state,
                &mut gpu.rng,
                timing,
                &SingleProcessReducer,
            )
            .unwrap();
        let got = trace::finish();
        assert_eq!(actual, expected);
        assert_eq!(
            events.iter().filter(|e| e[0] != 7).collect::<Vec<_>>(),
            got.iter().filter(|e| e[0] != 7).collect::<Vec<_>>()
        );
        assert_eq!(stage.gpu_words_consumed(), Some(replay.words_consumed()));
        assert_eq!(
            gpu.rng.state_snapshot(),
            seed_rng.state_snapshot(),
            "seed derived once; SFMT not used for proposals"
        );
    }
    assert!(
        timed.elapsed_ns.iter().any(|&ns| ns > 0),
        "timed inline sampling merges C timer sections"
    );
}
