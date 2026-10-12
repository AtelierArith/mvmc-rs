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

#[test]
#[ignore = "requires CUDA GPU; set MVMC_RS_XORWOW_DEVICE (default 1)"]
fn public_twenty_step_runs_restart_xorwow_but_keep_each_run_continuous() {
    use mvmc_core::reducer::{Reducer, SingleProcessReducer};
    use mvmc_core::sampler_backend::{
        register_sampler_backend, set_sampler_backend_override, SamplerBackendKind, SamplerStages,
        SamplerTiming,
    };
    use mvmc_core::sampling::driver::SampleStats;
    use mvmc_core::stage_backend::StageError;
    use mvmc_core::state::VmcOptimizationState;
    use mvmc_core::{ExpertModeData, RunConfig};
    use mvmc_gpu_cuda::sampler_stages::CudaSamplerStages;
    use sfmt19937::Sfmt19937Rng;
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct Point {
        consumed: u128,
        future: [u32; 64],
        cfg: Vec<i64>,
        counter: Vec<i64>,
    }
    struct Recording {
        inner: CudaSamplerStages,
        runs: Arc<Mutex<Vec<Vec<Point>>>>,
    }
    impl SamplerStages for Recording {
        fn label(&self) -> String {
            self.inner.label()
        }
        fn provider(&self) -> String {
            self.inner.provider()
        }
        fn begin_run(&mut self) {
            self.inner.begin_run();
            self.runs.lock().unwrap().push(Vec::new());
        }
        fn sample_real(
            &mut self,
            data: &ExpertModeData,
            state: &mut VmcOptimizationState,
            rng: &mut Sfmt19937Rng,
            timing: &mut dyn SamplerTiming,
            reducer: &dyn Reducer,
        ) -> Result<SampleStats, StageError> {
            let result = self.inner.sample_real(data, state, rng, timing, reducer)?;
            let mut future = [0; 64];
            assert!(self.inner.gpu_words_preview(&mut future));
            self.runs.lock().unwrap().last_mut().unwrap().push(Point {
                consumed: self.inner.gpu_words_consumed().unwrap(),
                future,
                cfg: state.electron_config.ele_idx.clone(),
                counter: state.electron_config.counter.to_vec(),
            });
            Ok(result)
        }
    }
    let device = std::env::var("MVMC_RS_XORWOW_DEVICE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    let runs = Arc::new(Mutex::new(Vec::new()));
    let kind = SamplerBackendKind::Cuda(device);
    mvmc_gpu_cuda::install();
    assert!(register_sampler_backend(
        kind,
        Box::new(Recording {
            inner: CudaSamplerStages::with_rng(device, true, Some(11272)).unwrap(),
            runs: runs.clone(),
        })
    ));
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let mut outcomes = Vec::new();
    for _ in 0..2 {
        set_sampler_backend_override(Some(kind));
        let mut config = RunConfig::new(20, "real");
        config.nsmp = Some(20);
        config.seed = Some(7);
        let result =
            mvmc_core::run_para_opt_from_namelist_observed(&path, config, &SingleProcessReducer);
        set_sampler_backend_override(None);
        let (_, state, rng) = result.unwrap();
        outcomes.push((state.opt_data, rng.state_snapshot(), rng.words_consumed()));
    }
    let recorded = runs.lock().unwrap();
    assert_eq!(
        recorded.len(),
        2,
        "one lifecycle reset per complete public run"
    );
    for points in recorded.iter() {
        assert_eq!(points.len(), 20, "no reset between public SR iterations");
        // Replay is a separate CPU implementation, validated by NVIDIA-header
        // fixtures above; it is not an external full-trajectory oracle.
        let mut replay = XorwowRng::new(11272);
        for point in points {
            assert!(point.consumed > replay.words_consumed());
            while replay.words_consumed() < point.consumed {
                replay.gen_rand32();
            }
            let mut expected = [0; 64];
            replay.dump_rand32(&mut expected);
            assert_eq!(
                point.future, expected,
                "future words continue across SR boundaries"
            );
        }
    }
    for (first, second) in recorded[0].iter().zip(&recorded[1]) {
        assert_eq!(first.consumed, second.consumed);
        assert_eq!(first.future, second.future);
        assert_eq!(first.cfg, second.cfg);
        assert_eq!(first.counter, second.counter);
    }
    assert_eq!(outcomes[0].1, outcomes[1].1, "caller SFMT state");
    assert_eq!(outcomes[0].2, outcomes[1].2, "caller SFMT draw count");
    assert_eq!(outcomes[0].0.len(), 20);
    assert_eq!(outcomes[1].0.len(), 20);
    for (first, second) in outcomes[0].0.iter().zip(&outcomes[1].0) {
        assert!((first.energy - second.energy).norm() < 1e-9);
        assert!((first.energy_squared - second.energy_squared).norm() < 1e-9);
        assert_eq!(first.parameters.len(), second.parameters.len());
        for (a, b) in first.parameters.iter().zip(&second.parameters) {
            assert!((*a - *b).norm() < 1e-9);
        }
    }
}
