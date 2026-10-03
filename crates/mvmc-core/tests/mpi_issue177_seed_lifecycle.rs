//! Bounded public negative-seed lifecycle. No replacement clocks or seed payloads.
#[cfg(feature = "mpi")]
mod enabled {
    use mvmc_core::{mpi::MpiContext, Reducer};
    use num_complex::Complex64;
    use std::{cell::RefCell, path::Path, rc::Rc};

    struct ObserveSeed<'a> {
        inner: &'a dyn Reducer,
        payloads: RefCell<Vec<Vec<i64>>>,
        offset_fault: Option<usize>,
    }
    impl Reducer for ObserveSeed<'_> {
        fn broadcast_i64(&self, root: usize, b: &mut [i64]) -> Result<(), String> {
            self.inner.broadcast_i64(root, b)?;
            self.payloads.borrow_mut().push(b.to_vec());
            Ok(())
        }
        fn broadcast_f64(&self, root: usize, b: &mut [f64]) -> Result<(), String> {
            self.inner.broadcast_f64(root, b)
        }
        fn broadcast_c64(&self, root: usize, b: &mut [Complex64]) {
            self.inner.broadcast_c64(root, b);
        }
        fn barrier(&self) {
            self.inner.barrier();
        }
        fn allreduce_sum_f64(&self, b: &mut [f64]) {
            self.inner.allreduce_sum_f64(b);
        }
        fn allreduce_sum_c64(&self, b: &mut [Complex64]) {
            self.inner.allreduce_sum_c64(b);
        }
        fn allreduce_sum_i64(&self, b: &mut [i64]) {
            self.inner.allreduce_sum_i64(b);
        }
        fn reduce_counters(&self, b: &mut [i64]) {
            self.inner.reduce_counters(b);
        }
        fn sampling_sum_f64(&self, b: &mut [f64]) {
            self.inner.sampling_sum_f64(b);
        }
        fn sampling_sum_c64(&self, b: &mut [Complex64]) {
            self.inner.sampling_sum_c64(b);
        }
        fn sampling_any_failure(&self, b: bool) -> bool {
            self.inner.sampling_any_failure(b)
        }
        fn sampling_qp_range(&self, n: usize) -> std::ops::Range<usize> {
            self.inner.sampling_qp_range(n)
        }
        fn any_failure(&self, b: bool) -> bool {
            self.inner.any_failure(b)
        }
        fn rank(&self) -> usize {
            self.inner.rank()
        }
        fn world_size(&self) -> usize {
            self.inner.world_size()
        }
        fn reduction_size(&self) -> usize {
            self.inner.reduction_size()
        }
        fn seed_offset(&self) -> usize {
            self.offset_fault
                .unwrap_or_else(|| self.inner.seed_offset())
        }
        fn supports_grouped_sampling(&self) -> bool {
            self.inner.supports_grouped_sampling()
        }
        fn is_output_root(&self) -> bool {
            self.inner.is_output_root()
        }
    }
    fn words(rng: &sfmt19937::Sfmt19937Rng) -> Vec<i64> {
        let mut b = [0; 624];
        rng.dump_rand32(&mut b);
        b.map(i64::from).to_vec()
    }
    fn saved_planes(ec: &mvmc_core::state::ElectronConfiguration) -> Vec<Vec<i64>> {
        vec![
            ec.ele_idx.clone(),
            ec.ele_cfg.clone(),
            ec.ele_num.clone(),
            ec.ele_proj_cnt.clone(),
            ec.ele_spn.clone(),
            ec.counter.to_vec(),
        ]
    }
    fn base(observer: &ObserveSeed<'_>) -> i64 {
        let payloads = observer.payloads.borrow();
        // Both public APIs receive an explicit output path; seed resolution is
        // their only integer broadcast in this bounded negative-seed call.
        assert_eq!(payloads.len(), 1, "unexpected integer broadcast protocol");
        assert_eq!(payloads[0].len(), 2);
        assert_eq!(payloads[0][0], 0);
        let seed = payloads[0][1];
        assert!(seed > 0 && seed <= i64::from(u32::MAX));
        seed
    }
    fn group_equal(world: &MpiContext, width: usize, local: &[i64]) {
        let mut failed = false;
        for peer in 0..world.world_size() {
            let mut size = [if world.rank() == peer {
                local.len() as i64
            } else {
                0
            }];
            world.broadcast_i64(peer, &mut size).unwrap();
            let mut b = if world.rank() == peer {
                local.to_vec()
            } else {
                vec![0; size[0] as usize]
            };
            world.broadcast_i64(peer, &mut b).unwrap();
            if world.rank() / width == peer / width {
                failed |= b != local;
            }
        }
        assert!(
            !world.any_failure(failed),
            "within-group actual sampler stream/config differs"
        );
    }
    fn checked_checkpoint(events: &[Vec<i64>]) -> &[i64] {
        let checkpoints: Vec<_> = events.iter().filter(|event| event[0] == 8).collect();
        assert_eq!(checkpoints.len(), 1, "one complete sampling frame required");
        let checkpoint = checkpoints[0];
        let mut offset = 1;
        for field in 0..7 {
            let length = usize::try_from(*checkpoint.get(offset).unwrap()).unwrap();
            offset += 1;
            if field == 6 {
                assert_eq!(length, 624);
            }
            offset = offset.checked_add(length).unwrap();
            assert!(offset <= checkpoint.len());
        }
        assert_eq!(offset, checkpoint.len());
        checkpoint
    }
    fn opt_outcome(
        result: &Result<mvmc_core::RunSummary, String>,
        systems: &[mvmc_core::sr::observer::DirectSolveObservation],
        events: &[Vec<i64>],
        world: &MpiContext,
    ) -> &'static str {
        checked_checkpoint(events);
        assert_eq!(systems.len(), 1, "exactly one real direct-SR observation");
        let system = &systems[0];
        assert_eq!(system.triangle, 'U');
        assert_eq!(system.rhs.len(), system.dimension);
        assert_eq!(system.matrix.len(), system.dimension * system.dimension);
        let factor_failed = system.status == Some(1)
            && system.not_solved.is_none()
            && system.dimension > 0
            && system
                .factor_info
                .is_some_and(|info| info > 0 && info as usize <= system.dimension)
            && system.solve_info.is_none();
        assert!(system
            .matrix
            .iter()
            .chain(system.rhs.iter())
            .all(|value| value.is_finite()));
        let any_factor_failed = world.any_failure(factor_failed);
        if let Err(reason) = result {
            assert_eq!(reason, &format!(
                "vmc_para_opt: direct SR failed at step 0 (local status {}); parameters were not updated",
                system.status.unwrap(),
            ), "unexpected OPT error");
            assert!(
                any_factor_failed,
                "no independently observed factorization failure"
            );
            if factor_failed {
                assert_eq!(
                    system.rhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    system
                        .increment
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            } else {
                assert_eq!(system.status, Some(0), "unrecognized peer error");
            }
            "step0-positive-factor-info"
        } else {
            assert!(!any_factor_failed);
            assert_eq!(system.status, Some(0));
            "success"
        }
    }
    #[derive(Default)]
    struct Samples(RefCell<Vec<Vec<i64>>>);
    impl mvmc_core::run::PhysCalGreenObserver for Samples {
        fn sample_completed(
            &self,
            _: &mvmc_core::ExpertModeData,
            _: usize,
            state: &mvmc_core::VmcOptimizationState,
            rng: &sfmt19937::Sfmt19937Rng,
        ) {
            let ec = &state.electron_config;
            let mut b = words(rng);
            b.push(rng.words_consumed() as i64);
            let (raw, cursor) = rng.state_snapshot();
            b.extend(raw.map(i64::from));
            b.push(cursor as i64);
            let planes: [&[i64]; 6] = [
                &ec.ele_idx,
                &ec.ele_cfg,
                &ec.ele_num,
                &ec.ele_proj_cnt,
                &ec.ele_spn,
                &ec.counter,
            ];
            for values in planes {
                b.push(values.len() as i64);
                b.extend(values);
            }
            self.0.borrow_mut().push(b);
        }
        fn accumulated(&self, _: mvmc_core::run::PhysCalGreenView<'_>) {}
    }
    fn prepare_inputs(from: &Path, to: &Path, width: usize, mode: i64) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
            }
        }
        let path = to.join("modpara.def");
        let original = std::fs::read_to_string(&path).unwrap();
        let mut text = String::new();
        for line in original.lines() {
            let key = line.split_whitespace().next().unwrap_or("");
            let value = match key {
                "RndSeed" => Some(-1),
                "NSplitSize" => Some(width as i64),
                "NVMCCalMode" => Some(mode),
                "NVMCSample" => Some(8),
                "NVMCWarmUp" | "NVMCInterval" | "NDataQtySmp" | "NSROptItrStep"
                | "NSROptItrSmp" => Some(1),
                "NSRCG" | "NStore" => Some(0),
                _ => None,
            };
            if let Some(value) = value {
                text.push_str(&format!("{key} {value}\n"));
            } else {
                text.push_str(line);
                text.push('\n');
            }
        }
        std::fs::write(path, text).unwrap();
        if mode == 0 {
            let path = to.join("namelist.def");
            let text = std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .filter(|line| line.split_whitespace().next() != Some("TwoBodyGEx"))
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(path, format!("{text}\n")).unwrap();
        }
    }
    #[test]
    #[ignore = "MPI177_EVIDENCE_DIR must be NEW; explicit mpiexec -n 2/4 with timeout"]
    fn public_negative_seed_opt_and_physcal_lifecycle() {
        let world = MpiContext::initialize().unwrap();
        assert!(matches!(world.world_size(), 2 | 4));
        let root = std::path::PathBuf::from(std::env::var("MPI177_EVIDENCE_DIR").unwrap());
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
        let setup = if world.is_root() {
            std::fs::create_dir(&root).map(|_| {
                for width in [1, 2] {
                    for mode in [0, 1] {
                        prepare_inputs(
                            &fixture.join("inputs"),
                            &root.join(format!("w{width}-m{mode}")),
                            width,
                            mode,
                        );
                    }
                }
            })
        } else {
            Ok(())
        };
        assert!(
            !world.any_failure(setup.is_err()),
            "exclusive evidence setup failed"
        );
        world.barrier();
        for width in [1, 2] {
            let group = (width == 2).then(|| world.split_groups(width).unwrap());
            let inner: &dyn Reducer = group.as_ref().map_or(&world as &dyn Reducer, |g| g);
            assert_eq!(inner.seed_offset(), world.rank() / width);
            std::thread::sleep(std::time::Duration::from_millis(world.rank() as u64 * 25));
            let opt = ObserveSeed {
                inner,
                payloads: RefCell::new(Vec::new()),
                offset_fault: None,
            };
            let namelist = root.join(format!("w{width}-m0/namelist.def"));
            let mut config = mvmc_core::RunConfig::new(1, "real");
            config.nsmp = Some(1);
            config.output_dir = Some(root.join(format!("opt-w{width}")));
            let mut explicit_config = config.clone();
            let sr_capture = mvmc_core::sr::observer::capture().unwrap();
            mvmc_core::sampling::driver::trace::start();
            let result =
                mvmc_core::run_para_opt_from_namelist_with_reducer(&namelist, config, &opt);
            let events = mvmc_core::sampling::driver::trace::finish();
            let systems = sr_capture.finish();
            let outcome = opt_outcome(&result, &systems, &events, &world);
            let seed = base(&opt);
            group_equal(&world, world.world_size(), &[seed]);
            // Separate public fixed-seed baseline; never replace the negative
            // run's broadcast payload or reseed its live sampler.
            explicit_config.seed = Some(seed);
            explicit_config.output_dir = Some(root.join(format!("opt-fixed-w{width}")));
            let explicit_sr_capture = mvmc_core::sr::observer::capture().unwrap();
            mvmc_core::sampling::driver::trace::start();
            let explicit_result = mvmc_core::run_para_opt_from_namelist_with_reducer(
                &namelist,
                explicit_config,
                inner,
            );
            let explicit_events = mvmc_core::sampling::driver::trace::finish();
            let explicit_systems = explicit_sr_capture.finish();
            let explicit_outcome = opt_outcome(
                &explicit_result,
                &explicit_systems,
                &explicit_events,
                &world,
            );
            assert_eq!(outcome, explicit_outcome);
            assert_eq!(systems, explicit_systems);
            assert_eq!(events, explicit_events);
            group_equal(&world, width, checked_checkpoint(&events));
            if result.is_err() {
                assert!(!root.join(format!("opt-w{width}/zqp_opt.dat")).exists());
                assert!(!root
                    .join(format!("opt-fixed-w{width}/zqp_opt.dat"))
                    .exists());
            }
            std::fs::write(
                root.join(format!("opt-w{width}-rank{}.txt", world.rank())),
                format!(
                    "base={seed} offset={} resolved_seed={} outcome={outcome} actual_error={:?} systems={systems:?} events={events:?}\n",
                    inner.seed_offset(), seed + inner.seed_offset() as i64, result.as_ref().err()
                ),
            )
            .unwrap();

            let phy = ObserveSeed {
                inner,
                payloads: RefCell::new(Vec::new()),
                offset_fault: None,
            };
            let namelist = root.join(format!("w{width}-m1/namelist.def"));
            let prepared = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
                &namelist,
                fixture.join("zqp_opt.dat"),
                "real",
                None,
                &phy,
            )
            .unwrap();
            let seed = base(&phy);
            group_equal(&world, world.world_size(), &[seed]);
            let expected = sfmt19937::Sfmt19937Rng::new(
                u32::try_from(seed + inner.seed_offset() as i64).unwrap(),
            );
            assert_eq!(prepared.rng.state_snapshot(), expected.state_snapshot());
            assert_eq!(prepared.rng.words_consumed(), 0);
            assert_eq!(words(&prepared.rng), words(&expected));
            let samples = Rc::new(Samples::default());
            let guard = mvmc_core::run::install_physcal_green_observer(samples.clone()).unwrap();
            let result = mvmc_core::vmc_phys_cal_with_reducer(
                prepared,
                Some(&root.join(format!("phys-w{width}"))),
                &phy,
            )
            .unwrap();
            drop(guard);
            assert_eq!(result.iterations, 1);
            assert_eq!(samples.0.borrow().len(), 1);
            group_equal(&world, width, &samples.0.borrow()[0]);
            group_equal(&world, width, &words(&result.final_rng));
            let explicit_prepared = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
                &namelist,
                fixture.join("zqp_opt.dat"),
                "real",
                Some(seed),
                inner,
            )
            .unwrap();
            let explicit_result = mvmc_core::vmc_phys_cal_with_reducer(
                explicit_prepared,
                Some(&root.join(format!("phys-fixed-w{width}"))),
                inner,
            )
            .unwrap();
            assert_eq!(
                result.final_rng.state_snapshot(),
                explicit_result.final_rng.state_snapshot()
            );
            assert_eq!(
                result.final_rng.words_consumed(),
                explicit_result.final_rng.words_consumed()
            );
            assert_eq!(words(&result.final_rng), words(&explicit_result.final_rng));
            assert_eq!(
                saved_planes(&result.state.electron_config),
                saved_planes(&explicit_result.state.electron_config)
            );
            std::fs::write(root.join(format!("phys-w{width}-rank{}.txt", world.rank())), format!("base={seed} offset={} samples={:?} final_raw={:?} final_next624={:?} words={}\n", inner.seed_offset(), samples.0.borrow(), result.final_rng.state_snapshot(), words(&result.final_rng), result.final_rng.words_consumed())).unwrap();
            eprintln!(
                "MPI177_LIFECYCLE rank={} world={} width={width} OPT=1 PHYSCAL=1",
                world.rank(),
                world.world_size()
            );
            // Fault-only cases use public overrides/invalid offset metadata,
            // never mutate the real negative-clock broadcast payload.
            for case in ["upper", "peer-upper", "peer-offset"] {
                let last = world.rank() + 1 == world.world_size();
                let seed = if case == "upper" || (case == "peer-upper" && last) {
                    i64::from(u32::MAX) + 1
                } else {
                    7
                };
                let fault = ObserveSeed {
                    inner,
                    payloads: RefCell::new(Vec::new()),
                    offset_fault: (case == "peer-offset" && last).then_some(usize::MAX),
                };
                let output = root.join(format!("boundary-opt-{case}-w{width}"));
                let mut config = mvmc_core::RunConfig::new(1, "real");
                config.nsmp = Some(1);
                config.seed = Some(seed);
                config.output_dir = Some(output.clone());
                mvmc_core::sampling::driver::trace::start();
                let error = mvmc_core::run_para_opt_from_namelist_with_reducer(
                    root.join(format!("w{width}-m0/namelist.def")),
                    config,
                    &fault,
                )
                .unwrap_err();
                let events = mvmc_core::sampling::driver::trace::finish();
                assert!(events.is_empty(), "seed failure entered sampler");
                assert!(fault.payloads.borrow().is_empty());
                assert!(!output.exists());
                assert!(error.contains(if case == "peer-offset" {
                    "seed offset"
                } else {
                    "seed"
                }));
                let error = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
                    &namelist,
                    fixture.join("zqp_opt.dat"),
                    "real",
                    Some(seed),
                    &fault,
                )
                .unwrap_err();
                assert!(fault.payloads.borrow().is_empty());
                assert!(error.contains(if case == "peer-offset" {
                    "seed offset"
                } else {
                    "seed"
                }));
                std::fs::write(
                    root.join(format!("boundary-{case}-w{width}-rank{}.txt", world.rank())),
                    error,
                )
                .unwrap();
            }
        }
    }
}

#[cfg(not(feature = "mpi"))]
#[test]
#[ignore = "requires mpi feature"]
fn public_negative_seed_opt_and_physcal_lifecycle() {
    panic!("Unsupported: #177 lifecycle requires mpi feature");
}
