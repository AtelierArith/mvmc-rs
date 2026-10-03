//! Live protocol gates. Launch this exact ignored test on 2 and 4 MPI ranks.
//! Synthetic sampling/SR faults test agreement boundaries, not kernel parity.

use super::*;
use std::cell::Cell;

struct FaultReducer<'a> {
    inner: &'a dyn Reducer,
    fail_here: bool,
    sampling: bool,
    sr: bool,
    checks: Cell<usize>,
}

impl Reducer for FaultReducer<'_> {
    fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        self.inner.broadcast_i64(root, values)
    }
    fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
        self.inner.broadcast_c64(root, values);
    }
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        self.inner.allreduce_sum_f64(values);
    }
    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        self.inner.allreduce_sum_c64(values);
    }
    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        if self.sr && values.len() == 1 {
            assert_eq!(
                values[0], 0,
                "local SR must succeed before injected peer failure"
            );
            values[0] += i64::from(self.fail_here);
        }
        self.inner.allreduce_sum_i64(values);
    }
    fn sampling_qp_range(&self, length: usize) -> std::ops::Range<usize> {
        self.inner.sampling_qp_range(length)
    }
    fn sampling_sum_f64(&self, values: &mut [f64]) {
        self.inner.sampling_sum_f64(values);
    }
    fn sampling_sum_c64(&self, values: &mut [Complex64]) {
        self.inner.sampling_sum_c64(values);
    }
    fn reduce_counters(&self, values: &mut [i64]) {
        self.inner.reduce_counters(values);
    }
    fn any_failure(&self, failed: bool) -> bool {
        let check = self.checks.get() + 1;
        self.checks.set(check);
        self.inner
            .any_failure(failed || (self.sampling && check == 3 && self.fail_here))
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
    fn supports_grouped_sampling(&self) -> bool {
        self.inner.supports_grouped_sampling()
    }
    fn seed_offset(&self) -> usize {
        self.inner.seed_offset()
    }
    fn is_output_root(&self) -> bool {
        self.inner.is_output_root()
    }
}

#[test]
#[ignore = "requires an explicit 2/4-rank MPI launch with a timeout"]
fn negative_clock_and_asymmetric_runner_failures() {
    let world = crate::mpi::MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let mut root_pid = [if world.is_root() {
        i64::from(std::process::id())
    } else {
        0
    }];
    world.broadcast_i64(0, &mut root_pid).unwrap();
    let directory = std::env::temp_dir().join(format!("mvmc-runtime-mpi-{}", root_pid[0]));
    let root_setup = if world.is_root() {
        std::fs::create_dir(&directory).map_err(|e| e.to_string())
    } else {
        Ok(())
    };
    collective_result(root_setup, &world, "live test setup").unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
    );
    let namelist = fixture.join("inputs/namelist.def");
    let fixed = fixture.join("zqp_opt.dat");
    // Optimization must use its own supported input, not the PhysCal-only
    // TwoBodyGEx fixture. Keep the actual runtime rejection policy intact.
    let optimization_namelist = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def");
    let optimization_data = mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
        &optimization_namelist,
        false,
    )
    .unwrap();
    crate::validation::validate_para_opt(&optimization_data).unwrap();
    let missing = directory.join("missing.def");
    let last_rank = world.rank() + 1 == world.world_size();

    // NSplitSize=1 world execution plus groups with one and multiple local ranks.
    let group_one = world.split_groups(1).unwrap();
    let group_two = world.split_groups(2).unwrap();
    for (case, reducer, split) in [
        ("world", &world as &dyn Reducer, 1),
        ("group-one", &group_one as &dyn Reducer, 1),
        ("group-two", &group_two as &dyn Reducer, 2),
    ] {
        let clocks = Cell::new(0);
        if !world.is_root() {
            std::thread::sleep(std::time::Duration::from_millis(10 * world.rank() as u64));
        }
        let seed = resolve_seed_with_reducer(-1, None, reducer.seed_offset(), reducer, || {
            clocks.set(clocks.get() + 1);
            Ok(1_700_000_000 + world.rank() as i64 * 100)
        })
        .unwrap();
        assert_eq!(clocks.get(), usize::from(world.is_root()));
        assert_eq!(seed, 1_700_000_000 + reducer.seed_offset() as i64);
        let mut actual = seeded_rng_with_reducer(seed, reducer).unwrap();
        let mut expected = Sfmt19937Rng::new((1_700_000_000 + reducer.seed_offset()) as u32);
        for _ in 0..624 {
            assert_eq!(actual.gen_rand32(), expected.gen_rand32(), "{case}");
        }
        assert!(
            resolve_seed_with_reducer(-1, None, reducer.seed_offset(), reducer, || {
                assert!(world.is_root());
                Err("injected root clock failure".into())
            })
            .unwrap_err()
            .contains("time-based RndSeed")
        );
        assert!(seeded_rng_with_reducer(if last_rank { -1 } else { 7 }, reducer).is_err());

        // Actual rank-local parse/load errors must stop before output creation.
        assert!(prepare_phys_cal_from_namelist_with_reducer(
            if last_rank { &missing } else { &namelist },
            &fixed,
            "real",
            Some(1),
            reducer,
        )
        .is_err());
        assert!(prepare_phys_cal_from_namelist_with_reducer(
            &namelist,
            if last_rank { &missing } else { &fixed },
            "real",
            Some(1),
            reducer,
        )
        .is_err());
        let parse_output = directory.join(format!("{case}-parse"));
        let mut config = RunConfig::new(1, "real");
        config.nsmp = Some(1);
        config.seed = Some(1);
        config.output_dir = Some(parse_output.clone());
        assert!(run_para_opt_from_namelist_with_reducer(
            if last_rank {
                &missing
            } else {
                &optimization_namelist
            },
            config,
            reducer,
        )
        .is_err());
        assert!(!parse_output.exists());

        let load_output = directory.join(format!("{case}-load"));
        let mut config = RunConfig::new(1, "real");
        config.nsmp = Some(1);
        config.seed = Some(1);
        config.output_dir = Some(load_output.clone());
        config.initial_def = if last_rank {
            InitialDef::Path(missing.clone())
        } else {
            InitialDef::None
        };
        assert!(
            run_para_opt_from_namelist_with_reducer(&optimization_namelist, config, reducer,)
                .is_err()
        );
        assert!(!load_output.exists());

        let prepared = || {
            let mut prepared = prepare_phys_cal_from_namelist_with_reducer(
                &namelist,
                &fixed,
                "real",
                Some(1),
                reducer,
            )
            .unwrap();
            prepared.data.modpara.nsplit_size = split;
            prepared.data.modpara.nvmc_sample = 8;
            prepared.data.modpara.nvmc_warmup = 1;
            prepared.data.modpara.nvmc_interval = 1;
            prepared.data.modpara.n_data_qty_smp = 2;
            prepared
        };
        let invalid_output = directory.join(format!("{case}-validation"));
        let mut invalid = prepared();
        if last_rank {
            invalid.data.modpara.lanczos_mode = 3;
        }
        assert!(vmc_phys_cal_with_reducer(invalid, Some(&invalid_output), reducer).is_err());
        assert!(!invalid_output.exists());

        // Only root writes: make root's output path a file, and peers' paths valid.
        let blocked = directory.join(format!("{case}-blocked"));
        let root_setup = if world.is_root() {
            std::fs::write(&blocked, b"blocked").map_err(|e| e.to_string())
        } else {
            Ok(())
        };
        collective_result(root_setup, reducer, "blocked output fixture").unwrap();
        let peer_output = directory.join(format!("{case}-peer-output"));
        let output = if world.is_root() {
            &blocked
        } else {
            &peer_output
        };
        assert!(vmc_phys_cal_with_reducer(prepared(), Some(output), reducer).is_err());
        assert!(!peer_output.exists());

        // Directory setup succeeds, but root's first actual file write fails.
        let sample_output = directory.join(format!("{case}-sample-output"));
        let setup = if world.is_root() {
            std::fs::create_dir(&sample_output)
                .and_then(|()| std::fs::create_dir(sample_output.join("zvo_cisajs_001.dat")))
                .map_err(|error| error.to_string())
        } else {
            Ok(())
        };
        collective_result(setup, reducer, "sample output failure fixture").unwrap();
        let output_callback_calls = Cell::new(0);
        let mut callback = |_: usize, _: &ExpertModeData, _: Complex64, _: i32| {
            output_callback_calls.set(output_callback_calls.get() + 1);
            Ok(())
        };
        assert!(vmc_phys_cal_with_reducer_and_callback(
            prepared(),
            Some(&sample_output),
            reducer,
            Some(&mut callback),
        )
        .is_err());
        assert_eq!(output_callback_calls.get(), 0);
        assert!(!sample_output.join("zvo_cisajs_002.dat").exists());

        let sampling_output = directory.join(format!("{case}-sampling"));
        let fault = FaultReducer {
            inner: reducer,
            fail_here: last_rank,
            sampling: true,
            sr: false,
            checks: Cell::new(0),
        };
        assert!(vmc_phys_cal_with_reducer(prepared(), Some(&sampling_output), &fault).is_err());
        assert_eq!(std::fs::read_dir(&sampling_output).unwrap().count(), 0);

        // Some ranks have no callback, but still join the failure agreement.
        let callback_output = directory.join(format!("{case}-callback"));
        let calls = Cell::new(0);
        let mut callback = |sample, _: &ExpertModeData, _: Complex64, status| {
            calls.set(calls.get() + 1);
            assert_eq!((sample, status), (0, 0));
            Err("injected rank-local callback failure".into())
        };
        let callback: Option<&mut PhysCalCallback<'_>> =
            if last_rank { Some(&mut callback) } else { None };
        let error = vmc_phys_cal_with_reducer_and_callback(
            prepared(),
            Some(&callback_output),
            reducer,
            callback,
        )
        .unwrap_err();
        assert!(error.contains("callback"));
        assert_eq!(calls.get(), usize::from(last_rank));
        assert!(callback_output.join("zvo_cisajs_001.dat").exists());
        assert!(!callback_output.join("zvo_cisajs_002.dat").exists());

        // A peer failure after a locally successful solve restores parameters.
        let mut data = optimization_data.clone();
        data.modpara.nsplit_size = split;
        data.modpara.nvmc_sample = 8;
        data.modpara.nvmc_warmup = 1;
        data.modpara.nvmc_interval = 1;
        data.modpara.nsr_opt_itr_step = 1;
        data.modpara.nsr_opt_itr_smp = 1;
        let mut rng = seeded_rng_with_reducer(1 + reducer.seed_offset() as i64, reducer).unwrap();
        init_parameter(&mut data, &mut rng);
        sync_modified_parameter(&mut data, true);
        init_qp_weight(&mut data);
        let before = data.clone();
        let mut state = state_from_data(&data);
        let sr_output = directory.join(format!("{case}-sr"));
        let setup = if world.is_root() {
            std::fs::create_dir(&sr_output).map_err(|e| e.to_string())
        } else {
            Ok(())
        };
        collective_result(setup, reducer, "SR output fixture").unwrap();
        let fault = FaultReducer {
            inner: reducer,
            fail_here: last_rank,
            sampling: false,
            sr: true,
            checks: Cell::new(0),
        };
        let error = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&sr_output),
            &fault,
            OptimizationOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("local status 0"), "{error}");
        assert_eq!(data.slater_params, before.slater_params);
        assert_eq!(data.projection_parameters(), before.projection_parameters());
        assert!(!sr_output.join("zqp_opt.dat").exists());
        assert_eq!(
            std::fs::read_to_string(sr_output.join("zvo_out.dat"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        assert!(
            !reducer.any_failure(false),
            "all ranks must remain usable after faults"
        );
    }
    // All peers have finished checking files before root removes its temporary tree.
    assert!(!world.any_failure(false));
    let cleanup = if world.is_root() {
        std::fs::remove_dir_all(&directory).map_err(|e| e.to_string())
    } else {
        Ok(())
    };
    collective_result(cleanup, &world, "live test cleanup").unwrap();
}
