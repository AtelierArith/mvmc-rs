//! Phase 1: real kernel/sampler failure, before any MPI implementation.
//! Finite overflow operands are deliberate failure-boundary input, not a claim
//! that huge parameters are a supported C physical workload. No NaN injection,
//! FaultReducer, oracle runtime, or production observation hook is used.

use mvmc_core::{
    pfaffian::calc_m_all_fsz_real, sampling::vmc_make_sample_fsz_real,
    state::ThreadedPfaPackWorkspace, CalcMAllError, ExpertModeData, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use mvmc_core::{
    vmc_para_opt, vmc_phys_cal_in_place, OptimizationOptions, Reducer, SingleProcessReducer,
};

fn data() -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 2;
    data.modpara.two_sz = 0;
    data.modpara.nvmc_warmup = 1;
    data.modpara.nvmc_sample = 3;
    data.modpara.nvmc_interval = 1;
    data.i_flg_orbital_general = 1;
    data.n_gutzwiller_idx = 1;
    data.gutzwiller_idx = vec![0; 2];
    data.gutzwiller_terms = (0..2)
        .map(|site| mvmc_expert_parsers::GutzwillerTerm {
            site,
            value: Complex64::new(0.0, 0.0),
            is_complex: false,
        })
        .collect();
    data
}

/// Complete declared General mapping; the public runner, not this test,
/// refreshes the actual Slater table. No fixed state-only corruption.
fn public_data(scale: f64, width: usize, physcal: bool) -> ExpertModeData {
    let mut data = data();
    data.modpara.nsp_gauss_leg = 1;
    data.modpara.nmp_trans = 1;
    data.modpara.nsplit_size = width as i64;
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
    data.modpara.n_data_qty_smp = 1;
    data.modpara.vmc_calc_mode = i64::from(physcal);
    data.modpara.nsrcg = 0;
    data.modpara.lanczos_mode = 0;
    data.n_qp_opt_trans = 1;
    data.n_qp_trans = 1;
    data.qp_trans_entries = vec![mvmc_expert_parsers::QPTransEntry {
        weight: Complex64::new(1.0, 0.0),
        site_map: vec![0, 1],
        site_sign: vec![1, 1],
    }];
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0)];
    data.qp_opt_trans = vec![vec![0, 1]];
    data.qp_opt_trans_sgn = vec![vec![1, 1]];
    data.slater_params = vec![Complex64::new(scale, 0.0)];
    data.modpara.n_orbital_idx = 1;
    for i in 0..4 {
        for j in i + 1..4 {
            data.orbital_terms.push(mvmc_expert_parsers::OrbitalTerm {
                site1: i,
                site2: j,
                idx: 0,
                is_complex: false,
                sign: 1,
            });
        }
    }
    data.ensure_optimization_flags(data.count_variational_parameters());
    data.optimization_flags.fill(0);
    mvmc_core::qp::init_qp_weight(&mut data);
    data
}

fn public_state(data: &ExpertModeData) -> VmcOptimizationState {
    VmcOptimizationState::zeros(
        2,
        2,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        1,
        3,
        false,
        true,
    )
}

#[derive(Debug, PartialEq, Eq)]
struct PublicCheckpoint {
    raw: [u32; 624],
    cursor: usize,
    count: u128,
    next624: [u32; 624],
    saved_and_scratch: String,
}

struct SerialOutput(PathBuf);
impl SerialOutput {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "issue178-sampling-control-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("{error}"),
            }
        }
    }
}
impl Drop for SerialOutput {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0); // Only this exclusively created path.
    }
}

fn checkpoint(state: &VmcOptimizationState, rng: &Sfmt19937Rng) -> PublicCheckpoint {
    let (raw, cursor) = rng.state_snapshot();
    let count = rng.words_consumed();
    let mut next624 = [0; 624];
    rng.dump_rand32(&mut next624);
    assert_eq!(rng.state_snapshot(), (raw, cursor));
    assert_eq!(rng.words_consumed(), count);
    PublicCheckpoint {
        raw,
        cursor,
        count,
        next624,
        saved_and_scratch: format!("{:?}", state.electron_config),
    }
}

fn public_call(
    physcal: bool,
    healthy_control: bool,
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output: Option<&Path>,
    reducer: &impl Reducer,
) -> (Result<(), String>, usize) {
    let mut calls = 0;
    let result = if physcal {
        let mut callback = |_: usize, _: &ExpertModeData, _: Complex64, _: i32| {
            calls += 1;
            Ok(())
        };
        vmc_phys_cal_in_place(data, state, rng, output, reducer, Some(&mut callback)).map(|_| ())
    } else {
        let mut callback = |_: usize, _: &mut ExpertModeData, _: Complex64, _: i32| {
            calls += 1;
            Ok(())
        };
        vmc_para_opt(
            data,
            state,
            rng,
            output,
            reducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                // Only the healthy control stops before SR. Failure cells leave
                // SR enabled, proving that the sampler error prevents the update.
                skip_sr: healthy_control,
                file_flush_interval: None,
            },
        )
    };
    (result, calls)
}

fn assert_unpublished_failure(state: &VmcOptimizationState) {
    // These zeros are the allocated runner entry state, NOT a promise to
    // preserve a caller sentinel through PhysCal's documented reallocation.
    let config = &state.electron_config;
    assert!(config.ele_idx.iter().all(|&v| v == 0));
    assert!(config.ele_cfg.iter().all(|&v| v == 0));
    assert!(config.ele_num.iter().all(|&v| v == 0));
    assert!(config.ele_proj_cnt.iter().all(|&v| v == 0));
    assert!(config.ele_spn.iter().all(|&v| v == 0));
    assert!(config.counter.iter().all(|&v| v == 0));
    assert_eq!(config.tmp_ele_num, vec![1; 4]);
    assert_eq!(config.tmp_ele_proj_cnt, vec![2]);
    assert!(config
        .tmp_ele_idx
        .iter()
        .all(|&site| (0..2).contains(&site)));
    assert!(state.slater_matrix.pf_m_real.iter().all(|&v| v == 0.0));
    assert!(state
        .slater_matrix
        .inv_m_real
        .as_slice()
        .iter()
        .all(|&v| v == 0.0));
    assert!(state
        .slater_matrix
        .pf_m
        .iter()
        .all(|&v| v == Complex64::new(0.0, 0.0)));
    assert!(state
        .slater_matrix
        .inv_m
        .as_slice()
        .iter()
        .all(|&v| v == Complex64::new(0.0, 0.0)));
}

#[test]
fn public_serial_finite_overflow_failure_and_healthy_control() {
    for physcal in [false, true] {
        let mut repeated = None;
        for trial in 0..2 {
            let mut data = public_data(2.0_f64.powi(600), 1, physcal);
            let coefficients = data.slater_params.clone();
            let flags = data.optimization_flags.clone();
            let mut state = public_state(&data);
            let sr = format!("{:?}", state.sr_opt);
            let mut rng = Sfmt19937Rng::new(1);
            rng.gen_rand32();
            let (result, callbacks) = public_call(
                physcal,
                false,
                &mut data,
                &mut state,
                &mut rng,
                None,
                &SingleProcessReducer,
            );
            let error = result.unwrap_err();
            assert_eq!(
                error,
                CalcMAllError::NonFinitePfaffian { qp: 0 }.to_string()
            );
            assert_eq!(callbacks, 0);
            assert_eq!(data.slater_params, coefficients);
            assert_eq!(data.optimization_flags, flags);
            assert_eq!(format!("{:?}", state.sr_opt), sr);
            assert!(state.opt_data.is_empty());
            assert_unpublished_failure(&state);
            assert!(state
                .slater_matrix
                .slater_elm
                .as_slice()
                .iter()
                .all(|z| z.re.is_finite() && z.im.is_finite()));
            let actual = checkpoint(&state, &rng);
            if let Some(previous) = &repeated {
                assert_eq!(&actual, previous);
            }
            println!("ISSUE178_PUBLIC_SERIAL physcal={physcal} trial={trial} error={error} callbacks={callbacks} count={} cursor={}", actual.count, actual.cursor);
            repeated = Some(actual);
        }
        let mut data = public_data(1.0, 1, physcal);
        let mut state = public_state(&data);
        let mut rng = Sfmt19937Rng::new(1);
        rng.gen_rand32();
        let output = SerialOutput::new();
        let (result, callbacks) = public_call(
            physcal,
            true,
            &mut data,
            &mut state,
            &mut rng,
            Some(&output.0),
            &SingleProcessReducer,
        );
        result.unwrap();
        assert_eq!(callbacks, 1);
        assert!(state.electron_config.ele_num.iter().all(|&n| n == 1));
        println!(
            "ISSUE178_PUBLIC_SERIAL healthy physcal={physcal} count={}",
            rng.words_consumed()
        );
    }
}

#[cfg(feature = "mpi")]
mod mpi_phase2 {
    use super::*;
    use std::{cell::RefCell, fs, path::PathBuf};

    /// Observation only. Every operation delegates to the real MPI group;
    /// notably neither the default serial sampling range nor local-only
    /// failure agreement is allowed to replace the comm1 implementation.
    struct Observer<'a> {
        inner: &'a dyn Reducer,
        comm1: RefCell<Vec<bool>>,
        global: RefCell<Vec<bool>>,
    }

    impl Reducer for Observer<'_> {
        fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
            self.inner.sampling_max_info(info)
        }
        fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
            self.inner.broadcast_f64(root, values)
        }
        fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
            self.inner.broadcast_i64(root, values)
        }
        fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
            self.inner.broadcast_c64(root, values);
        }
        fn barrier(&self) {
            self.inner.barrier();
        }
        fn allreduce_sum_f64(&self, values: &mut [f64]) {
            self.inner.allreduce_sum_f64(values);
        }
        fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
            self.inner.allreduce_sum_c64(values);
        }
        fn allreduce_sum_i64(&self, values: &mut [i64]) {
            self.inner.allreduce_sum_i64(values);
        }
        fn reduce_counters(&self, counters: &mut [i64]) {
            self.inner.reduce_counters(counters);
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
        fn sampling_any_failure(&self, failed: bool) -> bool {
            self.comm1.borrow_mut().push(failed);
            self.inner.sampling_any_failure(failed)
        }
        fn any_failure(&self, failed: bool) -> bool {
            self.global.borrow_mut().push(failed);
            self.inner.any_failure(failed)
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
    #[ignore = "explicit real MPI 2/4-rank sampling failure gate; bounded external timeout"]
    fn actual_sampling_failure_reaches_comm1_then_all_global_ranks() {
        let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
        let ranks = world.world_size();
        assert!(matches!(ranks, 2 | 4));
        let width: usize = std::env::var("MPI_ISSUE178_WIDTH")
            .unwrap()
            .parse()
            .unwrap();
        assert!(matches!(width, 1 | 2));
        let physcal = match std::env::var("MPI_ISSUE178_API").unwrap().as_str() {
            "opt" => false,
            "physcal" => true,
            other => panic!("unknown API {other}"),
        };
        assert!(
            !physcal || width == 1,
            "grouped FSZ PhysCal is intentionally unsupported"
        );
        let bad_rank = match std::env::var("MPI_ISSUE178_FAIL_RANK").unwrap().as_str() {
            "root" => 0,
            "last-owner" => ranks - width,
            other => panic!("unknown fault selector {other}"),
        };
        let rank = world.rank();
        let failing_group = rank / width == bad_rank / width;
        let group = world.split_groups(width).unwrap();
        let observer = Observer {
            inner: &group,
            comm1: RefCell::new(Vec::new()),
            global: RefCell::new(Vec::new()),
        };
        let range = observer.sampling_qp_range(1);
        assert_eq!(
            range,
            if rank.is_multiple_of(width) {
                0..1
            } else {
                1..1
            }
        );
        assert!(bad_rank % width == 0, "fault must be an actual QP owner");
        let parent = PathBuf::from(std::env::var_os("MPI_ISSUE178_OUTPUT").unwrap());
        let setup = if world.is_root() {
            fs::create_dir(&parent).and_then(|()| fs::write(parent.join("sentinel"), b"untouched"))
        } else {
            Ok(())
        };
        assert!(
            !observer.any_failure(setup.is_err()),
            "exclusive output setup failed"
        );
        observer.barrier();

        let mut repeated = None;
        for trial in 0..2 {
            observer.comm1.borrow_mut().clear();
            observer.global.borrow_mut().clear();
            let scale = if rank == bad_rank {
                2.0_f64.powi(600)
            } else {
                1.0
            };
            let mut data = public_data(scale, width, physcal);
            let parameters = (
                data.projection_parameters(),
                data.slater_params.clone(),
                data.optimization_flags.clone(),
            );
            let mut state = public_state(&data);
            // PhysCal replaces the complete caller state. Its failure checks
            // below refer to the new entry allocation, not these sentinels.
            if physcal {
                state.electron_config.ele_idx.fill(71);
                state.slater_matrix.pf_m_real.fill(19.0);
                state.slater_matrix.inv_m_real.as_mut_slice().fill(23.0);
            }
            let sr_before = format!("{:?}", public_state(&data).sr_opt);
            let mut rng = Sfmt19937Rng::new(1 + observer.seed_offset() as u32);
            rng.gen_rand32();
            let before_rng = rng.state_snapshot();
            let output = parent.join(format!("trial{trial}"));
            let (result, callbacks) = public_call(
                physcal,
                false,
                &mut data,
                &mut state,
                &mut rng,
                Some(&output),
                &observer,
            );
            // Report the actual return before assertions, preserving evidence
            // if one rank incorrectly succeeds or returns an unrelated error.
            println!("ISSUE178_MPI_RETURN rank={rank} ranks={ranks} width={width} physcal={physcal} bad_rank={bad_rank} trial={trial} range={range:?} result={result:?} callbacks={callbacks} comm1={:?} global={:?}", observer.comm1.borrow(), observer.global.borrow());
            let error = result.unwrap_err();
            if rank == bad_rank {
                assert_eq!(
                    error,
                    CalcMAllError::NonFinitePfaffian { qp: 0 }.to_string()
                );
            } else if failing_group {
                assert_eq!(
                    error,
                    "sampling initialization failed on another comm1 rank"
                );
            } else {
                assert_eq!(
                    error,
                    if physcal {
                        "sample 0 failed on another MPI rank"
                    } else {
                        "sample step 0 failed on another MPI rank"
                    }
                );
            }
            assert_eq!(callbacks, 0);
            assert_eq!(data.projection_parameters(), parameters.0);
            assert_eq!(data.slater_params, parameters.1);
            assert_eq!(data.optimization_flags, parameters.2);
            assert_eq!(format!("{:?}", state.sr_opt), sr_before);
            assert!(state.opt_data.is_empty());
            assert_eq!(observer.global.borrow().last(), Some(&failing_group));
            if failing_group {
                assert_unpublished_failure(&state);
                assert_eq!(observer.comm1.borrow().len(), 101);
                assert!(observer
                    .comm1
                    .borrow()
                    .iter()
                    .all(|&local| local == (rank == bad_rank)));
            } else {
                // A healthy group can finish its saved chain before the
                // global error. Do not assert rollback or zero RNG draws.
                assert!(state.electron_config.ele_num.iter().all(|&v| v == 1));
                assert!(observer.comm1.borrow().iter().all(|&local| !local));
            }
            let actual = checkpoint(&state, &rng);
            assert_ne!((actual.raw, actual.cursor), before_rng);
            if let Some(previous) = &repeated {
                assert_eq!(&actual, previous);
            }
            println!("ISSUE178_MPI_CHECKPOINT rank={rank} trial={trial} count={} cursor={} raw={:?} next624={:?} configuration={}", actual.count, actual.cursor, actual.raw, actual.next624, actual.saved_and_scratch);
            repeated = Some(actual);
            observer.barrier();
            assert_eq!(fs::read(parent.join("sentinel")).unwrap(), b"untouched");
            if output.exists() {
                assert_eq!(fs::read_dir(&output).unwrap().count(), 0);
            }
            observer.barrier();
        }
        if world.is_root() {
            let expected = if physcal { 3 } else { 1 };
            assert_eq!(fs::read_dir(&parent).unwrap().count(), expected);
        }
        observer.barrier();
        println!("ISSUE178_MPI_DONE rank={rank} ranks={ranks} width={width} physcal={physcal} bad_rank={bad_rank}");
        // Retain all directories/artifacts; never remove launcher-owned paths.
    }
}

fn state() -> VmcOptimizationState {
    let mut state = VmcOptimizationState::zeros(2, 2, 1, 1, 1, 3, false, true);
    let scale = 2.0_f64.powi(600);
    assert!(scale.is_finite());
    for i in 0..4 {
        for j in i + 1..4 {
            state
                .slater_matrix
                .slater_elm
                .set(0, i, j, Complex64::new(scale, 0.0));
            state
                .slater_matrix
                .slater_elm
                .set(0, j, i, Complex64::new(-scale, 0.0));
            // Prepare C's real-table input explicitly; public runners still
            // own their parameter-to-table refresh. No historical kernel overlay.
            state.slater_matrix.slater_elm_real.set(0, i, j, scale);
            state.slater_matrix.slater_elm_real.set(0, j, i, -scale);
        }
    }
    assert!(state
        .slater_matrix
        .slater_elm
        .as_slice()
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite()));
    // Saved planes/counters are separate from initializer-owned tmp planes.
    state.electron_config.ele_idx.fill(7);
    state.electron_config.ele_cfg.fill(8);
    state.electron_config.ele_num.fill(9);
    state.electron_config.ele_proj_cnt.fill(10);
    state.electron_config.ele_spn.fill(11);
    state.electron_config.counter.fill(12);
    state.electron_config.counter[9] = 0; // Force genuine initialization.
    state
}

#[test]
fn finite_overflow_pfaffian_and_real_sampler_return_actual_error() {
    // Every principal 4x4 skew matrix here has Pfaffian +/- scale^2.
    // 2^1200 exceeds f64's exponent range, though every operand is finite.
    let mut kernel = state();
    let pool = ThreadedPfaPackWorkspace::new(4, 1);
    let error = calc_m_all_fsz_real(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &mut kernel.slater_matrix,
        0,
        1,
        2,
        2,
        &pool,
    )
    .unwrap_err();
    assert_eq!(error, CalcMAllError::NonFinitePfaffian { qp: 0 });

    let data = data();
    let data_before = format!("{data:?}");
    let mut repeated_checkpoint = None;
    for trial in 0..2 {
        let mut state = state();
        let saved = state.electron_config.clone();
        let slater_bits = |state: &VmcOptimizationState| {
            state
                .slater_matrix
                .slater_elm
                .as_slice()
                .iter()
                .map(|z| (z.re.to_bits(), z.im.to_bits()))
                .collect::<Vec<_>>()
        };
        let slater_before = slater_bits(&state);
        let inverse_before = state.slater_matrix.inv_m.as_slice().to_vec();
        let pf_before = state.slater_matrix.pf_m.clone();
        let inverse_real_before = state.slater_matrix.inv_m_real.as_slice().to_vec();
        let pf_real_before = state.slater_matrix.pf_m_real.clone();
        let scratch_before = state.electron_config.tmp_ele_idx.clone();
        let mut rng = Sfmt19937Rng::new(1);
        rng.gen_rand32(); // Nonzero caller cursor/count, not a reseed repair.
        let before = rng.state_snapshot();
        let before_count = rng.words_consumed();
        let error = vmc_make_sample_fsz_real(&data, &mut state, &mut rng).unwrap_err();
        assert_eq!(error, CalcMAllError::NonFinitePfaffian { qp: 0 });
        assert_eq!(state.electron_config.ele_idx, saved.ele_idx);
        assert_eq!(state.electron_config.ele_cfg, saved.ele_cfg);
        assert_eq!(state.electron_config.ele_num, saved.ele_num);
        assert_eq!(state.electron_config.ele_proj_cnt, saved.ele_proj_cnt);
        assert_eq!(state.electron_config.ele_spn, saved.ele_spn);
        assert_eq!(state.electron_config.counter, saved.counter);
        assert_eq!(slater_bits(&state), slater_before);
        assert_eq!(format!("{data:?}"), data_before);
        // The 101 actual initializer attempts consume placement draws and may
        // publish tmp configuration and the real Slater copy, never saved
        // samples/counters. Failed factorization is staged privately and does
        // not publish inverse/Pfaffian results. The entire state/RNG is not unchanged.
        assert_ne!(state.electron_config.tmp_ele_idx, scratch_before);
        assert_eq!(state.electron_config.tmp_ele_num, vec![1; 4]);
        assert_eq!(state.electron_config.tmp_ele_spn, vec![0, 0, 1, 1]);
        assert_eq!(state.electron_config.tmp_ele_proj_cnt, vec![2]);
        assert_eq!(
            state.slater_matrix.inv_m.as_slice(),
            inverse_before,
            "failed factorization must not publish complex results"
        );
        assert_eq!(state.slater_matrix.pf_m, pf_before);
        assert_eq!(
            state.slater_matrix.inv_m_real.as_slice(),
            inverse_real_before
        );
        assert_eq!(state.slater_matrix.pf_m_real, pf_real_before);
        assert_eq!(
            state.slater_matrix.slater_elm_real.as_slice(),
            state
                .slater_matrix
                .slater_elm
                .as_slice()
                .iter()
                .map(|z| z.re)
                .collect::<Vec<_>>()
        );
        assert!(state
            .electron_config
            .tmp_ele_idx
            .iter()
            .all(|&site| (0..2).contains(&site)));
        assert!(rng.words_consumed() >= before_count + 101 * 4);
        let after = rng.state_snapshot();
        let after_count = rng.words_consumed();
        assert_ne!(after, before);
        let mut next624 = [0; 624];
        rng.dump_rand32(&mut next624);
        assert_eq!(
            rng.state_snapshot(),
            after,
            "peek must not advance raw state/cursor"
        );
        assert_eq!(rng.words_consumed(), after_count);
        let checkpoint = (after, after_count, next624);
        if let Some(previous) = &repeated_checkpoint {
            assert_eq!(
                &checkpoint, previous,
                "same input/seed failure is repeatable"
            );
        }
        repeated_checkpoint = Some(checkpoint);
        println!("ISSUE178_SERIAL trial={trial} error={error} before_count={before_count} after_count={after_count} before_cursor={} after_cursor={} scratch={:?}",
            before.1, after.1, state.electron_config.tmp_ele_idx);
        if std::env::var_os("MVMC_ISSUE178_SERIAL_DIAGNOSTIC").is_some() {
            println!(
                "ISSUE178_SERIAL before_raw={:?} after_raw={:?} next624={next624:?}",
                before.0, after.0
            );
        }
    }
}
