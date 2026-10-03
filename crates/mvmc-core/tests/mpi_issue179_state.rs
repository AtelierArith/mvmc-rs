//! Optional MPI state evidence worker; independent references run outside Cargo.
#![cfg(feature = "mpi")]
use mvmc_core::{mpi::MpiContext, Reducer};
use num_complex::Complex64;
use std::rc::Rc;
use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::Write,
    path::PathBuf,
};

type CgEvent = Vec<(String, Vec<f64>)>;

#[derive(Default)]
struct CgRecording {
    events: RefCell<Vec<CgEvent>>,
}
impl CgRecording {
    fn event(&self, kind: f64, fields: Vec<(String, Vec<f64>)>) {
        let mut event = vec![("kind".into(), vec![kind])];
        event.extend(fields);
        self.events.borrow_mut().push(event);
    }
}
impl mvmc_core::sr_cg::CgObserver for CgRecording {
    fn prepared(&self, mapping: &[usize], op: &mvmc_core::sr_cg::SampledSrOperator, g: &[f64]) {
        self.event(
            0.0,
            vec![
                (
                    "mapping".into(),
                    mapping.iter().map(|&i| i as f64).collect(),
                ),
                ("mean".into(), op.mean.clone()),
                ("diagonal".into(), op.diagonal.clone()),
                ("gradient".into(), g.to_vec()),
                ("real-samples".into(), op.real_samples.clone()),
                ("imag-samples".into(), op.imag_samples.clone()),
            ],
        );
    }
    fn product(&self, phase: mvmc_core::sr_cg::CgProductPhase, x: &[f64], z: &[f64]) {
        let phase = match phase {
            mvmc_core::sr_cg::CgProductPhase::RootSearch => 0.0,
            mvmc_core::sr_cg::CgProductPhase::Local => 1.0,
            mvmc_core::sr_cg::CgProductPhase::Global => 2.0,
            mvmc_core::sr_cg::CgProductPhase::Corrected => 3.0,
        };
        self.event(
            1.0,
            vec![
                ("phase".into(), vec![phase]),
                ("search".into(), x.to_vec()),
                ("product".into(), z.to_vec()),
            ],
        );
    }
    fn iteration(&self, view: mvmc_core::sr_cg::CgIterationView<'_>) {
        self.event(
            2.0,
            vec![
                ("iteration".into(), vec![view.iteration as f64]),
                ("solution".into(), view.solution.to_vec()),
                ("residual".into(), view.residual.to_vec()),
                ("direction".into(), view.direction.to_vec()),
                ("delta".into(), vec![view.delta]),
                ("alpha".into(), view.alpha.into_iter().collect()),
            ],
        );
    }
    fn finished(&self, result: &mvmc_core::sr_cg::CgSolution) {
        self.event(
            3.0,
            vec![
                ("iterations".into(), vec![result.iterations as f64]),
                ("solution".into(), result.solution.clone()),
                ("residual".into(), result.residual.clone()),
                ("direction".into(), result.direction.clone()),
            ],
        );
    }
}

fn parameters(data: &mvmc_core::ExpertModeData) -> Vec<Complex64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

struct Recording<'a> {
    inner: &'a dyn Reducer,
    file: RefCell<File>,
    complex_call: RefCell<usize>,
    real_call: RefCell<usize>,
    sr_complex: bool,
    step: Cell<usize>,
    initiating_thread: std::thread::ThreadId,
    collective_calls: Cell<usize>,
}
impl Recording<'_> {
    fn assert_funneled(&self) {
        assert_eq!(std::thread::current().id(), self.initiating_thread);
        assert!(
            rayon::current_thread_index().is_none(),
            "MPI entered from Rayon pool"
        );
        self.collective_calls.set(self.collective_calls.get() + 1);
    }
    fn step_key(&self, key: &str) -> String {
        if self.step.get() == 0 {
            key.to_string()
        } else {
            format!("step-{}-{key}", self.step.get())
        }
    }
    fn complex(&self, key: &str, buf: &[Complex64]) {
        if buf.is_empty() {
            return;
        }
        let mut f = self.file.borrow_mut();
        write!(f, "n:{key}").unwrap();
        for x in buf {
            write!(f, " {:.17e} {:.17e}", x.re, x.im).unwrap();
        }
        writeln!(f).unwrap();
    }
    fn real(&self, key: &str, buf: &[f64]) {
        if buf.is_empty() {
            return;
        }
        let mut f = self.file.borrow_mut();
        write!(f, "n:{key}").unwrap();
        for x in buf {
            write!(f, " {x:.17e}").unwrap();
        }
        writeln!(f).unwrap();
    }
    fn discrete(&self, key: &str, buf: &[i64]) {
        if buf.is_empty() {
            return;
        }
        let mut f = self.file.borrow_mut();
        write!(f, "d:{key}").unwrap();
        for x in buf {
            write!(f, " {x}").unwrap();
        }
        writeln!(f).unwrap();
    }
}
impl Reducer for Recording<'_> {
    fn sampling_any_failure(&self, failed: bool) -> bool {
        self.assert_funneled();
        self.inner.sampling_any_failure(failed)
    }
    fn sampling_qp_range(&self, length: usize) -> std::ops::Range<usize> {
        self.inner.sampling_qp_range(length)
    }
    fn sampling_sum_f64(&self, buf: &mut [f64]) {
        self.assert_funneled();
        self.inner.sampling_sum_f64(buf);
    }
    fn sampling_sum_c64(&self, buf: &mut [Complex64]) {
        self.assert_funneled();
        self.inner.sampling_sum_c64(buf);
    }
    fn reduce_counters(&self, buf: &mut [i64]) {
        self.assert_funneled();
        self.discrete(&self.step_key("local-counter"), buf);
        self.inner.reduce_counters(buf);
        self.discrete(&self.step_key("reduced-counter"), buf);
    }
    fn allreduce_sum_c64(&self, buf: &mut [Complex64]) {
        self.assert_funneled();
        let mut call = self.complex_call.borrow_mut();
        let key = match *call {
            0 => "energy",
            1 => "oo",
            2 => "ho",
            _ => panic!("unexpected complex accumulator collective {}", *call),
        };
        self.complex(&self.step_key(&format!("local-{key}")), buf);
        self.inner.allreduce_sum_c64(buf);
        self.complex(&self.step_key(&format!("reduced-{key}")), buf);
        *call += 1;
    }
    fn allreduce_sum_f64(&self, buf: &mut [f64]) {
        self.assert_funneled();
        let mut call = self.real_call.borrow_mut();
        let key = if !self.sr_complex && *call < 2 {
            if *call == 0 { "oo-real" } else { "ho-real" }.to_string()
        } else {
            let product = *call - if self.sr_complex { 0 } else { 2 };
            format!("collective-cg-product-{product:06}")
        };
        self.real(&self.step_key(&format!("local-{key}")), buf);
        self.inner.allreduce_sum_f64(buf);
        self.real(&self.step_key(&format!("reduced-{key}")), buf);
        *call += 1;
    }
    fn allreduce_sum_i64(&self, buf: &mut [i64]) {
        self.assert_funneled();
        if buf.len() == 10 {
            self.discrete(&self.step_key("local-counter"), buf);
        }
        self.inner.allreduce_sum_i64(buf);
        if buf.len() == 10 {
            self.discrete(&self.step_key("reduced-counter"), buf);
        }
    }
    fn broadcast_i64(&self, root: usize, buf: &mut [i64]) -> Result<(), String> {
        self.assert_funneled();
        self.inner.broadcast_i64(root, buf)
    }
    fn broadcast_f64(&self, root: usize, buf: &mut [f64]) -> Result<(), String> {
        self.assert_funneled();
        self.inner.broadcast_f64(root, buf)
    }
    fn barrier(&self) {
        self.assert_funneled();
        self.inner.barrier();
    }
    fn broadcast_c64(&self, root: usize, buf: &mut [Complex64]) {
        self.assert_funneled();
        self.inner.broadcast_c64(root, buf);
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn seed_offset(&self) -> usize {
        self.inner.seed_offset()
    }
    fn reduction_size(&self) -> usize {
        self.inner.reduction_size()
    }
    fn supports_grouped_sampling(&self) -> bool {
        self.inner.supports_grouped_sampling()
    }
    fn is_output_root(&self) -> bool {
        self.inner.is_output_root()
    }
    fn any_failure(&self, failed: bool) -> bool {
        self.assert_funneled();
        self.inner.any_failure(failed)
    }
}

#[test]
#[ignore = "MPI179_INPUT and MPI179_STATE_DIR required; mpirun -n 2/4"]
fn issue179_state() {
    use mvmc_expert_parsers::utils::{
        parameter_init::init_parameter, qp_weight::init_qp_weight,
        read_input_parameters::read_input_parameters,
    };
    let world = MpiContext::initialize().unwrap();
    assert!(
        matches!(world.world_size(), 2 | 4),
        "real 2/4-rank MPI world required; launcher/library may be incompatible"
    );
    let input = PathBuf::from(std::env::var("MPI179_INPUT").unwrap());
    let dir = PathBuf::from(std::env::var("MPI179_STATE_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let enable_opt_trans = std::fs::read_to_string(&input)
        .unwrap()
        .lines()
        .any(|line| line.split_whitespace().next() == Some("OptTrans"));
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, enable_opt_trans)
            .unwrap();
    assert!(data.inter_all_terms.is_empty(), "InterAll excluded");
    assert!(
        data.modpara.rnd_seed >= 0,
        "bounded deterministic seed required"
    );
    // Match the CLI's communicator choice exactly, including NSplitSize=1.
    // Do not repair runner group arithmetic inside the evidence harness.
    let group = (data.modpara.nsplit_size > 1).then(|| {
        world
            .split_groups(data.modpara.nsplit_size as usize)
            .unwrap()
    });
    let inner: &dyn Reducer = group.as_ref().map_or(&world as &dyn Reducer, |g| g);
    let recording = Recording {
        inner,
        file: RefCell::new(File::create(dir.join(format!("rank-{}.txt", world.rank()))).unwrap()),
        complex_call: RefCell::new(0),
        real_call: RefCell::new(0),
        sr_complex: mvmc_core::get_all_complex_flag(&data),
        step: Cell::new(0),
        initiating_thread: std::thread::current().id(),
        collective_calls: Cell::new(0),
    };
    let seed = data.modpara.rnd_seed + recording.seed_offset() as i64;
    recording.discrete("seed", &[seed]);
    recording.discrete(
        "group",
        &[
            recording.seed_offset() as i64,
            if group.is_some() {
                recording.rank() as i64
            } else {
                0
            },
            if group.is_some() {
                recording.world_size() as i64
            } else {
                1
            },
        ],
    );
    recording.discrete("configured-samples", &[data.modpara.nvmc_sample]);
    recording.discrete("requested-width", &[data.modpara.nsplit_size]);
    let mut rng = sfmt19937::Sfmt19937Rng::new(seed as u32);
    let (seeded_words, seeded_cursor) = rng.state_snapshot();
    recording.discrete("before-init-raw-rng", &seeded_words.map(i64::from));
    recording.discrete("before-init-rng-cursor", &[seeded_cursor as i64]);
    init_parameter(&mut data, &mut rng);
    let initial = input.parent().unwrap().join("initial.def");
    if initial.is_file() && !enable_opt_trans {
        assert!(mvmc_core::read_initial_def(&mut data, initial).unwrap());
    }
    read_input_parameters(&mut data, &input).unwrap();
    // Match the public runner's C comm0 parameter broadcast and local gauge fix.
    mvmc_core::sync::sync_modified_parameter(&mut data, &recording);
    init_qp_weight(&mut data);
    recording.discrete(
        "initial-draw-count",
        &[i64::try_from(rng.words_consumed()).unwrap()],
    );
    let mut initial_rng = rng.clone();
    recording.discrete(
        "initial-rng",
        &(0..624)
            .map(|_| i64::from(initial_rng.gen_rand32()))
            .collect::<Vec<_>>(),
    );
    recording.complex("initial-parameters", &parameters(&data));
    let (initial_state, initial_cursor) = rng.state_snapshot();
    recording.discrete("initial-raw-rng", &initial_state.map(i64::from));
    recording.discrete("initial-rng-cursor", &[initial_cursor as i64]);
    let steps = std::env::var("MPI179_STEPS")
        .unwrap_or_else(|_| "1".into())
        .parse::<i64>()
        .unwrap();
    assert!(
        matches!(steps, 1..=3 | 20),
        "prefix1/2/3 or long baseline20 required"
    );
    data.modpara.nsr_opt_itr_step = steps;
    data.modpara.nsr_opt_itr_smp = steps;
    recording.discrete("steps", &[steps]);
    let mut state = mvmc_core::VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        data.modpara.nsp_gauss_leg.max(1) as usize
            * data.modpara.nmp_trans.unsigned_abs() as usize
            * data.n_qp_opt_trans.max(1) as usize,
        data.modpara.nvmc_sample as usize,
        mvmc_core::get_all_complex_flag(&data),
        data.i_flg_orbital_general != 0,
    );
    let failure_rank = std::env::var("MPI179_FAIL_RANK")
        .ok()
        .map(|s| s.parse::<usize>().unwrap());
    if let Some(rank) = failure_rank {
        assert!(rank < world.world_size());
        recording.discrete("callback-injected-rank", &[rank as i64]);
    }
    let callback_calls = Cell::new(0_i64);
    let callback_errors = Cell::new(0_i64);
    let last_synchronized_parameters = RefCell::new(parameters(&data));
    let mut callback = |step, data: &mut mvmc_core::ExpertModeData, _, _| {
        callback_calls.set(callback_calls.get() + 1);
        recording.complex(&format!("sr-step-{step}"), &parameters(data));
        *last_synchronized_parameters.borrow_mut() = parameters(data);
        recording.step.set(step + 1);
        *recording.complex_call.borrow_mut() = 0;
        *recording.real_call.borrow_mut() = 0;
        if failure_rank == Some(world.rank()) {
            callback_errors.set(callback_errors.get() + 1);
            Err(format!(
                "issue179 injected rank {} callback failure",
                world.rank()
            ))
        } else {
            Ok(())
        }
    };
    let worker_observer = mvmc_core::threading::start_observation();
    let cg_recording = Rc::new(CgRecording::default());
    let cg_guard = mvmc_core::sr_cg::install_cg_observer(cg_recording.clone()).unwrap();
    let sr_observer = mvmc_core::sr::observer::capture().unwrap();
    mvmc_core::sampling::driver::trace::start();
    let result = mvmc_core::vmc_para_opt(
        &mut data,
        &mut state,
        &mut rng,
        Some(&dir),
        &recording,
        mvmc_core::OptimizationOptions {
            callback: Some(&mut callback),
            skip_sr: false,
        },
    );
    let trace = mvmc_core::sampling::driver::trace::finish();
    if result
        .as_ref()
        .is_err_and(|reason| reason.contains("direct SR failed"))
    {
        let bits = |values: &[Complex64]| {
            values
                .iter()
                .map(|z| (z.re.to_bits(), z.im.to_bits()))
                .collect::<Vec<_>>()
        };
        recording.complex(
            "failed-last-synchronized-parameters",
            &last_synchronized_parameters.borrow(),
        );
        recording.complex("failed-return-parameters", &parameters(&data));
        assert_eq!(
            bits(&parameters(&data)),
            bits(&last_synchronized_parameters.borrow())
        );
    }
    drop(cg_guard);
    let mut cg_file = File::create(dir.join(format!("cg-rank-{}.txt", world.rank()))).unwrap();
    writeln!(cg_file, "d:events {}", cg_recording.events.borrow().len()).unwrap();
    for (index, event) in cg_recording.events.borrow().iter().enumerate() {
        for (name, values) in event {
            if values.is_empty() {
                continue;
            }
            let discrete = matches!(
                name.as_str(),
                "kind" | "phase" | "mapping" | "iteration" | "iterations"
            );
            write!(
                cg_file,
                "{}:event-{index:06}-{name}",
                if discrete { "d" } else { "n" }
            )
            .unwrap();
            for value in values {
                if discrete {
                    write!(cg_file, " {}", *value as i64).unwrap();
                } else {
                    write!(cg_file, " {value:.17e}").unwrap();
                }
            }
            writeln!(cg_file).unwrap();
        }
    }
    if let Some(rank) = failure_rank {
        recording.discrete("callback-calls", &[callback_calls.get()]);
        recording.discrete("callback-local-errors", &[callback_errors.get()]);
        assert_eq!(callback_calls.get(), 1);
        assert_eq!(callback_errors.get(), i64::from(world.rank() == rank));
        let reason = result.as_ref().unwrap_err();
        assert!(reason.contains("callback"));
        if world.rank() == rank {
            assert!(reason.contains(&format!("issue179 injected rank {rank} callback failure")));
        }
        std::fs::write(
            dir.join(format!("callback-result-rank-{}.txt", world.rank())),
            reason,
        )
        .unwrap();
    }
    let worker_execution = worker_observer.finish();
    let sr_systems = sr_observer.finish();
    recording.discrete("sr-kind", &[data.modpara.nsrcg]);
    recording.discrete("sr-systems", &[sr_systems.len() as i64]);
    for (index, system) in sr_systems.iter().enumerate() {
        let key = format!("sr-system-{index:06}");
        recording.discrete(&format!("{key}-dimension"), &[system.dimension as i64]);
        recording.discrete(&format!("{key}-triangle"), &[system.triangle as i64]);
        // Direct SR solves a single RHS. Presence flags distinguish an absent
        // POTRS invocation from a successful invocation returning INFO=0.
        recording.discrete(&format!("{key}-nrhs"), &[1]);
        recording.discrete(
            &format!("{key}-factor-info"),
            &system
                .factor_info
                .map(i64::from)
                .into_iter()
                .collect::<Vec<_>>(),
        );
        recording.discrete(
            &format!("{key}-solve-info"),
            &system
                .solve_info
                .map(i64::from)
                .into_iter()
                .collect::<Vec<_>>(),
        );
        recording.discrete(
            &format!("{key}-not-solved"),
            &[match system.not_solved {
                None => 0,
                Some(mvmc_core::sr::observer::NotSolvedReason::NoParameters) => 1,
                Some(mvmc_core::sr::observer::NotSolvedReason::NoActiveComponents) => 2,
            }],
        );
        recording.discrete(
            &format!("{key}-status"),
            &[i64::from(system.status.unwrap())],
        );
        recording.discrete(
            &format!("{key}-active"),
            &system
                .active_indices
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>(),
        );
        recording.discrete(&format!("{key}-flags"), &system.flags);
        recording.discrete(
            &format!("{key}-settings"),
            &[
                system.settings.input_seed,
                system.settings.steps,
                system.settings.window,
                system.settings.nsrcg,
                system.settings.nstore,
            ],
        );
        recording.real(
            &format!("{key}-regularization"),
            &[
                system.settings.diagonal_shift,
                system.settings.redundant_cut,
                system.settings.step_dt,
            ],
        );
        recording.real(&format!("{key}-matrix"), &system.matrix);
        recording.real(&format!("{key}-rhs"), &system.rhs);
        recording.real(&format!("{key}-increment"), &system.increment);
    }
    recording.discrete(
        "total-draw-count",
        &[i64::try_from(rng.words_consumed()).unwrap()],
    );
    recording.discrete(
        "sampling-draw-count",
        &[trace.iter().filter(|event| event[0] == 9).count() as i64],
    );
    recording.discrete("trace-events", &[trace.len() as i64]);
    recording.discrete(
        "acceptance-events",
        &[
            trace
                .iter()
                .filter(|event| event[0] == 7 && event[1] == 1)
                .count() as i64,
            trace
                .iter()
                .filter(|event| event[0] == 7 && event[1] == 0)
                .count() as i64,
        ],
    );
    for (index, event) in trace.iter().enumerate() {
        recording.discrete(&format!("trace-{index:06}"), event);
    }
    // Separate provenance: worker settings deliberately differ across runs,
    // so they must not be mixed into exact trajectory comparison records.
    // Probe capacity after execution; this does not assert a kernel used it.
    let config = mvmc_core::threading::inner_thread_config();
    let pool_threads = mvmc_core::threading::install(rayon::current_num_threads);
    let qp_work = state.slater_matrix.pf_m.len();
    let mut worker_file =
        File::create(dir.join(format!("workers-rank-{}.txt", world.rank()))).unwrap();
    writeln!(
        worker_file,
        "requested={} configured={} threshold={} observed_pool={} qp_work={} effective_qp_workers={} capacity_probe_after_run=1",
        std::env::var("MVMC_RS_INNER_THREADS").unwrap_or_else(|_| "1".into()),
        config.threads,
        config.threshold,
        pool_threads,
        qp_work,
        mvmc_core::threading::inner_worker_count(qp_work),
    )
    .unwrap();
    writeln!(worker_file,
        "kernel_parallel_calls={} kernel_serial_calls={} kernel_qp_items={} kernel_parallel_qp_items={} kernel_serial_qp_items={} kernel_term_items={} kernel_parallel_term_items={} kernel_serial_term_items={} kernel_worker_entries={} kernel_workers_seen={} kernel_worker_ids={:?}",
        worker_execution.parallel_calls, worker_execution.serial_calls,
        worker_execution.executed_qp_items, worker_execution.parallel_qp_items, worker_execution.serial_qp_items,
        worker_execution.executed_term_items, worker_execution.parallel_term_items, worker_execution.serial_term_items,
        worker_execution.worker_entries, worker_execution.distinct_workers, worker_execution.worker_ids,
    ).unwrap();
    recording.discrete("status", &[i64::from(result.is_err())]);
    recording.discrete(
        "main-thread-collective-calls",
        &[recording.collective_calls.get() as i64],
    );
    recording.discrete("chain-samples", &[data.modpara.nvmc_sample]);
    recording.discrete("ele_idx", &state.electron_config.ele_idx);
    recording.discrete("ele_cfg", &state.electron_config.ele_cfg);
    recording.discrete("ele_num", &state.electron_config.ele_num);
    recording.discrete("ele_proj_cnt", &state.electron_config.ele_proj_cnt);
    recording.discrete("ele_spn", &state.electron_config.ele_spn);
    recording.discrete("counter", &state.electron_config.counter);
    recording.complex("parameters", &parameters(&data));
    recording.complex(
        "energy",
        &[
            state.energy.wc,
            state.energy.etot,
            state.energy.etot2,
            state.energy.sztot,
            state.energy.sztot2,
        ],
    );
    let mut final_words = [0; 624];
    let (raw_state, cursor) = rng.state_snapshot();
    recording.discrete("raw-rng", &raw_state.map(i64::from));
    recording.discrete("rng-cursor", &[cursor as i64]);
    rng.dump_rand32(&mut final_words);
    assert_eq!(rng.state_snapshot(), (raw_state, cursor));
    recording.discrete("rng", &final_words.map(i64::from));
    if failure_rank.is_some() {
        assert!(result.unwrap_err().contains("callback"));
    } else {
        result.unwrap();
    }
}
