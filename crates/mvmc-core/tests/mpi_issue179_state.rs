//! Optional MPI state evidence worker; independent references run outside Cargo.
#![cfg(feature = "mpi")]
use mvmc_core::{mpi::MpiContext, Reducer};
use num_complex::Complex64;
use std::{cell::RefCell, fs::File, io::Write, path::PathBuf};

struct Recording<'a> {
    inner: &'a dyn Reducer,
    file: RefCell<File>,
    complex_call: RefCell<usize>,
    real_call: RefCell<usize>,
}
impl Recording<'_> {
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
    fn allreduce_sum_c64(&self, buf: &mut [Complex64]) {
        let mut call = self.complex_call.borrow_mut();
        let key = match *call {
            0 => "energy",
            1 => "oo",
            2 => "ho",
            _ => "extra",
        };
        self.complex(&format!("local-{key}"), buf);
        self.inner.allreduce_sum_c64(buf);
        self.complex(&format!("reduced-{key}"), buf);
        *call += 1;
    }
    fn allreduce_sum_f64(&self, buf: &mut [f64]) {
        let mut call = self.real_call.borrow_mut();
        let key = if *call == 0 { "oo-real" } else { "ho-real" };
        self.real(&format!("local-{key}"), buf);
        self.inner.allreduce_sum_f64(buf);
        self.real(&format!("reduced-{key}"), buf);
        *call += 1;
    }
    fn allreduce_sum_i64(&self, buf: &mut [i64]) {
        self.inner.allreduce_sum_i64(buf);
    }
    fn broadcast_i64(&self, root: usize, buf: &mut [i64]) -> Result<(), String> {
        self.inner.broadcast_i64(root, buf)
    }
    fn broadcast_c64(&self, root: usize, buf: &mut [Complex64]) {
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
    let input = PathBuf::from(std::env::var("MPI179_INPUT").unwrap());
    let dir = PathBuf::from(std::env::var("MPI179_STATE_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(&input).unwrap();
    assert!(data.inter_all_terms.is_empty(), "InterAll excluded");
    assert!(
        data.modpara.rnd_seed >= 0,
        "bounded deterministic seed required"
    );
    let group = world
        .split_groups(data.modpara.nsplit_size as usize)
        .unwrap();
    let recording = Recording {
        inner: &group,
        file: RefCell::new(File::create(dir.join(format!("rank-{}.txt", world.rank()))).unwrap()),
        complex_call: RefCell::new(0),
        real_call: RefCell::new(0),
    };
    let seed = data.modpara.rnd_seed + recording.seed_offset() as i64;
    recording.discrete("seed", &[seed]);
    let mut rng = sfmt19937::Sfmt19937Rng::new(seed as u32);
    init_parameter(&mut data, &mut rng);
    let initial = input.parent().unwrap().join("initial.def");
    if initial.is_file() {
        assert!(mvmc_core::read_initial_def(&mut data, initial).unwrap());
    }
    read_input_parameters(&mut data, &input).unwrap();
    mvmc_core::sync::sync_modified_parameter(&mut data, &recording);
    init_qp_weight(&mut data);
    let mut initial_rng = rng.clone();
    recording.discrete(
        "initial-rng",
        &(0..624)
            .map(|_| i64::from(initial_rng.gen_rand32()))
            .collect::<Vec<_>>(),
    );
    recording.complex(
        "initial-parameters",
        &mvmc_core::sync::pack_variational_parameters(&data),
    );
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
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
    let mut callback = |_, _: &mut mvmc_core::ExpertModeData, _, _| {
        if failure_rank == Some(world.rank()) {
            Err("issue179 injected rank callback failure".to_string())
        } else {
            Ok(())
        }
    };
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
    recording.discrete("status", &[i64::from(result.is_err())]);
    recording.discrete("ele_idx", &state.electron_config.ele_idx);
    recording.discrete("ele_cfg", &state.electron_config.ele_cfg);
    recording.discrete("ele_num", &state.electron_config.ele_num);
    recording.discrete("ele_proj_cnt", &state.electron_config.ele_proj_cnt);
    recording.discrete("ele_spn", &state.electron_config.ele_spn);
    recording.discrete("counter", &state.electron_config.counter);
    recording.complex(
        "parameters",
        &mvmc_core::sync::pack_variational_parameters(&data),
    );
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
    recording.discrete(
        "rng",
        &(0..624)
            .map(|_| i64::from(rng.gen_rand32()))
            .collect::<Vec<_>>(),
    );
    if failure_rank.is_some() {
        assert!(result.unwrap_err().contains("callback"));
    } else {
        result.unwrap();
    }
}
