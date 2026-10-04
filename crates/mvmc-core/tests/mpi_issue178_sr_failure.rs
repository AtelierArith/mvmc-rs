//! Genuine finite-input direct-SR failure; no fake reducer or fabricated moments.
//! C stcopt_dposv.c multiplies covariance diagonals by 1 + DSROptStaDel.
//! A -2 shift is a defined error control, not recommended physical regularization.
use mvmc_core::{
    read_opt_para_file, vmc_para_opt, ExpertModeData, OptimizationOptions, Reducer,
    SingleProcessReducer, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

fn owned() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    loop {
        let p = std::env::temp_dir().join(format!(
            "issue178-sr-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&p) {
            Ok(()) => return p,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("{e}"),
        }
    }
}
fn parameters(data: &ExpertModeData) -> Vec<Complex64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}
fn loaded(width: usize, shift: f64) -> ExpertModeData {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files(root.join("inputs/namelist.def")).unwrap();
    read_opt_para_file(&mut data, root.join("zqp_opt.dat")).unwrap();
    data.namelist.retain(|(kind, _)| kind != "TwoBodyGEx");
    data.green_two_ex_terms.clear();
    data.modpara.vmc_calc_mode = 0;
    data.modpara.nsplit_size = width as i64;
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
    data.modpara.nvmc_warmup = 1;
    data.modpara.nvmc_sample = 8;
    data.modpara.nvmc_interval = 1;
    data.modpara.nsrcg = 0;
    data.modpara.dsr_opt_sta_del = shift;
    data.ensure_optimization_flags(data.count_variational_parameters());
    // The low-level OPT runner refreshes existing QP weights, but does not
    // initialize them. Complete the public caller-owned preparation explicitly.
    mvmc_core::qp::init_qp_weight(&mut data);
    assert!(data.qp_weights.is_some());
    data
}
fn checkpoint(state: &VmcOptimizationState, rng: &Sfmt19937Rng) -> String {
    let raw = rng.state_snapshot();
    let count = rng.words_consumed();
    let mut next = [0; 624];
    rng.dump_rand32(&mut next);
    assert_eq!(rng.state_snapshot(), raw);
    assert_eq!(rng.words_consumed(), count);
    format!(
        "raw={:?} cursor={} count={count} next624={next:?} config={:?}",
        raw.0, raw.1, state.electron_config
    )
}

fn exercise(
    out: &Path,
    reducer: &impl Reducer,
    width: usize,
    faulty: bool,
    collective_failure: bool,
    rank: usize,
) -> String {
    let mut data = loaded(width, if faulty { -2.0 } else { 0.02 });
    assert!(!mvmc_core::run::get_all_complex_flag(&data).unwrap());
    // Own same-call fixed-parameter anchor: this low-level runner does not
    // initialize/load parameters before SR. The solve observer captures flags,
    // matrix and statuses, but does not expose a pre-SR parameter snapshot.
    let fixed = parameters(&data);
    let flags = data.optimization_flags.clone();
    let mut state = VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        data.modpara.nsp_gauss_leg.max(1) as usize * data.modpara.nmp_trans.unsigned_abs() as usize,
        data.modpara.nvmc_sample as usize,
        false,
        false,
    );
    let mut rng = Sfmt19937Rng::new(11272 + reducer.seed_offset() as u32);
    if reducer.is_output_root() {
        fs::create_dir(out).unwrap();
    }
    let mut callbacks = 0;
    let mut callback = |_, _: &mut ExpertModeData, _: Complex64, status| {
        callbacks += 1;
        assert_eq!(status, 0);
        Ok(())
    };
    let capture = mvmc_core::sr::observer::capture().unwrap();
    let result = vmc_para_opt(
        &mut data,
        &mut state,
        &mut rng,
        Some(out),
        reducer,
        OptimizationOptions {
            callback: Some(&mut callback),
            skip_sr: false,
        },
    );
    let records = capture.finish();
    println!("ISSUE178_SR_RETURN rank={rank} width={width} faulty={faulty} result={result:?} callbacks={callbacks}");
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert!(
        record.dimension > 0,
        "healthy active covariance is mandatory"
    );
    assert!(record.not_solved.is_none());
    assert_eq!(record.flags, flags);
    assert!(record.matrix.iter().all(|value| value.is_finite()));
    let diagonal = record.matrix[0];
    if faulty {
        assert!(
            diagonal < 0.0,
            "actual negative active variance: {diagonal}"
        );
        assert!(record.factor_info.is_some_and(|info| info > 0));
        assert_eq!(
            record.solve_info, None,
            "POTRS must not follow failed POTRF"
        );
        assert_eq!(record.status, Some(1));
    } else {
        assert!(
            diagonal > 0.0,
            "genuine healthy active covariance: {diagonal}"
        );
        assert_eq!(record.factor_info, Some(0));
        assert_eq!(record.solve_info, Some(0));
        assert_eq!(record.status, Some(0));
    }
    if collective_failure {
        assert!(result.unwrap_err().contains("direct SR failed at step 0"));
        assert_eq!(callbacks, 0);
        assert_eq!(
            parameters(&data),
            fixed,
            "successful peers must roll back too"
        );
        assert_eq!(data.optimization_flags, flags);
        assert!(state.opt_data.is_empty());
        assert!(!out.join("zqp_opt.dat").exists());
    } else {
        result.unwrap();
        assert_eq!(callbacks, 1);
    }
    if reducer.is_output_root() {
        let text = fs::read_to_string(out.join("zvo_out.dat")).unwrap();
        assert_eq!(text.lines().count(), 1, "measurement output precedes SR");
        assert_eq!(out.join("zqp_opt.dat").exists(), !collective_failure);
    } else {
        assert!(!out.exists());
    }
    assert!(rng.words_consumed() > 0);
    let retained = checkpoint(&state, &rng);
    println!("ISSUE178_SR_CHECK rank={rank} width={width} faulty={faulty} collective_failure={collective_failure} diagonal={diagonal:?} factor={:?} solve={:?} callbacks={callbacks} {retained}", record.factor_info, record.solve_info);
    retained
}

#[test]
fn finite_negative_regularizer_fails_actual_public_sr_after_healthy_control() {
    let root = owned();
    exercise(
        &root.join("healthy"),
        &SingleProcessReducer,
        1,
        false,
        false,
        0,
    );
    let first = exercise(
        &root.join("failure0"),
        &SingleProcessReducer,
        1,
        true,
        true,
        0,
    );
    let second = exercise(
        &root.join("failure1"),
        &SingleProcessReducer,
        1,
        true,
        true,
        0,
    );
    assert_eq!(first, second, "same-input fixed-seed actual repeat");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "explicit native MPI worlds2/4; external bounded launcher required"]
fn actual_rank_local_nonpd_sr_restores_successful_peers_before_callback() {
    // Diagnostic only: allowlisted startup metadata, never the whole environment.
    for name in [
        "PMI_RANK",
        "PMI_SIZE",
        "PMI_FD",
        "PMI_VERSION",
        "PMI_SUBVERSION",
        "PMIX_RANK",
    ] {
        let value = std::env::var(name).ok();
        let safe = value
            .as_deref()
            .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()));
        eprintln!(
            "ISSUE234_MPI_DIAG env={name} value={}",
            safe.unwrap_or(if value.is_some() {
                "NONNUMERIC_REDACTED"
            } else {
                "UNSET"
            })
        );
    }
    for name in [
        "MPIR_CVAR_PMI_VERSION",
        "MPICH_PMI_VERSION",
        "MPIR_PARAM_PMI_VERSION",
    ] {
        let value = std::env::var(name).ok();
        let safe = value
            .as_deref()
            .filter(|value| matches!(*value, "1" | "2" | "x"));
        eprintln!(
            "ISSUE234_MPI_DIAG env={name} value={}",
            safe.unwrap_or(if value.is_some() {
                "OTHER_REDACTED"
            } else {
                "UNSET"
            })
        );
    }
    for name in [
        "PMI_PORT",
        "PMIX_NAMESPACE",
        "PMIX_SERVER_URI",
        "PMIX_SERVER_URI2",
        "PMIX_SERVER_URI21",
    ] {
        eprintln!(
            "ISSUE234_MPI_DIAG env={name} present={}",
            std::env::var_os(name).is_some()
        );
    }
    let fd_kind = std::env::var("PMI_FD")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(|fd| match fs::read_link(format!("/proc/self/fd/{fd}")) {
            Ok(link) if link.to_string_lossy().starts_with("socket:") => "socket",
            Ok(link) if link.to_string_lossy().starts_with("pipe:") => "pipe",
            Ok(_) => "other",
            Err(_) => "unavailable",
        })
        .unwrap_or("unset_or_invalid");
    eprintln!(
        "ISSUE234_MPI_DIAG before_init pid={} thread={:?} pmi_fd_kind={fd_kind}",
        std::process::id(),
        std::thread::current().name()
    );
    let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
    eprintln!(
        "ISSUE234_MPI_DIAG after_init rank={} world_size={} threading={:?}",
        world.rank(),
        world.world_size(),
        mpi::environment::threading_support()
    );
    assert!([2, 4].contains(&world.world_size()));
    let root = PathBuf::from(std::env::var_os("MPI_ISSUE178_SR_OUTPUT").unwrap());
    let setup = if world.is_root() {
        fs::create_dir(&root)
    } else {
        Ok(())
    };
    assert!(!world.any_failure(setup.is_err()), "{setup:?}");
    let local = root.join(format!("rank-{}", world.rank()));
    fs::create_dir(&local).unwrap();
    for width in [1, 2] {
        let group = world.split_groups(width).unwrap();
        exercise(
            &local.join(format!("healthy-{width}")),
            &group,
            width,
            false,
            false,
            world.rank(),
        );
        world.barrier();
        for bad_rank in [0, world.world_size() - 1] {
            let mut first = None;
            for trial in 0..2 {
                let retained = exercise(
                    &local.join(format!("failure-{width}-{bad_rank}-{trial}")),
                    &group,
                    width,
                    world.rank() == bad_rank,
                    true,
                    world.rank(),
                );
                if trial == 0 {
                    first = Some(retained);
                } else {
                    assert_eq!(first.as_ref(), Some(&retained));
                }
                println!("ISSUE178_SR_REPEAT rank={} world={} width={width} bad_rank={bad_rank} trial={trial} exact=true", world.rank(), world.world_size());
                world.barrier();
            }
        }
    }
    println!(
        "ISSUE178_SR_DONE rank={} world={}",
        world.rank(),
        world.world_size()
    );
}
