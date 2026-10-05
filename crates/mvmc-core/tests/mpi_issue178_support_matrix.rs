//! Public supported/rejected combinations and actual callback/filesystem errors.
//! Only the grouped combination that is undefined in mVMC C (SR-CG, #349) is
//! rejected; FSZ, OptTrans and Lanczos grouped runs are accepted as in C.
use mvmc_core::{
    read_opt_para_file, vmc_para_opt, vmc_phys_cal_in_place, ExpertModeData, OptimizationOptions,
    Reducer, SingleProcessReducer, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
#[cfg(feature = "mpi")]
use std::collections::BTreeMap;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

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
#[cfg(feature = "mpi")]
fn repeated(records: &mut BTreeMap<String, String>, key: String, value: String, trial: usize) {
    if trial == 0 {
        assert!(records.insert(key, value).is_none());
    } else {
        assert_eq!(
            records.get(&key),
            Some(&value),
            "same-input rank-local repeat: {key}"
        );
    }
}

fn loaded(model: &str, phys: bool) -> ExpertModeData {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(model);
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files(root.join("inputs/namelist.def")).unwrap();
    read_opt_para_file(&mut data, root.join("zqp_opt.dat")).unwrap();
    if !phys {
        data.namelist.retain(|(kind, _)| kind != "TwoBodyGEx");
        data.green_two_ex_terms.clear();
    }
    data.modpara.vmc_calc_mode = i64::from(phys);
    data.modpara.nsr_opt_itr_step = 1;
    data.modpara.nsr_opt_itr_smp = 1;
    data.modpara.nvmc_warmup = 1;
    data.modpara.nvmc_sample = 3;
    data.modpara.nvmc_interval = 1;
    data.modpara.n_data_qty_smp = 1;
    if !phys {
        // The low-level ParaOpt runner refreshes existing QP weights but does
        // not create them (PhysCal does); complete the caller-owned preparation.
        mvmc_core::qp::init_qp_weight(&mut data);
    }
    data
}
fn owned() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    loop {
        let path = std::env::temp_dir().join(format!(
            "issue178-matrix-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&path) {
            Ok(()) => return path,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("{e}"),
        }
    }
}
fn state(data: &ExpertModeData) -> VmcOptimizationState {
    VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        data.modpara.nsp_gauss_leg.max(1) as usize
            * data.modpara.nmp_trans.unsigned_abs() as usize
            * data.n_qp_opt_trans.max(1) as usize,
        data.modpara.nvmc_sample as usize,
        mvmc_core::run::get_all_complex_flag(data).unwrap(),
        data.i_flg_orbital_general != 0,
    )
}
fn call(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    out: &Path,
    phys: bool,
    reducer: &impl Reducer,
) -> Result<(), String> {
    if phys {
        vmc_phys_cal_in_place(data, state, rng, Some(out), reducer, None).map(|_| ())
    } else {
        vmc_para_opt(
            data,
            state,
            rng,
            Some(out),
            reducer,
            OptimizationOptions {
                skip_sr: true,
                file_flush_interval: None,
                callback: None,
            },
        )
    }
}
fn rejected_case(case: &str, phys: bool, width: usize) -> (ExpertModeData, &'static str) {
    let model = "heisenberg_chain_real";
    let mut data = loaded(model, phys);
    data.modpara.nsplit_size = width as i64;
    let diagnostic = match case {
        "cg" => {
            data.modpara.nsrcg = 1;
            "NSRCG != 0 is undefined in mVMC C"
        }
        "zero-trans" => {
            data.modpara.nmp_trans = 0;
            "NMPTrans must be nonzero"
        }
        "invalid-split" => {
            data.modpara.nsplit_size = 0;
            "NSplitSize must be >= 1"
        }
        "invalid-lanczos" => {
            data.modpara.lanczos_mode = 3;
            "NLanczosMode must be"
        }
        "invalid-cg" => {
            // NSRCG >= 2 selects CG in C; without O storage it is undefined there.
            // PhysCal never runs SR, so the case is only a ParaOpt rejection.
            data.modpara.nsrcg = 2;
            data.modpara.nstore_o = 0;
            "undefined in mVMC C"
        }
        other => panic!("unknown case {other}"),
    };
    (data, diagnostic)
}
fn rejection(
    mut data: ExpertModeData,
    diagnostic: &str,
    phys: bool,
    out: &Path,
    reducer: &impl Reducer,
    record_rank: usize,
) {
    let data_before = format!("{data:?}");
    // Nonempty sentinels catch replacement or mutation of every existing plane.
    let mut state = VmcOptimizationState::zeros(2, 1, 2, 3, 3, 2, false, true);
    state.energy.etot = Complex64::new(7.0, -2.0);
    state
        .slater_matrix
        .slater_elm
        .as_mut_slice()
        .fill(Complex64::new(3.0, 1.0));
    state
        .slater_matrix
        .inv_m
        .as_mut_slice()
        .fill(Complex64::new(4.0, 2.0));
    state.slater_matrix.slater_elm_real.as_mut_slice().fill(5.0);
    state.slater_matrix.inv_m_real.as_mut_slice().fill(6.0);
    let state_before = format!("{state:?}");
    let planes = (
        state.slater_matrix.slater_elm.as_slice().to_vec(),
        state.slater_matrix.inv_m.as_slice().to_vec(),
        state.slater_matrix.slater_elm_real.as_slice().to_vec(),
        state.slater_matrix.inv_m_real.as_slice().to_vec(),
    );
    let mut rng = Sfmt19937Rng::new(11272);
    rng.gen_rand32();
    let raw = rng.state_snapshot();
    let count = rng.words_consumed();
    let mut expected = [0; 624];
    rng.dump_rand32(&mut expected);
    let result = call(&mut data, &mut state, &mut rng, out, phys, reducer);
    println!(
        "ISSUE178_MATRIX_RETURN rank={} phys={phys} result={result:?}",
        record_rank
    );
    let error = result.unwrap_err();
    assert!(
        error.contains(diagnostic),
        "expected {diagnostic:?}, got {error:?}"
    );
    assert_eq!(format!("{data:?}"), data_before);
    assert_eq!(format!("{state:?}"), state_before);
    assert_eq!(state.slater_matrix.slater_elm.as_slice(), planes.0);
    assert_eq!(state.slater_matrix.inv_m.as_slice(), planes.1);
    assert_eq!(state.slater_matrix.slater_elm_real.as_slice(), planes.2);
    assert_eq!(state.slater_matrix.inv_m_real.as_slice(), planes.3);
    assert_eq!(rng.state_snapshot(), raw);
    assert_eq!(rng.words_consumed(), count);
    let mut actual = [0; 624];
    rng.dump_rand32(&mut actual);
    assert_eq!(actual, expected);
    assert_eq!(rng.state_snapshot(), raw);
    assert_eq!(rng.words_consumed(), count);
    assert!(!out.exists());
    println!(
        "ISSUE178_MATRIX_UNCHANGED rank={} phys={phys} {}",
        record_rank,
        checkpoint(&state, &rng)
    );
}

#[test]
fn shared_public_guards_reject_invalid_values_without_mutation() {
    let root = owned();
    for phys in [false, true] {
        for case in [
            "zero-trans",
            "invalid-split",
            "invalid-lanczos",
            "invalid-cg",
        ] {
            if phys && case == "invalid-cg" {
                continue; // PhysCal ignores SR controls (accepted, as in C).
            }
            let (data, diagnostic) = rejected_case(case, phys, 1);
            rejection(
                data,
                diagnostic,
                phys,
                &root.join(format!("{phys}-{case}")),
                &SingleProcessReducer,
                0,
            );
        }
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn signed_projection_and_physcal_cg_metadata_reach_actual_sampling() {
    let root = owned();
    for sign in [1, -1] {
        let mut data = loaded("heisenberg_chain_real", true);
        data.modpara.nmp_trans = sign;
        data.modpara.nsrcg = 1; // Fixed measurement does not invoke an SR solver.
        let mut state = state(&data);
        let mut rng = Sfmt19937Rng::new(11272);
        let out = root.join(format!("signed-{sign}"));
        call(
            &mut data,
            &mut state,
            &mut rng,
            &out,
            true,
            &SingleProcessReducer,
        )
        .unwrap();
        assert!(rng.words_consumed() > 0);
        assert!(state.energy.etot.re.is_finite() && state.energy.etot.im.is_finite());
        assert!(out.join("zvo_out_001.dat").is_file());
    }
    fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "explicit native 2/4 rank supported/rejected and ParaOpt boundary matrix"]
fn public_grouped_matrix_and_paraopt_callback_output_failures() {
    let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let root = PathBuf::from(std::env::var_os("MPI_ISSUE178_MATRIX_OUTPUT").unwrap());
    let setup = if world.is_root() {
        fs::create_dir(&root)
    } else {
        Ok(())
    };
    assert!(!world.any_failure(setup.is_err()), "{setup:?}");
    let local = root.join(format!("rank-{}", world.rank()));
    fs::create_dir(&local).unwrap();
    let mut records = BTreeMap::new();
    for trial in 0..2 {
        for width in [1, 2] {
            let group = world.split_groups(width).unwrap();
            {
                for phys in [false, true] {
                    let mut cases = vec![
                        "zero-trans",
                        "invalid-split",
                        "invalid-lanczos",
                        "invalid-cg",
                    ];
                    if phys {
                        cases.retain(|case| *case != "invalid-cg");
                    }
                    if width == 2 && !phys {
                        cases.push("cg");
                    }
                    for case in cases {
                        for bad_rank in [0, world.world_size() - 1] {
                            let (data, diagnostic) = rejected_case(case, phys, width);
                            let (data, diagnostic) = if world.rank() == bad_rank {
                                (data, diagnostic)
                            } else {
                                let mut healthy = loaded("heisenberg_chain_real", phys);
                                healthy.modpara.nsplit_size = width as i64;
                                (
                                    healthy,
                                    if phys {
                                        "PhysCal validation failed on another MPI rank"
                                    } else {
                                        "optimization validation failed on another MPI rank"
                                    },
                                )
                            };
                            rejection(
                                data,
                                diagnostic,
                                phys,
                                &local.join(format!(
                                    "reject-{trial}-{width}-{phys}-{case}-{bad_rank}"
                                )),
                                &group,
                                world.rank(),
                            );
                            println!("ISSUE178_MATRIX_CHECK rank={} trial={trial} width={width} phys={phys} case={case} bad_rank={bad_rank} unchanged=true output_absent=true", world.rank());
                            world.barrier();
                        }
                    }
                }
            }
            // Positive normal multi-QP paths are real public calculations, not
            // validate_grouped_runtime(Ok) substituted for runner coverage.
            let mut positives = vec![
                ("normal-opt", "heisenberg_chain_real", false, 0, 0),
                (
                    "normal-phys-cg-metadata",
                    "heisenberg_chain_real",
                    true,
                    1,
                    0,
                ),
                ("fsz-opt", "heisenberg_chain_fsz", false, 0, 0),
            ];
            // C defines every grouped combination below (#349).
            positives.extend([
                ("fsz-phys", "heisenberg_chain_fsz", true, 0, 0),
                ("opttrans-opt", "hubbard_chain_dh_opttrans", false, 0, 0),
                ("opttrans-phys", "hubbard_chain_dh_opttrans", true, 0, 0),
                ("lanczos1-phys", "heisenberg_chain_real", true, 0, 1),
                ("lanczos2-phys", "heisenberg_chain_real", true, 0, 2),
            ]);
            if width == 1 {
                positives.push((
                    "normal-opt-cg-sampling",
                    "heisenberg_chain_real",
                    false,
                    1,
                    0,
                ));
            }
            for (case, model, phys, cg, lanczos) in positives {
                let mut data = loaded(model, phys);
                data.modpara.nsplit_size = width as i64;
                data.modpara.nsrcg = cg;
                data.modpara.lanczos_mode = lanczos;
                let mut state = state(&data);
                let mut rng = Sfmt19937Rng::new(11272 + group.seed_offset() as u32);
                let out = local.join(format!("positive-{trial}-{width}-{case}"));
                if !phys {
                    let setup = if group.is_output_root() {
                        fs::create_dir(&out)
                    } else {
                        Ok(())
                    };
                    assert!(
                        !world.any_failure(setup.is_err()),
                        "OPT output setup: {setup:?}"
                    );
                }
                call(&mut data, &mut state, &mut rng, &out, phys, &group).unwrap();
                assert!(rng.words_consumed() > 0);
                assert!(state.energy.etot.re.is_finite() && state.energy.etot.im.is_finite());
                assert_eq!(out.exists(), group.is_output_root());
                let captured = checkpoint(&state, &rng);
                repeated(
                    &mut records,
                    format!("positive-{width}-{case}"),
                    captured.clone(),
                    trial,
                );
                println!("ISSUE178_MATRIX_REPEAT rank={} trial={trial} width={width} case={case} phys={phys} {captured}", world.rank());
                println!("ISSUE178_MATRIX_POSITIVE rank={} width={width} case={case} phys={phys} sampling_only_opt={} words={} raw={:?} cursor={} config={:?}",
                world.rank(), !phys, rng.words_consumed(), rng.state_snapshot().0, rng.state_snapshot().1, state.electron_config);
            }
            for bad_rank in [0, world.world_size() - 1] {
                let mut data = loaded("heisenberg_chain_real", false);
                data.modpara.nsplit_size = width as i64;
                data.modpara.nsr_opt_itr_step = 2;
                data.ensure_optimization_flags(data.count_variational_parameters());
                data.optimization_flags.fill(0); // Real successful zero-active SR, no fabricated peer failure.
                let mut state = state(&data);
                let mut rng = Sfmt19937Rng::new(11272 + group.seed_offset() as u32);
                let mut calls = 0;
                let mut callback = |step, _: &mut ExpertModeData, _: Complex64, status| {
                    calls += 1;
                    assert_eq!(step, 0);
                    assert_eq!(status, 0);
                    Err("issue178 actual ParaOpt callback failure".into())
                };
                let out = local.join(format!("callback-{trial}-{width}-{bad_rank}"));
                let setup = if group.is_output_root() {
                    fs::create_dir(&out)
                } else {
                    Ok(())
                };
                assert!(
                    !world.any_failure(setup.is_err()),
                    "callback output setup: {setup:?}"
                );
                let result = vmc_para_opt(
                    &mut data,
                    &mut state,
                    &mut rng,
                    Some(&out),
                    &group,
                    OptimizationOptions {
                        skip_sr: false,
                        file_flush_interval: None,
                        callback: if world.rank() == bad_rank {
                            Some(&mut callback)
                        } else {
                            None
                        },
                    },
                );
                println!("ISSUE178_OPT_RETURN rank={} trial={trial} width={width} bad_rank={bad_rank} result={result:?}", world.rank());
                let error = result.unwrap_err();
                assert_eq!(
                    error,
                    if world.rank() == bad_rank {
                        "issue178 actual ParaOpt callback failure"
                    } else {
                        "optimization callback/declaration failed on another MPI rank"
                    }
                );
                assert_eq!(calls, usize::from(world.rank() == bad_rank));
                assert_eq!(
                    data.optimization_flags
                        .iter()
                        .filter(|&&flag| flag != 0)
                        .count(),
                    0
                );
                if group.is_output_root() {
                    assert_eq!(
                        fs::read_to_string(out.join("zvo_out.dat"))
                            .unwrap()
                            .lines()
                            .count(),
                        1
                    );
                    assert!(!out.join("zqp_opt.dat").exists());
                } else {
                    assert!(!out.exists());
                }
                let captured = checkpoint(&state, &rng);
                repeated(
                    &mut records,
                    format!("callback-{width}-{bad_rank}"),
                    captured.clone(),
                    trial,
                );
                println!("ISSUE178_OPT_CHECK rank={} trial={trial} width={width} bad_rank={bad_rank} {captured}", world.rank());
                println!("ISSUE178_OPT_CALLBACK rank={} width={width} bad_rank={bad_rank} callbacks={calls} error={error:?} words={} raw={:?} cursor={} config={:?}",
                world.rank(), rng.words_consumed(), rng.state_snapshot().0, rng.state_snapshot().1, state.electron_config);
            }
            // Only the output root writes: block its actual zvo_out.dat with a
            // directory. Peers have valid paths; their local filesystem cannot fail.
            let mut data = loaded("heisenberg_chain_real", false);
            data.modpara.nsplit_size = width as i64;
            let coefficients = data.slater_params.clone();
            let flags = data.optimization_flags.clone();
            let mut state = state(&data);
            let mut rng = Sfmt19937Rng::new(11272 + group.seed_offset() as u32);
            let out = local.join(format!("output-error-{trial}-{width}"));
            let setup = if group.is_output_root() {
                fs::create_dir(&out).and_then(|()| fs::create_dir(out.join("zvo_out.dat")))
            } else {
                Ok(())
            };
            assert!(!world.any_failure(setup.is_err()), "{setup:?}");
            let mut calls = 0;
            let mut callback = |_, _: &mut ExpertModeData, _: Complex64, _| {
                calls += 1;
                Ok(())
            };
            let result = vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&out),
                &group,
                OptimizationOptions {
                    skip_sr: false,
                    file_flush_interval: None,
                    callback: Some(&mut callback),
                },
            );
            println!(
                "ISSUE178_OPT_WRITE_RETURN rank={} trial={trial} width={width} result={result:?}",
                world.rank()
            );
            assert!(result.is_err());
            assert_eq!(calls, 0);
            assert_eq!(data.slater_params, coefficients);
            assert_eq!(data.optimization_flags, flags);
            assert!(state.opt_data.is_empty());
            assert_eq!(out.exists(), group.is_output_root());
            let captured = checkpoint(&state, &rng);
            repeated(
                &mut records,
                format!("output-{width}"),
                captured.clone(),
                trial,
            );
            println!(
                "ISSUE178_OPT_WRITE_CHECK rank={} trial={trial} width={width} {captured}",
                world.rank()
            );
            println!("ISSUE178_OPT_OUTPUT rank={} width={width} result={result:?} callbacks={calls} words={} raw={:?} cursor={} config={:?}",
            world.rank(), rng.words_consumed(), rng.state_snapshot().0, rng.state_snapshot().1, state.electron_config);
            world.barrier();
        }
    }
    println!(
        "ISSUE178_MATRIX_DONE rank={} world={}",
        world.rank(),
        world.world_size()
    );
}
