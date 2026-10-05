//! MPI PhysCal smoke gate. Run explicitly with an MPI launcher:
//!
//! See docs/OPTIONAL_GATES.md for building and selecting the exact MPI test binary.

mod support;
#[cfg(feature = "mpi")]
use mvmc_core::Reducer;
use support::{report_gate, require_gate, GateStatus};

#[cfg(feature = "mpi")]
fn next624(rng: &sfmt19937::Sfmt19937Rng) -> Vec<u32> {
    let mut copy = rng.clone();
    (0..624).map(|_| copy.gen_rand32()).collect()
}

#[cfg(feature = "mpi")]
fn fixed_parameters(data: &mvmc_core::ExpertModeData) -> Vec<(u64, u64)> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        // Immutable loaded values are a copy contract, not computed-float parity.
        // Preserve both signed zeros as well as every other stored bit.
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect()
}

#[cfg(feature = "mpi")]
fn close(a: f64, b: f64) -> bool {
    // Repeat-local elementary/BLAS arithmetic only; inherited from the scoped
    // S196 PhysCal repeat gate. No independent reference/solver budget is widened.
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 + 1e-12 * a.abs().max(b.abs())
}

#[cfg(feature = "mpi")]
fn output_agrees(a: &std::path::Path, b: &std::path::Path, indices: usize) -> bool {
    let (Ok(a), Ok(b)) = (std::fs::read_to_string(a), std::fs::read_to_string(b)) else {
        return false;
    };
    let a: Vec<_> = a.lines().collect();
    let b: Vec<_> = b.lines().collect();
    !a.is_empty()
        && a.iter().any(|line| !line.trim().is_empty())
        && a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            let a: Vec<_> = a.split_whitespace().collect();
            let b: Vec<_> = b.split_whitespace().collect();
            a.len() == b.len()
                && (a.is_empty()
                    || (a.len() > indices
                        && a[..indices] == b[..indices]
                        && a[indices..].iter().zip(&b[indices..]).all(|(a, b)| {
                            match (a.parse::<f64>(), b.parse::<f64>()) {
                                (Ok(a), Ok(b)) => close(a, b),
                                _ => false,
                            }
                        })))
        })
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "optional MPI gate; requires MVMC_RS_MPI_PHYSICAL=1, mpi feature and MPI launcher"]
fn mpi_physcal_reduces_fixed_parameter_samples() {
    require_gate("mpi-physcal", "MVMC_RS_MPI_PHYSICAL");

    // Offline copy has independent provenance; no reference runtime is required.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    for path in [root.join("inputs/namelist.def"), root.join("zqp_opt.dat")] {
        if !path.is_file() {
            support::missing_fixture("mpi-physcal", path.display().to_string());
        }
    }
    // This gate requires a multi-rank launch, unlike the library's valid
    // singleton use. Reject missing/singleton launcher metadata before MPI
    // initialization, which may itself be unavailable on the current host.
    let launch = mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok());
    if launch.is_none_or(|launch| launch.world_size < 2) {
        support::unsupported(
            "mpi-physcal",
            "recognized MPI launcher metadata with at least two ranks is required",
        );
    }
    let context = mvmc_core::mpi::MpiContext::initialize().expect("MPI initialization");
    if !matches!(context.world_size(), 2 | 4) {
        support::unsupported(
            "mpi-physcal",
            "this bounded gate requires an actual 2/4-rank MPI world",
        );
    }
    let output = std::path::PathBuf::from(
        std::env::var("MPI179_PHYSCAL_OUTPUT").expect("exclusive retained output root required"),
    );
    let setup = if context.is_root() {
        std::fs::create_dir(&output)
    } else {
        Ok(())
    };
    assert!(
        !context.any_failure(setup.is_err()),
        "output root must be new"
    );
    let group = context.split_groups(2).expect("MPI PhysCal group split");
    for (width, reducer) in [
        (1, &context as &dyn mvmc_core::Reducer),
        (2, &group as &dyn mvmc_core::Reducer),
    ] {
        let mut discrete = None;
        let mut numerical: Option<Vec<f64>> = None;
        for repeat in 1..=2 {
            let mut preparation = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
                root.join("inputs/namelist.def"),
                root.join("zqp_opt.dat"),
                "real",
                Some(1),
                reducer,
            )
            .expect("prepare normal PhysCal");
            let data = &mut preparation.data;
            assert_eq!(data.i_flg_orbital_general, 0);
            assert_eq!(data.modpara.lanczos_mode, 0);
            assert!(data.inter_all_terms.is_empty());
            assert!(data.n_qp_opt_trans <= 1 && data.opt_trans.len() <= 1);
            // Retain the fixture's nontrivial standard projection, not identity QP.
            assert_eq!(data.modpara.nsp_gauss_leg, 8);
            assert_eq!(data.modpara.nmp_trans, -1);
            data.modpara.nsplit_size = width;
            data.modpara.nvmc_sample = 3;
            data.modpara.nvmc_warmup = 1;
            data.modpara.n_data_qty_smp = 1;
            mvmc_core::validation::validate_phys_cal(data).unwrap();
            let fixed = fixed_parameters(data);
            assert_eq!(preparation.rng.words_consumed(), 0);
            let expected = sfmt19937::Sfmt19937Rng::new((1 + reducer.seed_offset()) as u32);
            assert_eq!(preparation.rng.state_snapshot(), expected.state_snapshot());
            assert_eq!(next624(&preparation.rng), next624(&expected));
            let dir = output.join(format!("width{width}-repeat{repeat}"));
            let result = mvmc_core::vmc_phys_cal_with_reducer(preparation, Some(&dir), reducer)
                .expect("supported normal MPI PhysCal");
            assert_eq!(result.iterations, 1);
            assert_eq!(fixed_parameters(&result.data), fixed);
            let config = &result.state.electron_config;
            let actual = (
                config.ele_idx.clone(),
                config.ele_cfg.clone(),
                config.ele_num.clone(),
                config.ele_proj_cnt.clone(),
                config.ele_spn.clone(),
                config.counter,
                result.final_rng.words_consumed(),
                result.final_rng.state_snapshot(),
                next624(&result.final_rng),
            );
            if let Some(before) = &discrete {
                assert_eq!(&actual, before);
            }
            discrete = Some(actual);
            let phys = result.state.phys_quantities.as_ref().unwrap();
            let e = result.state.energy;
            let values: Vec<_> = [e.wc, e.etot, e.etot2, e.sztot, e.sztot2]
                .into_iter()
                .chain(phys.phys_cis_ajs.iter().copied())
                .chain(phys.phys_cis_ajs_ckt_alt.iter().copied())
                .chain(phys.phys_cis_ajs_ckt_alt_dc.iter().copied())
                .flat_map(|z| [z.re, z.im])
                .collect();
            assert!(values.iter().all(|v| v.is_finite()));
            if let Some(before) = &numerical {
                assert_eq!(values.len(), before.len());
                assert!(values.iter().zip(before).all(|(&a, &b)| close(a, b)));
            }
            numerical = Some(values);
            context.barrier();
            let mut bad_output = false;
            if context.is_root() {
                let previous = output.join(format!("width{width}-repeat1"));
                let mut names: Vec<_> = std::fs::read_dir(&dir)
                    .unwrap()
                    .map(|v| v.unwrap().file_name())
                    .collect();
                let mut expected: Vec<_> = [
                    "zvo_out_001.dat",
                    "zvo_var_001.dat",
                    "zvo_cisajs_001.dat",
                    "zvo_cisajscktalt_001.dat",
                    "zvo_cisajscktaltex_001.dat",
                    "zvo_time_001.dat",
                ]
                .into_iter()
                .map(std::ffi::OsString::from)
                .collect();
                names.sort();
                expected.sort();
                bad_output |= names != expected;
                for (name, indices) in [
                    ("zvo_out_001.dat", 0),
                    ("zvo_var_001.dat", 0),
                    ("zvo_cisajs_001.dat", 4),
                    ("zvo_cisajscktalt_001.dat", 8),
                    ("zvo_cisajscktaltex_001.dat", 0),
                ] {
                    bad_output |= !output_agrees(&dir.join(name), &previous.join(name), indices);
                }
            }
            assert!(
                !context.any_failure(bad_output),
                "finite indexed output repeatability"
            );
            println!(
                "MPI_PHYSCAL rank={} world={} width={width} repeat={repeat} words={}",
                context.rank(),
                context.world_size(),
                result.final_rng.words_consumed()
            );
        }
        // Explicit manual CI observes actual communicator sizes only after
        // both fixed-parameter/output repeatability checks completed.
        println!(
            "OPTIONAL183_MPI rank={} world={} width={width} group_size={}",
            context.rank(),
            context.world_size(),
            reducer.world_size()
        );
    }
    let mut grouped = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
        &group,
    )
    .expect("prepare grouped PhysCal");
    // Independent input negative: an invalid NLanczosMode. Grouped Lanczos
    // itself is defined in C and accepted (#349), so it is no longer a negative.
    grouped.data.modpara.nsplit_size = 2;
    grouped.data.modpara.lanczos_mode = 3;
    let before_data = format!("{:?}", grouped.data);
    let before_fixed = fixed_parameters(&grouped.data);
    let before_rng = next624(&grouped.rng);
    let before_raw_rng = grouped.rng.state_snapshot();
    let words = grouped.rng.words_consumed();
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, false, false);
    let before_state = format!("{state:?}");
    let rejected_output = output.join("rejected-lanczos");
    let error = mvmc_core::vmc_phys_cal_in_place(
        &mut grouped.data,
        &mut state,
        &mut grouped.rng,
        Some(&rejected_output),
        &group,
        None,
    )
    .unwrap_err();
    assert!(error.contains("NLanczosMode must be"), "{error}");
    assert_eq!(format!("{:?}", grouped.data), before_data);
    assert_eq!(fixed_parameters(&grouped.data), before_fixed);
    assert_eq!(format!("{state:?}"), before_state);
    assert_eq!(next624(&grouped.rng), before_rng);
    assert_eq!(grouped.rng.state_snapshot(), before_raw_rng);
    assert_eq!(grouped.rng.words_consumed(), words);
    assert!(!rejected_output.exists());
    report_gate(
        "mpi-physcal",
        GateStatus::Pass,
        "MPI world/grouped normal PhysCal repeated fixed/discrete/indexed outputs; invalid NLanczosMode rejected before mutation",
    );
}

#[cfg(not(feature = "mpi"))]
#[test]
#[ignore = "optional MPI gate; requires MVMC_RS_MPI_PHYSICAL=1 and mpi feature"]
fn mpi_physcal_requires_feature() {
    require_gate("mpi-physcal", "MVMC_RS_MPI_PHYSICAL");
    report_gate(
        "mpi-physcal",
        GateStatus::Unsupported,
        "mvmc-core mpi feature is disabled",
    );
    panic!("MPI PhysCal gate selected but the mpi feature is disabled");
}

/// Selecting the MPI gate without a multi-rank launch (or without the `mpi`
/// feature) must fail with an explicit Unsupported status, never pass.
#[test]
fn selected_mpi_gate_without_launcher_fails_as_unsupported() {
    let test = if cfg!(feature = "mpi") {
        "mpi_physcal_reduces_fixed_parameter_samples"
    } else {
        "mpi_physcal_requires_feature"
    };
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.env_clear();
    if let Some(path) = std::env::var_os("LD_LIBRARY_PATH") {
        command.env("LD_LIBRARY_PATH", path);
    }
    let output = command
        .args(["--ignored", "--exact", test, "--nocapture"])
        .env("MVMC_RS_MPI_PHYSICAL", "1")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Unsupported"));
}
