//! Combined MPI/inner-worker test; one worker setting per native process.
//! Rank-local worker invariance is not full C sampler or independent MPI parity.
#![cfg(feature = "mpi")]

use std::{fs, io::Write, path::PathBuf};

use mvmc_core::{
    mpi::MpiContext,
    prepare_phys_cal_from_namelist_with_reducer,
    threading::{inner_thread_config, start_observation},
    vmc_phys_cal_in_place, Reducer, VmcOptimizationState,
};
use num_complex::Complex64;
use serde_json::{json, Value};
use sfmt19937::Sfmt19937Rng;
use sha2::{Digest, Sha256};

fn complex(values: &[Complex64]) -> Vec<[f64; 2]> {
    values.iter().map(|z| [z.re, z.im]).collect()
}

fn rng_boundary(rng: &Sfmt19937Rng) -> Value {
    let raw = rng.state_snapshot();
    let count = rng.words_consumed();
    let mut future = [0; 624];
    rng.dump_rand32(&mut future);
    assert_eq!(rng.state_snapshot(), raw, "nonconsuming future raw state");
    assert_eq!(rng.words_consumed(), count, "nonconsuming future count");
    assert!(raw.1 <= 624);
    json!({"raw624":raw.0.to_vec(),"cursor":raw.1,"words":count.to_string(),"future624":future.to_vec()})
}

#[test]
#[ignore = "explicit native worlds2/4 x workers1/2/4; reviewed provider/owner wrapper required"]
fn two_frame_public_physcal_keeps_rank_local_rng_and_executes_green_workers() {
    let workers = inner_thread_config();
    assert!(matches!(workers.threads, 1 | 2 | 4));
    assert_eq!(workers.threshold, 32);
    for name in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ] {
        assert_eq!(std::env::var(name).as_deref(), Ok("1"), "{name}");
    }
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let pins = [
        (
            "inputs/namelist.def",
            "3df55df5d3e65cf3f31846bb69f2e16f6c2cdd410d241f87596a489ca1440116",
        ),
        (
            "inputs/greenone.def",
            "79855770f5dec7e429e53a65ba25a8edee2070d717c4d75033ab3d6efd9584d6",
        ),
        (
            "inputs/greentwo.def",
            "691a473126df3c13e933921c9576acbbac0ca72480aeefb2448eb50936b9518b",
        ),
        (
            "inputs/modpara.def",
            "96d5aee8c5d317cfb7d2fe747bac314ed59cff0020a16aaedcbc9bb8aefc885b",
        ),
        (
            "zqp_opt.dat",
            "943def3230d2d545284862d7e7570c4e564c78864cfe07de6ed51c1cc3156a78",
        ),
    ];
    for (name, expected) in pins {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(fixture.join(name)).unwrap())
            ),
            expected
        );
    }
    let root = PathBuf::from(
        std::env::var_os("ISSUE182_MPI_RECEIPT_ROOT").expect("exclusive receipt root"),
    );
    assert!(root.is_absolute() && root.is_dir());
    let width: usize = std::env::var("ISSUE182_MPI_GROUP_WIDTH")
        .unwrap()
        .parse()
        .unwrap();
    assert!(matches!(width, 1 | 2));
    let world = MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let group = world.split_groups(width).unwrap();
    // Committed runner output is global group0/local0 only. Still give each
    // group an exclusive namespace; never share a group leader's output path.
    let public = root.join(format!("public-group-{}", group.group()));
    let directory_error = if group.rank() == 0 {
        fs::create_dir(&public).err()
    } else {
        None
    };
    assert!(
        !world.any_failure(directory_error.is_some()),
        "exclusive group output directory: {directory_error:?}"
    );
    let mut preparation = prepare_phys_cal_from_namelist_with_reducer(
        fixture.join("inputs/namelist.def"),
        fixture.join("zqp_opt.dat"),
        "real",
        None,
        &group,
    )
    .unwrap();
    let data = &mut preparation.data;
    assert_eq!(data.modpara.rnd_seed, 1);
    assert_eq!(data.i_flg_orbital_general, 0);
    assert_eq!(data.count_rbm_parameters(), 0);
    assert_eq!(data.modpara.lanczos_mode, 0);
    assert_eq!(data.green_one_terms.len(), 2);
    assert_eq!(
        data.green_two_terms.len(),
        36,
        "original descriptor width, no synthetic duplication"
    );
    assert_eq!(data.modpara.n_data_qty_smp, 1);
    // Original supported model, two consecutive public frames. The original
    // file declares one; this explicit run-length overlay is not fixture regeneration.
    data.modpara.n_data_qty_smp = 2;
    data.modpara.nsplit_size = i64::try_from(width).unwrap();
    let before = rng_boundary(&preparation.rng);
    let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    let mut callbacks = Vec::new();
    let mut callback = |frame: usize,
                        callback_data: &mvmc_core::ExpertModeData,
                        energy: Complex64,
                        status: i32| {
        if frame != callbacks.len() || status != 0 || callback_data.green_two_terms.len() != 36 {
            return Err("unexpected actual PhysCal frame/status/descriptor width".into());
        }
        callbacks.push(json!({"frame":frame,"status":status,"energy":[energy.re,energy.im]}));
        Ok(())
    };
    let observer = start_observation();
    let result = vmc_phys_cal_in_place(
        data,
        &mut state,
        &mut preparation.rng,
        Some(&public),
        &group,
        Some(&mut callback),
    );
    let actual = observer.finish();
    // A rank-local capture panic or I/O error must not strand peers at the
    // final collective. No collective occurs inside this capture closure.
    let capture = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<(), String> {
            let after = rng_boundary(&preparation.rng);
            let phys = state.phys_quantities.as_ref();
            let cfg = &state.electron_config;
            let dimensions = json!({"sites":data.modpara.nsite,"electronsPerSpin":data.modpara.nelec,"projection":data.projection_layout().n_proj,"samples":data.modpara.nvmc_sample,"factored":data.green_two_ex_indices.len()});
            let discrete = json!({"initial":before,"final":after,"callbacks":callbacks.iter().map(|r|json!({"frame":r["frame"],"status":r["status"]})).collect::<Vec<_>>(),
        "saved":{"idx":cfg.ele_idx,"cfg":cfg.ele_cfg,"num":cfg.ele_num,"proj":cfg.ele_proj_cnt,"spin":cfg.ele_spn},
        "burn":{"idx":cfg.burn_ele_idx,"cfg":cfg.burn_ele_cfg,"num":cfg.burn_ele_num,"proj":cfg.burn_ele_proj_cnt,"spin":cfg.burn_ele_spn},
        "scratch":{"idx":cfg.tmp_ele_idx,"cfg":cfg.tmp_ele_cfg,"num":cfg.tmp_ele_num,"proj":cfg.tmp_ele_proj_cnt,"spin":cfg.tmp_ele_spn},"counter":cfg.counter});
            let numeric = json!({"callbacks":callbacks.iter().map(|r|r["energy"].clone()).collect::<Vec<_>>(),
        "energy":complex(&[state.energy.wc,state.energy.etot,state.energy.etot2,state.energy.sztot,state.energy.sztot2]),
        "one":phys.map(|p|complex(&p.phys_cis_ajs)),"factored":phys.map(|p|complex(&p.phys_cis_ajs_ckt_alt)),"direct":phys.map(|p|complex(&p.phys_cis_ajs_ckt_alt_dc))});
            let record = json!({"schema":"issue182-mpi-inner-physcal-v1","world":world.world_size(),"rank":world.rank(),"groupWidth":width,"group":group.group(),"groupRank":group.rank(),"seedOffset":group.seed_offset(),"workers":workers.threads,"threshold":32,"frames":2,"originalOne":2,"originalDirect":36,
        "dimensions":dimensions,"result":result.as_ref().copied().map_err(Clone::clone),"discrete":discrete,"numeric":numeric,
        "observation":{"parallelCalls":actual.parallel_calls,"serialCalls":actual.serial_calls,"parallelEntries":actual.parallel_entry_items,"serialEntries":actual.serial_entry_items,"workerEntries":actual.worker_entries,"workerIds":actual.worker_ids},
        "scope":"initial/final rank-local boundaries and two public callbacks; no intermediate raw RNG, no standalone MPI oracle, no timer/performance claim"});
            let path = root.join(format!("rank-{}.json", world.rank()));
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|error| error.to_string())?;
            serde_json::to_writer(&mut file, &record).map_err(|error| error.to_string())?;
            file.write_all(b"\n").map_err(|error| error.to_string())?;
            if workers.threads == 1 {
                assert_eq!(actual.worker_entries, 0);
                assert_eq!(actual.parallel_entry_items, 0);
                assert!(
                    actual.serial_entry_items >= 72,
                    "actual two-frame entry work"
                );
            } else {
                assert!(
                    actual.parallel_entry_items >= 72,
                    "must actually execute original width36 producer work"
                );
                assert!(actual.worker_entries >= 72);
                assert!(!actual.worker_ids.is_empty());
                assert!(actual.worker_ids.iter().all(|&id| id < workers.threads));
            }
            for (name, expected) in pins {
                assert_eq!(
                    format!(
                        "{:x}",
                        Sha256::digest(fs::read(fixture.join(name)).unwrap())
                    ),
                    expected
                );
            }
            Ok(())
        },
    ));
    let capture_error = match capture {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error),
        Err(_) => Some("rank-local capture/validation panicked; partial record retained".into()),
    };
    let failed = capture_error.is_some()
        || result.as_ref().map_or(true, |&n| n != 2)
        || callbacks.len() != 2;
    let any_failed = world.any_failure(failed);
    assert!(
        !any_failed,
        "actual public result: {result:?}; capture: {capture_error:?}"
    );
    println!(
        "ISSUE182_MPI_INNER_DONE world={} rank={} groupWidth={} group={} workers={} frames=2",
        world.world_size(),
        world.rank(),
        width,
        group.group(),
        workers.threads
    );
}
