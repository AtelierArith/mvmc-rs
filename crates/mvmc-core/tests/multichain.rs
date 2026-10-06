//! Multi-chain walker runner contract (issue #425): C-compatible per-walker seeds, per-walker
//! draw order/count equal to the serial path, `W = 1` equal to the serial run, results
//! independent of the pool size.

use std::path::{Path, PathBuf};

use mvmc_core::multichain::{
    pack_walker_tables, run_phys_cal_multichain, walker_seed, MultiChainConfig, WalkerPlanes,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
    )
}

fn config(walkers: usize, group_base: usize, threads: usize) -> MultiChainConfig {
    let root = root();
    assert!(root.is_dir(), "Julia-mVMC PhysCal fixture is required");
    MultiChainConfig {
        namelist: root.join("inputs/namelist.def"),
        opt_para: Some(root.join("zqp_opt.dat")),
        mode: "real".to_string(),
        seed: Some(1),
        group_base,
        walkers,
        enable_opt_trans: true,
        samples: Some(3),
        threads,
    }
}

/// The existing serial path for a given seed offset, with decision recording.
fn serial(offset: usize) -> (mvmc_core::PhysCalResult, Vec<(f64, f64)>) {
    let cfg = config(1, 0, 1);
    let mut prep = mvmc_core::prepare_phys_cal_with_seed_offset(
        &cfg.namelist,
        cfg.opt_para.as_deref(),
        "real",
        Some(1),
        true,
        offset,
    )
    .unwrap();
    prep.data.modpara.n_data_qty_smp = 3;
    mvmc_core::sampling::driver::trace::start_decisions();
    let result = mvmc_core::vmc_phys_cal(prep).unwrap();
    (
        result,
        mvmc_core::sampling::driver::trace::finish_decisions(),
    )
}

fn same_run(a: &mvmc_core::PhysCalResult, b: &mvmc_core::PhysCalResult) {
    assert_eq!(a.iterations, b.iterations);
    // exact RNG stream state: complete generator block and position
    assert_eq!(a.final_rng.state_snapshot(), b.final_rng.state_snapshot());
    assert_eq!(a.final_rng.words_consumed(), b.final_rng.words_consumed());
    // whole sampling/observable state, byte-for-byte (same code path, same arithmetic order)
    assert_eq!(format!("{:?}", a.state), format!("{:?}", b.state));
}

#[test]
fn single_walker_reproduces_the_serial_run_exactly() {
    let multi = run_phys_cal_multichain(&config(1, 0, 1)).unwrap();
    let (reference, decisions) = serial(0);
    assert_eq!(multi.walkers.len(), 1);
    assert_eq!(multi.walkers[0].seed, 1);
    same_run(&multi.walkers[0].result, &reference);
    assert_eq!(multi.walkers[0].decisions, decisions);
}

#[test]
fn walker_w_is_the_serial_run_with_group_offset_and_equal_draw_count() {
    let group_base = 2;
    let multi = run_phys_cal_multichain(&config(4, group_base, 4)).unwrap();
    assert_eq!(multi.walkers.len(), 4);
    for (w, walker) in multi.walkers.iter().enumerate() {
        assert_eq!(walker.walker, w);
        assert_eq!(walker.seed, walker_seed(1, group_base, w).unwrap());
        assert_eq!(walker.seed, 1 + (group_base + w) as i64);
        let (reference, decisions) = serial(group_base + w);
        same_run(&walker.result, &reference);
        // same decisions => same draw order and count: the stream positions are equal and
        // the recorded (weight, draw) sequences are identical
        assert_eq!(walker.rng_words, reference.final_rng.words_consumed());
        assert_eq!(walker.decisions, decisions);
        assert!(walker.summary.proposals > 0);
        assert!(walker.summary.min_margin >= 0.0);
    }
}

#[test]
fn walkers_own_distinct_streams() {
    let multi = run_phys_cal_multichain(&config(3, 0, 3)).unwrap();
    let states: Vec<_> = multi
        .walkers
        .iter()
        .map(|w| w.result.final_rng.state_snapshot())
        .collect();
    assert_ne!(states[0], states[1]);
    assert_ne!(states[1], states[2]);
    assert_ne!(states[0], states[2]);
}

#[test]
fn results_do_not_depend_on_the_worker_pool_size() {
    let one = run_phys_cal_multichain(&config(4, 0, 1)).unwrap();
    let four = run_phys_cal_multichain(&config(4, 0, 4)).unwrap();
    for (a, b) in one.walkers.iter().zip(&four.walkers) {
        same_run(&a.result, &b.result);
        assert_eq!(a.decisions, b.decisions);
    }
}

#[test]
fn walker_tables_pack_into_the_batched_layout() {
    let multi = run_phys_cal_multichain(&config(3, 0, 3)).unwrap();
    let packed = pack_walker_tables(&multi.walkers).unwrap();
    let WalkerPlanes::Real(planes) = packed else {
        panic!("the fixture is a real wavefunction");
    };
    assert_eq!(planes.capacity(), 3);
    for (slot, walker) in multi.walkers.iter().enumerate() {
        let state = &walker.result.state;
        for qp in 0..planes.n_qp() {
            assert_eq!(
                planes.plane(slot, qp),
                state.slater_matrix.inv_m_real.qp_matrix_slice(qp)
            );
        }
        assert_eq!(
            planes.pf(slot),
            &state.slater_matrix.pf_m_real[..planes.n_qp()]
        );
    }
}

#[test]
fn zero_walkers_is_rejected() {
    assert!(run_phys_cal_multichain(&config(0, 0, 1)).is_err());
}

/// Walker scaling on the CPU (optional, ignored): wall time of `W` walkers on a pool of `W`
/// threads. `MVMC_RS_MULTICHAIN_SAMPLES` (default 200) sets the measurement samples per walker,
/// `MVMC_RS_MULTICHAIN_OUT` the report path.
#[test]
#[ignore = "optional scaling measurement"]
fn multichain_scaling_report() {
    use mvmc_core::accel_validation::BenchMetadata;
    let samples: i64 = std::env::var("MVMC_RS_MULTICHAIN_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut rows = String::from(
        "| walkers W | threads | wall ms (median of 5) | ms per walker-chain | scaling efficiency | proposals/walker | min margin (worst walker) |\n|---|---|---|---|---|---|---|\n",
    );
    let mut t1 = 0.0;
    for w in [1usize, 2, 4, 8, 16, 32] {
        if w > available {
            continue;
        }
        let mut cfg = config(w, 0, w);
        cfg.samples = Some(samples);
        let mut times = Vec::new();
        let mut last = None;
        for rep in 0..6 {
            let start = std::time::Instant::now();
            let res = run_phys_cal_multichain(&cfg).unwrap();
            if rep > 0 {
                times.push(start.elapsed().as_secs_f64() * 1e3);
            }
            last = Some(res);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let wall = times[times.len() / 2];
        if w == 1 {
            t1 = wall;
        }
        let res = last.unwrap();
        let min_margin = res
            .walkers
            .iter()
            .map(|x| x.summary.min_margin)
            .fold(f64::INFINITY, f64::min);
        rows += &format!(
            "| {w} | {w} | {wall:.1} | {:.1} | {:.2} | {} | {min_margin:.2e} |\n",
            wall / w as f64,
            t1 / wall,
            res.walkers[0].summary.proposals
        );
    }
    let meta = BenchMetadata::collect(
        "CPU multichain (rayon over walkers, serial walker, BLAS 1 thread)",
        "f64",
        1,
        1,
        5,
        false,
        None,
    );
    let text = format!(
        "## Multi-chain CPU scaling\n\n```\n{}```\n\nsamples per walker {samples}; scaling \
         efficiency = T(1) / T(W) for W walkers of identical work on W threads (1.0 is perfect).\n\n{rows}",
        meta.render()
    );
    println!("{text}");
    if let Ok(path) = std::env::var("MVMC_RS_MULTICHAIN_OUT") {
        std::fs::write(path, &text).unwrap();
    }
}
