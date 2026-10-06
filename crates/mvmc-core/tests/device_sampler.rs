//! Lock-step device sampler protocol (issue #434) on the host service: the sampler driven through
//! `WalkerStage` + `HostService` in lock-step must equal the plain CPU sampler exactly (same
//! draws, decisions, configurations, RNG state), for hopping (Hubbard) and exchange (Heisenberg).

use std::path::{Path, PathBuf};

use mvmc_core::device_sampler::{run_lockstep_real, HostService, LockstepOptions, Teacher};
use mvmc_core::run::{prepare_sampling_walker, SamplingWalker};
use mvmc_core::sampling::driver::{trace, vmc_make_sample_real};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Case {
    namelist: PathBuf,
    opt: Option<PathBuf>,
    seed: i64,
    samples: i64,
}

fn heisenberg() -> Case {
    let root = repo()
        .join("extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref");
    assert!(root.is_dir(), "Julia-mVMC fixture required");
    Case {
        namelist: root.join("inputs/namelist.def"),
        opt: Some(root.join("zqp_opt.dat")),
        seed: 1,
        samples: 4,
    }
}

fn hubbard() -> Case {
    Case {
        namelist: repo().join("benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def"),
        opt: None,
        seed: 7,
        samples: 6,
    }
}

fn walker(c: &Case, offset: usize) -> SamplingWalker {
    let mut prep = mvmc_core::prepare_phys_cal_with_seed_offset(
        &c.namelist,
        c.opt.as_deref(),
        "real",
        Some(c.seed),
        true,
        offset,
    )
    .unwrap();
    prep.data.modpara.nvmc_sample = c.samples;
    prepare_sampling_walker(prep).unwrap()
}

fn check(c: &Case, walkers: usize) {
    // CPU references
    let mut reference = Vec::new();
    for w in 0..walkers {
        let mut sw = walker(c, w);
        trace::start_decisions();
        let stats = vmc_make_sample_real(&sw.data, &mut sw.state, &mut sw.rng).unwrap();
        let decisions = trace::finish_decisions();
        reference.push((stats, decisions, sw));
    }
    // lock-step on the host service
    let mut ws: Vec<SamplingWalker> = (0..walkers).map(|w| walker(c, w)).collect();
    let mut service = HostService::new();
    let (runs, stats) =
        run_lockstep_real(&mut ws, &mut service, LockstepOptions::default()).unwrap();
    assert!(stats.passes > 0);
    for (w, run) in runs.iter().enumerate() {
        let (ref_stats, ref_dec, ref_walker) = &reference[w];
        assert_eq!(run.stats.as_ref().unwrap(), ref_stats, "walker {w} stats");
        assert_eq!(&run.decisions, ref_dec, "walker {w} decisions");
        assert_eq!(
            ws[w].rng.state_snapshot(),
            ref_walker.rng.state_snapshot(),
            "walker {w} rng state"
        );
        assert_eq!(ws[w].rng.words_consumed(), ref_walker.rng.words_consumed());
        assert_eq!(
            ws[w].state.electron_config.tmp_ele_idx,
            ref_walker.state.electron_config.tmp_ele_idx
        );
        assert_eq!(
            ws[w].state.electron_config.ele_idx,
            ref_walker.state.electron_config.ele_idx
        );
    }
    eprintln!(
        "lock-step: {walkers} walkers, {} passes, {} proposals, {} accepts, {} recomputes, max batch {}",
        stats.passes, stats.proposals, stats.accepts, stats.recomputes, stats.max_batch
    );

    // teacher-forced replay on the host service: no flips, draws identical
    let teachers: Vec<Teacher> = reference
        .iter()
        .map(|(_, d, _)| Teacher {
            reference: d.clone(),
            weight_abs: 1e-12,
            weight_rel: 1e-10,
        })
        .collect();
    let mut ws: Vec<SamplingWalker> = (0..walkers).map(|w| walker(c, w)).collect();
    let (runs, _) = run_lockstep_real(
        &mut ws,
        &mut HostService::new(),
        LockstepOptions {
            teachers,
            ..LockstepOptions::default()
        },
    )
    .unwrap();
    for (w, run) in runs.iter().enumerate() {
        let rep = run.teacher.as_ref().unwrap();
        assert_eq!(rep.compared, reference[w].1.len());
        assert_eq!(rep.flips, 0);
        assert_eq!(rep.defects, 0);
        assert_eq!(rep.draw_mismatches, 0);
        assert_eq!(rep.overrun, 0);
        assert_eq!(rep.max_weight_abs, 0.0, "host service is bit-identical");
    }
}

#[test]
fn hubbard_hopping_lockstep_equals_cpu_sampler() {
    check(&hubbard(), 3);
}

#[test]
fn heisenberg_exchange_lockstep_equals_cpu_sampler() {
    check(&heisenberg(), 3);
}

#[test]
fn single_walker_lockstep_equals_cpu_sampler() {
    check(&hubbard(), 1);
}
