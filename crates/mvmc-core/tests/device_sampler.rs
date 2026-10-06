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

/// The flip detector works: a reference decision that disagrees with the computed weight by far
/// more than the weight error is reported as a flip and as a defect, while the walker follows the
/// reference decision.
#[test]
fn teacher_reports_a_forced_flip_as_a_defect() {
    let c = hubbard();
    let mut sw = walker(&c, 0);
    trace::start_decisions();
    vmc_make_sample_real(&sw.data, &mut sw.state, &mut sw.rng).unwrap();
    let mut reference = trace::finish_decisions();
    // find a decision with a comfortable margin and invert it
    let i = reference
        .iter()
        .position(|&(w, u)| (w - u).abs() > 0.1)
        .expect("a decision with margin");
    let (w, u) = reference[i];
    reference[i].0 = if w > u { u - 0.1 } else { u + 0.1 };
    let mut ws = vec![walker(&c, 0)];
    let (runs, _) = run_lockstep_real(
        &mut ws,
        &mut HostService::new(),
        LockstepOptions {
            teachers: vec![Teacher {
                reference,
                weight_abs: 1e-12,
                weight_rel: 1e-10,
            }],
            ..LockstepOptions::default()
        },
    )
    .unwrap();
    let rep = runs[0].teacher.as_ref().unwrap();
    // the first disagreement is the forced one; afterwards the walker follows the reference
    // decisions while the configurations differ, so later decisions disagree as well
    assert!(rep.flips >= 1);
    assert!(rep.defects >= 1);
}

fn device_run(c: &Case) -> (SamplingWalker, SamplingWalker) {
    let mut cpu = walker(c, 0);
    vmc_make_sample_real(&cpu.data, &mut cpu.state, &mut cpu.rng).unwrap();
    let mut ws = vec![walker(c, 0)];
    run_lockstep_real(&mut ws, &mut HostService::new(), LockstepOptions::default()).unwrap();
    (ws.pop().unwrap(), cpu)
}

/// Issue #454: a device-resident run leaves the host inverse table marked stale, and any read
/// of it fails loudly instead of returning out-of-date numbers.
#[test]
#[should_panic(expected = "stale inverse table read")]
fn stale_host_inverse_read_after_device_run_panics() {
    let (w, _) = device_run(&hubbard());
    assert!(w.state.slater_matrix.inv_m_real.is_stale());
    let _ = w.state.slater_matrix.inv_m_real.as_slice();
}

#[test]
#[should_panic(expected = "stale inverse table read")]
fn stale_host_inverse_element_read_panics() {
    let (w, _) = device_run(&heisenberg());
    let _ = w.state.slater_matrix.inv_m_real.get(0, 0, 1);
}

/// The Pfaffian buffer is kept current by the stage (hop, exchange and recompute copies), so it
/// equals the CPU sampler's, for hopping and for exchange moves; the CPU tables are not stale.
#[test]
fn host_pfaffians_after_device_run_equal_cpu_sampler() {
    for c in [hubbard(), heisenberg()] {
        let (dev, cpu) = device_run(&c);
        assert!(!cpu.state.slater_matrix.inv_m_real.is_stale());
        let (a, b) = (
            &dev.state.slater_matrix.pf_m_real,
            &cpu.state.slater_matrix.pf_m_real,
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            // same implementation and operation order: identical up to rounding of the copy
            assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "pf {x} vs {y}");
        }
    }
}

/// Recomputing is the way back to a valid table: the next sampling call recomputes every plane
/// first, which clears the flag, and the tables match the CPU sampler's.
#[test]
fn recompute_after_device_run_makes_the_table_valid_again() {
    let c = hubbard();
    let (mut dev, mut cpu) = device_run(&c);
    assert!(dev.state.slater_matrix.inv_m_real.is_stale());
    vmc_make_sample_real(&dev.data, &mut dev.state, &mut dev.rng).unwrap();
    vmc_make_sample_real(&cpu.data, &mut cpu.state, &mut cpu.rng).unwrap();
    assert!(!dev.state.slater_matrix.inv_m_real.is_stale());
    let (a, b) = (
        dev.state.slater_matrix.inv_m_real.as_slice(),
        cpu.state.slater_matrix.inv_m_real.as_slice(),
    );
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-9 * y.abs().max(1.0), "inv {x} vs {y}");
    }
}
