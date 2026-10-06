//! Multi-walker optimization against the native-C ungrouped MPI matrix (issue #435).
//!
//! `W` walkers of `run_para_opt_multichain` are the `W` ranks of an ungrouped C run
//! (`NSplitSize = 1`): walker `w` is seeded `RndSeed + w`, `HO`/`OO`/energy are summed over the
//! walkers and every walker applies the same SR update. The expected values are the instrumented
//! native C `vmc.out` outputs of `tests/fixtures/mpi_matrix_179` (world 2 and 4, width 1; see its
//! README and PROVENANCE) and, for the two-group equivalence, the grouped cells (4 ranks,
//! `NSplitSize = 2`: two chains, seeds 0 and 1). No MPI, C or Julia is run.
//!
//! Exact per walker: `Counter[0..6]`, every saved `EleIdx`, the complete SFMT state. Reduced
//! operands and the step energy: `|a - b| <= 1e-13 + 1e-12 |b|` (last-bit summation order;
//! the Rust reduction is a rank-order left fold, `MPI_Allreduce` leaves the order to the MPI
//! library). Beyond step 1 (ill-conditioned SR, #358): repeatability and `W = 1` equals serial.
//! C defects handled as in the MPI matrix: solver `d1`/`cg` ungrouped cells are well defined
//! (width 1); SR failure cells (`c_sr_error`) must fail in Rust too.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use mvmc_core::multichain::{
    run_para_opt_multichain, ParaOptMultiChainConfig, ParaOptWalker, ThreadComm, ThreadReducer,
};
use mvmc_core::{InitialDef, Reducer, RunConfig, VmcOptimizationState};
use num_complex::Complex64;

#[derive(Clone)]
#[allow(dead_code)]
struct Cell {
    id: String,
    dir: String,
    opttrans: bool,
    model: String,
    solver: String,
    ranks: usize,
    width: usize,
    c_sr_error: bool,
    overrides: Vec<(String, String)>,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mpi_matrix_179")
}

fn cells() -> Vec<Cell> {
    fs::read_to_string(fixture_root().join("cells.txt"))
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<_> = line.split_whitespace().collect();
            assert_eq!(f.len(), 10, "{line}");
            Cell {
                id: f[0].into(),
                dir: f[1].into(),
                opttrans: f[2] == "1",
                model: f[3].into(),
                solver: f[4].into(),
                ranks: f[5].parse().unwrap(),
                width: f[6].parse().unwrap(),
                c_sr_error: f[8] == "1",
                overrides: f[9]
                    .split(',')
                    .map(|kv| {
                        let (k, v) = kv.split_once('=').unwrap();
                        (k.to_owned(), v.to_owned())
                    })
                    .collect(),
            }
        })
        .collect()
}

/// One rank's sampler state after the step, as dumped by C.
#[derive(Debug, PartialEq, Eq, Clone)]
struct RankState {
    counter: Vec<i64>,
    ele_idx: Vec<Vec<i64>>,
    rng_index: usize,
    rng_words: Vec<u32>,
}

fn rank_state(cell: &Cell, rank: usize) -> RankState {
    let text = fs::read_to_string(
        fixture_root()
            .join(&cell.id)
            .join(format!("rank{rank}.txt")),
    )
    .unwrap();
    let mut state = RankState {
        counter: vec![],
        ele_idx: vec![],
        rng_index: 0,
        rng_words: vec![],
    };
    for line in text.lines() {
        let mut tokens = line.split_whitespace();
        match tokens.next().unwrap() {
            "COUNTER" => {
                state.counter = tokens.skip(2).map(|t| t.parse().unwrap()).collect();
            }
            "ELEIDX" => {
                state
                    .ele_idx
                    .push(tokens.skip(3).map(|t| t.parse().unwrap()).collect());
            }
            "RNG" => {
                let mut tokens = tokens.skip(2);
                state.rng_index = tokens
                    .next()
                    .unwrap()
                    .strip_prefix("idx=")
                    .unwrap()
                    .parse()
                    .unwrap();
                state.rng_words = tokens.map(|t| t.parse().unwrap()).collect();
            }
            other => panic!("unexpected record {other}"),
        }
    }
    assert_eq!(state.counter.len(), 6);
    assert_eq!(state.rng_words.len(), 624);
    state
}

/// Reduced operands: name -> (re, im) pairs, plus the step energy.
fn operands(cell: &Cell) -> (f64, BTreeMap<String, Vec<(f64, f64)>>) {
    let text = fs::read_to_string(fixture_root().join(&cell.id).join("operands.txt")).unwrap();
    let mut energy = 0.0;
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let (name, rest) = line.split_once(' ').unwrap();
        let values: Vec<f64> = rest
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        if name == "energy" {
            energy = values[0];
        } else {
            map.insert(
                name.to_owned(),
                values.chunks(2).map(|c| (c[0], c[1])).collect(),
            );
        }
    }
    (energy, map)
}

#[allow(dead_code)]
fn max_scaled(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x.0 - y.0).abs().max((x.1 - y.1).abs())) / (1.0 + y.0.abs().max(y.1.abs())))
        .fold(0.0, f64::max)
}

const ABS: f64 = 1e-13;
const REL: f64 = 1e-12;

fn prepare(cell: &Cell, dest: &Path, solver: &str) -> PathBuf {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(&cell.dir)
        .join("inputs");
    fs::create_dir_all(dest).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), dest.join(entry.file_name())).unwrap();
    }
    let modpara = dest.join("modpara.def");
    let mut lines: Vec<String> = fs::read_to_string(&modpara)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let mut overrides = cell.overrides.clone();
    if solver != cell.solver {
        // Same cell with the direct solver (stored-vs-direct control).
        assert_eq!(solver, "d0");
        for (k, v) in &mut overrides {
            match k.as_str() {
                "NStore" | "NSRCG" => *v = "0".into(),
                _ => {}
            }
        }
    }
    for (key, value) in overrides {
        let line = lines
            .iter_mut()
            .find(|line| line.split_whitespace().next() == Some(key.as_str()))
            .unwrap_or_else(|| panic!("{key} missing from modpara.def"));
        *line = format!("{key} {value}");
    }
    fs::write(&modpara, lines.join("\n") + "\n").unwrap();
    dest.join("namelist.def")
}

struct Prepared {
    namelist: PathBuf,
    run: RunConfig,
    all_complex: bool,
}

fn setup(cell: &Cell, dest: &Path, steps_override: Option<(i64, i64, i64)>) -> Prepared {
    let namelist = prepare(cell, dest, &cell.solver);
    if let Some((steps, smp, samples)) = steps_override {
        let modpara = dest.join("modpara.def");
        let mut lines: Vec<String> = fs::read_to_string(&modpara)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        for (key, value) in [
            ("NSROptItrStep", steps),
            ("NSROptItrSmp", smp),
            ("NVMCSample", samples),
        ] {
            let line = lines
                .iter_mut()
                .find(|l| l.split_whitespace().next() == Some(key))
                .unwrap();
            *line = format!("{key} {value}");
        }
        fs::write(&modpara, lines.join("\n") + "\n").unwrap();
    }
    let data =
        mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist, cell.opttrans)
            .unwrap();
    let all_complex = mvmc_core::get_all_complex_flag(&data).unwrap();
    let mode = if data.i_flg_orbital_general != 0 {
        "fsz"
    } else if all_complex {
        "cmp"
    } else {
        "real"
    };
    let steps = steps_override.map_or_else(
        || {
            cell.overrides
                .iter()
                .find(|(k, _)| k == "NSROptItrStep")
                .map_or(1, |(_, v)| v.parse().unwrap())
        },
        |(s, _, _)| s,
    );
    let mut run = RunConfig::new(steps, mode);
    run.initial_def = InitialDef::None;
    run.output_dir = Some(dest.join("out"));
    run.enable_opt_trans = Some(cell.opttrans);
    Prepared {
        namelist,
        run,
        all_complex,
    }
}

fn multi(p: &Prepared, walkers: usize) -> Result<Vec<ParaOptWalker>, String> {
    run_para_opt_multichain(&ParaOptMultiChainConfig {
        namelist: p.namelist.clone(),
        run: p.run.clone(),
        group_base: 0,
        walkers,
    })
}

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= ABS + REL * b.abs()
}

/// Compare one array against the C dump (leading elements use C's layout).
fn compare_operand(
    cell: &Cell,
    name: &str,
    actual: &[Complex64],
    expected: &[(f64, f64)],
    size: usize,
) -> Result<(), String> {
    if actual.len() < expected.len() {
        return Err(format!("{name}: {} < {}", actual.len(), expected.len()));
    }
    // Real-mode OptTrans derivative slots are mis-offset in C (#370).
    let width = if cell.model == "cmp" || cell.model.starts_with("fsz") {
        2 * size
    } else {
        size
    };
    let skip = |i: usize| {
        cell.opttrans && {
            let matrix = expected.len() == width * width;
            (i % width >= width - 2) || (matrix && i / width >= width - 2)
        }
    };
    for (i, (c, a)) in expected.iter().zip(actual).enumerate() {
        if skip(i) {
            continue;
        }
        if !close(a.re, c.0) || !close(a.im, c.1) {
            return Err(format!("{name}[{i}]: Rust {a} vs C ({}, {})", c.0, c.1));
        }
    }
    Ok(())
}

fn as_complex(state: &VmcOptimizationState, all_complex: bool, name: &str) -> Vec<Complex64> {
    let s = &state.sr_opt;
    let real = |v: &[f64]| v.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    match (all_complex, name) {
        (true, "ho") => s.sr_opt_ho.clone(),
        (true, "o") => s.sr_opt_o.clone(),
        (true, "oo") => s.sr_opt_oo.clone(),
        (false, "ho") => real(&s.sr_opt_ho_real),
        (false, "o") => real(&s.sr_opt_o_real),
        (false, "oo") => real(&s.sr_opt_oo_real),
        _ => unreachable!(),
    }
}

/// The SR backend override is process-wide: serialize the tests of this file.
fn lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mvmc-435-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Check one ungrouped C cell (`ranks` walkers) against the C fixture, walker by walker.
fn check_cell(cell: &Cell) -> Result<(), String> {
    let dir = scratch(&cell.id);
    let prepared = setup(cell, &dir, None);
    let outcome = multi(&prepared, cell.ranks);
    let result = if cell.c_sr_error {
        // C reports "Error: StcOpt" for this tiny sample count; Rust must fail too.
        match outcome {
            Err(error) if error.to_lowercase().contains("sr") || error.contains("solve") => Ok(()),
            Err(error) => Err(format!("unexpected error text: {error}")),
            Ok(_) => Err("C fails the SR solve here; Rust succeeded".into()),
        }
    } else {
        outcome.and_then(|walkers| check_walkers(cell, &walkers, prepared.all_complex))
    };
    let _ = fs::remove_dir_all(&dir);
    result
}

fn check_walkers(cell: &Cell, walkers: &[ParaOptWalker], all_complex: bool) -> Result<(), String> {
    if walkers.len() != cell.ranks {
        return Err("walker count".into());
    }
    let (energy, c_operands) = operands(cell);
    for walker in walkers {
        let state = &walker.state;
        let expected = rank_state(cell, walker.walker);
        let counter = state.electron_config.counter[..6].to_vec();
        if counter != expected.counter {
            return Err(format!(
                "walker {} counter {counter:?} vs C {:?}",
                walker.walker, expected.counter
            ));
        }
        for (k, idx) in expected.ele_idx.iter().enumerate() {
            if state.electron_config.ele_idx_slice(k) != idx.as_slice() {
                return Err(format!(
                    "walker {} EleIdx sample {k} differs from C",
                    walker.walker
                ));
            }
        }
        let (words, index) = walker.rng.state_snapshot();
        if index != expected.rng_index || words.to_vec() != expected.rng_words {
            return Err(format!("walker {} RNG state differs from C", walker.walker));
        }
        if !close(state.energy.etot.re, energy) {
            return Err(format!("energy {} vs C {energy}", state.energy.etot.re));
        }
        let size = state.sr_opt.sr_opt_size;
        for name in ["ho", "o", "oo"] {
            // C's SROptO is the rank-local last-sample derivative; C dumps rank 0's.
            if name == "o" && walker.walker != 0 {
                continue;
            }
            let mut actual = as_complex(state, all_complex, name);
            if name == "oo" && all_complex && cell.solver == "d1" {
                actual.iter_mut().for_each(|v| *v = v.conj());
            }
            compare_operand(cell, name, &actual, &c_operands[name], size)
                .map_err(|e| format!("walker {}: {e}", walker.walker))?;
        }
    }
    Ok(())
}

#[test]
fn walkers_match_native_c_ungrouped_cells_two_ranks() {
    let _guard = lock();
    let mut checked = 0;
    for cell in cells().iter().filter(|c| c.width == 1 && c.ranks == 2) {
        check_cell(cell).unwrap_or_else(|e| panic!("{}: {e}", cell.id));
        checked += 1;
    }
    assert!(checked >= 15, "{checked}");
}

#[test]
fn walkers_match_native_c_ungrouped_cells_four_ranks() {
    let _guard = lock();
    let mut checked = 0;
    for cell in cells().iter().filter(|c| c.width == 1 && c.ranks == 4) {
        check_cell(cell).unwrap_or_else(|e| panic!("{}: {e}", cell.id));
        checked += 1;
    }
    assert!(checked >= 15, "{checked}");
}

/// Two walkers equal the two groups of the 4-rank, `NSplitSize = 2` C run (chains with seeds 0
/// and 1; the intra-group split only changes the reduction order).
#[test]
fn two_walkers_equal_the_two_groups_of_the_four_rank_split_two_c_cells() {
    let _guard = lock();
    let all = cells();
    let mut checked = 0;
    for grouped in all
        .iter()
        .filter(|c| c.ranks == 4 && c.width == 2 && c.solver == "d0" && !c.c_sr_error)
    {
        let dir = scratch(&grouped.id);
        let mut two = grouped.clone();
        two.ranks = 2;
        two.width = 1;
        for (k, v) in &mut two.overrides {
            if k == "NSplitSize" {
                *v = "1".into();
            }
        }
        let prepared = setup(&two, &dir, None);
        let walkers = multi(&prepared, 2).unwrap();
        let (energy, c_operands) = operands(grouped);
        let state = &walkers[0].state;
        assert!(
            close(state.energy.etot.re, energy),
            "{}: energy",
            grouped.id
        );
        let size = state.sr_opt.sr_opt_size;
        for name in ["ho", "oo"] {
            let actual = as_complex(state, prepared.all_complex, name);
            compare_operand(&two, name, &actual, &c_operands[name], size)
                .unwrap_or_else(|e| panic!("{}: {e}", grouped.id));
        }
        // each group leader of the C run has the chain of the matching walker
        #[allow(clippy::needless_range_loop)]
        for g in 0..2 {
            let leader = rank_state(grouped, g * 2);
            let walker = &walkers[g].state;
            for (k, idx) in leader.ele_idx.iter().enumerate() {
                assert_eq!(
                    walker.electron_config.ele_idx_slice(k),
                    idx.as_slice(),
                    "{} g{g}",
                    grouped.id
                );
            }
            assert_eq!(
                walkers[g].rng.state_snapshot().0.to_vec(),
                leader.rng_words,
                "{} g{g}",
                grouped.id
            );
        }
        let _ = fs::remove_dir_all(&dir);
        checked += 1;
    }
    assert!(checked >= 4, "{checked}");
}

/// `W = 1` is the serial optimization, byte for byte (output files, state, RNG).
#[test]
fn single_walker_is_byte_identical_to_the_serial_optimization() {
    let _guard = lock();
    let all = cells();
    for model in ["real", "cmp", "fsz2", "ot"] {
        let base = all
            .iter()
            .find(|c| c.model == model && c.ranks == 2 && c.width == 1 && c.solver == "d0")
            .unwrap();
        let dir_s = scratch(&format!("serial-{model}"));
        let dir_m = scratch(&format!("multi-{model}"));
        let steps = Some((4, 2, 40));
        let serial = setup(base, &dir_s, steps);
        let multi_p = setup(base, &dir_m, steps);
        let (s_sum, s_state, s_rng) = mvmc_core::run_para_opt_from_namelist_observed(
            &serial.namelist,
            serial.run.clone(),
            &mvmc_core::SingleProcessReducer,
        )
        .unwrap();
        let walkers = multi(&multi_p, 1).unwrap();
        let w = &walkers[0];
        assert_eq!(
            w.rng.state_snapshot(),
            s_rng.state_snapshot(),
            "{model} rng"
        );
        assert_eq!(
            format!("{:?}", w.state),
            format!("{:?}", s_state),
            "{model} state"
        );
        assert_eq!(w.summary.zvo_first_n, s_sum.zvo_first_n, "{model} summary");
        assert!(!s_sum.zvo_first_n.is_empty());
        for entry in fs::read_dir(dir_s.join("out")).unwrap() {
            let entry = entry.unwrap();
            // wall-clock timing files contain timestamps; everything else must be identical
            if entry.file_name().to_string_lossy().contains("time") {
                continue;
            }
            let a = fs::read(entry.path()).unwrap();
            let b = fs::read(dir_m.join("out").join(entry.file_name())).unwrap();
            assert_eq!(a, b, "{model}: {:?}", entry.file_name());
        }
        let _ = fs::remove_dir_all(&dir_s);
        let _ = fs::remove_dir_all(&dir_m);
    }
}

/// Beyond step 1: a four-step multi-walker run is bitwise repeatable (thread scheduling does
/// not change the rank-ordered reductions), walkers agree on the reduced output, and the
/// reduced SR output of two walkers is close to the serial chain of the same seed in the
/// well-conditioned first step only (checked above against C).
#[test]
fn multi_step_multi_walker_runs_are_repeatable_and_walkers_agree() {
    let _guard = lock();
    let all = cells();
    for model in ["real", "cmp", "fsz2", "ot"] {
        let base = all
            .iter()
            .find(|c| c.model == model && c.ranks == 4 && c.width == 1 && c.solver == "d0")
            .unwrap();
        let mut outputs = Vec::new();
        for run in 0..2 {
            let dir = scratch(&format!("rep-{model}-{run}"));
            let prepared = setup(base, &dir, Some((4, 2, 40)));
            let walkers = multi(&prepared, 4).unwrap();
            // the reduced energy is identical on every walker (allreduce); only the output
            // root reads the output files for its summary
            for w in &walkers[1..] {
                assert_eq!(w.state.energy.etot, walkers[0].state.energy.etot, "{model}");
                assert!(
                    w.summary.zvo_first_n.is_empty(),
                    "{model}: non-root summary"
                );
            }
            assert_eq!(walkers[0].summary.zvo_first_n.len(), 4, "{model}");
            let zvo = fs::read(dir.join("out/zvo_out.dat")).unwrap();
            let rngs: Vec<_> = walkers.iter().map(|w| w.rng.state_snapshot()).collect();
            let energies: Vec<_> = walkers.iter().map(|w| w.state.energy.etot).collect();
            outputs.push((zvo, rngs, energies));
            let _ = fs::remove_dir_all(&dir);
        }
        assert_eq!(outputs[0], outputs[1], "{model}: repeat differs");
    }
}

/// The thread reducer implements the collective contract.
#[test]
fn thread_reducer_collectives_follow_the_mpi_contract() {
    let comm = ThreadComm::new(3);
    let handles: Vec<_> = (0..3)
        .map(|rank| {
            let reducer = ThreadReducer::new(comm.clone(), rank, 5);
            std::thread::spawn(move || {
                let mut f = [rank as f64 + 0.5, 1.0];
                reducer.allreduce_sum_f64(&mut f);
                let mut c = [Complex64::new(rank as f64, -(rank as f64))];
                reducer.allreduce_sum_c64(&mut c);
                let mut i = [rank as i64 + 1];
                reducer.allreduce_sum_i64(&mut i);
                let mut b = [rank as i64 * 10];
                reducer.broadcast_i64(1, &mut b).unwrap();
                let max = reducer.sampling_max_info(rank as i32 - 1).unwrap();
                reducer.barrier();
                (
                    f,
                    c,
                    i,
                    b,
                    max,
                    reducer.rank(),
                    reducer.world_size(),
                    reducer.seed_offset(),
                )
            })
        })
        .collect();
    for (rank, h) in handles.into_iter().enumerate() {
        let (f, c, i, b, max, r, size, seed) = h.join().unwrap();
        assert_eq!(f, [(0.5 + 1.5) + 2.5, 3.0]);
        assert_eq!(c[0], Complex64::new(3.0, -3.0));
        assert_eq!(i[0], 6);
        assert_eq!(b[0], 10);
        assert_eq!(max, 1);
        assert_eq!((r, size, seed), (rank, 3, 5 + rank));
    }
}

/// Optional criterion of #435: the SR assembly through the tenferro backend (#421) with
/// walkers. Sampling (counters, configurations, RNG) and the reduced operands are independent
/// of the SR solver at step 1, so the C comparison holds unchanged; with `W = 1` the tenferro
/// path equals the serial tenferro run byte for byte.
#[test]
fn multi_walker_optimization_runs_through_the_tenferro_sr_backend() {
    use mvmc_core::sr_backend::{set_sr_backend_override, SrBackendKind};
    let _guard = lock();
    set_sr_backend_override(Some(SrBackendKind::Tenferro));
    let result = std::panic::catch_unwind(|| {
        let all = cells();
        for cell in all
            .iter()
            .filter(|c| c.width == 1 && c.ranks == 2 && c.solver == "d0")
        {
            check_cell(cell).unwrap_or_else(|e| panic!("{} (tenferro SR): {e}", cell.id));
        }
        let base = all
            .iter()
            .find(|c| c.model == "real" && c.ranks == 2 && c.width == 1 && c.solver == "d0")
            .unwrap();
        let dir_s = scratch("tf-serial");
        let dir_m = scratch("tf-multi");
        let serial = setup(base, &dir_s, Some((4, 2, 40)));
        let multi_p = setup(base, &dir_m, Some((4, 2, 40)));
        let (_, s_state, s_rng) = mvmc_core::run_para_opt_from_namelist_observed(
            &serial.namelist,
            serial.run.clone(),
            &mvmc_core::SingleProcessReducer,
        )
        .unwrap();
        let walkers = multi(&multi_p, 1).unwrap();
        assert_eq!(walkers[0].rng.state_snapshot(), s_rng.state_snapshot());
        assert_eq!(format!("{:?}", walkers[0].state), format!("{:?}", s_state));
        // two walkers: well-defined result, repeatable
        let a = multi(&setup(base, &scratch("tf-a"), Some((4, 2, 40))), 2).unwrap();
        let b = multi(&setup(base, &scratch("tf-b"), Some((4, 2, 40))), 2).unwrap();
        assert_eq!(a[0].state.energy.etot, b[0].state.energy.etot);
        assert_eq!(a[1].rng.state_snapshot(), b[1].rng.state_snapshot());
    });
    set_sr_backend_override(None);
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
