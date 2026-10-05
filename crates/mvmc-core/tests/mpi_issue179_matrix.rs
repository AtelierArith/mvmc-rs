//! Rank-wise MPI optimization matrix against native C (issue #179).
//!
//! Cells: model (real, cmp, FSZ with NQPFull 1 and 2, OptTrans with NQPOptTrans 3)
//! x solver (direct NStore 0, direct NStore 1, SR-CG) x world size 2/4 x
//! NSplitSize 1/2/4, plus uneven and empty-work cells (3 saved samples on 4 ranks,
//! width-4 groups with NQPFull = 1). Each cell runs ONE optimization step of the
//! production entry point; the expected values are the instrumented native C
//! `vmc.out` (`c_toolbox/mpi_matrix_179/`, provenance in
//! `tests/fixtures/mpi_matrix_179/PROVENANCE.txt`). Ordinary tests never run C.
//!
//! Exact comparison, per rank (integers and the generator state are not floats):
//! `Counter[0..6]` after `ReduceCounter`, every saved `EleIdx`, and the complete
//! SFMT state (624 words and the cursor). Computed values (the reduced SR operands
//! `<HO>`, `<O>`, `<OO>` and the step energy) use explicit tolerances.
//!
//! C defects (documented, not reproduced): `NSplitSize > 1` with stored O
//! (`NStore != 0`) or SR-CG reads unwritten memory in `calculateOO_Store`
//! (`vmccal.c:314-318`, tmisawa/Julia-mVMC#61). Their `<OO>` is not compared with C;
//! Rust grouped stored `<OO>` is asserted against Rust grouped direct `<OO>`, and
//! grouped SR-CG must be rejected. Real-mode OptTrans derivative slots follow C's
//! mis-offset layout (#370); those slots are excluded as in the serial gate. PhysCal
//! native cells belong to #397.
//!
//! The always-running test checks invariants of the C fixtures themselves.
//! The explicit MPI gate runs with `mpiexec -n 2` and `-n 4`:
//! `MPI_ISSUE179_MATRIX_OUTPUT=<new dir> mpiexec -n N <binary> --ignored --exact
//! mpi_matrix::rank_wise_matrix_matches_native_c --nocapture`.

use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Clone)]
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
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

fn find<'a>(all: &'a [Cell], like: &Cell, solver: &str, width: usize) -> Option<&'a Cell> {
    all.iter().find(|c| {
        c.model == like.model
            && c.solver == solver
            && c.ranks == like.ranks
            && c.width == width
            && c.id.ends_with(like.id.rsplit('-').next().unwrap())
    })
}

fn max_scaled(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x.0 - y.0).abs().max((x.1 - y.1).abs())) / (1.0 + y.0.abs().max(y.1.abs())))
        .fold(0.0, f64::max)
}

#[test]
fn c_fixtures_have_the_group_and_solver_invariants_of_the_c_contract() {
    let all = cells();
    assert!(all.len() >= 80);
    let mut grouped_store_defects = 0;
    for cell in &all {
        let states: Vec<_> = (0..cell.ranks).map(|r| rank_state(cell, r)).collect();
        // Ranks of one group make identical draws, hence identical chains.
        for group in 0..cell.ranks / cell.width {
            for member in 1..cell.width {
                let (a, b) = (
                    &states[group * cell.width],
                    &states[group * cell.width + member],
                );
                assert_eq!(a.ele_idx, b.ele_idx, "{} group {group}", cell.id);
                assert_eq!(a.rng_words, b.rng_words, "{} group {group}", cell.id);
                assert_eq!(a.rng_index, b.rng_index, "{} group {group}", cell.id);
            }
            // Group g of a grouped run is the width-1 chain of rank g (seed + g).
            if cell.width > 1 {
                if let Some(serial) = find(&all, cell, &cell.solver, 1) {
                    let reference = rank_state(serial, group);
                    let leader = &states[group * cell.width];
                    assert_eq!(leader.ele_idx, reference.ele_idx, "{} g{group}", cell.id);
                    assert_eq!(
                        leader.rng_words, reference.rng_words,
                        "{} g{group}",
                        cell.id
                    );
                }
            }
        }
        // The solver does not influence step-0 sampling: even the C-defective
        // stored-O and CG cells have the d0 chain.
        if cell.solver != "d0" {
            if let Some(direct) = find(&all, cell, "d0", cell.width) {
                for (r, a) in states.iter().enumerate() {
                    let b = &rank_state(direct, r);
                    assert_eq!(a.ele_idx, b.ele_idx, "{} rank {r}", cell.id);
                    assert_eq!(a.rng_words, b.rng_words, "{} rank {r}", cell.id);
                }
            }
        }
        // <HO>, <O> and the energy are unaffected by the C stored-O defect; <OO> is.
        if cell.solver == "d1" && !cell.c_sr_error {
            let direct = find(&all, cell, "d0", cell.width).unwrap();
            let (e1, o1) = operands(cell);
            let (e0, o0) = operands(direct);
            assert!((e1 - e0).abs() <= 1e-14 * (1.0 + e0.abs()), "{}", cell.id);
            assert!(max_scaled(&o1["ho"], &o0["ho"]) <= 1e-13, "{}", cell.id);
            assert!(max_scaled(&o1["o"], &o0["o"]) <= 1e-13, "{}", cell.id);
            let oo = max_scaled(&o1["oo"], &o0["oo"]);
            if cell.width == 1 {
                assert!(oo <= 1e-13, "{} stored vs direct <OO> {oo:e}", cell.id);
            } else if oo > 1e-6 {
                grouped_store_defects += 1;
            }
        }
    }
    // C defect evidence (calloc'd zeros on this build): grouped stored <OO>
    // loses the samples of ranks after the first in every grouped stored cell.
    assert!(grouped_store_defects >= 8, "{grouped_store_defects}");
}

#[cfg(feature = "mpi")]
mod mpi_matrix {
    use super::*;
    use mvmc_core::{
        run_para_opt_from_namelist_observed, InitialDef, Reducer, RunConfig, VmcOptimizationState,
    };
    use num_complex::Complex64;
    use sfmt19937::Sfmt19937Rng;
    use std::path::Path;

    /// Rust vs C reduced operands. First divergence is the last-bit summation
    /// order of the sampled sums (<= ~2 ulp serially, docs/NUMERICAL_COMPARISONS.md);
    /// ABS covers exact-zero C entries holding roundoff.
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

    type Observed = (VmcOptimizationState, Sfmt19937Rng);

    fn run<R: Reducer + ?Sized>(
        cell: &Cell,
        dest: &Path,
        solver: &str,
        reducer: &R,
    ) -> Result<(Observed, bool), String> {
        let namelist = prepare(cell, dest, solver);
        let data =
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist, cell.opttrans)
                .map_err(|e| e.to_string())?;
        let all_complex = mvmc_core::get_all_complex_flag(&data)?;
        let mode = if data.i_flg_orbital_general != 0 {
            "fsz"
        } else if all_complex {
            "cmp"
        } else {
            "real"
        };
        let steps = cell
            .overrides
            .iter()
            .find(|(k, _)| k == "NSROptItrStep")
            .map_or(1, |(_, v)| v.parse().unwrap());
        let mut config = RunConfig::new(steps, mode);
        config.initial_def = InitialDef::None;
        config.output_dir = Some(dest.join("out"));
        config.enable_opt_trans = Some(cell.opttrans);
        let (_, state, rng) = run_para_opt_from_namelist_observed(&namelist, config, reducer)?;
        Ok(((state, rng), all_complex))
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

    fn check_cell(
        world: &mvmc_core::mpi::MpiContext,
        cell: &Cell,
        local: &Path,
        all: &[Cell],
    ) -> Result<(), String> {
        let group = world.split_groups(cell.width)?;
        let dest = local.join(&cell.id);
        let cg_grouped = cell.solver == "cg" && cell.width > 1;
        let outcome = run(cell, &dest, &cell.solver, &group);
        if cg_grouped {
            // C is undefined here (stored O read from unwritten memory): rejected.
            let error = outcome.err().ok_or("grouped SR-CG must be rejected")?;
            return if error.contains("undefined in mVMC C") {
                Ok(())
            } else {
                Err(format!("wrong rejection: {error}"))
            };
        }
        if cell.c_sr_error {
            // C reports "Error: StcOpt" for this tiny sample count; Rust must fail too.
            return match outcome {
                Err(error) if error.to_lowercase().contains("sr") || error.contains("solve") => {
                    Ok(())
                }
                Err(error) => Err(format!("unexpected error text: {error}")),
                Ok(_) => Err("C fails the SR solve here; Rust succeeded".into()),
            };
        }
        let ((state, rng), all_complex) = outcome?;
        let store_defect = cell.solver == "d1" && cell.width > 1;
        // Collective control run first, so a local mismatch cannot desynchronize ranks.
        let control = if store_defect {
            Some(run(
                cell,
                &local.join(format!("{}-direct", cell.id)),
                "d0",
                &group,
            )?)
        } else {
            None
        };
        // Exact: this rank's counters, saved configurations and generator state.
        let expected = rank_state(cell, world.rank());
        let counter = state.electron_config.counter[..6].to_vec();
        if counter != expected.counter {
            return Err(format!("counter {counter:?} vs C {:?}", expected.counter));
        }
        for (k, idx) in expected.ele_idx.iter().enumerate() {
            if state.electron_config.ele_idx_slice(k) != idx.as_slice() {
                return Err(format!("EleIdx sample {k} differs from C"));
            }
        }
        let (words, index) = rng.state_snapshot();
        if index != expected.rng_index || words.to_vec() != expected.rng_words {
            return Err(format!(
                "RNG state differs from C (cursor {index} vs {})",
                expected.rng_index
            ));
        }
        // Reduced operands on every rank (allreduce: duplicate-reduction check).
        let (energy, c_operands) = operands(cell);
        if !close(state.energy.etot.re, energy) {
            return Err(format!("energy {} vs C {energy}", state.energy.etot.re));
        }
        let size = state.sr_opt.sr_opt_size;
        for name in ["ho", "o", "oo"] {
            if name == "oo" && store_defect {
                continue;
            }
            // C's SROptO is the rank-local last-sample derivative (WeightAverageSROpt
            // reduces OO and HO only); C dumps rank 0's, so compare it on rank 0 only.
            if name == "o" && world.rank() != 0 {
                continue;
            }
            let mut actual = as_complex(&state, all_complex, name);
            if name == "oo" && all_complex && cell.solver == "d1" {
                // Rust's complex stored <OO> keeps the conjugate-transposed memory
                // layout of C's ZGEMM('N','C') result; the real parts, the only
                // part SR uses, agree (serial d0 and d1 trajectories are bit-identical).
                actual.iter_mut().for_each(|v| *v = v.conj());
            }
            compare_operand(cell, name, &actual, &c_operands[name], size)?;
        }
        if let Some(((state0, _), _)) = control {
            // Rust grouped stored <OO> equals Rust grouped direct <OO> (the C
            // value is wrong; vmccal.c:314-318).
            let direct = find(all, cell, "d0", cell.width).unwrap();
            let mut a = as_complex(&state, all_complex, "oo");
            if all_complex {
                a.iter_mut().for_each(|v| *v = v.conj());
            }
            let b = as_complex(&state0, all_complex, "oo");
            let pairs: Vec<_> = a.iter().map(|x| (x.re, x.im)).collect();
            let reference: Vec<_> = b.iter().map(|x| (x.re, x.im)).collect();
            let diff = max_scaled(&pairs, &reference);
            if diff > 1e-12 {
                return Err(format!(
                    "grouped stored vs direct <OO> {diff:e} ({})",
                    direct.id
                ));
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "explicit native 2/4-rank MPI matrix; see the module documentation"]
    fn rank_wise_matrix_matches_native_c() {
        let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
        let ranks = world.world_size();
        assert!(matches!(ranks, 2 | 4), "run with mpiexec -n 2 or 4");
        let root = PathBuf::from(std::env::var_os("MPI_ISSUE179_MATRIX_OUTPUT").unwrap());
        let setup = if world.is_root() {
            fs::create_dir(&root)
        } else {
            Ok(())
        };
        assert!(!world.any_failure(setup.is_err()), "{setup:?}");
        let local = root.join(format!("rank-{}", world.rank()));
        fs::create_dir(&local).unwrap();
        let all = cells();
        let mut failures = Vec::new();
        let mut checked = 0;
        for cell in all.iter().filter(|c| c.ranks == ranks) {
            let result = check_cell(&world, cell, &local, &all);
            let failed = world.any_failure(result.is_err());
            println!(
                "ISSUE179 rank={} cell={} ok={}",
                world.rank(),
                cell.id,
                result.is_ok()
            );
            if failed {
                failures.push(format!("{}: {:?}", cell.id, result));
            }
            checked += 1;
            world.barrier();
        }
        assert!(checked >= 30, "{checked}");
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Variant of a matrix cell with a longer run (several SR steps, 40 samples).
    fn long_variant(cell: &Cell, solver: &str, width: usize) -> Cell {
        let mut c = cell.clone();
        c.solver = solver.into();
        c.width = width;
        for (k, v) in &mut c.overrides {
            match k.as_str() {
                "NSROptItrStep" => *v = "4".into(),
                "NSROptItrSmp" => *v = "2".into(),
                "NVMCSample" => *v = "40".into(),
                "NSplitSize" => *v = width.to_string(),
                "NStore" => *v = "0".into(),
                "NSRCG" => *v = if solver == "cg" { "1" } else { "0" }.into(),
                _ => {}
            }
        }
        c
    }

    fn scaled_file_difference(a: &Path, b: &Path) -> f64 {
        let read = |p: &Path| -> Vec<f64> {
            fs::read_to_string(p)
                .unwrap()
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect()
        };
        let (x, y) = (read(a), read(b));
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(&y)
            .map(|(u, v)| (u - v).abs() / (1.0 + v.abs()))
            .fold(0.0, f64::max)
    }

    /// Beyond step 1 (SR trajectories are ill-conditioned, #358): a four-step run
    /// is bitwise repeatable on every rank, and one group equals the serial Rust
    /// chain of the same seed to explicit tolerances. SR-CG is only repeatable.
    #[test]
    #[ignore = "explicit native 2/4-rank MPI matrix; see the module documentation"]
    fn multi_step_runs_are_repeatable_and_one_group_equals_serial() {
        let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
        let ranks = world.world_size();
        assert!(matches!(ranks, 2 | 4), "run with mpiexec -n 2 or 4");
        let root = PathBuf::from(std::env::var_os("MPI_ISSUE179_MATRIX_OUTPUT").unwrap());
        let setup = if world.is_root() {
            fs::create_dir(&root)
        } else {
            Ok(())
        };
        assert!(!world.any_failure(setup.is_err()), "{setup:?}");
        let local = root.join(format!("rank-{}", world.rank()));
        fs::create_dir(&local).unwrap();
        let all = cells();
        let mut failures = Vec::new();
        for model in ["real", "cmp", "fsz2", "ot"] {
            let base = all
                .iter()
                .find(|c| c.model == model && c.ranks == ranks && c.solver == "d0")
                .unwrap();
            for (solver, width) in [("d0", 1), ("d0", ranks), ("cg", 1)] {
                let cell = long_variant(base, solver, width);
                let result = (|| -> Result<(), String> {
                    let group = world.split_groups(width)?;
                    let first = run(
                        &cell,
                        &local.join(format!("{model}-{solver}-{width}-a")),
                        solver,
                        &group,
                    )?;
                    let second = run(
                        &cell,
                        &local.join(format!("{model}-{solver}-{width}-b")),
                        solver,
                        &group,
                    )?;
                    let ((s1, r1), _) = first;
                    let ((s2, r2), _) = second;
                    if r1.state_snapshot() != r2.state_snapshot()
                        || s1.electron_config.counter != s2.electron_config.counter
                        || s1.energy.etot != s2.energy.etot
                    {
                        return Err("same-input repeat differs".into());
                    }
                    if world.is_root() {
                        let a = local.join(format!("{model}-{solver}-{width}-a/out/zvo_out.dat"));
                        let b = local.join(format!("{model}-{solver}-{width}-b/out/zvo_out.dat"));
                        if fs::read(&a).unwrap() != fs::read(&b).unwrap() {
                            return Err("zvo_out.dat repeat differs".into());
                        }
                    }
                    Ok(())
                })();
                let mut serial_result: Result<(), String> = Ok(());
                if solver == "d0" && width == ranks {
                    // Collective part done; root alone compares with a serial chain.
                    let group_out =
                        local.join(format!("{model}-{solver}-{width}-a/out/zvo_out.dat"));
                    if world.is_root() {
                        let mut serial = cell.clone();
                        serial.width = 1;
                        for (k, v) in &mut serial.overrides {
                            if k == "NSplitSize" {
                                *v = "1".into();
                            }
                        }
                        serial_result = run(
                            &serial,
                            &local.join(format!("{model}-serial")),
                            solver,
                            &mvmc_core::SingleProcessReducer,
                        )
                        .and_then(|_| {
                            // Observed grouped-vs-serial maxima: <= 7e-10 (FSZ), <= 3e-13 else.
                            let diff = scaled_file_difference(
                                &group_out,
                                &local.join(format!("{model}-serial/out/zvo_out.dat")),
                            );
                            if diff <= 1e-8 {
                                Ok(())
                            } else {
                                Err(format!("grouped vs serial zvo_out scaled diff {diff:e}"))
                            }
                        });
                    }
                }
                let combined = result.and(serial_result);
                if world.any_failure(combined.is_err()) {
                    failures.push(format!("{model} {solver} w{width}: {combined:?}"));
                }
                world.barrier();
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
