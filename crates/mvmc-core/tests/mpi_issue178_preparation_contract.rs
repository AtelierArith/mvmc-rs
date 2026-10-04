//! Real public parse/load failures, not a FaultReducer or C malformed-fscanf claim.
//! PhysCal fixed load is preseed; OPT initial.def load is after initialization.
//! Native gates use externally supplied exclusive output roots and bounded timeouts.
use mvmc_core::{prepare_phys_cal_from_namelist_with_reducer, Reducer, SingleProcessReducer};
#[cfg(feature = "mpi")]
use mvmc_core::{run_para_opt_from_namelist_with_reducer, InitialDef, RunConfig};
use num_complex::Complex64;
use std::{
    cell::Cell,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Observer<'a> {
    inner: &'a dyn Reducer,
    broadcasts: Cell<usize>,
    reductions: Cell<usize>,
}
impl<'a> Observer<'a> {
    fn new(inner: &'a dyn Reducer) -> Self {
        Self {
            inner,
            broadcasts: Cell::new(0),
            reductions: Cell::new(0),
        }
    }
}
impl Reducer for Observer<'_> {
    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        self.inner.sampling_max_info(info)
    }
    fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        self.broadcasts.set(self.broadcasts.get() + 1);
        self.inner.broadcast_i64(root, values)
    }
    fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
        self.broadcasts.set(self.broadcasts.get() + 1);
        self.inner.broadcast_f64(root, values)
    }
    fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
        self.broadcasts.set(self.broadcasts.get() + 1);
        self.inner.broadcast_c64(root, values);
    }
    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        self.reductions.set(self.reductions.get() + 1);
        self.inner.allreduce_sum_i64(values);
    }
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        self.reductions.set(self.reductions.get() + 1);
        self.inner.allreduce_sum_f64(values);
    }
    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        self.reductions.set(self.reductions.get() + 1);
        self.inner.allreduce_sum_c64(values);
    }
    fn any_failure(&self, failed: bool) -> bool {
        self.inner.any_failure(failed)
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn reduction_size(&self) -> usize {
        self.inner.reduction_size()
    }
    fn supports_grouped_sampling(&self) -> bool {
        self.inner.supports_grouped_sampling()
    }
    fn seed_offset(&self) -> usize {
        self.inner.seed_offset()
    }
    fn is_output_root(&self) -> bool {
        self.inner.is_output_root()
    }
}

fn copy_input(parent: &std::path::Path) -> (PathBuf, PathBuf) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let input = parent.join("inputs");
    fs::create_dir(&input).unwrap();
    for entry in fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), input.join(entry.file_name())).unwrap();
    }
    let fixed = parent.join("fixed.dat");
    fs::copy(source.join("zqp_opt.dat"), &fixed).unwrap();
    // OPT cannot consume the PhysCal-only TwoBodyGEx section. The remaining
    // complete standard input and anti-periodic signs are unchanged.
    let original = fs::read_to_string(input.join("namelist.def")).unwrap();
    let opt_modpara = input.join("opt-modpara.def");
    let opt_settings = fs::read_to_string(input.join("modpara.def")).unwrap();
    fs::write(
        &opt_modpara,
        format!(
            "{}\nNVMCCalMode 0\n",
            opt_settings
                .lines()
                .filter(|line| line.split_whitespace().next() != Some("NVMCCalMode"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    )
    .unwrap();
    fs::write(
        input.join("opt.def"),
        original
            .lines()
            .filter(|line| {
                !line
                    .split_whitespace()
                    .next()
                    .is_some_and(|s| s == "TwoBodyGEx")
            })
            .map(|line| {
                if line.split_whitespace().next() == Some("ModPara") {
                    "ModPara opt-modpara.def"
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    (input.join("namelist.def"), fixed)
}

fn owned_dir() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    loop {
        let path = std::env::temp_dir().join(format!(
            "issue178-preparation-{}-{}",
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

#[test]
fn public_preparation_positive_and_malformed_fixed_controls_are_preseed() {
    let parent = owned_dir();
    let (input, fixed) = copy_input(&parent);
    let positive = Observer::new(&SingleProcessReducer);
    let prepared =
        prepare_phys_cal_from_namelist_with_reducer(&input, &fixed, "real", Some(11272), &positive)
            .unwrap();
    // An explicit seed bypasses broadcast; the returned RNG proves seeding.
    assert_eq!(
        prepared.rng.state_snapshot(),
        sfmt19937::Sfmt19937Rng::new(11272).state_snapshot()
    );
    assert_eq!(prepared.rng.words_consumed(), 0);
    for (name, contents, diagnostic) in [
        ("short", "0 0 0 0 0 0", "too short"),
        ("token", "0 0 invalid 0 0 0", "non-numeric token"),
    ] {
        let bad = parent.join(name);
        fs::write(&bad, contents).unwrap();
        let observer = Observer::new(&SingleProcessReducer);
        let error = prepare_phys_cal_from_namelist_with_reducer(
            &input,
            &bad,
            "real",
            Some(11272),
            &observer,
        )
        .unwrap_err();
        assert!(error.contains(diagnostic), "{error}");
        assert_eq!(observer.broadcasts.get(), 0);
        assert_eq!(observer.reductions.get(), 0);
        assert_eq!(fs::read_to_string(&bad).unwrap(), contents);
    }
    fs::remove_dir_all(parent).unwrap(); // Exclusively created by this invocation.
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "explicit native 2/4 rank preparation matrix with external timeout"]
fn rank_local_parse_load_errors_stop_before_seed_and_output() {
    let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let root = PathBuf::from(std::env::var_os("MPI_ISSUE178_PREPARATION_OUTPUT").unwrap());
    let setup = if world.is_root() {
        fs::create_dir(&root)
    } else {
        Ok(())
    };
    assert!(
        !world.any_failure(setup.is_err()),
        "exclusive setup: {setup:?}"
    );
    // Each rank owns its files: filesystem corruption cannot accidentally
    // corrupt the peer's valid input or hide a local-error propagation defect.
    let local = root.join(format!("rank-{}", world.rank()));
    fs::create_dir(&local).unwrap();
    let (input, fixed) = copy_input(&local);
    fs::write(local.join("sentinel"), b"untouched").unwrap();
    let before_input = fs::read(&input).unwrap();
    let before_fixed = fs::read(&fixed).unwrap();
    let short = local.join("short.dat");
    fs::write(&short, "0 0 0 0 0 0").unwrap();
    let token = local.join("token.dat");
    fs::write(&token, "0 0 bad 0 0 0").unwrap();
    let missing = local.join("missing.def");
    let overlay = input.parent().unwrap().join("overlay.def");
    fs::write(
        &overlay,
        format!(
            "{}\nInGutzwiller invalid-utf8.def\n",
            fs::read_to_string(&input).unwrap()
        ),
    )
    .unwrap();
    let opt_overlay = input.parent().unwrap().join("opt-overlay.def");
    fs::write(
        &opt_overlay,
        format!(
            "{}\nInGutzwiller invalid-utf8.def\n",
            fs::read_to_string(input.parent().unwrap().join("opt.def")).unwrap()
        ),
    )
    .unwrap();
    let invalid_utf8 = input.parent().unwrap().join("invalid-utf8.def");
    fs::write(&invalid_utf8, [0xff]).unwrap();
    for width in [1, 2] {
        let group = world.split_groups(width).unwrap();
        // Positive public preparation must reach seed before negative controls.
        let positive = Observer::new(&group);
        let prepared = prepare_phys_cal_from_namelist_with_reducer(
            &input,
            &fixed,
            "real",
            Some(11272),
            &positive,
        )
        .unwrap();
        assert_eq!(
            prepared.rng.state_snapshot(),
            sfmt19937::Sfmt19937Rng::new(11272 + group.seed_offset() as u32).state_snapshot()
        );
        assert_eq!(prepared.rng.words_consumed(), 0);
        for bad_rank in [0, world.world_size() - 1] {
            for case in [
                "phys-parse",
                "phys-fixed",
                "phys-short",
                "phys-token",
                "opt-parse",
                "opt-initial",
                "phys-overlay",
                "opt-overlay",
            ] {
                let observer = Observer::new(&group);
                let bad = world.rank() == bad_rank;
                let output = local.join(format!("out-{width}-{bad_rank}-{case}"));
                let result = if case.starts_with("phys") {
                    let name = if bad && case == "phys-parse" {
                        &missing
                    } else if bad && case == "phys-overlay" {
                        &overlay
                    } else {
                        &input
                    };
                    let params = if bad {
                        match case {
                            "phys-fixed" => &missing,
                            "phys-short" => &short,
                            "phys-token" => &token,
                            _ => &fixed,
                        }
                    } else {
                        &fixed
                    };
                    prepare_phys_cal_from_namelist_with_reducer(
                        name,
                        params,
                        "real",
                        Some(11272),
                        &observer,
                    )
                    .map(|_| ())
                } else {
                    let mut config = RunConfig::new(1, "real");
                    config.nsmp = Some(1);
                    config.seed = Some(11272);
                    config.output_dir = Some(output.clone());
                    config.initial_def = if bad && case == "opt-initial" {
                        InitialDef::Path(missing.clone())
                    } else {
                        InitialDef::None
                    };
                    let opt = if bad && case == "opt-overlay" {
                        opt_overlay.clone()
                    } else {
                        input.parent().unwrap().join("opt.def")
                    };
                    run_para_opt_from_namelist_with_reducer(
                        if bad && case == "opt-parse" {
                            &missing
                        } else {
                            &opt
                        },
                        config,
                        &observer,
                    )
                    .map(|_| ())
                };
                println!("ISSUE178_PREP_RETURN rank={} world={} width={width} bad_rank={bad_rank} case={case} result={result:?} broadcasts={} reductions={}",
                    world.rank(), world.world_size(), observer.broadcasts.get(), observer.reductions.get());
                let error = result.unwrap_err();
                if !bad {
                    assert!(error.contains("failed on another MPI rank"), "{error}");
                } else if case.ends_with("short") {
                    assert!(error.contains("too short"), "{error}");
                } else if case.ends_with("token") {
                    assert!(error.contains("non-numeric token"), "{error}");
                } else if case == "opt-initial" {
                    assert!(error.contains("explicitly requested path"), "{error}");
                } else if case.ends_with("overlay") {
                    assert!(
                        error.contains("utf-8") || error.contains("UTF-8"),
                        "{error}"
                    );
                } else {
                    assert!(
                        error.contains("not found") || error.contains("No such file"),
                        "{error}"
                    );
                }
                // Explicit seeds require no seed broadcast. OPT initialization
                // precedes its load by source order, not by this observer count.
                if !matches!(case, "opt-initial" | "opt-overlay") {
                    assert_eq!(
                        observer.broadcasts.get(),
                        0,
                        "parse/fixed failure must be preseed"
                    );
                }
                assert_eq!(observer.reductions.get(), 0);
                assert!(!output.exists());
                assert_eq!(fs::read(local.join("sentinel")).unwrap(), b"untouched");
                assert_eq!(fs::read(&input).unwrap(), before_input);
                assert_eq!(fs::read(&fixed).unwrap(), before_fixed);
                assert_eq!(fs::read(&invalid_utf8).unwrap(), [0xff]);
                println!("ISSUE178_PREP_CHECK rank={} width={width} bad_rank={bad_rank} case={case} preseed={} output_absent=true inputs_unchanged=true", world.rank(), !matches!(case, "opt-initial" | "opt-overlay"));
                world.barrier();
            }
        }
    }
    println!(
        "ISSUE178_PREP_DONE rank={} world={}",
        world.rank(),
        world.world_size()
    );
}
