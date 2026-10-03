//! Actual public-library preflight failure, not CLI or numerical-kernel parity.
//! Run on 2/4 ranks with MPI_ISSUE178_FIELD=mode|nsteps|nsmp,
//! MPI_ISSUE178_FAIL_RANK=0|last, MPI_ISSUE178_WIDTH=1|2 and a fresh
//! MPI_ISSUE178_OUTPUT path. Timeout belongs to the explicit MPI launcher.
#![cfg(feature = "mpi")]

use std::{cell::Cell, fs, path::PathBuf};

use mvmc_core::{run_para_opt_from_namelist_with_reducer, InitialDef, Reducer, RunConfig};
use num_complex::Complex64;

struct PreflightObserver<'a> {
    inner: &'a dyn Reducer,
    agreements: Cell<usize>,
    broadcasts: Cell<usize>,
    reductions: Cell<usize>,
}

impl Reducer for PreflightObserver<'_> {
    fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        self.broadcasts.set(self.broadcasts.get() + 1);
        self.inner.broadcast_i64(root, values)
    }
    fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
        self.broadcasts.set(self.broadcasts.get() + 1);
        self.inner.broadcast_c64(root, values);
    }
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        self.reductions.set(self.reductions.get() + 1);
        self.inner.allreduce_sum_f64(values);
    }
    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        self.reductions.set(self.reductions.get() + 1);
        self.inner.allreduce_sum_c64(values);
    }
    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        self.reductions.set(self.reductions.get() + 1);
        self.inner.allreduce_sum_i64(values);
    }
    fn any_failure(&self, failed: bool) -> bool {
        self.agreements.set(self.agreements.get() + 1);
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

#[test]
#[ignore = "explicit actual MPI 2/4-rank preflight gate with bounded timeout"]
fn one_rank_invalid_public_options_stop_all_ranks_before_seed_and_output() {
    let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let width: usize = std::env::var("MPI_ISSUE178_WIDTH")
        .unwrap()
        .parse()
        .unwrap();
    assert!(matches!(width, 1 | 2));
    let group = world.split_groups(width).unwrap();
    let observer = PreflightObserver {
        inner: &group,
        agreements: Cell::new(0),
        broadcasts: Cell::new(0),
        reductions: Cell::new(0),
    };
    let field = std::env::var("MPI_ISSUE178_FIELD").unwrap();
    let fail_rank = match std::env::var("MPI_ISSUE178_FAIL_RANK").unwrap().as_str() {
        "0" => 0,
        "last" => world.world_size() - 1,
        other => panic!("unsupported failure rank {other}"),
    };
    let parent = PathBuf::from(std::env::var_os("MPI_ISSUE178_OUTPUT").unwrap());
    let setup = if world.is_root() {
        fs::create_dir(&parent)
            .and_then(|()| fs::write(parent.join("sentinel"), b"untouched"))
            .map_err(|error| error.to_string())
    } else {
        Ok(())
    };
    assert!(
        !world.any_failure(setup.is_err()),
        "exclusive fixture setup failed: {setup:?}"
    );
    let output = parent.join("must-not-exist");
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def");
    let input_before = fs::read(&input).unwrap();
    let mut config = RunConfig::new(1, "real");
    config.nsmp = Some(1);
    config.seed = Some(11272);
    config.initial_def = InitialDef::None;
    config.output_dir = Some(output.clone());
    let diagnostic = match field.as_str() {
        "mode" => {
            if world.rank() == fail_rank {
                config.mode = "invalid".into();
            }
            "mode must"
        }
        "nsteps" => {
            if world.rank() == fail_rank {
                config.nsteps = 0;
            }
            "nsteps must"
        }
        "nsmp" => {
            if world.rank() == fail_rank {
                config.nsmp = Some(0);
            }
            "nsmp must"
        }
        other => panic!("unsupported invalid field {other}"),
    };
    let result = run_para_opt_from_namelist_with_reducer(&input, config, &observer);
    // Print before any further collective: the old implementation returns only
    // on the offending rank while peers remain inside the runner's agreement.
    println!("ISSUE178_RETURN rank={} world={} width={width} field={field} fail_rank={fail_rank} error={:?} agreements={} broadcasts={} reductions={}",
        world.rank(), world.world_size(), result.as_ref().err(), observer.agreements.get(),
        observer.broadcasts.get(), observer.reductions.get());
    let error = result.unwrap_err();
    if world.rank() == fail_rank {
        assert!(error.contains(diagnostic), "{error}");
    } else {
        assert!(
            error.contains("optimization configuration failed on another MPI rank"),
            "{error}"
        );
    }
    assert_eq!(observer.agreements.get(), 1);
    assert_eq!(
        observer.broadcasts.get(),
        0,
        "must not reach seed/parameter broadcasts"
    );
    assert_eq!(
        observer.reductions.get(),
        0,
        "must not reach sampling/SR reductions"
    );
    assert!(!output.exists());
    assert_eq!(fs::read(parent.join("sentinel")).unwrap(), b"untouched");
    assert_eq!(fs::read(&input).unwrap(), input_before);
    assert_eq!(fs::read_dir(&parent).unwrap().count(), 1);
    // Caller supplies no RNG/state to this high-level API and Err returns none:
    // these are pre-seed reachability assertions, not exported raw-RNG proof.
    world.barrier();
}
