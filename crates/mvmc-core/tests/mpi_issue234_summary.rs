//! Public high-level readback contract, not a C trajectory oracle.
//! Root readback I/O is deliberately broken after the optimizer writes its final
//! parameter file. The reducer delegates every collective and rank decision.
#![cfg(feature = "mpi")]

use mvmc_core::{run_para_opt_from_namelist_with_reducer, InitialDef, Reducer, RunConfig};
use num_complex::Complex64;
use std::{
    cell::Cell,
    fs,
    path::{Path, PathBuf},
};

struct ReadbackFault<'a> {
    inner: &'a dyn Reducer,
    out: &'a Path,
    enabled: bool,
    fired: Cell<bool>,
}

impl Reducer for ReadbackFault<'_> {
    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        self.inner.sampling_max_info(info)
    }
    fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
        self.inner.broadcast_f64(root, values)
    }
    fn broadcast_i64(&self, root: usize, values: &mut [i64]) -> Result<(), String> {
        self.inner.broadcast_i64(root, values)
    }
    fn broadcast_c64(&self, root: usize, values: &mut [Complex64]) {
        self.inner.broadcast_c64(root, values);
    }
    fn barrier(&self) {
        self.inner.barrier();
    }
    fn allreduce_sum_f64(&self, values: &mut [f64]) {
        self.inner.allreduce_sum_f64(values);
    }
    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        self.inner.allreduce_sum_c64(values);
    }
    fn allreduce_sum_i64(&self, values: &mut [i64]) {
        self.inner.allreduce_sum_i64(values);
    }
    fn sampling_qp_range(&self, length: usize) -> std::ops::Range<usize> {
        self.inner.sampling_qp_range(length)
    }
    fn sampling_sum_f64(&self, values: &mut [f64]) {
        self.inner.sampling_sum_f64(values);
    }
    fn sampling_sum_c64(&self, values: &mut [Complex64]) {
        self.inner.sampling_sum_c64(values);
    }
    fn sampling_any_failure(&self, failed: bool) -> bool {
        self.inner.sampling_any_failure(failed)
    }
    fn reduce_counters(&self, values: &mut [i64]) {
        self.inner.reduce_counters(values);
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn reduction_size(&self) -> usize {
        self.inner.reduction_size()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn supports_grouped_sampling(&self) -> bool {
        self.inner.supports_grouped_sampling()
    }
    fn seed_offset(&self) -> usize {
        self.inner.seed_offset()
    }
    fn any_failure(&self, failed: bool) -> bool {
        self.inner.any_failure(failed)
    }
    fn is_output_root(&self) -> bool {
        let root = self.inner.is_output_root();
        if root && self.enabled && !self.fired.get() && self.out.join("zqp_opt.dat").is_file() {
            // Fresh directory; final parameter file proves all numerical steps
            // and root output completed. Inject only a subsequent readback fault.
            fs::remove_file(self.out.join("zvo_out.dat")).unwrap();
            self.fired.set(true);
        }
        root
    }
}

fn inputs(root: &Path, width: usize) -> PathBuf {
    fs::create_dir(root).unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    for entry in fs::read_dir(fixture.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), root.join(entry.file_name())).unwrap();
        }
    }
    let namelist = root.join("namelist.def");
    let text = fs::read_to_string(&namelist).unwrap();
    fs::write(
        &namelist,
        text.lines()
            .filter(|line| !line.contains("TwoBodyGEx"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    let path = root.join("modpara.def");
    let text = fs::read_to_string(&path).unwrap();
    let text = text
        .lines()
        .map(|line| {
            let key = line.split_whitespace().next().unwrap_or("");
            let value = match key {
                "NVMCCalMode" => Some(0),
                "NSplitSize" => Some(width),
                "NVMCSample" => Some(8),
                "NVMCWarmUp" | "NVMCInterval" => Some(1),
                "NSRCG" => Some(0),
                _ => None,
            };
            value.map_or_else(|| line.to_owned(), |value| format!("{key} {value}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(path, text).unwrap();
    fs::copy(fixture.join("zqp_opt.dat"), root.join("initial.def")).unwrap();
    namelist
}

fn inventory(path: &Path) -> Vec<String> {
    let mut names = fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            entry.file_name().into_string().unwrap()
        })
        .collect::<Vec<_>>();
    // C InitFile also creates the run-log `_time_` and `_SRinfo` files (formats are
    // covered by run_log_files.rs); this inventory concerns the summary outputs.
    names.retain(|name| !name.contains("_time_") && !name.contains("_SRinfo"));
    names.sort();
    names
}

#[test]
#[ignore = "explicit native MPI worlds2/4 with bounded launcher and fresh output parent"]
fn public_rank_local_summary_and_root_readback_failure() {
    exercise_summary();
}

fn exercise_summary() {
    let world = mvmc_core::mpi::MpiContext::initialize().unwrap();
    assert!([2, 4].contains(&world.world_size()));
    let parent = PathBuf::from(std::env::var_os("MPI_ISSUE234_SUMMARY_OUTPUT").unwrap());
    let setup = if world.is_root() {
        fs::create_dir(&parent)
    } else {
        Ok(())
    };
    assert!(!world.any_failure(setup.is_err()), "{setup:?}");
    let local = parent.join(format!("rank-{}", world.rank()));
    fs::create_dir(&local).unwrap();
    for width in [1, 2] {
        let group = world.split_groups(width).unwrap();
        let namelist = inputs(&local.join(format!("inputs-{width}")), width);
        let mut root_parameters: Option<Vec<u8>> = None;
        for faulty in [false, true] {
            let out = local.join(format!("out-{width}-{faulty}"));
            let reducer = ReadbackFault {
                inner: &group,
                out: &out,
                enabled: faulty,
                fired: Cell::new(false),
            };
            let config = RunConfig {
                output_dir: Some(out.clone()),
                nsmp: Some(1),
                seed: Some(11272),
                initial_def: InitialDef::Path(namelist.parent().unwrap().join("initial.def")),
                ..RunConfig::new(1, "real")
            };
            let result = run_para_opt_from_namelist_with_reducer(&namelist, config, &reducer);
            if faulty {
                let error = result.unwrap_err();
                if world.is_root() {
                    assert!(reducer.fired.get());
                    assert!(!out.join("zvo_out.dat").exists());
                    let expected = fs::read_to_string(out.join("zvo_out.dat")).unwrap_err();
                    assert_eq!(expected.kind(), std::io::ErrorKind::NotFound);
                    assert_eq!(error, expected.to_string());
                    assert_eq!(inventory(&out), ["zqp_opt.dat", "zvo_var.dat"]);
                    assert_eq!(
                        fs::read(out.join("zqp_opt.dat")).unwrap(),
                        root_parameters.as_ref().unwrap().clone()
                    );
                } else {
                    assert!(!reducer.fired.get());
                    assert_eq!(
                        error,
                        "optimization final output/summary failed on another MPI rank"
                    );
                }
            } else {
                let summary = result.unwrap();
                assert_eq!(summary.status, 0);
                assert_eq!(summary.output_dir, fs::canonicalize(&out).unwrap());
                assert_eq!((summary.effective_nsteps, summary.effective_nsmp), (1, 1));
                if world.is_root() {
                    assert_eq!(
                        inventory(&out),
                        ["zqp_opt.dat", "zvo_out.dat", "zvo_var.dat"]
                    );
                    assert_eq!(summary.zvo_first_n.len(), 1);
                    assert_eq!(summary.ctest_values.len(), 2);
                    assert!(summary.final_energy_per_site.is_finite());
                    root_parameters = Some(fs::read(out.join("zqp_opt.dat")).unwrap());
                } else {
                    assert!(summary.zvo_first_n.is_empty());
                    assert!(summary.ctest_values.is_empty());
                    assert!(summary.final_energy_per_site.is_nan());
                }
            }
            if !world.is_root() {
                assert_eq!(fs::read_dir(&out).unwrap().count(), 0);
            }
            println!("ISSUE234_SUMMARY_RETURN rank={} world={} width={width} faulty={faulty} validated=true", world.rank(), world.world_size());
            world.barrier();
        }
    }
}
