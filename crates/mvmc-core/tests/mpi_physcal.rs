//! MPI PhysCal smoke gate. Run explicitly with an MPI launcher:
//!
//! Resolve the executable with Cargo JSON, then use `--ignored` under mpiexec.
//! See docs/OPTIONAL_GATES.md for the exact invocation.

mod support;

#[cfg(feature = "mpi")]
#[test]
#[ignore = "optional MPI gate: mpi feature and MVMC_RS_MPI_PHYSICAL required"]
fn mpi_physcal_reduces_fixed_parameter_samples() {
    support::require_gate("mpi-physcal", "MVMC_RS_MPI_PHYSICAL");

    let root = support::julia_mvmc_root()
        .unwrap_or_else(|| support::missing_fixture("mpi-physcal", "Julia-mVMC checkout not found"))
        .join("test/integration/reference/heisenberg_chain_real/physcal_ref");
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
    if context.world_size() < 2 {
        support::unsupported(
            "mpi-physcal",
            "actual MPI world must contain at least two ranks",
        );
    }
    let preparation = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
        &context,
    )
    .expect("prepare PhysCal");
    let result =
        mvmc_core::vmc_phys_cal_with_reducer(preparation, None, &context).expect("MPI PhysCal");
    assert_eq!(result.iterations, 1);

    let group = context.split_groups(2).expect("MPI PhysCal group split");
    let mut grouped = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
        &group,
    )
    .expect("prepare grouped PhysCal");
    // The committed fixture is serial by design; changing only this runtime
    // control exercises the comm1/comm2 reducer path without modifying files.
    grouped.data.modpara.nsplit_size = 2;
    let result =
        mvmc_core::vmc_phys_cal_with_reducer(grouped, None, &group).expect("grouped MPI PhysCal");
    assert_eq!(result.iterations, 1);
}

#[cfg(not(feature = "mpi"))]
#[test]
#[ignore = "optional MPI gate: mpi feature and MVMC_RS_MPI_PHYSICAL required"]
fn mpi_physcal_requires_feature() {
    support::require_gate("mpi-physcal", "MVMC_RS_MPI_PHYSICAL");
    support::unsupported("mpi-physcal", "mvmc-core mpi feature is disabled");
}
