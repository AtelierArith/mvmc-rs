//! MPI PhysCal smoke gate. Run explicitly with an MPI launcher:
//!
//! `MVMC_RS_MPI_PHYSICAL=1 mpiexec -n 2 target/debug/deps/mpi_physcal-* --nocapture`

#[cfg(feature = "mpi")]
#[test]
fn mpi_physcal_reduces_fixed_parameter_samples() {
    if std::env::var_os("MVMC_RS_MPI_PHYSICAL").is_none() {
        eprintln!("set MVMC_RS_MPI_PHYSICAL=1 to run the MPI PhysCal smoke gate");
        return;
    }

    let context = mvmc_core::mpi::MpiContext::initialize().expect("MPI initialization");
    assert!(context.world_size() >= 2);
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
    );
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
fn mpi_physcal_requires_feature() {
    eprintln!("mvmc-core mpi feature is disabled; MPI PhysCal smoke gate skipped");
}
