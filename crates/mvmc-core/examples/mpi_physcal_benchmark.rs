//! Optional production PhysCal MPI benchmark; startup and warmups are excluded.
use mvmc_core::{mpi::MpiContext, run, Reducer};
use std::{path::PathBuf, time::Instant};

fn benchmark(world: &MpiContext) -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "usage: mpi_physcal_benchmark NAMELIST PARAMS GROUPS WARMUPS REPS OUTPUT RANKS".into(),
        );
    }
    let parse = |i: usize| args[i].parse::<usize>().map_err(|e| e.to_string());
    let (groups, warmups, reps, ranks) = (parse(2)?, parse(3)?, parse(4)?, parse(6)?);
    if groups == 0 || warmups == 0 || reps == 0 || ranks != world.world_size() {
        return Err("positive counts and matching MPI world required".into());
    }
    #[cfg(mvmc_blas_openblas)]
    let blas_threads = {
        unsafe extern "C" {
            fn openblas_get_num_threads() -> std::ffi::c_int;
        }
        unsafe { openblas_get_num_threads() }
    };
    #[cfg(not(mvmc_blas_openblas))]
    let blas_threads = 0;
    if blas_threads != 1 {
        return Err("benchmark requires OpenBLAS with one thread per rank".into());
    }
    println!("WORLD {} {}", world.rank(), ranks);
    println!(
        "THREADS {} {}",
        world.rank(),
        mvmc_core::threading::inner_thread_config().threads
    );
    println!("BLAS_THREADS {} {}", world.rank(), blas_threads);
    for iteration in 0..warmups + reps {
        let output = PathBuf::from(&args[5]).join(format!("run-{iteration}"));
        world.barrier();
        let start = Instant::now();
        let mut preparation = run::prepare_phys_cal_from_namelist_with_reducer(
            &args[0], &args[1], "real", None, world,
        )?;
        preparation.data.modpara.n_data_qty_smp =
            i64::try_from(groups).map_err(|e| e.to_string())?;
        let result = run::vmc_phys_cal_with_reducer(preparation, Some(&output), world)?;
        world.barrier();
        let mut times = vec![0.0; ranks];
        times[world.rank()] = start.elapsed().as_secs_f64();
        world.allreduce_sum_f64(&mut times);
        if result.iterations != groups {
            return Err("incomplete PhysCal measurement groups".into());
        }
        if world.is_root() && iteration >= warmups {
            println!(
                "BENCH {} {:.9} {}",
                iteration - warmups + 1,
                times.into_iter().fold(0.0, f64::max),
                result.iterations
            );
        }
    }
    Ok(())
}

fn run() {
    let world = MpiContext::initialize().expect("MPI initialization");
    if let Err(error) = benchmark(&world) {
        eprintln!("MPI PhysCal benchmark failed: {error}");
        use mpi::traits::Communicator;
        mpi::topology::SimpleCommunicator::world().abort(1);
    }
}

fn main() {
    if mvmc_core::threading::inner_thread_config().threads > 1 {
        mvmc_core::threading::install(run);
    } else {
        run();
    }
}
