//! Optional developer benchmark: production optimization, synchronized MPI timings.
use mvmc_core::{mpi::MpiContext, Reducer, RunConfig};
use std::{path::PathBuf, time::Instant};

fn blas_thread_count() -> Result<std::ffi::c_int, String> {
    #[cfg(mvmc_blas_openblas)]
    {
        unsafe extern "C" {
            fn openblas_get_num_threads() -> std::ffi::c_int;
        }
        Ok(unsafe { openblas_get_num_threads() })
    }
    #[cfg(not(mvmc_blas_openblas))]
    {
        Err("this optional benchmark requires the OpenBLAS provider".into())
    }
}

fn benchmark(world: &MpiContext) -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: mpi_benchmark NAMELIST STEPS WARMUPS REPS OUTPUT RANKS".into());
    }
    let parse = |i: usize| args[i].parse::<usize>().map_err(|e| e.to_string());
    let (steps, warmups, reps, ranks) = (parse(1)?, parse(2)?, parse(3)?, parse(5)?);
    if steps == 0 || reps == 0 || ranks != world.world_size() {
        return Err(
            "positive steps/reps and actual MPI world matching requested ranks required".into(),
        );
    }
    let steps = i64::try_from(steps).map_err(|e| e.to_string())?;
    let blas_threads = blas_thread_count()?;
    if blas_threads != 1 {
        return Err("BLAS must use one thread per process".into());
    }
    println!("BLAS_THREADS {} {}", world.rank(), blas_threads);
    println!("WORLD {} {}", world.rank(), world.world_size());
    println!(
        "THREADS {} {}",
        world.rank(),
        mvmc_core::threading::inner_thread_config().threads
    );
    for iteration in 0..warmups + reps {
        let mut config = RunConfig::new(steps, "real");
        config.nsmp = Some(steps);
        config.output_dir = Some(PathBuf::from(&args[4]).join(format!("run-{iteration}")));
        world.barrier();
        let start = Instant::now();
        let result = mvmc_core::run_para_opt_from_namelist_with_reducer(&args[0], config, world)?;
        if iteration == 0 {
            // Keep per-item observational locks outside the measured runs and
            // bound their cost even when all tiny regions are forced parallel.
            let diagnostic_steps = steps.min(20);
            let mut diagnostic = RunConfig::new(diagnostic_steps, "real");
            diagnostic.nsmp = Some(diagnostic_steps);
            diagnostic.output_dir = Some(PathBuf::from(&args[4]).join("run-observe"));
            let observer = mvmc_core::threading::start_observation();
            mvmc_core::run_para_opt_from_namelist_with_reducer(&args[0], diagnostic, world)?;
            let evidence = observer.finish();
            println!(
                "EXECUTION {} {} {}",
                world.rank(),
                evidence.parallel_calls,
                evidence.distinct_workers
            );
            println!("EXECUTION_DETAIL {} {:?}", world.rank(), evidence);
        }
        world.barrier();
        let mut times = vec![0.0; ranks];
        times[world.rank()] = start.elapsed().as_secs_f64();
        world.allreduce_sum_f64(&mut times);
        if world.is_root() {
            if result.effective_nsteps as i64 != steps || !result.final_energy_per_site.is_finite()
            {
                return Err("incomplete optimization or nonfinite energy".into());
            }
            if iteration >= warmups {
                let seconds = times.into_iter().fold(0.0, f64::max);
                println!(
                    "BENCH {} {:.9} {:.17e}",
                    iteration - warmups + 1,
                    seconds,
                    result.final_energy_per_site
                );
            }
        }
    }
    Ok(())
}

fn run() {
    let world = MpiContext::initialize().expect("MPI initialization");
    if let Err(error) = benchmark(&world) {
        eprintln!("MPI benchmark failed: {error}");
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
