//! Public Opt runner timing worker: a command on stdin runs one complete optimization.
//! Keeping the process alive separates the first CUDA initialization from warm repetitions.
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert!(
        (2..=3).contains(&args.len()),
        "usage: opt_timing_worker NAMELIST OUTPUT_DIRECTORY [WALKERS]"
    );
    let namelist = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let walkers: usize = args
        .get(2)
        .map_or(1, |n| n.to_str().unwrap().parse().unwrap());
    assert!(walkers > 0);
    let seed: i64 =
        std::env::var("MVMC_OPT_TIMING_SEED").map_or(79, |n| n.parse().expect("integer seed"));
    mvmc_gpu_cuda::install();
    for key in [
        "MVMC_RS_SAMPLER_BACKEND",
        "MVMC_RS_SR_BACKEND",
        "MVMC_RS_MEASURE_BACKEND",
        "MVMC_RS_MEASURE_PF_BACKEND",
    ] {
        println!("{key}={}", std::env::var(key).unwrap_or_default());
    }
    println!("walkers={walkers} seed={seed}");
    println!("READY");
    std::io::stdout().flush().unwrap();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        let rep: usize = line.parse().expect("integer repetition");
        let mut run = mvmc_core::RunConfig::new(20, "real");
        run.nsmp = Some(1);
        run.seed = Some(seed);
        run.output_dir = Some(output.join(format!("rep-{rep}")));
        let start = Instant::now();
        let result = if walkers == 1 {
            mvmc_core::run_para_opt_from_namelist(&namelist, run)
        } else {
            // The same public rank runner and communicator as run_para_opt_multichain,
            // retaining its timer-enabled entry point for a per-stage profile.
            let comm = mvmc_core::multichain::ThreadComm::new(walkers);
            let results = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..walkers)
                    .map(|rank| {
                        let reducer =
                            mvmc_core::multichain::ThreadReducer::new(comm.clone(), rank, 0);
                        let run = run.clone();
                        let namelist = &namelist;
                        scope.spawn(move || {
                            mvmc_core::run_para_opt_from_namelist_with_reducer(
                                namelist, run, &reducer,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("walker thread"))
                    .collect::<Result<Vec<_>, _>>()
            });
            results.map(|mut runs| runs.remove(0))
        }
        .expect("GPU Opt");
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(result.status, 0);
        assert!(result.final_energy_per_site.is_finite());
        println!(
            "rep={rep} seconds={elapsed:.9} energy_per_site={} output={}",
            result.final_energy_per_site,
            result.output_dir.display()
        );
        println!("DONE {rep}");
        std::io::stdout().flush().unwrap();
    }
}
