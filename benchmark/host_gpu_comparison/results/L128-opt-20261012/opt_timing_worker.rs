//! Public Opt runner timing worker: a command on stdin runs one complete optimization.
//! Keeping the process alive separates the first CUDA initialization from warm repetitions.
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(
        args.len(),
        2,
        "usage: opt_timing_worker NAMELIST OUTPUT_DIRECTORY"
    );
    let namelist = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    mvmc_gpu_cuda::install();
    for key in [
        "MVMC_RS_SAMPLER_BACKEND",
        "MVMC_RS_SR_BACKEND",
        "MVMC_RS_MEASURE_BACKEND",
        "MVMC_RS_MEASURE_PF_BACKEND",
    ] {
        println!("{key}={}", std::env::var(key).unwrap_or_default());
    }
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
        run.seed = Some(79);
        run.output_dir = Some(output.join(format!("rep-{rep}")));
        let start = Instant::now();
        let result = mvmc_core::run_para_opt_from_namelist(&namelist, run).expect("GPU Opt");
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
