//! Persistent-process E2E comparison of the single-owner GPU Opt API.
//! Commands on stdin are repetition numbers; `quit` exits. All setup, initial
//! sampling, warmup, optimization, and final downloads are inside the timer.
use mvmc_gpu_cuda::opt::{run_gpu_para_opt_from_namelist, OptConfig};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use tenferro_ext::raw_exec::raw_exec_counters;

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert!((2..=6).contains(&args.len()), "usage: resident_opt_worker NAMELIST OUTPUT [WALKERS=320] [STEPS=20] [WARMUP=128] [INTERVAL=128]");
    let namelist = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let value = |i: usize, default: usize| {
        args.get(i)
            .map(|v| v.to_str().unwrap().parse().unwrap())
            .unwrap_or(default)
    };
    let config = OptConfig {
        walkers: Some(value(2, 320)),
        nsteps: value(3, 20),
        warmup: Some(value(4, 128)),
        interval: Some(value(5, 128)),
        seed: Some(79),
        ..Default::default()
    };
    std::fs::create_dir_all(&output).unwrap();
    println!("config={config:?}\nREADY");
    std::io::stdout().flush().unwrap();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        let rep: usize = line.parse().expect("integer repetition");
        let before = raw_exec_counters();
        let start = std::time::Instant::now();
        let result =
            run_gpu_para_opt_from_namelist(&namelist, config.clone()).expect("resident GPU Opt");
        let seconds = start.elapsed().as_secs_f64();
        let traffic = raw_exec_counters() - before;
        assert!(result
            .energies
            .iter()
            .chain(&result.parameters)
            .all(|x| x.is_finite()));
        let mut file = std::fs::File::create(output.join(format!("rep-{rep}.txt"))).unwrap();
        writeln!(
            file,
            "seconds {seconds:.9}\nenergies {}",
            result.energies.len()
        )
        .unwrap();
        for x in &result.energies {
            writeln!(file, "{x:.17e}").unwrap();
        }
        writeln!(file, "parameters {}", result.parameters.len()).unwrap();
        for x in &result.parameters {
            writeln!(file, "{x:.17e}").unwrap();
        }
        println!(
            "rep={rep} seconds={seconds:.9} energy={} traffic={traffic:?}\nDONE {rep}",
            result.energies.last().unwrap()
        );
        std::io::stdout().flush().unwrap();
    }
}
