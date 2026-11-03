//! Warm public-API benchmark. Repetition zero is warmup; save results after timing.
use mvmc_gpu_cuda::physcal::{run_gpu_phys_cal_from_namelist, PhysCalConfig};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use tenferro_ext::raw_exec::raw_exec_counters;

fn profiler(start: bool) {
    // Optional developer-only Nsight capture range; absent in speed runs.
    unsafe {
        let library = libloading::Library::new("libcuda.so.1").unwrap();
        let symbol = if start {
            b"cuProfilerStart\0".as_slice()
        } else {
            b"cuProfilerStop\0".as_slice()
        };
        let call: libloading::Symbol<unsafe extern "C" fn() -> i32> = library.get(symbol).unwrap();
        assert_eq!(call(), 0, "CUDA profiler API");
    }
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert!((3..=7).contains(&args.len()), "usage: resident_physcal_worker NAMELIST PARAMS OUTPUT [WALKERS=320] [WINDOWS=1] [WARMUP=128] [INTERVAL=128]");
    let namelist = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[2]);
    let value = |i: usize, default: usize| {
        args.get(i)
            .map(|v| v.to_str().unwrap().parse().unwrap())
            .unwrap_or(default)
    };
    let config = PhysCalConfig {
        walkers: Some(value(3, 320)),
        nsamples: Some(value(4, 1)),
        warmup: Some(value(5, 128)),
        interval: Some(value(6, 128)),
        seed: Some(79),
        opt_para: Some(PathBuf::from(&args[1])),
        ..Default::default()
    };
    std::fs::create_dir_all(&output).unwrap();
    println!("CUDA versions: {:?}", mvmc_gpu_cuda::versions::query());
    println!("config={config:?}\nREADY");
    std::io::stdout().flush().unwrap();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        let rep: usize = line.parse().expect("integer repetition");
        let profile = std::env::var("MVMC_PHYSCAL_PROFILE_REP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            == Some(rep);
        if profile {
            profiler(true);
        }
        let before = raw_exec_counters();
        let start = std::time::Instant::now();
        let result = run_gpu_phys_cal_from_namelist(&namelist, config.clone())
            .expect("resident GPU PhysCal");
        let seconds = start.elapsed().as_secs_f64();
        if profile {
            profiler(false);
        }
        let traffic = raw_exec_counters() - before;
        let mut file = std::fs::File::create(output.join(format!("rep-{rep}.txt"))).unwrap();
        writeln!(file, "seconds {seconds:.9}\nenergy {}", result.energy).unwrap();
        for (name, values) in [
            ("one", &result.green_one),
            ("two", &result.green_two),
            ("two_ex", &result.green_two_ex),
        ] {
            writeln!(file, "{name} {}", values.len()).unwrap();
            for x in values {
                writeln!(file, "{x:.17e}").unwrap();
            }
        }
        println!(
            "BENCH rep={rep} seconds={seconds:.9} energy={} traffic={traffic:?}\nDONE {rep}",
            result.energy
        );
        std::io::stdout().flush().unwrap();
    }
}
