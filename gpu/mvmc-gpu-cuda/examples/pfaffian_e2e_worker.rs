//! Matched Pfaffian-backend E2E probe, including complex outputs and traffic.
//! One fresh public API call per stdin repetition; repetition zero warms JIT.
use std::io::{BufRead, Write};
use std::path::PathBuf;

use mvmc_gpu_cuda::{opt, physcal};
use tenferro_ext::raw_exec::raw_exec_counters;

fn section(file: &mut std::fs::File, name: &str, values: &[f64]) {
    assert!(values.iter().all(|x| x.is_finite()), "{name}");
    writeln!(file, "{name} {}", values.len()).unwrap();
    for x in values {
        writeln!(file, "{x:.17e}").unwrap();
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!((3..=7).contains(&args.len()),
        "pfaffian_e2e_worker opt|phys NAMELIST NEW_OUTPUT [WALKERS=320] [STEPS_OR_WINDOWS] [WARMUP=128] [INTERVAL=128]");
    let mode = &args[0];
    assert!(mode == "opt" || mode == "phys");
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    std::fs::create_dir(&output).expect("output must be a fresh directory");
    let value = |i: usize, default: usize| args.get(i).map_or(default, |v| v.parse().unwrap());
    let walkers = value(3, 320);
    let steps = value(4, if mode == "opt" { 20 } else { 1 });
    let warmup = value(5, 128);
    let interval = value(6, 128);
    println!("versions {:?}", mvmc_gpu_cuda::versions::query());
    println!("mode {mode} walkers {walkers} steps_or_windows {steps} warmup {warmup} interval {interval} seed 79");
    println!("rep,seconds,upload_calls,upload_bytes,download_calls,download_bytes,synchronizations,scopes");
    std::io::stdout().flush().unwrap();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        let rep: usize = line.parse().unwrap();
        let before = raw_exec_counters();
        let start = std::time::Instant::now();
        let (seconds, traffic, result) = if mode == "opt" {
            let result = opt::run_gpu_para_opt_from_namelist(
                &input,
                opt::OptConfig {
                    nsteps: steps,
                    walkers: Some(walkers),
                    warmup: Some(warmup),
                    interval: Some(interval),
                    seed: Some(79),
                    ..Default::default()
                },
            )
            .expect("resident Opt");
            (
                start.elapsed().as_secs_f64(),
                raw_exec_counters() - before,
                Ok(result),
            )
        } else {
            // No external opt_para: fixed initial parameters and initialization
            // follow the supplied input definitions and seed on both revisions.
            let result = physcal::run_gpu_phys_cal_from_namelist(
                &input,
                physcal::PhysCalConfig {
                    nsamples: Some(steps),
                    walkers: Some(walkers),
                    warmup: Some(warmup),
                    interval: Some(interval),
                    seed: Some(79),
                    ..Default::default()
                },
            )
            .expect("resident PhysCal");
            (
                start.elapsed().as_secs_f64(),
                raw_exec_counters() - before,
                Err(result),
            )
        };
        let mut file = std::fs::File::create(output.join(format!("rep-{rep}.txt"))).unwrap();
        match result {
            Ok(r) => {
                writeln!(file, "seed {} walkers {}", r.seed, r.walkers).unwrap();
                section(&mut file, "energies", &r.energies);
                section(&mut file, "parameters", &r.parameters);
                section(&mut file, "parameters_imag", &r.parameters_imag);
                writeln!(
                    file,
                    "cg_iterations {:?}\ncg_fallback {:?}",
                    r.cg_iterations, r.cg_fallback
                )
                .unwrap();
            }
            Err(r) => {
                writeln!(
                    file,
                    "seed {} walkers {} windows {}",
                    r.seed, r.walkers, r.nsamples
                )
                .unwrap();
                section(&mut file, "energy", &[r.energy, r.energy_imag]);
                section(&mut file, "one", &r.green_one);
                section(&mut file, "one_imag", &r.green_one_imag);
                section(&mut file, "two", &r.green_two);
                section(&mut file, "two_imag", &r.green_two_imag);
                section(&mut file, "two_ex", &r.green_two_ex);
                section(&mut file, "two_ex_imag", &r.green_two_ex_imag);
            }
        }
        println!(
            "{rep},{seconds:.9},{},{},{},{},{},{}",
            traffic.upload_calls,
            traffic.upload_bytes,
            traffic.download_calls,
            traffic.download_bytes,
            traffic.synchronizations,
            traffic.scopes
        );
        std::io::stdout().flush().unwrap();
    }
}
