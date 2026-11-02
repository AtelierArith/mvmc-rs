//! Linux-only exploratory accounting of Rayon worker CPU during repeated broadcasts.
#![cfg(target_os = "linux")]
use rayon::ThreadPoolBuilder;
use std::time::{Duration, Instant};

fn ticks(path: &str) -> u64 {
    let line = std::fs::read_to_string(path).unwrap();
    let fields: Vec<_> = line
        .rsplit_once(')')
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap()
}

#[test]
#[ignore]
fn rayon_idle_probe() {
    let threads: usize = std::env::var("PROBE_THREADS").unwrap().parse().unwrap();
    let micros: u64 = std::env::var("PROBE_GAP_US").unwrap().parse().unwrap();
    let inside = std::env::var("PROBE_INSIDE").unwrap() == "1";
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.broadcast(|_| {});
    let run = || {
        let process_start = ticks("/proc/self/stat");
        let caller_start = ticks("/proc/thread-self/stat");
        let start = Instant::now();
        let mut calls = 0u64;
        while start.elapsed() < Duration::from_secs(3) {
            if micros != 0 {
                std::thread::sleep(Duration::from_micros(micros));
            }
            if inside {
                rayon::broadcast(|_| {});
            } else {
                pool.broadcast(|_| {});
            }
            calls += 1;
        }
        let wall = start.elapsed().as_secs_f64();
        let caller = ticks("/proc/thread-self/stat") - caller_start;
        let process = ticks("/proc/self/stat") - process_start;
        eprintln!("threads={threads} requested_sleep_us={micros} inside={inside} calls={calls} wall_s={wall:.6} process_ticks={process} caller_ticks={caller} other_ticks={}", process.saturating_sub(caller));
    };
    if inside {
        pool.install(run);
    } else {
        run();
    }
}
