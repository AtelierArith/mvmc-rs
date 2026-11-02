//! Linux-only exploratory accounting of Rayon worker CPU during repeated broadcasts.
#![cfg(target_os = "linux")]
use rayon::ThreadPoolBuilder;
use std::collections::BTreeMap;
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

fn other_sched_runtime() -> BTreeMap<u32, u64> {
    let caller: u32 = std::fs::read_link("/proc/thread-self")
        .unwrap()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let tid: u32 = entry.file_name().to_str()?.parse().ok()?;
            if tid == caller {
                return None;
            }
            let stat = std::fs::read_to_string(entry.path().join("schedstat")).ok()?;
            let runtime = stat.split_whitespace().next()?.parse().ok()?;
            Some((tid, runtime))
        })
        .collect()
}

#[test]
#[ignore]
fn schedstat_busy_positive_control() {
    let read = || -> u64 {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let before = read();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(20) {
        std::hint::spin_loop();
    }
    let after = read();
    assert!(after > before, "schedstat did not record busy CPU time");
    eprintln!("positive_control_cpu_ns={}", after - before);
}

#[test]
#[ignore]
fn rayon_post_broadcast_idle_probe() {
    let threads: usize = std::env::var("PROBE_THREADS").unwrap().parse().unwrap();
    let inside = std::env::var("PROBE_INSIDE").unwrap() == "1";
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.broadcast(|_| {});
    let run = || {
        let initial_process = ticks("/proc/self/stat");
        let initial_caller = ticks("/proc/thread-self/stat");
        let initial_sched = other_sched_runtime();
        let start = Instant::now();
        for ms in [10u64, 50, 100, 500, 1000] {
            let target = Duration::from_millis(ms);
            if let Some(remaining) = target.checked_sub(start.elapsed()) {
                std::thread::sleep(remaining);
            }
            let wall = start.elapsed().as_secs_f64();
            let process = ticks("/proc/self/stat") - initial_process;
            let caller = ticks("/proc/thread-self/stat") - initial_caller;
            let sched = other_sched_runtime();
            let missing = initial_sched
                .keys()
                .filter(|tid| !sched.contains_key(tid))
                .count();
            let added = sched
                .keys()
                .filter(|tid| !initial_sched.contains_key(tid))
                .count();
            let other_ns: u64 = initial_sched
                .iter()
                .filter_map(|(tid, initial)| {
                    sched.get(tid).and_then(|now| now.checked_sub(*initial))
                })
                .sum();
            eprintln!("idle threads={threads} inside={inside} requested_ms={ms} elapsed_s={wall:.6} process_ticks={process} caller_ticks={caller} other_ticks={} other_ns={other_ns} baseline_threads={} added={added} missing={missing}", process.saturating_sub(caller), initial_sched.len());
        }
    };
    if inside {
        pool.install(run);
    } else {
        run();
    }
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
