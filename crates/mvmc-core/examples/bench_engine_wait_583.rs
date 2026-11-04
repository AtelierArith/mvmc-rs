//! Synthetic CpuEngine lock/dispatch probe; timings include instrumentation and scheduling.
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: bench_engine_wait_583 callers iterations gap_us default|0"
    );
    let callers: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let gap: u64 = args[3].parse().unwrap();
    assert!(callers > 0 && iterations > 0);
    let mut builder = tenferro_cpu::Pool::builder().threads(4);
    if args[4] != "default" {
        builder = builder.spin_polls(args[4].parse().unwrap());
    }
    let pool = builder.build().expect("pool");
    let team = tenferro_cpu::CpuEngine::new(pool);
    let team = Arc::new(Mutex::new(team));
    let barrier = Barrier::new(callers);
    let total = AtomicU64::new(0);
    let lock_ns = AtomicU64::new(0);
    let dispatch_ns = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..callers {
            let team = &team;
            let barrier = &barrier;
            let total = &total;
            let lock_ns = &lock_ns;
            let dispatch_ns = &dispatch_ns;
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..iterations {
                    if gap > 0 {
                        std::thread::sleep(Duration::from_micros(gap));
                    }
                    let start = Instant::now();
                    let mut engine = team.lock().unwrap();
                    lock_ns.fetch_add(
                        start.elapsed().as_nanos().try_into().unwrap(),
                        Ordering::Relaxed,
                    );
                    let start = Instant::now();
                    engine
                        .with_exec(|exec| {
                            exec.token().for_ranges(64, NonZeroUsize::MIN, |range| {
                                let mut sum = 0u64;
                                for i in range {
                                    sum += i as u64 + 1;
                                }
                                total.fetch_add(sum, Ordering::Relaxed);
                            });
                        })
                        .expect("Engine operation");
                    dispatch_ns.fetch_add(
                        start.elapsed().as_nanos().try_into().unwrap(),
                        Ordering::Relaxed,
                    );
                }
            });
        }
    });
    let expected = callers as u64 * iterations as u64 * (64 * 65 / 2);
    assert_eq!(total.load(Ordering::Relaxed), expected);
    println!(
        "calls={} sum={} lock_ns={} dispatch_ns={}",
        callers * iterations,
        expected,
        lock_ns.load(Ordering::Relaxed),
        dispatch_ns.load(Ordering::Relaxed)
    );
}
