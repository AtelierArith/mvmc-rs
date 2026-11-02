//! Synthetic threadteam wait-counter probe, not an mVMC application benchmark.
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: bench_team_wait_583 callers iterations gap_us default|0"
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
    let team = pool.into_team(4).expect("reserve 4 participants");
    let team = Arc::new(Mutex::new(team));
    let barrier = Barrier::new(callers);
    let total = AtomicU64::new(0);
    std::thread::scope(|scope| {
        for _ in 0..callers {
            let team = &team;
            let barrier = &barrier;
            let total = &total;
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..iterations {
                    if gap > 0 {
                        std::thread::sleep(Duration::from_micros(gap));
                    }
                    let team = team.lock().unwrap();
                    team.exec(4).for_ranges(64, NonZeroUsize::MIN, |range| {
                        let mut sum = 0u64;
                        for i in range {
                            sum += i as u64 + 1;
                        }
                        total.fetch_add(sum, Ordering::Relaxed);
                    });
                }
            });
        }
    });
    let expected = callers as u64 * iterations as u64 * (64 * 65 / 2);
    assert_eq!(total.load(Ordering::Relaxed), expected);
    let team = team.lock().unwrap();
    let counts = team.pool().wait_counters();
    println!(
        "calls={} sum={} join_sleeps={} barrier_sleeps={} member_parks={}",
        callers * iterations,
        expected,
        counts.join_sleeps,
        counts.barrier_sleeps,
        counts.member_parks
    );
}
