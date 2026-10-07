//! Issue #361: the work-estimate and problem-size gates of the inner pool.
//!
//! Dispatching a tiny region to the Rayon pool costs more than running it, so
//! without `MVMC_RS_INNER_THRESHOLD` a region is pooled only when its estimated
//! serial work (`items * cost_ns`) reaches `MVMC_RS_INNER_MIN_WORK_NS`. An explicit
//! threshold keeps the plain item-count gate that the worker-invariance tests use
//! to force pooled execution of small inputs. The configuration is frozen per
//! process, so every case runs in its own child process.

use std::process::Command;

use mvmc_core::threading::{
    default_min_size, inner_parallel_enabled, inner_parallel_work, inner_thread_config,
    scaled_cost_ns, DEFAULT_MIN_PARALLEL_SIZE, DEFAULT_MIN_PARALLEL_WORK_NS,
};

/// (name, threads, threshold env, min-work env)
type Case = (
    &'static str,
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
);

const CASES: [Case; 6] = [
    ("hoisted-gate", "4", None, None),
    ("serial-never-pools", "1", None, None),
    ("auto-default", "4", None, None),
    ("auto-custom-min-work", "4", None, Some("1000")),
    ("explicit-threshold-is-item-count", "4", Some("4"), None),
    ("invalid-threshold-falls-back-to-auto", "4", Some("0"), None),
];

#[test]
fn work_estimate_gate_matches_independent_expectations() {
    for (name, threads, threshold, min_work) in CASES {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--ignored",
                "--exact",
                "gate_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("ISSUE361_CASE", name)
            .env("MVMC_RS_INNER_THREADS", threads)
            .env_remove("MVMC_RS_INNER_THRESHOLD")
            .env_remove("MVMC_RS_INNER_MIN_WORK_NS")
            .env_remove("MVMC_RS_INNER_MIN_SIZE");
        if name == "auto-custom-min-work" {
            command.env("MVMC_RS_INNER_MIN_SIZE", "8");
        }
        if let Some(value) = threshold {
            command.env("MVMC_RS_INNER_THRESHOLD", value);
        }
        if let Some(value) = min_work {
            command.env("MVMC_RS_INNER_MIN_WORK_NS", value);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{name}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(&format!("GATE_COMPLETED {name}")),
            "{name}: no completion evidence"
        );
    }
}

#[ignore = "child process of work_estimate_gate_matches_independent_expectations"]
#[test]
fn gate_child() {
    let name = std::env::var("ISSUE361_CASE").expect("child case");
    let config = inner_thread_config();
    match name.as_str() {
        "serial-never-pools" => {
            assert_eq!(config.threads, 1);
            for (items, cost) in [(0, 1), (2, usize::MAX), (1 << 20, 1 << 20)] {
                assert!(!inner_parallel_work(items, cost));
            }
        }
        "auto-default" => {
            assert_eq!(config.min_work_ns, DEFAULT_MIN_PARALLEL_WORK_NS);
            // 120 * w / (w - 1): four workers need 160, two workers 240, eight 138.
            assert_eq!(config.min_size, 160);
            assert_eq!(default_min_size(4), 160);
            assert_eq!(default_min_size(2), 240);
            assert_eq!(default_min_size(8), 138);
            assert!(default_min_size(1) >= DEFAULT_MIN_PARALLEL_SIZE);
            assert!(!config.threshold_explicit);
            // Matrices below the size gate stay serial whatever the estimate says.
            assert_eq!(scaled_cost_ns(config.min_size - 1, 1 << 30), 0);
            assert_eq!(scaled_cost_ns(config.min_size, 1 << 30), 1 << 30);
            assert!(!inner_parallel_work(
                8,
                scaled_cost_ns(config.min_size - 1, 1 << 30)
            ));
            // 100 us of estimated work is the boundary.
            assert!(!inner_parallel_work(100, 999));
            assert!(inner_parallel_work(100, 1000));
            assert!(!inner_parallel_work(8, 12_499));
            assert!(inner_parallel_work(8, 12_500));
            // A single item has nothing to split; zero items never pool.
            assert!(!inner_parallel_work(1, usize::MAX));
            assert!(!inner_parallel_work(0, usize::MAX));
            // Cheap elementwise work needs a huge range to pay for a dispatch.
            assert!(!inner_parallel_work(32, 2));
            assert!(!inner_parallel_work(40_000, 2));
            assert!(inner_parallel_work(50_000, 2));
            // The item-count gate used by the legacy predicate is unchanged.
            assert!(inner_parallel_enabled(32));
            assert!(!inner_parallel_enabled(31));
        }
        "hoisted-gate" => {
            use mvmc_core::threading::{
                inner_parallel_work_cold, install, HOISTED_COLD_MIN_PARALLEL_WORK_NS,
                HOISTED_MIN_PARALLEL_WORK_NS,
            };
            assert_eq!(HOISTED_MIN_PARALLEL_WORK_NS, 40_000);
            assert_eq!(HOISTED_COLD_MIN_PARALLEL_WORK_NS, 400_000);
            // From outside the pool the Rayon-calibrated gate and size gate apply.
            assert!(!inner_parallel_work(8, 12_499));
            assert_eq!(scaled_cost_ns(config.min_size - 1, 1 << 30), 0);
            // Inside the pool (a hoisted driver) regions are cold by default (400 us) and the
            // size gate is off; the sampler marks its loop hot (40 us, checked in the library).
            install(|| {
                assert!(!inner_parallel_work(8, 49_999));
                assert!(inner_parallel_work(8, 50_000));
                assert!(!inner_parallel_work_cold(8, 49_999));
                assert!(inner_parallel_work_cold(8, 50_000));
                assert!(!inner_parallel_work(1, usize::MAX));
                assert_eq!(scaled_cost_ns(1, 1 << 30), 1 << 30);
                assert_eq!(scaled_cost_ns(0, 1 << 30), 0);
            });
            assert!(!inner_parallel_work(8, 12_499));
        }
        "auto-custom-min-work" => {
            assert_eq!(config.min_work_ns, 1000);
            assert_eq!(config.min_size, 8);
            assert_eq!(scaled_cost_ns(7, 5), 0);
            assert_eq!(scaled_cost_ns(8, 5), 5);
            assert!(inner_parallel_work(10, 100));
            assert!(!inner_parallel_work(10, 99));
        }
        "explicit-threshold-is-item-count" => {
            assert!(config.threshold_explicit);
            assert_eq!(config.threshold, 4);
            // Cost is ignored: only the item count decides, as before #361.
            assert_eq!(
                scaled_cost_ns(1, 7),
                0,
                "size gate zeroes only the estimate"
            );
            assert!(inner_parallel_work(4, 0));
            assert!(inner_parallel_work(4, 1));
            assert!(!inner_parallel_work(3, usize::MAX));
        }
        "invalid-threshold-falls-back-to-auto" => {
            assert!(!config.threshold_explicit);
            assert_eq!(config.threshold, 32);
            assert!(!inner_parallel_work(1000, 1));
            assert!(inner_parallel_work(1000, 1000));
        }
        other => panic!("unknown case {other}"),
    }
    println!("GATE_COMPLETED {name}");
}
