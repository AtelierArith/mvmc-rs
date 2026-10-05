//! Rust's documented integer worker controls, not Julia boolean/PFAPACK parity.
//! Each configuration owns a subprocess; normal tests never mutate shared env.
use mvmc_core::threading::{
    inner_parallel_enabled, inner_thread_config, inner_worker_count, install, InnerThreadConfig,
};
use std::process::Command;

type ControlCase = (
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
    usize,
    usize,
);

const CASES: [ControlCase; 12] = [
    ("absent", None, None, 1, 32),
    ("empty", Some(""), Some(""), 1, 32),
    ("zero", Some("0"), Some("0"), 1, 32),
    ("negative", Some("-1"), Some("-1"), 1, 32),
    ("invalid", Some("true"), Some("auto"), 1, 32),
    (
        "overflow",
        Some("9999999999999999999999999999999999999999"),
        Some("9999999999999999999999999999999999999999"),
        1,
        32,
    ),
    ("one-default", Some("1"), Some("32"), 1, 32),
    ("two-default", Some("2"), Some("32"), 2, 32),
    ("four-default", Some("4"), Some("32"), 4, 32),
    ("one-small", Some("1"), Some("1"), 1, 1),
    ("two-small", Some("2"), Some("1"), 2, 1),
    ("four-small", Some("4"), Some("1"), 4, 1),
];

fn command() -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--ignored",
        "--exact",
        "control_child",
        "--nocapture",
        "--test-threads=1",
    ]);
    command.env_remove("MVMC_RS_INNER_THREADS");
    command.env_remove("MVMC_RS_INNER_THRESHOLD");
    command.env_remove("ISSUE184_CONTROL_CASE");
    command
}

#[test]
fn integer_controls_and_frozen_pool_match_all_twelve_configurations() {
    let mut completed = 0;
    for (name, threads, threshold, _, _) in CASES {
        let mut command = command();
        command.env("ISSUE184_CONTROL_CASE", name);
        if let Some(value) = threads {
            command.env("MVMC_RS_INNER_THREADS", value);
        }
        if let Some(value) = threshold {
            command.env("MVMC_RS_INNER_THRESHOLD", value);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            stdout
                .lines()
                .filter(|line| *line == format!("CONTROL_COMPLETED {name} work_checks=6"))
                .count(),
            1,
            "{name}: exact completion evidence"
        );
        completed += 1;
    }
    assert_eq!(completed, 12);
}

#[test]
fn internal_control_child_missing_selector_fails_closed() {
    let output = command().output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("internal control child requires ISSUE184_CONTROL_CASE"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("CONTROL_COMPLETED"));
}

#[test]
#[ignore = "internal isolated configuration probe; ISSUE184_CONTROL_CASE required"]
fn control_child() {
    let name = std::env::var("ISSUE184_CONTROL_CASE")
        .expect("internal control child requires ISSUE184_CONTROL_CASE");
    let (_, _, threshold_env, threads, threshold) = CASES
        .into_iter()
        .find(|case| case.0 == name)
        .expect("unknown internal control case");
    // An invalid or zero `MVMC_RS_INNER_THRESHOLD` falls back to the default and is not
    // explicit; the work-estimate gate (#361) is then used by the kernels.
    let expected = InnerThreadConfig {
        threads,
        threshold,
        threshold_explicit: threshold_env
            .is_some_and(|value| value.parse::<usize>().is_ok_and(|v| v > 0)),
        min_work_ns: mvmc_core::threading::DEFAULT_MIN_PARALLEL_WORK_NS,
        min_size: mvmc_core::threading::default_min_size(threads),
    };
    assert_eq!(inner_thread_config(), expected);
    // Only this single-test child mutates its env, before any pool is created.
    // Both config and subsequently built pool must retain the captured values.
    std::env::set_var("MVMC_RS_INNER_THREADS", "99");
    std::env::set_var("MVMC_RS_INNER_THRESHOLD", "99");
    assert_eq!(inner_thread_config(), expected);
    assert_eq!(install(rayon::current_num_threads), threads);
    let mut checks = 0;
    for work in [0, 1, 2, 31, 32, 33] {
        let enabled = threads > 1 && work >= threshold;
        assert_eq!(inner_parallel_enabled(work), enabled);
        assert_eq!(
            inner_worker_count(work),
            if enabled { threads.min(work) } else { 1 }
        );
        assert_eq!(install(rayon::current_num_threads), threads);
        checks += 1;
    }
    assert_eq!(checks, 6);
    println!("\nCONTROL_COMPLETED {name} work_checks={checks}");
}
