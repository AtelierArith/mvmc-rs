//! CTimer contracts from Julia c_timer.jl, with an injected deterministic clock.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::c_timer::{CTimer, TimerEnv, CTIMER_N};
use std::fs;

#[test]
fn inclusive_sections_accumulate_and_reset_both_arrays() {
    let mut timer = CTimer::<true>::new();
    timer.start_at(0, 100);
    timer.start_at(3, 200);
    timer.stop_at(3, 700);
    timer.stop_at(0, 1000);
    assert_eq!(timer.elapsed_ns[0], 900);
    assert_eq!(timer.elapsed_ns[3], 500);
    timer.start_at(3, 1500);
    timer.stop_at(3, 2000);
    assert_eq!(timer.elapsed_ns[3], 1000);
    numerical_comparison::assert_close(
        timer.seconds(3),
        0.000001,
        0.0,
        4.0 * f64::EPSILON,
        "nanoseconds to seconds",
    );
    timer.reset();
    assert!(timer.elapsed_ns.iter().all(|&value| value == 0));
    assert!(timer.start_ns.iter().all(|&value| value == 0));
    assert_eq!(CTIMER_N, 1000);
}
#[test]
fn disabled_operations_do_not_access_slots_or_read_the_clock() {
    let mut timer = CTimer::<false>::new();
    timer.start(usize::MAX);
    timer.stop(usize::MAX);
    timer.start_at(usize::MAX, 10);
    timer.stop_at(usize::MAX, 20);
    assert!(timer.elapsed_ns.iter().all(|&value| value == 0));
    assert!(timer.start_ns.iter().all(|&value| value == 0));
}
#[test]
fn environment_semantics_enable_every_value_except_literal_zero() {
    for value in [
        None,
        Some("0"),
        Some(""),
        Some("false"),
        Some("1"),
        Some("00"),
    ] {
        let flags = TimerEnv::from_lookup(|key| {
            if key == "MVMC_C_TIMER" {
                value.map(str::to_owned)
            } else {
                None
            }
        });
        assert_eq!(flags.enabled(), value.is_some_and(|value| value != "0"));
    }
    for key in [
        "MVMC_CALHAM1_DIAG",
        "MVMC_SLATER_DIAG",
        "MVMC_MAINCAL_DIAG",
        "MVMC_WEIGHTAVG_DIAG",
    ] {
        let flags = TimerEnv::from_lookup(|name| (name == key).then(|| "1".to_owned()));
        assert!(flags.enabled());
        assert!(flags.any_diag());
    }
    let flags = TimerEnv::from_lookup(|key| (key == "MVMC_TIMER").then(|| "1".to_owned()));
    assert!(flags.enabled());
    assert!(flags.legacy_warning());
}
#[test]
fn reports_match_julia_zero_fixtures_and_respect_prefix() {
    let dir = std::env::temp_dir().join(format!("mvmc-timer-reports-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let timer = CTimer::<false>::new();
    let path = timer.write_para_opt(&dir, "custom").unwrap();
    assert_eq!(path, dir.join("custom_CalcTimer.dat"));
    assert_eq!(
        fs::read_to_string(path).unwrap(),
        include_str!("../../../tests/fixtures/timers/julia_para_opt_zero.dat")
    );
    let path = timer.write_diag(&dir, "custom").unwrap();
    assert_eq!(
        fs::read_to_string(path).unwrap(),
        include_str!("../../../tests/fixtures/timers/julia_diag_zero.dat")
    );
    let mut nonzero = CTimer::<true>::new();
    nonzero.start_at(0, 100);
    nonzero.stop_at(0, 9123456889);
    let path = nonzero.write_para_opt(&dir, "nonzero").unwrap();
    assert_eq!(
        fs::read_to_string(path).unwrap().lines().next().unwrap(),
        "All                         [0]      9.12346"
    );
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn worker_elapsed_times_merge_without_changing_parent_start_slots() {
    let mut parent = CTimer::<true>::new();
    let mut worker = CTimer::<true>::new();
    parent.start_at(0, 10);
    worker.start_at(3, 20);
    worker.stop_at(3, 50);
    parent.merge_elapsed(&worker);
    assert_eq!(parent.elapsed_ns[3], 30);
    assert_eq!(parent.start_ns[0], 10);
    assert_eq!(parent.start_ns[3], 0);
}
