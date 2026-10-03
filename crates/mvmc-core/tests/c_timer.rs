//! CTimer contracts from Julia c_timer.jl, with an injected deterministic clock.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::c_timer::{CTimer, TimerEnv, CTIMER_N};
use std::fs;

#[test]
fn type_level_enabled_query_and_diagnostic_predicates() {
    let mut enabled = CTimer::<true>::new();
    let mut disabled = CTimer::<false>::new();
    assert!(enabled.enabled());
    assert!(!disabled.enabled());
    enabled.start_diag(usize::MAX, false);
    enabled.stop_diag(usize::MAX, false);
    disabled.start_diag(usize::MAX, true);
    disabled.stop_diag(usize::MAX, true);
    assert!(enabled.start_ns.iter().all(|&value| value == 0));
    assert!(enabled.elapsed_ns.iter().all(|&value| value == 0));
    assert!(disabled.start_ns.iter().all(|&value| value == 0));
    assert!(disabled.elapsed_ns.iter().all(|&value| value == 0));
}

#[test]
fn unsigned_timer_arithmetic_wraps_and_large_seconds_are_defined() {
    let mut timer = CTimer::<true>::new();
    timer.start_at(3, u64::MAX - 2);
    timer.stop_at(3, 4);
    assert_eq!(timer.elapsed_ns[3], 7);
    timer.elapsed_ns[3] = u64::MAX - 3;
    timer.start_at(3, 10);
    timer.stop_at(3, 15);
    assert_eq!(timer.elapsed_ns[3], 1);
    timer.elapsed_ns[0] = 1_u64 << 53;
    numerical_comparison::assert_close(
        timer.seconds(0),
        9_007_199.254_740_993,
        0.0,
        4.0 * f64::EPSILON,
        "2^53 nanoseconds",
    );
    timer.elapsed_ns[0] = u64::MAX;
    // Binary64 rounds u64::MAX to 2^64 before conversion to seconds.
    numerical_comparison::assert_close(
        timer.seconds(0),
        18_446_744_073.709_553,
        0.0,
        4.0 * f64::EPSILON,
        "u64 maximum nanoseconds",
    );
}

#[test]
fn both_writers_use_original_prefix_and_propagate_directory_errors() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "mvmc-timer-conditions-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    let mut timer = CTimer::<true>::new();
    timer.start_at(0, 100);
    timer.stop_at(0, 1_000_000_100);
    timer.start_at(920, 100);
    timer.stop_at(920, 2_000_000_100);
    let para = timer.write_para_opt(&dir, "zvo").unwrap();
    let diag = timer.write_diag(&dir, "zvo").unwrap();
    assert_eq!(para, dir.join("zvo_CalcTimer.dat"));
    assert_eq!(diag, dir.join("zvo_CalcTimerDiag.dat"));
    assert_eq!(
        fs::read_to_string(&para).unwrap().lines().next().unwrap(),
        "All                         [0]      1.00000"
    );
    assert_eq!(
        fs::read_to_string(&diag).unwrap().lines().next().unwrap(),
        "CalH1 GreenFunc1Real       [920]      2.00000"
    );
    let absent = dir.join("absent");
    assert_eq!(
        timer.write_para_opt(&absent, "zvo").unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
    assert_eq!(
        timer.write_diag(&absent, "zvo").unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
    assert!(timer.write_para_opt(&para, "zvo").is_err());
    assert!(timer.write_diag(&diag, "zvo").is_err());
    fs::remove_file(para).unwrap();
    fs::remove_file(diag).unwrap();
    fs::remove_dir(dir).unwrap();
}

#[test]
fn every_diagnostic_environment_value_and_legacy_precedence_are_explicit() {
    for key in [
        "MVMC_CALHAM1_DIAG",
        "MVMC_SLATER_DIAG",
        "MVMC_MAINCAL_DIAG",
        "MVMC_WEIGHTAVG_DIAG",
    ] {
        for value in [
            None,
            Some("0"),
            Some(""),
            Some("false"),
            Some("1"),
            Some("00"),
        ] {
            let flags = TimerEnv::from_lookup(|name| {
                (name == key).then(|| value.map(str::to_owned)).flatten()
            });
            let expected = value.is_some_and(|value| value != "0");
            assert_eq!(flags.any_diag(), expected);
            assert_eq!(flags.enabled(), expected);
        }
    }
    for primary in [false, true] {
        for legacy in [false, true] {
            let flags = TimerEnv::from_lookup(|name| match name {
                "MVMC_C_TIMER" if primary => Some("1".into()),
                "MVMC_TIMER" if legacy => Some("1".into()),
                _ => None,
            });
            assert_eq!(flags.enabled(), primary || legacy);
            assert_eq!(flags.legacy_warning(), legacy && !primary);
        }
    }
}

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
