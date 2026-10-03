//! Original Julia threading assertions M0888/M0889, using Rust's existing
//! public timer API. No new local-accumulator API or reference runtime.
use mvmc_core::c_timer::{CTimer, CTIMER_N};

#[test]
fn original_two_local_timer_sums_are_25_and_20() {
    // test_unit_threading.jl:233–241: local1 ID4 +=10,
    // local2 ID4 +=15 and ID7 +=20, then merge in (local1,local2) order.
    // Julia storage [5]/[8] and Rust storage [4]/[7] share C timer IDs.
    // Rust records explicit elapsed intervals rather than an add_elapsed API.
    let mut parent = CTimer::<true>::new();
    let mut local1 = CTimer::<true>::new();
    let mut local2 = CTimer::<true>::new();
    local1.start_at(4, 0);
    local1.stop_at(4, 10);
    local2.start_at(4, 0);
    local2.stop_at(4, 15);
    local2.start_at(7, 0);
    local2.stop_at(7, 20);
    parent.merge_elapsed(&local1);
    parent.merge_elapsed(&local2);
    assert_eq!(parent.elapsed_ns[4], 25);
    assert_eq!(parent.elapsed_ns[7], 20);
    let mut expected = [0; CTIMER_N];
    expected[4] = 25;
    expected[7] = 20;
    assert_eq!(parent.elapsed_ns, expected);
    assert!(parent.start_ns.iter().all(|&value| value == 0));
    assert_eq!(
        local1.elapsed_ns[4], 10,
        "merge leaves local payload intact"
    );
    assert_eq!(local2.elapsed_ns[4], 15);
    assert_eq!(local2.elapsed_ns[7], 20);
    parent.reset();
    assert!(parent.elapsed_ns.iter().all(|&value| value == 0));
    assert!(parent.start_ns.iter().all(|&value| value == 0));
}

#[test]
fn enabled_timer_rejects_original_upper_bound_id_before_publication() {
    // M0891's ID=CTIMER_N rejection, adapted to the existing Rust API:
    // bounds panic, not Julia's ErrorException from ctimer_add_elapsed!.
    let mut timer = CTimer::<true>::new();
    timer.start_at(4, 0);
    timer.stop_at(4, 25);
    let elapsed = timer.elapsed_ns;
    let starts = timer.start_ns;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        timer.start_at(CTIMER_N, 1);
    }))
    .is_err());
    assert_eq!(timer.elapsed_ns, elapsed);
    assert_eq!(timer.start_ns, starts);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        timer.stop_at(CTIMER_N, 1);
    }))
    .is_err());
    assert_eq!(timer.elapsed_ns, elapsed);
    assert_eq!(timer.start_ns, starts);
}
