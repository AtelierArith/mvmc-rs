//! `mvmc` binary: see the `mvmc_cli` library for the driver and its options.

fn main() {
    let observation = std::env::var("MVMC_RS_INNER_OBSERVE")
        .is_ok_and(|value| value == "1")
        .then(mvmc_core::threading::start_observation);
    // With inner threads requested (`MVMC_RS_INNER_THREADS > 1`) the whole driver runs on one
    // inner-pool worker: kernel regions are then dispatched from inside the pool (no latch
    // sleep of the caller, workers stay hot between nearby regions), which is what lets the
    // Rayon pool scale (issue #479). The closure captures nothing, MPI (when enabled) is
    // initialized and finalized on this same thread, and the results do not depend on which
    // thread runs the driver.
    if mvmc_core::threading::inner_thread_config().threads > 1 {
        mvmc_core::threading::install(mvmc_cli::run_cli);
    } else {
        mvmc_cli::run_cli();
    }
    if let Some(observation) = observation {
        eprintln!("inner-execution: {:?}", observation.finish());
    }
}
