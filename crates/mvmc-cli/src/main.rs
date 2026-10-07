//! `mvmc` binary: see the `mvmc_cli` library for the driver and its options.

fn main() {
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
}
