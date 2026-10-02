//! Runtime control for deterministic shared-memory inner kernels.

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Default minimum number of independent QP items before inner threading is
/// worthwhile. The sequential path remains the default for small workloads.
pub const DEFAULT_INNER_THRESHOLD: usize = 32;

/// Runtime configuration for independent inner work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnerThreadConfig {
    /// Number of worker threads requested by `MVMC_RS_INNER_THREADS`.
    pub threads: usize,
    /// Minimum work items required to enable parallel execution.
    pub threshold: usize,
}

/// Read the process-wide inner-kernel controls once.
///
/// `MVMC_RS_INNER_THREADS` defaults to one, preserving the historical
/// sequential execution and RNG/Markov-chain contract. `MVMC_RS_INNER_THRESHOLD`
/// defaults to [`DEFAULT_INNER_THRESHOLD`]. Invalid or zero values fall back
/// to those defaults.
pub fn inner_thread_config() -> InnerThreadConfig {
    static CONFIG: OnceLock<InnerThreadConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| InnerThreadConfig {
        threads: std::env::var("MVMC_RS_INNER_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(1),
        threshold: std::env::var("MVMC_RS_INNER_THRESHOLD")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(DEFAULT_INNER_THRESHOLD),
    })
}

/// Return whether a range should use the shared-memory worker pool.
pub fn inner_parallel_enabled(work_items: usize) -> bool {
    let config = inner_thread_config();
    config.threads > 1 && work_items >= config.threshold
}

/// Number of workers to provision for a work range.
pub fn inner_worker_count(work_items: usize) -> usize {
    if inner_parallel_enabled(work_items) {
        inner_thread_config().threads.min(work_items)
    } else {
        1
    }
}

/// Execute a closure on the configured deterministic Rayon pool.
pub fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(inner_thread_config().threads.max(1))
            .thread_name(|index| format!("mvmc-inner-{index}"))
            .build()
            .expect("inner Rayon pool must build")
    });
    pool.install(operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_default_is_safe_for_small_work() {
        // Environment-independent assertion: a zero-sized range never needs
        // a worker, even when a caller configured multiple threads.
        assert_eq!(inner_worker_count(0), 1);
    }
}
