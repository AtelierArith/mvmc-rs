//! BLAS execution policy for the deterministic serial optimizer.
use std::sync::Once;

#[cfg(mvmc_blas_openblas)]
extern "C" {
    fn openblas_set_num_threads(threads: i32);
}

#[cfg(mvmc_blas_mkl)]
extern "C" {
    fn MKL_Set_Num_Threads(threads: i32);
}

/// Environment variable overriding the BLAS thread count (`MVMC_RS_BLAS_THREADS`).
///
/// The **default is 1** and must stay 1: the deterministic serial parity runs pin the BLAS to
/// one thread so that every BLAS-backed kernel (PfaPack `trmm`, SR Gram/solve) keeps the C
/// summation order. Values above 1 are a **benchmark-only policy** (the 4-thread CPU baseline of
/// issue #518); they change the arithmetic order of every BLAS call and are not parity numbers.
pub const BLAS_THREADS_VARIABLE: &str = "MVMC_RS_BLAS_THREADS";

/// The thread count selected by [`BLAS_THREADS_VARIABLE`], defaulting to the serial policy `1`.
///
/// An unparsable or zero value is treated as the default (the selector is a benchmark knob, not
/// a validation input); negative values are not meaningful and are rejected the same way.
pub fn configured_threads() -> i32 {
    std::env::var(BLAS_THREADS_VARIABLE)
        .ok()
        .and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(1)
}

/// Use the one-thread BLAS setting of the pinned Julia serial parity runs.
///
/// Set this before any SR BLAS call. OpenMP OpenBLAS builds ignore
/// OPENBLAS_NUM_THREADS, so an environment variable alone cannot establish the
/// arithmetic order. The setting is process-wide, as in Julia's BLAS API.
/// Only the OpenBLAS and MKL providers expose a runtime thread setter; other providers
/// (Accelerate, netlib) are used as linked (`MVMC_BLAS_PROVIDER`, issue #474).
///
/// [`BLAS_THREADS_VARIABLE`] overrides the count (benchmark policy); the default remains the
/// deterministic serial `1`.
pub(crate) fn initialize() {
    static INITIALIZE: Once = Once::new();
    INITIALIZE.call_once(|| {
        let threads = configured_threads();
        #[cfg(mvmc_blas_openblas)]
        unsafe {
            openblas_set_num_threads(threads)
        };
        #[cfg(mvmc_blas_mkl)]
        unsafe {
            MKL_Set_Num_Threads(threads)
        };
        #[cfg(not(any(mvmc_blas_openblas, mvmc_blas_mkl)))]
        let _ = threads;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The selector parses positive values and falls back to the serial default otherwise.
    #[test]
    fn blas_threads_selector_defaults_to_one() {
        let guard = std::env::var_os(BLAS_THREADS_VARIABLE);
        // SAFETY: the test runs single-threaded here and restores the variable.
        unsafe {
            std::env::set_var(BLAS_THREADS_VARIABLE, "4");
        }
        assert_eq!(configured_threads(), 4);
        unsafe {
            std::env::set_var(BLAS_THREADS_VARIABLE, "0");
        }
        assert_eq!(configured_threads(), 1);
        unsafe {
            std::env::set_var(BLAS_THREADS_VARIABLE, "nope");
        }
        assert_eq!(configured_threads(), 1);
        unsafe {
            std::env::remove_var(BLAS_THREADS_VARIABLE);
        }
        assert_eq!(configured_threads(), 1);
        if let Some(value) = guard {
            unsafe {
                std::env::set_var(BLAS_THREADS_VARIABLE, value);
            }
        }
    }
}
