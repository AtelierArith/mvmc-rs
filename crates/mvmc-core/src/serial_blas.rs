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

/// Use the one-thread BLAS setting of the pinned Julia serial parity runs.
///
/// Set this before any SR BLAS call. OpenMP OpenBLAS builds ignore
/// OPENBLAS_NUM_THREADS, so an environment variable alone cannot establish the
/// arithmetic order. The setting is process-wide, as in Julia's BLAS API.
/// Only the OpenBLAS and MKL providers expose a runtime thread setter; other providers
/// (Accelerate, netlib) are used as linked (`MVMC_BLAS_PROVIDER`, issue #474).
pub(crate) fn initialize() {
    static INITIALIZE: Once = Once::new();
    INITIALIZE.call_once(|| {
        #[cfg(mvmc_blas_openblas)]
        unsafe {
            openblas_set_num_threads(1)
        };
        #[cfg(mvmc_blas_mkl)]
        unsafe {
            MKL_Set_Num_Threads(1)
        };
    });
}
