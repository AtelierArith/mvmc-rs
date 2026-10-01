//! BLAS execution policy for the deterministic serial optimizer.
use std::sync::Once;

extern "C" {
    fn openblas_set_num_threads(threads: i32);
}

/// Use the one-thread BLAS setting of the pinned Julia serial parity runs.
///
/// Set this before any SR BLAS call. OpenMP OpenBLAS builds ignore
/// OPENBLAS_NUM_THREADS, so an environment variable alone cannot establish the
/// arithmetic order. The setting is process-wide, as in Julia's BLAS API.
pub(crate) fn initialize() {
    static INITIALIZE: Once = Once::new();
    INITIALIZE.call_once(|| unsafe { openblas_set_num_threads(1) });
}
