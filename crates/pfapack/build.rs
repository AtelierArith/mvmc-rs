fn main() {
    if !cfg!(feature = "blas-backend") {
        return;
    }
    // The provider is selected by `MVMC_BLAS_PROVIDER` (openblas by default, see
    // build_support/blas_provider.rs). Without Homebrew OpenBLAS the standalone crate keeps
    // its historical macOS fallback to Accelerate.
    println!("cargo:rerun-if-changed=../../build_support/blas_provider.rs");
    mvmc_link_blas(true);
}

include!("../../build_support/blas_provider.rs");
