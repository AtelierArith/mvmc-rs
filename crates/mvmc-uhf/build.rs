fn main() {
    // Link the BLAS/LAPACK provider (LP64) selected by `MVMC_BLAS_PROVIDER`
    // (openblas by default, see build_support/blas_provider.rs).
    println!("cargo:rerun-if-changed=../../build_support/blas_provider.rs");
    mvmc_link_blas(false);
}

include!("../../build_support/blas_provider.rs");
