fn main() {
    // Recorded in benchmark/validation metadata (`accel_validation::BenchMetadata`).
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "rustc unknown".to_string());
    println!("cargo:rustc-env=MVMC_RS_RUSTC_VERSION={version}");
    // Link a BLAS/LAPACK provider (LP64) for CG dgemv_ and direct-SR dpotrf_ / dpotrs_.
    // `MVMC_BLAS_PROVIDER` selects it at the binary leaf (openblas by default; accelerate,
    // mkl, netlib; see build_support/blas_provider.rs and the manual's build chapter).
    //
    // Julia uses OpenBLAS with the ILP64 integer ABI. The Rust BLAS/LAPACK wrappers use
    // LP64; argument widths must match their linked backend. ABI width alone does not
    // determine floating-point parity. Backend version, CPU kernels, reduction order, and
    // solver inputs all matter. Fixed-input CG fixtures use explicit numerical bounds and
    // independently materialized backward residuals against Julia 1.13.1 references.
    println!("cargo:rerun-if-changed=../../build_support/blas_provider.rs");
    mvmc_link_blas(false);
}

include!("../../build_support/blas_provider.rs");
