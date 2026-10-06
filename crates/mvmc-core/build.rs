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
    // Link OpenBLAS (LP64) for CG dgemv_ and direct-SR dpotrf_ / dpotrs_.
    //
    // On macOS the Homebrew openblas formula installs to a keg-only prefix
    // (/usr/local/opt/openblas on Intel, /opt/homebrew/opt/openblas on ARM)
    // because macOS provides its own BLAS/LAPACK via Accelerate.  We must
    // therefore add the library search path explicitly.
    //
    // Julia uses OpenBLAS with the ILP64 integer ABI. The Rust BLAS/LAPACK
    // wrappers use LP64; argument widths must match their linked backend.
    // ABI width alone does not determine floating-point parity. Backend
    // version, CPU kernels, reduction order, and solver inputs all matter.
    // Fixed-input CG fixtures use explicit numerical bounds and independently
    // materialized backward residuals against Julia 1.13.1 references.
    //
    // Linux: a system `libopenblas-dev` / `liblapack-dev` package satisfies
    // `-l openblas` just the same.
    if cfg!(target_os = "macos") {
        // Homebrew keg-only prefix — Intel vs Apple Silicon.
        let brew_prefix = if cfg!(target_arch = "x86_64") {
            "/usr/local/opt/openblas"
        } else {
            "/opt/homebrew/opt/openblas"
        };
        println!("cargo:rustc-link-search=native={brew_prefix}/lib");
        println!("cargo:rustc-link-lib=dylib=openblas");
    } else {
        // Linux: OpenBLAS or reference LAPACK installed system-wide.
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
