// Shared BLAS/LAPACK provider selection for the build scripts of every crate that calls
// BLAS/LAPACK symbols (`include!`d from `crates/*/build.rs`, issue #474).
//
// The provider is chosen at the binary leaf with the environment variable
// `MVMC_BLAS_PROVIDER` (a build-time setting, exported for the whole `cargo` invocation):
//
//   openblas    (default) `-lopenblas`; on macOS the Homebrew keg-only prefix is searched.
//   accelerate  macOS Accelerate framework (macOS only).
//   mkl         Intel oneMKL single dynamic library `-lmkl_rt` (LP64 interface; `MKLROOT`
//               adds `$MKLROOT/lib` and `$MKLROOT/lib/intel64` to the search path).
//   netlib      reference/netlib `-lblas -llapack`.
//
// All providers must export the LP64 Fortran symbols (`dgemm_`, `dpotrf_`, ...).
// Linux OpenBLAS stays the default and the numerical reference environment: with the
// variable unset the emitted directives are exactly the former ones (plus a `rustc-cfg`
// that only enables `openblas_set_num_threads`, as before).

fn mvmc_blas_provider() -> String {
    println!("cargo:rerun-if-env-changed=MVMC_BLAS_PROVIDER");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    std::env::var("MVMC_BLAS_PROVIDER")
        .map(|v| v.trim().to_ascii_lowercase())
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "openblas".to_string())
}

/// Emit the link directives. `accelerate_fallback`: with the default provider on macOS and no
/// Homebrew OpenBLAS, use Accelerate (the historical behavior of the standalone `pfapack`).
fn mvmc_link_blas(accelerate_fallback: bool) {
    let macos = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    let x86_64 = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "x86_64");
    let provider = mvmc_blas_provider();
    println!("cargo::rustc-check-cfg=cfg(mvmc_blas_openblas)");
    println!("cargo::rustc-check-cfg=cfg(mvmc_blas_mkl)");
    match provider.as_str() {
        "openblas" => {
            println!("cargo:rustc-cfg=mvmc_blas_openblas");
            if macos {
                // Homebrew keg-only prefix: Intel vs Apple Silicon.
                let prefix = if x86_64 {
                    "/usr/local/opt/openblas"
                } else {
                    "/opt/homebrew/opt/openblas"
                };
                let present = std::path::Path::new(prefix).join("lib").exists();
                if present || !accelerate_fallback {
                    println!("cargo:rustc-link-search=native={prefix}/lib");
                    println!("cargo:rustc-link-lib=dylib=openblas");
                } else {
                    println!("cargo:rustc-link-lib=framework=Accelerate");
                }
            } else {
                println!("cargo:rustc-link-lib=dylib=openblas");
            }
        }
        "accelerate" => {
            assert!(
                macos,
                "MVMC_BLAS_PROVIDER=accelerate is only available on macOS"
            );
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        "mkl" => {
            println!("cargo:rustc-cfg=mvmc_blas_mkl");
            if let Ok(root) = std::env::var("MKLROOT") {
                for sub in ["lib", "lib/intel64"] {
                    println!("cargo:rustc-link-search=native={root}/{sub}");
                }
            }
            println!("cargo:rustc-link-lib=dylib=mkl_rt");
        }
        "netlib" => {
            println!("cargo:rustc-link-lib=dylib=blas");
            println!("cargo:rustc-link-lib=dylib=lapack");
        }
        other => panic!(
            "unknown MVMC_BLAS_PROVIDER `{other}` (expected openblas, accelerate, mkl or netlib)"
        ),
    }
}
