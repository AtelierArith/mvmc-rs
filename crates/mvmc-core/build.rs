fn main() {
    // Link against OpenBLAS (LP64) to get dpotrf_ / dpotrs_.
    //
    // On macOS the Homebrew openblas formula installs to a keg-only prefix
    // (/usr/local/opt/openblas on Intel, /opt/homebrew/opt/openblas on ARM)
    // because macOS provides its own BLAS/LAPACK via Accelerate.  We must
    // therefore add the library search path explicitly.
    //
    // Julia uses OpenBLAS (ILP64 variant) internally, so linking the LP64
    // Homebrew build brings us closer to Julia's dpotrf/dpotrs floating-point
    // operation ordering than Accelerate does.  Full bit-identical agreement
    // would require the ILP64 variant with 64-bit integer arguments; LP64 is
    // the practical next-best option without coupling to Julia's private libs.
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
