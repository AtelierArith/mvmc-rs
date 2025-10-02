fn main() {
    // Link against LAPACK library
    // This assumes LAPACK is installed via Homebrew on macOS or system package manager on Linux

    #[cfg(target_os = "macos")]
    {
        // On macOS, use Accelerate framework which includes LAPACK
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(target_os = "linux")]
    {
        // On Linux, link against system LAPACK/BLAS
        println!("cargo:rustc-link-lib=lapack");
        println!("cargo:rustc-link-lib=blas");
    }

    #[cfg(target_os = "windows")]
    {
        // On Windows, you might need to install and link LAPACK manually
        // This is a placeholder - adjust based on your LAPACK installation
        println!("cargo:rustc-link-lib=lapack");
        println!("cargo:rustc-link-lib=blas");
    }
}
