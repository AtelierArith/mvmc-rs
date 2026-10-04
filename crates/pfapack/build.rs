fn main() {
    if !cfg!(feature = "blas-backend") {
        return;
    }

    if cfg!(target_os = "macos") {
        let brew_prefix = if cfg!(target_arch = "x86_64") {
            "/usr/local/opt/openblas"
        } else {
            "/opt/homebrew/opt/openblas"
        };

        if std::path::Path::new(brew_prefix).join("lib").exists() {
            println!("cargo:rustc-link-search=native={brew_prefix}/lib");
            println!("cargo:rustc-link-lib=dylib=openblas");
        } else {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
