//! `mvmc-cuda`: command line for device-resident Opt and PhysCal.
//! The CPU C-order runner is the `mvmc` binary of `mvmc-cli`.

#[path = "mvmc_cuda/opt.rs"]
mod opt;
#[path = "mvmc_cuda/physcal.rs"]
mod physcal;

const USAGE: &str = "usage: mvmc-cuda resident-physcal NAMELIST PARAMS NEW_OUTPUT_DIR [--green-windows]\n       mvmc-cuda resident-opt NAMELIST NEW_OUTPUT_DIR [INITPARA] [OPTIONS]";

/// Dispatch one subcommand; only the resident GPU subcommands exist.
fn run(args: &[std::ffi::OsString]) -> Result<(), String> {
    match args.first() {
        Some(arg) if arg == "resident-physcal" => physcal::run(&args[1..]),
        Some(arg) if arg == "resident-opt" => opt::run(&args[1..]),
        Some(arg) if arg == "--help" || arg == "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        Some(arg) => Err(format!(
            "unknown subcommand {}\n{USAGE}",
            arg.to_string_lossy()
        )),
        None => Err(USAGE.into()),
    }
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cpu_runner_invocation_and_unknown_subcommands() {
        assert!(run(&[]).unwrap_err().contains("resident-physcal"));
        let error = run(&["namelist.def".into()]).unwrap_err();
        assert!(error.contains("unknown subcommand namelist.def"));
    }

    #[test]
    fn resident_physcal_requires_explicit_paths_before_cuda() {
        assert!(physcal::run(&[]).unwrap_err().contains("NAMELIST PARAMS"));
    }

    #[test]
    fn resident_physcal_preserves_existing_output_before_cuda() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output = std::env::temp_dir().join(format!("mvmc-resident-cli-{nonce}"));
        std::fs::create_dir(&output).unwrap();
        std::fs::write(output.join("keep"), "user data").unwrap();
        let error = physcal::run(&[
            "missing-namelist".into(),
            "missing-params".into(),
            output.clone().into_os_string(),
        ])
        .unwrap_err();
        assert!(error.contains("already exists"));
        assert_eq!(
            std::fs::read_to_string(output.join("keep")).unwrap(),
            "user data"
        );
        std::fs::remove_dir_all(output).unwrap();
    }
}
