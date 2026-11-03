//! `mvmc-cuda`: the normal `mvmc` command line with the CUDA provider registered (issue #464),
//! so `MVMC_RS_SR_BACKEND=cuda[:N]` works. `resident-physcal` explicitly selects
//! the Julia-compatible GPU ensemble and exports its averaged observables.

fn resident_physcal(args: &[std::ffi::OsString]) -> Result<(), String> {
    let usage = "usage: mvmc-cuda resident-physcal NAMELIST PARAMS NEW_OUTPUT_DIR";
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{usage}\nOutputs: energy.dat, green_one.dat, green_two.dat, green_two_ex.dat (ensemble means in parsed operator order).");
        return Ok(());
    }
    if args.len() != 3 {
        return Err(usage.into());
    }
    let output = std::path::PathBuf::from(&args[2]);
    if output.exists() {
        return Err(format!("output already exists: {}", output.display()));
    }
    let result = mvmc_gpu_cuda::physcal::run_gpu_phys_cal_from_namelist(
        std::path::PathBuf::from(&args[0]),
        mvmc_gpu_cuda::physcal::PhysCalConfig {
            opt_para: Some(std::path::PathBuf::from(&args[1])),
            ..Default::default()
        },
    )?;
    std::fs::create_dir(&output).map_err(|e| e.to_string())?;
    for (name, values) in [
        ("energy.dat", std::slice::from_ref(&result.energy)),
        ("green_one.dat", result.green_one.as_slice()),
        ("green_two.dat", result.green_two.as_slice()),
        ("green_two_ex.dat", result.green_two_ex.as_slice()),
    ] {
        let text = values
            .iter()
            .map(|v| format!("{v:.17e}\n"))
            .collect::<String>();
        std::fs::write(output.join(name), text).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        output.join("sampling.txt"),
        format!(
            "walkers {}\nwindows {}\nseed {}\nrng cuRAND XORWOW\n",
            result.walkers, result.nsamples, result.seed
        ),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "Completed resident PhysCal: {} walkers, {} windows; energy={:.17e}",
        result.walkers, result.nsamples, result.energy
    );
    Ok(())
}

fn main() {
    mvmc_gpu_cuda::install();
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "resident-physcal") {
        if let Err(error) = resident_physcal(&args[1..]) {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        eprintln!("GPU extension: mvmc-cuda resident-physcal NAMELIST PARAMS NEW_OUTPUT_DIR\n");
    }
    mvmc_cli::run_cli();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_physcal_requires_explicit_paths_before_cuda() {
        assert!(resident_physcal(&[])
            .unwrap_err()
            .contains("NAMELIST PARAMS"));
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
        let error = resident_physcal(&[
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
