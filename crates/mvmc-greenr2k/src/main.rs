//! `greenr2k <NameList> <Geometry>`: Fourier transform of mVMC/HPhi correlation
//! functions (port of `tool/greenr2k.F90`). Works in the current directory.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <namelist.def> <geometry.dat>",
            args.first().map_or("greenr2k", |s| s)
        );
        return ExitCode::from(2);
    }
    let stdout = std::io::stdout();
    let mut lock = std::io::BufWriter::new(stdout.lock());
    let result = mvmc_greenr2k::run(&args[1], &args[2], Path::new("."), &mut lock);
    let _ = lock.flush();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(mvmc_greenr2k::Error::MissingIndices) => {
            eprintln!("STOP Missing indices for the Green function.");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("greenr2k: error: {e}");
            ExitCode::from(2)
        }
    }
}
