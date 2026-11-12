//! PhysCal CLI and window output, kept outside the timed library API.
use mvmc_gpu_cuda::physcal::{run_gpu_phys_cal_from_namelist, PhysCalConfig, PhysCalResult};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HELP: &str =
    "usage: mvmc-cuda resident-physcal NAMELIST PARAMS NEW_OUTPUT_DIR [--green-windows]
Outputs: energy.dat, energy_imag.dat, energy_second_moment.dat, energy_windows.dat,
green_one.dat, green_two.dat, green_two_ex.dat, sampling.txt.
--green-windows also exports window/operator/real/imaginary Green rows.";

pub(super) fn run(args: &[OsString]) -> Result<(), String> {
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let window_greens = args.iter().any(|s| s == "--green-windows");
    let paths: Vec<_> = args
        .iter()
        .filter(|s| *s != "--green-windows")
        .map(PathBuf::from)
        .collect();
    if paths.len() != 3 {
        return Err(HELP.into());
    }
    if paths[2].exists() {
        return Err(format!("output already exists: {}", paths[2].display()));
    }
    let result = run_gpu_phys_cal_from_namelist(
        &paths[0],
        PhysCalConfig {
            opt_para: Some(paths[1].clone()),
            window_greens,
            ..Default::default()
        },
    )?;
    write_result(&paths[2], &result).map_err(|e| e.to_string())?;
    println!(
        "Completed resident PhysCal: {} walkers, {} windows; energy={:.17e}",
        result.walkers, result.nsamples, result.energy
    );
    Ok(())
}

fn write_result(output: &Path, result: &PhysCalResult) -> std::io::Result<()> {
    fs::create_dir(output)?;
    for (name, values) in [
        ("energy.dat", std::slice::from_ref(&result.energy)),
        ("energy_imag.dat", std::slice::from_ref(&result.energy_imag)),
        (
            "energy_second_moment.dat",
            std::slice::from_ref(&result.energy_second_moment),
        ),
        ("green_one.dat", result.green_one.as_slice()),
        ("green_two.dat", result.green_two.as_slice()),
        ("green_two_ex.dat", result.green_two_ex.as_slice()),
    ] {
        let mut file = BufWriter::new(File::create(output.join(name))?);
        for v in values {
            writeln!(file, "{v:.17e}")?;
        }
        file.flush()?;
    }
    let mut file = BufWriter::new(File::create(output.join("energy_windows.dat"))?);
    writeln!(
        file,
        "# window energy_real energy_imag energy_second_moment"
    )?;
    for (window, e) in result.energy_windows.iter().enumerate() {
        writeln!(
            file,
            "{window} {:.17e} {:.17e} {:.17e}",
            e.energy, e.energy_imag, e.energy_second_moment
        )?;
    }
    file.flush()?;
    if let Some(windows) = &result.green_windows {
        for kind in 0..3 {
            let name = [
                "green_one_windows.dat",
                "green_two_windows.dat",
                "green_two_ex_windows.dat",
            ][kind];
            let mut file = BufWriter::new(File::create(output.join(name))?);
            writeln!(file, "# window operator real imaginary")?;
            for (window, g) in windows.iter().enumerate() {
                let (re, im) = [
                    (&g.green_one, &g.green_one_imag),
                    (&g.green_two, &g.green_two_imag),
                    (&g.green_two_ex, &g.green_two_ex_imag),
                ][kind];
                for (row, (&re, &im)) in re.iter().zip(im).enumerate() {
                    writeln!(file, "{window} {row} {re:.17e} {im:.17e}")?;
                }
            }
            file.flush()?;
        }
    }
    fs::write(output.join("sampling.txt"), format!(
        "walkers {}\nwindows {}\nwarmup {}\ninterval {}\nseed {}\nrng cuRAND XORWOW\nwindow_greens {}\nenergy_second_moment mean_abs_local_energy_squared\n",
        result.walkers,result.nsamples,result.warmup,result.interval,result.seed,result.green_windows.is_some()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvmc_gpu_cuda::physcal::{EnergyWindow, GreenWindow};

    #[test]
    fn window_output_has_explicit_indices_and_opt_in_green_rows() {
        let output = std::env::temp_dir().join(format!(
            "mvmc-cli729-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut result = PhysCalResult {
            energy: 1.5,
            energy_imag: 0.,
            energy_second_moment: 2.5,
            green_one: vec![0.3],
            green_two: vec![],
            green_two_ex: vec![],
            green_one_imag: vec![0.],
            green_two_imag: vec![],
            green_two_ex_imag: vec![],
            seed: 79,
            walkers: 8,
            nsamples: 2,
            warmup: 128,
            interval: 16,
            energy_windows: vec![
                EnergyWindow {
                    energy: 1.,
                    energy_imag: 0.,
                    energy_second_moment: 1.,
                },
                EnergyWindow {
                    energy: 2.,
                    energy_imag: 0.,
                    energy_second_moment: 4.,
                },
            ],
            green_windows: None,
        };
        write_result(&output, &result).unwrap();
        let text = fs::read_to_string(output.join("energy_windows.dat")).unwrap();
        let rows: Vec<Vec<f64>> = text
            .lines()
            .skip(1)
            .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
            .collect();
        assert_eq!(rows, [vec![0., 1., 0., 1.], vec![1., 2., 0., 4.]]);
        assert!(!output.join("green_one_windows.dat").exists());
        assert_eq!(
            fs::read_to_string(output.join("energy_second_moment.dat"))
                .unwrap()
                .trim()
                .parse::<f64>()
                .unwrap(),
            2.5
        );
        fs::remove_dir_all(&output).unwrap();
        result.green_windows = Some(vec![
            GreenWindow {
                green_one: vec![0.2],
                green_one_imag: vec![0.05],
                ..Default::default()
            },
            GreenWindow {
                green_one: vec![0.4],
                green_one_imag: vec![-0.05],
                ..Default::default()
            },
        ]);
        write_result(&output, &result).unwrap();
        let text = fs::read_to_string(output.join("green_one_windows.dat")).unwrap();
        let rows: Vec<Vec<f64>> = text
            .lines()
            .skip(1)
            .map(|l| l.split_whitespace().map(|s| s.parse().unwrap()).collect())
            .collect();
        assert_eq!(rows, [vec![0., 0., 0.2, 0.05], vec![1., 0., 0.4, -0.05]]);
        assert!(fs::read_to_string(output.join("sampling.txt"))
            .unwrap()
            .contains("warmup 128\ninterval 16\nseed 79"));
        assert!(write_result(&output, &result).is_err());
        fs::remove_dir_all(output).unwrap();
    }
}
