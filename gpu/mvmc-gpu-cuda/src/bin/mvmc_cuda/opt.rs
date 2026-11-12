//! Opt CLI: resolve input controls outside the timed library API, then export
//! final parameters in C's six-diagnostic / parameter-triple record layout.
use mvmc_core::{io::CDouble, InitialDef};
use mvmc_gpu_cuda::opt::{run_gpu_para_opt_from_namelist, OptConfig, OptResult};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HELP: &str = "usage: mvmc-cuda resident-opt NAMELIST NEW_OUTPUT_DIR [INITPARA] [OPTIONS]
Options: --walkers N --steps N --seed S --device N --warmup N --interval N
Warmup and interval overrides count proposals per walker.
Defaults: NVMCSample walkers, NSROptItrStep steps, RndSeed;
warmup = NVMCWarmUp * NVMCInterval * Nsite, interval = NVMCInterval * Nsite.
Outputs: zqp_opt.dat, zvo_out.dat, zvo_SRinfo.dat, sampling.txt.";

struct Command {
    input: PathBuf,
    output: PathBuf,
    config: OptConfig,
}

impl Command {
    fn parse(args: &[OsString]) -> Result<Self, String> {
        let mut paths = Vec::new();
        let mut overrides = Vec::new();
        let mut i = 0;
        while i < args.len() {
            if args[i].to_string_lossy().starts_with("--") {
                let flag = args[i].to_str().ok_or("invalid option name")?;
                if ![
                    "--walkers",
                    "--steps",
                    "--seed",
                    "--device",
                    "--warmup",
                    "--interval",
                ]
                .contains(&flag)
                {
                    return Err(format!("unknown option {flag}\n{HELP}"));
                }
                let value = args
                    .get(i + 1)
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| format!("{flag} needs a value"))?;
                overrides.push((flag, value));
                i += 2;
            } else {
                paths.push(PathBuf::from(&args[i]));
                i += 1;
            }
        }
        if !(2..=3).contains(&paths.len()) {
            return Err(HELP.into());
        }
        if paths[1].exists() {
            return Err(format!("output already exists: {}", paths[1].display()));
        }
        let data =
            mvmc_expert_parsers::parse_expert_mode_files(&paths[0]).map_err(|e| e.to_string())?;
        let m = &data.modpara;
        let length = |x: i64| usize::try_from(x).map_err(|_| "negative run length".to_string());
        let interval = length(m.nvmc_interval)?
            .checked_mul(length(m.nsite)?)
            .ok_or("interval overflow")?;
        let warmup = length(m.nvmc_warmup)?
            .checked_mul(interval)
            .ok_or("warmup overflow")?;
        let mut config = OptConfig {
            nsteps: length(m.nsr_opt_itr_step)?,
            walkers: Some(length(m.nvmc_sample)?),
            warmup: Some(warmup),
            interval: Some(interval),
            initial_def: paths
                .get(2)
                .cloned()
                .map(InitialDef::Path)
                .unwrap_or(InitialDef::Auto),
            ..Default::default()
        };
        for (flag, value) in overrides {
            if flag == "--seed" {
                config.seed = Some(value.parse().map_err(|_| "invalid --seed")?);
                continue;
            }
            let n = value
                .parse::<usize>()
                .map_err(|_| format!("invalid {flag}"))?;
            match flag {
                "--walkers" => config.walkers = Some(n),
                "--steps" => config.nsteps = n,
                "--device" => config.device = n,
                "--warmup" => config.warmup = Some(n),
                "--interval" => config.interval = Some(n),
                _ => unreachable!(),
            }
        }
        if config.nsteps == 0 || config.walkers == Some(0) {
            return Err("walkers and steps must be positive".into());
        }
        Ok(Self {
            input: paths[0].clone(),
            output: paths[1].clone(),
            config,
        })
    }
}

pub(super) fn run(args: &[OsString]) -> Result<(), String> {
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let cmd = Command::parse(args)?;
    let result = run_gpu_para_opt_from_namelist(&cmd.input, cmd.config.clone())?;
    write_result(&cmd.output, &cmd.config, &result).map_err(|e| e.to_string())?;
    println!(
        "Completed resident Opt: {} walkers, {} steps; energy={:.17e}",
        result.walkers,
        result.energies.len(),
        result.energies.last().unwrap()
    );
    Ok(())
}

fn write_result(output: &Path, config: &OptConfig, result: &OptResult) -> std::io::Result<()> {
    let steps = result.energies.len();
    if steps == 0
        || steps != config.nsteps
        || result.parameters.len() != result.parameters_imag.len()
        || !result.cg_iterations.is_empty()
            && (result.cg_iterations.len() != steps || result.cg_fallback.len() != steps)
        || result
            .energies
            .iter()
            .chain(&result.parameters)
            .chain(&result.parameters_imag)
            .any(|x| !x.is_finite())
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "incomplete or nonfinite Opt result",
        ));
    }
    // Atomic directory creation protects an existing run, including a race
    // between the preflight check and completion of the GPU call.
    fs::create_dir(output)?;
    let mut parameters = BufWriter::new(File::create(output.join("zqp_opt.dat"))?);
    // The API has no energy second moment or uncertainty. NaN diagnostics
    // explicitly mean unavailable; C's ReadInitParameter ignores these six fields.
    write!(
        parameters,
        "{} {} {} {} {} {} ",
        CDouble(result.energies[steps - 1]),
        CDouble(0.),
        CDouble(f64::NAN),
        CDouble(f64::NAN),
        CDouble(0.),
        CDouble(f64::NAN)
    )?;
    for (&re, &im) in result.parameters.iter().zip(&result.parameters_imag) {
        write!(
            parameters,
            "{} {} {} ",
            CDouble(re),
            CDouble(im),
            CDouble(0.)
        )?;
    }
    writeln!(parameters)?;
    parameters.flush()?;
    let mut history = BufWriter::new(File::create(output.join("zvo_out.dat"))?);
    let mut sr = BufWriter::new(File::create(output.join("zvo_SRinfo.dat"))?);
    writeln!(history, "# step energy_total_before_update")?;
    writeln!(sr, "# step cg_fallback_used cg_fallback_info cg_info")?;
    for (step, &energy) in result.energies.iter().enumerate() {
        writeln!(history, "{step} {}", CDouble(energy))?;
        let first = result.cg_iterations.get(step).copied().unwrap_or(0);
        let fallback = result.cg_fallback.get(step).copied().flatten();
        writeln!(
            sr,
            "{step} {} {} {first}",
            usize::from(fallback.is_some()),
            fallback.unwrap_or(0)
        )?;
    }
    history.flush()?;
    sr.flush()?;
    fs::write(output.join("sampling.txt"), format!(
        "walkers {}\nsteps {steps}\nwarmup {}\ninterval {}\nseed {}\nrng cuRAND XORWOW\ndevice {}\nsolver {}\nenergy_second_moment unavailable\nparameter_uncertainty unavailable\n",
        result.walkers, config.warmup.unwrap(), config.interval.unwrap(), result.seed, config.device,
        if result.cg_iterations.is_empty() { "direct" } else { "cg" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def")
    }

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mvmc-cli728-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn resolves_input_defaults_and_overrides_before_cuda() {
        let output = scratch("settings");
        let cmd = Command::parse(&[input().into(), output.clone().into()]).unwrap();
        assert_eq!(
            (
                cmd.config.walkers,
                cmd.config.nsteps,
                cmd.config.warmup,
                cmd.config.interval
            ),
            (Some(300), 300, Some(160), Some(16))
        );
        let cmd = Command::parse(&[
            input().into(),
            output.into(),
            "--walkers".into(),
            "8".into(),
            "--steps".into(),
            "2".into(),
            "--seed".into(),
            "79".into(),
            "--warmup".into(),
            "0".into(),
            "--interval".into(),
            "0".into(),
        ])
        .unwrap();
        assert_eq!(
            (
                cmd.config.walkers,
                cmd.config.nsteps,
                cmd.config.seed,
                cmd.config.warmup,
                cmd.config.interval
            ),
            (Some(8), 2, Some(79), Some(0), Some(0))
        );
        assert!(
            Command::parse(&["missing".into(), std::env::temp_dir().into()])
                .err()
                .unwrap()
                .contains("already exists")
        );
    }

    #[test]
    fn output_is_c_parameter_record_and_roundtrips_loaded_values() {
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(input()).unwrap();
        let n = data.count_variational_parameters();
        let result = OptResult {
            energies: vec![-12.25, -12.5],
            parameters: (0..n).map(|k| 0.125 + k as f64 / 17.).collect(),
            parameters_imag: (0..n).map(|k| -0.0625 - k as f64 / 31.).collect(),
            seed: 79,
            walkers: 8,
            cg_iterations: vec![7, -3],
            cg_fallback: vec![None, Some(5)],
        };
        let output = scratch("format");
        let config = OptConfig {
            nsteps: 2,
            warmup: Some(0),
            interval: Some(16),
            ..Default::default()
        };
        write_result(&output, &config, &result).unwrap();
        let params = output.join("zqp_opt.dat");
        assert_eq!(
            fs::read_to_string(&params)
                .unwrap()
                .split_whitespace()
                .count(),
            6 + 3 * n
        );
        assert_eq!(
            mvmc_core::read_opt_para_file(&mut data, &params).unwrap(),
            n
        );
        let loaded = mvmc_core::pack_parameters(&data).unwrap();
        for (k, value) in loaded.iter().enumerate() {
            // These are loaded decimal records, not computed numerical results.
            assert_eq!(value.re.to_bits(), result.parameters[k].to_bits());
            assert_eq!(value.im.to_bits(), result.parameters_imag[k].to_bits());
        }
        assert!(mvmc_core::read_initial_def(&mut data, &params).unwrap());
        assert_eq!(mvmc_core::pack_parameters(&data).unwrap(), loaded);
        let history = fs::read_to_string(output.join("zvo_out.dat")).unwrap();
        assert_eq!(history.lines().count(), 3);
        assert_eq!(
            history
                .lines()
                .nth(2)
                .unwrap()
                .split_whitespace()
                .last()
                .unwrap()
                .parse::<f64>()
                .unwrap(),
            -12.5
        );
        assert_eq!(
            fs::read_to_string(output.join("zvo_SRinfo.dat")).unwrap(),
            "# step cg_fallback_used cg_fallback_info cg_info\n0 0 0 7\n1 1 5 -3\n"
        );
        assert!(fs::read_to_string(output.join("sampling.txt"))
            .unwrap()
            .contains("warmup 0\ninterval 16\nseed 79\nrng cuRAND XORWOW"));
        assert!(write_result(&output, &config, &result).is_err());
        fs::remove_dir_all(output).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA; CLI/API parity, restart and resident PhysCal parameter handoff"]
    fn resident_opt_cli_matches_api_and_restarts() {
        let output = scratch("cuda");
        let args: Vec<OsString> = vec![
            input().into(),
            output.clone().into(),
            "--walkers".into(),
            "8".into(),
            "--steps".into(),
            "2".into(),
            "--seed".into(),
            "79".into(),
            "--warmup".into(),
            "0".into(),
            "--interval".into(),
            "0".into(),
        ];
        let cmd = Command::parse(&args).unwrap();
        let expected = run_gpu_para_opt_from_namelist(&cmd.input, cmd.config.clone()).unwrap();
        run(&args).unwrap();
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(input()).unwrap();
        let params = output.join("zqp_opt.dat");
        mvmc_core::read_opt_para_file(&mut data, &params).unwrap();
        for (actual, expected) in mvmc_core::pack_parameters(&data)
            .unwrap()
            .iter()
            .zip(&expected.parameters)
        {
            assert!((actual.re - expected).abs() <= 1e-12 + 1e-12 * expected.abs());
        }
        let actual_energy: Vec<f64> = fs::read_to_string(output.join("zvo_out.dat"))
            .unwrap()
            .lines()
            .skip(1)
            .map(|line| line.split_whitespace().nth(1).unwrap().parse().unwrap())
            .collect();
        for (a, b) in actual_energy.iter().zip(&expected.energies) {
            assert!((a - b).abs() <= 1e-12 + 1e-12 * b.abs());
        }
        let restart = scratch("restart");
        let mut restart_args = args.clone();
        restart_args[1] = restart.clone().into();
        restart_args.push(params.clone().into());
        run(&restart_args).unwrap();
        let phys = scratch("phys-input");
        fs::create_dir(&phys).unwrap();
        for entry in fs::read_dir(input().parent().unwrap()).unwrap() {
            let p = entry.unwrap().path();
            if p.is_file() {
                fs::copy(&p, phys.join(p.file_name().unwrap())).unwrap();
            }
        }
        let modpara = phys.join("modpara.def");
        fs::write(
            &modpara,
            fs::read_to_string(&modpara)
                .unwrap()
                .replace("NVMCCalMode    0", "NVMCCalMode    1")
                .replace("NVMCWarmUp     10", "NVMCWarmUp     0")
                .replace("NVMCSample     300", "NVMCSample     8"),
        )
        .unwrap();
        let measured = scratch("phys-output");
        super::super::resident_physcal(&[
            phys.join("namelist.def").into(),
            params.into(),
            measured.clone().into(),
        ])
        .unwrap();
        assert!(fs::read_to_string(measured.join("energy.dat"))
            .unwrap()
            .trim()
            .parse::<f64>()
            .unwrap()
            .is_finite());
        for p in [output, restart, phys, measured] {
            fs::remove_dir_all(p).unwrap();
        }
    }
}
