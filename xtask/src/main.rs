//! Workspace task runner.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MODELS: &[Model] = &[
    Model {
        name: "heisenberg_chain_real",
        julia_mode: "real",
        julia_input_dir: "heisenberg_chain_real",
    },
    Model {
        name: "heisenberg_chain_cmp",
        julia_mode: "cmp",
        julia_input_dir: "heisenberg_chain_cmp",
    },
    Model {
        name: "heisenberg_chain_fsz",
        julia_mode: "fsz",
        julia_input_dir: "heisenberg_chain_fsz",
    },
    Model {
        name: "hubbard_chain",
        julia_mode: "real",
        julia_input_dir: "hubbard_chain_real",
    },
];

#[derive(Debug, Clone, Copy)]
struct Model {
    name: &'static str,
    julia_mode: &'static str,
    julia_input_dir: &'static str,
}

/// Commit-checked Hubbard-chain inputs generated with C StdFace (see
/// `benchmark/hubbard_chain/README.md`). Used by `bench-hubbard`.
const HUBBARD_MODELS: &[HubbardModel] = &[
    HubbardModel {
        name: "hubbard_chain_L16",
    },
    HubbardModel {
        name: "hubbard_chain_L24",
    },
    HubbardModel {
        name: "hubbard_chain_L32",
    },
    HubbardModel {
        name: "hubbard_chain_L64",
    },
];

#[derive(Debug, Clone, Copy)]
struct HubbardModel {
    name: &'static str,
}

#[derive(Debug)]
struct BenchConfig {
    steps: usize,
    reps: usize,
    warmups: usize,
    models: Vec<Model>,
    julia_root: PathBuf,
    csv: PathBuf,
    keep_output: bool,
    threads: Option<usize>,
    inner: InnerEnv,
    julia_bin: PathBuf,
}

#[derive(Debug, Clone)]
struct Measurement {
    implementation: &'static str,
    model: String,
    rep: usize,
    steps: usize,
    seconds: f64,
    final_energy_per_site: Option<f64>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None | Some("help") | Some("--help") | Some("-h") => print_help(),
        Some("bench-julia") => {
            let rest: Vec<String> = args.collect();
            if let Err(e) = bench_julia_vs_rust(&rest) {
                eprintln!("xtask bench-julia: {e}");
                std::process::exit(1);
            }
        }
        Some("bench-hubbard") => {
            let rest: Vec<String> = args.collect();
            if let Err(e) = bench_hubbard(&rest) {
                eprintln!("xtask bench-hubbard: {e}");
                std::process::exit(1);
            }
        }
        Some("bench-physcal") => {
            let rest: Vec<String> = args.collect();
            if let Err(e) = bench_physcal(&rest) {
                eprintln!("xtask bench-physcal: {e}");
                std::process::exit(1);
            }
        }
        Some("bench-physcal-hubbard") => {
            let rest: Vec<String> = args.collect();
            if let Err(e) = bench_physcal_hubbard(&rest) {
                eprintln!("xtask bench-physcal-hubbard: {e}");
                std::process::exit(1);
            }
        }
        Some(task) => {
            eprintln!("xtask: unknown task `{task}`");
            print_help();
            std::process::exit(2);
        }
    }
}

fn bench_julia_vs_rust(args: &[String]) -> Result<(), String> {
    let workspace = workspace_root();
    let config = parse_bench_args(args, &workspace)?;

    println!("=== Julia-mVMC vs Rust benchmark ===");
    println!("steps      : {}", config.steps);
    println!("warmups    : {}", config.warmups);
    println!("reps       : {}", config.reps);
    match config.threads {
        Some(n) => println!("threads    : {n} (pinned on both Rust & Julia)"),
        None => println!("threads    : <inherit env> (consider --threads N for fairness)"),
    }
    println!("rust inner : {}", config.inner.describe());
    println!("pfapack    : BLAS/LAPACK (required for Julia numerical parity)");
    println!("SR backend : BLAS GEMV for CG; LAPACK for the direct solver");
    println!("julia root : {}", config.julia_root.display());
    println!("julia bin  : {}", config.julia_bin.display());
    println!("csv        : {}", config.csv.display());
    println!();

    build_rust_examples(&workspace)?;

    let run_root = workspace
        .join("target")
        .join("bench-output")
        .join(format!("run-{}", unix_timestamp_millis()));
    fs::create_dir_all(&run_root)
        .map_err(|e| format!("cannot create {}: {e}", run_root.display()))?;

    let julia_runner = write_julia_runner(&workspace)?;
    let mut measurements = Vec::new();

    for model in &config.models {
        println!("--- {} ---", model.name);
        let rust = run_rust_model(&workspace, &config.julia_root, &run_root, *model, &config)?;
        print_summary("rust", &rust);
        measurements.extend(rust);

        let julia = run_julia_model(
            &config.julia_root,
            &julia_runner,
            &run_root,
            *model,
            &config,
        )?;
        print_summary("julia", &julia);
        measurements.extend(julia);
        println!();
    }

    write_csv(&config.csv, &measurements)?;
    write_run_config(&config)?;
    print_comparison(&measurements);

    if config.keep_output {
        println!("kept outputs: {}", run_root.display());
    } else if let Err(e) = fs::remove_dir_all(&run_root) {
        eprintln!("warning: failed to remove {}: {e}", run_root.display());
    }

    Ok(())
}

fn parse_bench_args(args: &[String], workspace: &Path) -> Result<BenchConfig, String> {
    let mut steps = 10usize;
    let mut reps = 3usize;
    let mut warmups = 1usize;
    let mut model_names: Vec<String> = Vec::new();
    let mut julia_root = workspace.join("extern/Julia-mVMC");
    let mut csv = workspace
        .join("target")
        .join("bench")
        .join("julia_vs_rust.csv");
    let mut keep_output = false;
    let mut threads: Option<usize> = None;
    let mut inner = InnerEnv::default();
    let mut julia_bin = PathBuf::from("julia");

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                print_bench_help();
                std::process::exit(0);
            }
            "--steps" => {
                idx += 1;
                steps = parse_value(args.get(idx), "--steps")?;
            }
            "--reps" => {
                idx += 1;
                reps = parse_value(args.get(idx), "--reps")?;
            }
            "--warmups" => {
                idx += 1;
                warmups = parse_value(args.get(idx), "--warmups")?;
            }
            "--threads" => {
                idx += 1;
                threads = Some(parse_value(args.get(idx), "--threads")?);
            }
            "--model" => {
                idx += 1;
                model_names.push(
                    args.get(idx)
                        .ok_or_else(|| "--model requires a value".to_string())?
                        .clone(),
                );
            }
            "--julia-root" => {
                idx += 1;
                julia_root = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-root requires a value".to_string())?,
                );
            }
            "--julia-bin" => {
                idx += 1;
                julia_bin = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-bin requires a value".to_string())?,
                );
            }
            "--csv" => {
                idx += 1;
                csv = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--csv requires a value".to_string())?,
                );
            }
            "--keep-output" => keep_output = true,
            flag => {
                if !inner.parse_flag(flag, args.get(idx + 1))? {
                    return Err(format!("unknown bench-julia flag `{flag}`"));
                }
                idx += 1;
            }
        }
        idx += 1;
    }

    if steps == 0 {
        return Err("--steps must be positive".to_string());
    }
    if reps == 0 {
        return Err("--reps must be positive".to_string());
    }

    let julia_root = canonicalize_existing_dir(&julia_root, "Julia-mVMC root")?;
    let models = if model_names.is_empty() {
        MODELS.to_vec()
    } else {
        model_names
            .iter()
            .map(|name| {
                MODELS
                    .iter()
                    .copied()
                    .find(|model| model.name == name)
                    .ok_or_else(|| format!("unknown model `{name}`"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(BenchConfig {
        steps,
        reps,
        warmups,
        models,
        julia_root,
        csv,
        keep_output,
        threads,
        inner,
        julia_bin,
    })
}

fn parse_value<T>(value: Option<&String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<T>()
        .map_err(|_| format!("invalid value for {flag}"))
}

fn build_rust_examples(workspace: &Path) -> Result<(), String> {
    println!("building Rust release examples...");
    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--release")
        .arg("-p")
        .arg("mvmc-cli")
        .arg("--examples");
    let output = command
        .current_dir(workspace)
        .output()
        .map_err(|e| format!("failed to spawn cargo build: {e}"))?;
    ensure_success("cargo build --release -p mvmc-cli --examples", &output)?;
    Ok(())
}

fn run_rust_model(
    workspace: &Path,
    julia_root: &Path,
    run_root: &Path,
    model: Model,
    config: &BenchConfig,
) -> Result<Vec<Measurement>, String> {
    let binary = workspace
        .join("target")
        .join("release")
        .join("examples")
        .join(executable_name(model.name));
    if !binary.is_file() {
        return Err(format!(
            "Rust example binary not found: {}",
            binary.display()
        ));
    }

    for warmup in 0..config.warmups {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("warmup-{}", warmup + 1));
        run_rust_once(&binary, julia_root, &out, config)?;
    }

    let mut measurements = Vec::with_capacity(config.reps);
    for rep in 0..config.reps {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", rep + 1));
        let (duration, output) = run_rust_once(&binary, julia_root, &out, config)?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        measurements.push(Measurement {
            implementation: "rust",
            model: model.name.to_string(),
            rep: rep + 1,
            steps: config.steps,
            seconds: duration.as_secs_f64(),
            final_energy_per_site: parse_final_energy_per_site(&stdout),
        });
    }

    Ok(measurements)
}

fn run_rust_once(
    binary: &Path,
    julia_root: &Path,
    out_root: &Path,
    config: &BenchConfig,
) -> Result<(Duration, Output), String> {
    let (steps, threads) = (config.steps, config.threads);
    let mut command = config.inner.command(binary);
    command
        .env("JULIA_MVMC_ROOT", julia_root)
        .env("JULIA_MVMC_EXAMPLE_STEPS", steps.to_string())
        .env("MVMC_OUT_DIR", out_root)
        .env("RUST_BACKTRACE", "1");
    if let Some(n) = threads {
        apply_thread_env(&mut command, n);
    }
    config.inner.apply(&mut command);
    timed_output(command, binary.as_os_str())
}

fn run_julia_model(
    julia_root: &Path,
    runner: &Path,
    run_root: &Path,
    model: Model,
    config: &BenchConfig,
) -> Result<Vec<Measurement>, String> {
    let namelist = julia_root
        .join("examples")
        .join("inputs")
        .join(model.julia_input_dir)
        .join("namelist.def");
    if !namelist.is_file() {
        return Err(format!("namelist not found: {}", namelist.display()));
    }

    let julia_out_root = run_root.join("julia").join(model.name);
    fs::create_dir_all(&julia_out_root)
        .map_err(|e| format!("cannot create {}: {e}", julia_out_root.display()))?;

    let mut command = Command::new(&config.julia_bin);
    if config.julia_bin == Path::new("julia") {
        command.arg("+1.13.1");
    }
    command
        .arg(format!("--project={}", julia_root.display()))
        .arg("--startup-file=no")
        .arg("--history-file=no");
    if let Some(n) = config.threads {
        command.arg(format!("--threads={n}"));
        apply_thread_env(&mut command, n);
    }
    let output = command
        .arg(runner)
        .arg(model.name)
        .arg(model.julia_mode)
        .arg(namelist)
        .arg(config.steps.to_string())
        .arg(config.warmups.to_string())
        .arg(config.reps.to_string())
        .arg(julia_out_root)
        .current_dir(julia_root)
        .output()
        .map_err(|e| format!("failed to spawn Julia: {e}"))?;
    ensure_success("julia benchmark runner", &output)?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut measurements = Vec::with_capacity(config.reps);
    for line in stdout.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 6 || fields[0] != "BENCH" {
            continue;
        }
        let rep = fields[3]
            .parse::<usize>()
            .map_err(|_| format!("bad Julia rep in line: {line}"))?;
        let seconds = fields[4]
            .parse::<f64>()
            .map_err(|_| format!("bad Julia seconds in line: {line}"))?;
        let final_energy_per_site = fields[5].parse::<f64>().ok();
        measurements.push(Measurement {
            implementation: "julia",
            model: fields[2].to_string(),
            rep,
            steps: config.steps,
            seconds,
            final_energy_per_site,
        });
    }

    if measurements.len() != config.reps {
        return Err(format!(
            "Julia benchmark produced {} measurements, expected {}\nstdout:\n{}\nstderr:\n{}",
            measurements.len(),
            config.reps,
            stdout,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(measurements)
}

fn timed_output(mut command: Command, label: &OsStr) -> Result<(Duration, Output), String> {
    let start = Instant::now();
    let output = command
        .output()
        .map_err(|e| format!("failed to spawn {}: {e}", label.to_string_lossy()))?;
    let elapsed = start.elapsed();
    ensure_success(&label.to_string_lossy(), &output)?;
    Ok((elapsed, output))
}

fn ensure_success(label: &str, output: &Output) -> Result<(), String> {
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    ))
}

fn parse_final_energy_per_site(stdout: &str) -> Option<f64> {
    stdout.lines().find_map(|line| {
        line.split_once("Final energy / site =")
            .and_then(|(_, value)| value.trim().parse::<f64>().ok())
    })
}

fn print_summary(label: &str, measurements: &[Measurement]) {
    let values: Vec<f64> = measurements.iter().map(|m| m.seconds).collect();
    println!(
        "{label:>5}: mean={:.3}s median={:.3}s min={:.3}s stddev={:.3}s reps={}",
        mean(&values),
        median(&values),
        min(&values),
        stddev(&values),
        values.len(),
    );
}

fn print_comparison(measurements: &[Measurement]) {
    println!("=== Summary ===");
    for model in MODELS {
        let rust = seconds_for(measurements, "rust", model.name);
        let julia = seconds_for(measurements, "julia", model.name);
        if rust.is_empty() || julia.is_empty() {
            continue;
        }
        let rust_median = median(&rust);
        let julia_median = median(&julia);
        let rust_min = min(&rust);
        let julia_min = min(&julia);
        let energy_delta = energy_delta(measurements, model.name);
        let energy_str = match energy_delta {
            Some(d) => format!("|ΔE|={:>9.2e}", d),
            None => "|ΔE|=    n/a".to_string(),
        };
        println!(
            "{:<24} rust(med)={:>7.3}s julia(med)={:>7.3}s  speedup(med)={:>5.2}x  speedup(min)={:>5.2}x  {}",
            model.name,
            rust_median,
            julia_median,
            julia_median / rust_median,
            julia_min / rust_min,
            energy_str,
        );
        let rust_energy = final_energies_for(measurements, "rust", model.name);
        let julia_energy = final_energies_for(measurements, "julia", model.name);
        if !rust_energy.is_empty() && !julia_energy.is_empty() {
            let delta = (mean(&rust_energy) - mean(&julia_energy)).abs();
            let reference_std = stddev(&julia_energy);
            let failure = ctest_failure(delta, reference_std);
            println!(
                "{:<24} ctest |Δmean|={:.3e} reference_std={:.3e} threshold={:.3e} status={}",
                model.name,
                delta,
                reference_std,
                3.0 * reference_std,
                if failure { "FAIL" } else { "PASS" },
            );
        }
    }
}

fn seconds_for(measurements: &[Measurement], implementation: &str, model: &str) -> Vec<f64> {
    measurements
        .iter()
        .filter(|m| m.implementation == implementation && m.model == model)
        .map(|m| m.seconds)
        .collect()
}

fn energy_delta(measurements: &[Measurement], model: &str) -> Option<f64> {
    let rust = measurements
        .iter()
        .find(|m| m.implementation == "rust" && m.model == model)
        .and_then(|m| m.final_energy_per_site)?;
    let julia = measurements
        .iter()
        .find(|m| m.implementation == "julia" && m.model == model)
        .and_then(|m| m.final_energy_per_site)?;
    Some((rust - julia).abs())
}

fn final_energies_for(measurements: &[Measurement], implementation: &str, model: &str) -> Vec<f64> {
    measurements
        .iter()
        .filter(|m| m.implementation == implementation && m.model == model)
        .filter_map(|m| m.final_energy_per_site)
        .collect()
}

/// Upstream ctest-equivalent failure rule: both absolute thresholds must be
/// met. The statistical reference is the Julia repetition distribution.
fn ctest_failure(delta: f64, reference_std: f64) -> bool {
    delta.is_finite()
        && reference_std.is_finite()
        && delta >= 3.0 * reference_std
        && delta >= 1.0e-8
}

/// Sidecar `<csv>.config.txt` recording how the Rust side was configured, so a CSV
/// of a threaded run is never mistaken for a serial one.
fn write_run_config(config: &BenchConfig) -> Result<(), String> {
    let path = config.csv.with_extension("config.txt");
    let body = format!(
        "steps={}\nreps={}\nwarmups={}\nthreads={}\nrust_inner={}\nhost_load={}\ncpu={}\n",
        config.steps,
        config.reps,
        config.warmups,
        config
            .threads
            .map_or("<inherit env>".to_string(), |n| n.to_string()),
        config.inner.describe(),
        command_output("cat", &["/proc/loadavg"]),
        host_cpu_model(),
    );
    write_report(&path, &body)
}

/// CPU model string (Linux `/proc/cpuinfo`), or `unknown`.
fn host_cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name"))
                .and_then(|rest| rest.split_once(':'))
                .map(|(_, model)| model.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn write_csv(path: &Path, measurements: &[Measurement]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let mut body = String::from("implementation,model,rep,steps,seconds,final_energy_per_site\n");
    for measurement in measurements {
        let energy = measurement
            .final_energy_per_site
            .map(|v| v.to_string())
            .unwrap_or_default();
        body.push_str(&format!(
            "{},{},{},{},{:.9},{}\n",
            measurement.implementation,
            measurement.model,
            measurement.rep,
            measurement.steps,
            measurement.seconds,
            energy,
        ));
    }
    fs::write(path, body).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

// ── bench-hubbard: report-parameter Rust vs Julia comparison ────────────────

#[derive(Debug)]
struct HubbardBenchConfig {
    steps: usize,
    reps: usize,
    warmups: usize,
    threads: usize,
    inner: InnerEnv,
    keep_output: bool,
    julia_root: PathBuf,
    julia_bin: PathBuf,
    csv: PathBuf,
    report: PathBuf,
    models: Vec<HubbardModel>,
}

fn bench_hubbard(args: &[String]) -> Result<(), String> {
    let workspace = workspace_root();
    let config = parse_hubbard_args(args, &workspace)?;

    println!("=== Rust vs Julia Hubbard-chain benchmark ===");
    println!("steps      : {}", config.steps);
    println!("warmups    : {}", config.warmups);
    println!("reps       : {}", config.reps);
    println!(
        "threads    : {} (pinned on both Rust & Julia)",
        config.threads
    );
    println!("rust inner : {}", config.inner.describe());
    println!("inputs     : benchmark/hubbard_chain/inputs");
    println!("julia root : {}", config.julia_root.display());
    println!("julia bin  : {}", config.julia_bin.display());
    println!("csv        : {}", config.csv.display());
    println!("report     : {}", config.report.display());
    println!();

    build_rust_cli(&workspace)?;

    let run_root = workspace
        .join("target")
        .join("bench-output")
        .join(format!("hubbard-{}", unix_timestamp_millis()));
    fs::create_dir_all(&run_root)
        .map_err(|e| format!("cannot create {}: {e}", run_root.display()))?;

    let julia_runner = write_julia_runner(&workspace)?;
    let mut measurements = Vec::new();

    for model in &config.models {
        println!("--- {} ---", model.name);
        let rust = run_rust_hubbard(&workspace, &run_root, *model, &config)?;
        print_summary("rust", &rust);
        measurements.extend(rust);

        let julia = run_julia_hubbard(
            &config.julia_root,
            &julia_runner,
            &run_root,
            *model,
            &config,
        )?;
        print_summary("julia", &julia);
        measurements.extend(julia);
        println!();
    }

    write_csv(&config.csv, &measurements)?;
    print_hubbard_comparison(&measurements, &config.models);
    write_report(
        &config.report,
        &build_hubbard_report(&config, &measurements),
    )?;
    println!("csv    : {}", config.csv.display());
    println!("report : {}", config.report.display());

    if config.keep_output {
        println!("kept outputs: {}", run_root.display());
    } else if let Err(e) = fs::remove_dir_all(&run_root) {
        eprintln!("warning: failed to remove {}: {e}", run_root.display());
    }

    Ok(())
}

fn parse_hubbard_args(args: &[String], workspace: &Path) -> Result<HubbardBenchConfig, String> {
    let mut steps = 300usize;
    let mut reps = 3usize;
    let mut warmups = 1usize;
    let mut threads = 1usize;
    let mut inner = InnerEnv::default();
    let mut model_names: Vec<String> = Vec::new();
    let mut julia_root = workspace.join("extern/Julia-mVMC");
    let mut csv = workspace
        .join("target")
        .join("bench")
        .join("hubbard_chain.csv");
    let mut report = workspace
        .join("target")
        .join("bench")
        .join("hubbard_chain_report.md");
    let mut julia_bin = PathBuf::from("julia");
    let mut keep_output = false;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                print_hubbard_help();
                std::process::exit(0);
            }
            "--steps" => {
                idx += 1;
                steps = parse_value(args.get(idx), "--steps")?;
            }
            "--reps" => {
                idx += 1;
                reps = parse_value(args.get(idx), "--reps")?;
            }
            "--warmups" => {
                idx += 1;
                warmups = parse_value(args.get(idx), "--warmups")?;
            }
            "--threads" => {
                idx += 1;
                threads = parse_value(args.get(idx), "--threads")?;
            }
            "--model" => {
                idx += 1;
                model_names.push(
                    args.get(idx)
                        .ok_or_else(|| "--model requires a value".to_string())?
                        .clone(),
                );
            }
            "--julia-root" => {
                idx += 1;
                julia_root = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-root requires a value".to_string())?,
                );
            }
            "--julia-bin" => {
                idx += 1;
                julia_bin = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-bin requires a value".to_string())?,
                );
            }
            "--csv" => {
                idx += 1;
                csv = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--csv requires a value".to_string())?,
                );
            }
            "--report" => {
                idx += 1;
                report = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--report requires a value".to_string())?,
                );
            }
            "--keep-output" => keep_output = true,
            flag => {
                if !inner.parse_flag(flag, args.get(idx + 1))? {
                    return Err(format!("unknown bench-hubbard flag `{flag}`"));
                }
                idx += 1;
            }
        }
        idx += 1;
    }

    if steps == 0 {
        return Err("--steps must be positive".to_string());
    }
    if reps == 0 {
        return Err("--reps must be positive".to_string());
    }
    if threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    if warmups + reps == 0 {
        return Err("--warmups + --reps must be positive".to_string());
    }

    let julia_root = canonicalize_existing_dir(&julia_root, "Julia-mVMC root")?;
    let models = if model_names.is_empty() {
        HUBBARD_MODELS.to_vec()
    } else {
        model_names
            .iter()
            .map(|name| {
                HUBBARD_MODELS
                    .iter()
                    .copied()
                    .find(|model| model.name == name)
                    .ok_or_else(|| format!("unknown model `{name}`"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(HubbardBenchConfig {
        steps,
        reps,
        warmups,
        threads,
        inner,
        keep_output,
        julia_root,
        julia_bin,
        csv,
        report,
        models,
    })
}

fn build_rust_cli(workspace: &Path) -> Result<(), String> {
    println!("building mvmc-cli release binary...");
    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--release")
        .arg("-p")
        .arg("mvmc-cli");
    let output = command
        .current_dir(workspace)
        .output()
        .map_err(|e| format!("failed to spawn cargo build: {e}"))?;
    ensure_success("cargo build --release -p mvmc-cli", &output)?;
    Ok(())
}

fn hubbard_namelist(workspace: &Path, model: HubbardModel) -> PathBuf {
    workspace
        .join("benchmark")
        .join("hubbard_chain")
        .join("inputs")
        .join(model.name)
        .join("namelist.def")
}

fn run_rust_hubbard(
    workspace: &Path,
    run_root: &Path,
    model: HubbardModel,
    config: &HubbardBenchConfig,
) -> Result<Vec<Measurement>, String> {
    let binary = workspace
        .join("target")
        .join("release")
        .join(executable_name("mvmc"));
    if !binary.is_file() {
        return Err(format!("Rust CLI binary not found: {}", binary.display()));
    }

    for warmup in 0..config.warmups {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("warmup-{}", warmup + 1));
        run_rust_hubbard_once(workspace, &binary, model, &out, config)?;
    }

    let mut measurements = Vec::with_capacity(config.reps);
    for rep in 0..config.reps {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", rep + 1));
        let (seconds, energy, _output) =
            run_rust_hubbard_once(workspace, &binary, model, &out, config)?;
        measurements.push(Measurement {
            implementation: "rust",
            model: model.name.to_string(),
            rep: rep + 1,
            steps: config.steps,
            seconds,
            final_energy_per_site: energy,
        });
    }
    Ok(measurements)
}

fn run_rust_hubbard_once(
    workspace: &Path,
    binary: &Path,
    model: HubbardModel,
    out_root: &Path,
    config: &HubbardBenchConfig,
) -> Result<(f64, Option<f64>, Output), String> {
    let namelist = hubbard_namelist(workspace, model);
    let mut command = config.inner.command(binary);
    command
        .arg(&namelist)
        .arg("--mode")
        .arg("real")
        .arg("--nsteps")
        .arg(config.steps.to_string())
        .arg("--nsmp")
        .arg(config.steps.to_string())
        .arg("--out-dir")
        .arg(out_root);
    apply_thread_env(&mut command, config.threads);
    config.inner.apply(&mut command);
    let output = command
        .output()
        .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;
    ensure_success(&binary.to_string_lossy(), &output)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let seconds = parse_rust_seconds(&stdout).ok_or_else(|| {
        format!(
            "cannot parse Rust internal timing for {} from:\n{stdout}",
            model.name
        )
    })?;
    // Prefer the full-precision zvo_out.dat value; the CLI prints only 10 decimals.
    let energy =
        read_rust_energy_per_site(out_root, &namelist).or_else(|| parse_rust_final_energy(&stdout));
    Ok((seconds, energy, output))
}

/// Read the last `zvo_out.dat` row, divide the total energy by `Nsite`, and
/// return the energy per site at full printed precision.
fn read_rust_energy_per_site(out_root: &Path, namelist: &Path) -> Option<f64> {
    let text = fs::read_to_string(out_root.join("zvo_out.dat")).ok()?;
    let last = text.lines().rfind(|line| !line.trim().is_empty())?;
    let etot = last.split_whitespace().next()?.parse::<f64>().ok()?;
    let nsite = modpara_nsite(namelist)?;
    if nsite == 0 {
        return None;
    }
    Some(etot / nsite as f64)
}

/// Resolve `Nsite` through the namelist's `ModPara` entry.
fn modpara_nsite(namelist: &Path) -> Option<usize> {
    modpara_value(namelist, "Nsite")
}

/// Resolve `NVMCSample` through the namelist's `ModPara` entry.
fn modpara_nvmc_sample(namelist: &Path) -> Option<usize> {
    modpara_value(namelist, "NVMCSample")
}

fn modpara_value(namelist: &Path, key: &str) -> Option<usize> {
    let dir = namelist.parent()?;
    let namelist_text = fs::read_to_string(namelist).ok()?;
    let modpara_name = namelist_text.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next()) {
            (Some("ModPara"), Some(name)) => Some(name.to_string()),
            _ => None,
        }
    })?;
    let modpara_text = fs::read_to_string(dir.join(modpara_name)).ok()?;
    modpara_text.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next() == Some(key) {
            fields.next()?.parse().ok()
        } else {
            None
        }
    })
}

fn run_julia_hubbard(
    julia_root: &Path,
    runner: &Path,
    run_root: &Path,
    model: HubbardModel,
    config: &HubbardBenchConfig,
) -> Result<Vec<Measurement>, String> {
    let namelist = hubbard_namelist(&workspace_root(), model);
    if !namelist.is_file() {
        return Err(format!("namelist not found: {}", namelist.display()));
    }

    let out_root = run_root.join("julia").join(model.name);
    fs::create_dir_all(&out_root)
        .map_err(|e| format!("cannot create {}: {e}", out_root.display()))?;

    let mut command = Command::new(&config.julia_bin);
    if config.julia_bin == Path::new("julia") {
        command.arg("+1.13.1");
    }
    command
        .arg(format!("--project={}", julia_root.display()))
        .arg("--startup-file=no")
        .arg("--history-file=no")
        .arg(format!("--threads={}", config.threads));
    apply_thread_env(&mut command, config.threads);
    let output = command
        .arg(runner)
        .arg(model.name)
        .arg("real")
        .arg(&namelist)
        .arg(config.steps.to_string())
        .arg(config.warmups.to_string())
        .arg(config.reps.to_string())
        .arg(&out_root)
        .current_dir(julia_root)
        .output()
        .map_err(|e| format!("failed to spawn Julia: {e}"))?;
    ensure_success("julia bench-hubbard runner", &output)?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut measurements = Vec::with_capacity(config.reps);
    for line in stdout.lines() {
        if let Some(measurement) = parse_julia_bench_line(line, model.name, config.steps) {
            measurements.push(measurement);
        }
    }

    if measurements.len() != config.reps {
        return Err(format!(
            "Julia benchmark produced {} measurements, expected {}\nstdout:\n{}\nstderr:\n{}",
            measurements.len(),
            config.reps,
            stdout,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(measurements)
}

/// Parse `=== Completed 300 SR steps in 4.11s ===` into `4.11`.
fn parse_rust_seconds(stdout: &str) -> Option<f64> {
    stdout.lines().find_map(|line| {
        let rest = line.strip_prefix("=== Completed ")?;
        let (_, tail) = rest.split_once(" SR steps in ")?;
        let token = tail.split_whitespace().next()?;
        token.trim_end_matches('s').parse().ok()
    })
}

/// Parse `Final energy / site: -0.5418807042`.
fn parse_rust_final_energy(stdout: &str) -> Option<f64> {
    stdout.lines().find_map(|line| {
        line.split_once("Final energy / site:")
            .and_then(|(_, value)| value.trim().parse().ok())
    })
}

/// Parse one `BENCH\tjulia\t<model>\t<rep>\t<seconds>\t<energy>` line.
fn parse_julia_bench_line(line: &str, model: &str, steps: usize) -> Option<Measurement> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 6 || fields[0] != "BENCH" {
        return None;
    }
    Some(Measurement {
        implementation: "julia",
        model: model.to_string(),
        rep: fields[3].parse().ok()?,
        steps,
        seconds: fields[4].parse().ok()?,
        final_energy_per_site: fields[5].parse().ok(),
    })
}

fn print_hubbard_comparison(measurements: &[Measurement], models: &[HubbardModel]) {
    println!("=== Summary ===");
    for model in models {
        let rust = seconds_for(measurements, "rust", model.name);
        let julia = seconds_for(measurements, "julia", model.name);
        if rust.is_empty() || julia.is_empty() {
            continue;
        }
        let rust_median = median(&rust);
        let julia_median = median(&julia);
        let energy_str = match energy_delta(measurements, model.name) {
            Some(d) => format!("|ΔE|={d:.2e}"),
            None => "|ΔE|=   n/a".to_string(),
        };
        println!(
            "{:<24} rust(med)={:>7.3}s julia(med)={:>7.3}s  speedup(julia/rust)={:>5.2}x  {}",
            model.name,
            rust_median,
            julia_median,
            julia_median / rust_median,
            energy_str,
        );
    }
}

fn build_hubbard_report(config: &HubbardBenchConfig, measurements: &[Measurement]) -> String {
    let julia_version = if config.julia_bin == Path::new("julia") {
        command_output("julia", &["+1.13.1", "--version"])
    } else {
        command_output(&config.julia_bin.to_string_lossy(), &["--version"])
    };
    let mut report = String::new();
    report.push_str("# Rust vs Julia Hubbard-chain benchmark\n\n");
    report.push_str(&format!(
        "- generated: {}\n",
        command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
    ));
    report.push_str(&format!(
        "- platform: {}\n",
        command_output("uname", &["-sm"])
    ));
    report.push_str(&format!(
        "- rustc: {}\n",
        command_output("rustc", &["--version"])
    ));
    report.push_str(&format!("- julia: {julia_version}\n"));
    report.push_str(&format!(
        "- steps/reps/warmups/threads: {}/{}/{}/{}\n",
        config.steps, config.reps, config.warmups, config.threads
    ));
    report.push_str(&format!(
        "- rust inner kernels: {}\n",
        config.inner.describe()
    ));
    report.push_str(&format!(
        "- BLAS/OpenMP threads (both sides): {}; host load average at report time: {}\n",
        config.threads,
        command_output("cat", &["/proc/loadavg"])
    ));
    report.push_str(&format!("- cpu: {}\n", host_cpu_model()));
    report.push_str("- inputs: benchmark/hubbard_chain/inputs\n");
    report.push_str(
        "- timing: internal `run_para_opt_from_namelist` wall clock; Julia JIT and\n  process startup are excluded, and both sides are thread-pinned\n\n",
    );
    report
        .push_str("| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | |ΔE| |\n");
    report.push_str("|---|---:|---:|---:|---:|\n");
    for model in &config.models {
        let rust = seconds_for(measurements, "rust", model.name);
        let julia = seconds_for(measurements, "julia", model.name);
        if rust.is_empty() || julia.is_empty() {
            continue;
        }
        let rust_median = median(&rust);
        let julia_median = median(&julia);
        let energy = energy_delta(measurements, model.name)
            .map(|d| format!("{d:.2e}"))
            .unwrap_or_else(|| "n/a".to_string());
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3}x | {} |\n",
            model.name,
            rust_median,
            julia_median,
            julia_median / rust_median,
            energy,
        ));
    }
    report.push_str("\n`speedup = julia / rust`; values above `1.0x` mean Rust was faster.\n");
    report
}

fn write_report(path: &Path, body: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    fs::write(path, body).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if stdout.is_empty() {
                String::from_utf8_lossy(&output.stderr).trim().to_string()
            } else {
                stdout
            }
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

// ── bench-physcal: fixed-parameter PhysCal benchmark ────────────────────────

const PHYSCAL_MODELS: &[PhyscalModel] = &[
    PhyscalModel {
        name: "heisenberg_chain_real",
        mode: "real",
        fixture_dir: "heisenberg_chain_real",
    },
    PhyscalModel {
        name: "heisenberg_chain_cmp",
        mode: "cmp",
        fixture_dir: "heisenberg_chain_cmp",
    },
    PhyscalModel {
        name: "heisenberg_chain_fsz",
        mode: "fsz",
        fixture_dir: "heisenberg_chain_fsz",
    },
    PhyscalModel {
        name: "hubbard_chain_real",
        mode: "real",
        fixture_dir: "hubbard_chain_real",
    },
];

#[derive(Debug, Clone, Copy)]
struct PhyscalModel {
    name: &'static str,
    mode: &'static str,
    fixture_dir: &'static str,
}

#[derive(Debug)]
struct PhyscalBenchConfig {
    reps: usize,
    warmups: usize,
    threads: usize,
    keep_output: bool,
    julia_root: PathBuf,
    julia_bin: PathBuf,
    csv: PathBuf,
    models: Vec<PhyscalModel>,
}

fn bench_physcal(args: &[String]) -> Result<(), String> {
    let workspace = workspace_root();
    let config = parse_physcal_args(args, &workspace)?;

    println!("=== Rust vs Julia PhysCal benchmark ===");
    println!("warmups    : {}", config.warmups);
    println!("reps       : {}", config.reps);
    println!(
        "threads    : {} (pinned on both Rust & Julia)",
        config.threads
    );
    println!("inputs     : extern/Julia-mVMC/test/integration/reference/*/physcal_ref");
    println!("julia root : {}", config.julia_root.display());
    println!("julia bin  : {}", config.julia_bin.display());
    println!("csv        : {}", config.csv.display());
    println!();

    build_rust_cli(&workspace)?;

    let run_root = workspace
        .join("target")
        .join("bench-output")
        .join(format!("physcal-{}", unix_timestamp_millis()));
    fs::create_dir_all(&run_root)
        .map_err(|e| format!("cannot create {}: {e}", run_root.display()))?;

    let julia_runner = write_julia_physcal_runner(&workspace)?;
    let mut measurements = Vec::new();
    let mut observable_diffs: Vec<(String, Vec<ObservableDiff>)> = Vec::new();

    for model in &config.models {
        println!("--- {} ---", model.name);
        let nvmc_sample = physcal_nvmc_sample(&config.julia_root, *model);
        let rust = run_rust_physcal(&workspace, &run_root, *model, nvmc_sample, &config)?;
        print_summary("rust", &rust);
        measurements.extend(rust);

        let julia = run_julia_physcal(
            &config.julia_root,
            &julia_runner,
            &run_root,
            *model,
            nvmc_sample,
            &config,
        )?;
        print_summary("julia", &julia);
        measurements.extend(julia);

        let last_rust = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", config.reps));
        let last_julia = run_root
            .join("julia")
            .join(model.name)
            .join(format!("run_{}", config.warmups + config.reps));
        let diffs = compare_physcal_observables(&last_rust, &last_julia)?;
        observable_diffs.push((model.name.to_string(), diffs));
        println!();
    }

    write_csv(&config.csv, &measurements)?;
    print_physcal_comparison(&measurements, &config.models);
    print_physcal_observables(&observable_diffs);

    if config.keep_output {
        println!("kept outputs: {}", run_root.display());
    } else if let Err(e) = fs::remove_dir_all(&run_root) {
        eprintln!("warning: failed to remove {}: {e}", run_root.display());
    }

    Ok(())
}

fn parse_physcal_args(args: &[String], workspace: &Path) -> Result<PhyscalBenchConfig, String> {
    let mut reps = 3usize;
    let mut warmups = 1usize;
    let mut threads = 1usize;
    let mut model_names: Vec<String> = Vec::new();
    let mut julia_root = workspace.join("extern/Julia-mVMC");
    let mut csv = workspace
        .join("target")
        .join("bench")
        .join("physcal_chain.csv");
    let mut julia_bin = PathBuf::from("julia");
    let mut keep_output = false;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                print_physcal_help();
                std::process::exit(0);
            }
            "--reps" => {
                idx += 1;
                reps = parse_value(args.get(idx), "--reps")?;
            }
            "--warmups" => {
                idx += 1;
                warmups = parse_value(args.get(idx), "--warmups")?;
            }
            "--threads" => {
                idx += 1;
                threads = parse_value(args.get(idx), "--threads")?;
            }
            "--model" => {
                idx += 1;
                model_names.push(
                    args.get(idx)
                        .ok_or_else(|| "--model requires a value".to_string())?
                        .clone(),
                );
            }
            "--julia-root" => {
                idx += 1;
                julia_root = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-root requires a value".to_string())?,
                );
            }
            "--julia-bin" => {
                idx += 1;
                julia_bin = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-bin requires a value".to_string())?,
                );
            }
            "--csv" => {
                idx += 1;
                csv = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--csv requires a value".to_string())?,
                );
            }
            "--keep-output" => keep_output = true,
            flag => return Err(format!("unknown bench-physcal flag `{flag}`")),
        }
        idx += 1;
    }

    if reps == 0 {
        return Err("--reps must be positive".to_string());
    }
    if threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    if warmups + reps == 0 {
        return Err("--warmups + --reps must be positive".to_string());
    }

    let julia_root = canonicalize_existing_dir(&julia_root, "Julia-mVMC root")?;
    let models = if model_names.is_empty() {
        PHYSCAL_MODELS.to_vec()
    } else {
        model_names
            .iter()
            .map(|name| {
                PHYSCAL_MODELS
                    .iter()
                    .copied()
                    .find(|model| model.name == name)
                    .ok_or_else(|| format!("unknown model `{name}`"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(PhyscalBenchConfig {
        reps,
        warmups,
        threads,
        keep_output,
        julia_root,
        julia_bin,
        csv,
        models,
    })
}

fn physcal_fixture_dir(julia_root: &Path, model: PhyscalModel) -> PathBuf {
    julia_root
        .join("test")
        .join("integration")
        .join("reference")
        .join(model.fixture_dir)
        .join("physcal_ref")
}

fn physcal_namelist(julia_root: &Path, model: PhyscalModel) -> PathBuf {
    physcal_fixture_dir(julia_root, model)
        .join("inputs")
        .join("namelist.def")
}

fn physcal_opt_params(julia_root: &Path, model: PhyscalModel) -> PathBuf {
    physcal_fixture_dir(julia_root, model).join("zqp_opt.dat")
}

fn physcal_nvmc_sample(julia_root: &Path, model: PhyscalModel) -> usize {
    modpara_nvmc_sample(&physcal_namelist(julia_root, model)).unwrap_or(0)
}

fn run_rust_physcal(
    workspace: &Path,
    run_root: &Path,
    model: PhyscalModel,
    nvmc_sample: usize,
    config: &PhyscalBenchConfig,
) -> Result<Vec<Measurement>, String> {
    let binary = workspace
        .join("target")
        .join("release")
        .join(executable_name("mvmc"));
    if !binary.is_file() {
        return Err(format!("Rust CLI binary not found: {}", binary.display()));
    }
    let namelist = physcal_namelist(&config.julia_root, model);
    let opt_params = physcal_opt_params(&config.julia_root, model);

    for warmup in 0..config.warmups {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("warmup-{}", warmup + 1));
        run_rust_physcal_once(&binary, &namelist, &opt_params, model.mode, &out, config)?;
    }

    let mut measurements = Vec::with_capacity(config.reps);
    for rep in 0..config.reps {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", rep + 1));
        let (seconds, _output) =
            run_rust_physcal_once(&binary, &namelist, &opt_params, model.mode, &out, config)?;
        // PhysCal writes one indexed `zvo_out_NNN.dat` per sample; the
        // optimization reader's non-indexed `zvo_out.dat` does not exist here.
        let energy = read_rust_physcal_energy_per_site(&out, &namelist);
        measurements.push(Measurement {
            implementation: "rust",
            model: model.name.to_string(),
            rep: rep + 1,
            steps: nvmc_sample,
            seconds,
            final_energy_per_site: energy,
        });
    }
    Ok(measurements)
}

fn run_rust_physcal_once(
    binary: &Path,
    namelist: &Path,
    opt_params: &Path,
    mode: &str,
    out_root: &Path,
    config: &PhyscalBenchConfig,
) -> Result<(f64, Output), String> {
    let mut command = Command::new(binary);
    command
        .arg(namelist)
        .arg("--physcal")
        .arg(opt_params)
        .arg("--mode")
        .arg(mode)
        .arg("--out-dir")
        .arg(out_root);
    apply_thread_env(&mut command, config.threads);
    let (duration, output) = timed_output(command, binary.as_os_str())?;
    Ok((duration.as_secs_f64(), output))
}

fn run_julia_physcal(
    julia_root: &Path,
    runner: &Path,
    run_root: &Path,
    model: PhyscalModel,
    nvmc_sample: usize,
    config: &PhyscalBenchConfig,
) -> Result<Vec<Measurement>, String> {
    let namelist = physcal_namelist(julia_root, model);
    let opt_params = physcal_opt_params(julia_root, model);
    if !namelist.is_file() {
        return Err(format!("namelist not found: {}", namelist.display()));
    }
    if !opt_params.is_file() {
        return Err(format!("opt params not found: {}", opt_params.display()));
    }

    let out_root = run_root.join("julia").join(model.name);
    fs::create_dir_all(&out_root)
        .map_err(|e| format!("cannot create {}: {e}", out_root.display()))?;
    // The runner divides the sampled total energy by its 8th argument to report
    // energy per site; pass Nsite, not the PhysCal sample count.
    let nsite = modpara_nsite(&namelist).unwrap_or(0);

    let mut command = Command::new(&config.julia_bin);
    if config.julia_bin == Path::new("julia") {
        command.arg("+1.13.1");
    }
    command
        .arg(format!("--project={}", julia_root.display()))
        .arg("--startup-file=no")
        .arg("--history-file=no")
        .arg(format!("--threads={}", config.threads));
    apply_thread_env(&mut command, config.threads);
    let output = command
        .arg(runner)
        .arg(model.name)
        .arg(model.mode)
        .arg(&namelist)
        .arg(&opt_params)
        .arg(config.warmups.to_string())
        .arg(config.reps.to_string())
        .arg(&out_root)
        .arg(nsite.to_string())
        .current_dir(julia_root)
        .output()
        .map_err(|e| format!("failed to spawn Julia: {e}"))?;
    ensure_success("julia bench-physcal runner", &output)?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut measurements = Vec::with_capacity(config.reps);
    for line in stdout.lines() {
        if let Some(measurement) = parse_julia_bench_line(line, model.name, nvmc_sample) {
            measurements.push(measurement);
        }
    }

    if measurements.len() != config.reps {
        return Err(format!(
            "Julia benchmark produced {} measurements, expected {}\nstdout:\n{}\nstderr:\n{}",
            measurements.len(),
            config.reps,
            stdout,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(measurements)
}

fn print_physcal_comparison(measurements: &[Measurement], models: &[PhyscalModel]) {
    println!("=== Summary ===");
    for model in models {
        let rust = seconds_for(measurements, "rust", model.name);
        let julia = seconds_for(measurements, "julia", model.name);
        if rust.is_empty() || julia.is_empty() {
            continue;
        }
        let rust_median = median(&rust);
        let julia_median = median(&julia);
        let energy_str = match energy_delta(measurements, model.name) {
            Some(d) => format!("|ΔE|={d:.2e}"),
            None => "|ΔE|=   n/a".to_string(),
        };
        println!(
            "{:<24} rust(med)={:>7.3}s julia(med)={:>7.3}s  speedup(julia/rust)={:>5.2}x  {}",
            model.name,
            rust_median,
            julia_median,
            julia_median / rust_median,
            energy_str,
        );
    }
}

// ── PhysCal observable comparison ───────────────────────────────────────────

/// PhysCal output families compared between the Rust CLI and the Julia runner.
/// The numeric sample index (`_001`) is stripped before matching, so one family
/// groups every `NDataQtySmp` sample. `zvo_out`/`zvo_var` are excluded: their
/// naming differs (indexed on the Rust side, plain on the Julia side) and the
/// energy is already compared through the timing summary.
const PHYSCAL_OBSERVABLE_FAMILIES: &[&str] = &[
    "zvo_cisajs",
    "zvo_cisajscktalt",
    "zvo_cisajscktaltex",
    "zvo_ls_out",
    "zvo_ls_qqqq",
    "zvo_ls_cisajs",
    "zvo_ls_cisajscktalt",
    "zvo_ls_cisajscktaltex",
];

/// Absolute and relative bounds used to label a family `ok` in the observable
/// comparison. They follow the documented PhysCal comparison policy (one-body
/// `1e-10`, direct and factored two-body `1e-9`). A benchmark run reports the
/// observed maxima rather than aborting on them.
const PHYSCAL_OBSERVABLE_ATOL: f64 = 1.0e-10;
const PHYSCAL_OBSERVABLE_RTOL: f64 = 1.0e-9;

#[derive(Debug, Clone)]
struct ObservableDiff {
    family: String,
    files: usize,
    values: usize,
    max_abs: f64,
    max_rel: f64,
    /// Largest `|Δ| / (atol + rtol*scale)` across all compared values. A family
    /// is within tolerance when this is at most one, which is the componentwise
    /// absolute-plus-relative rule rather than two independent maxima.
    max_error_ratio: f64,
}

impl ObservableDiff {
    fn within_tolerance(&self) -> bool {
        self.max_error_ratio <= 1.0
    }
}

/// Map a PhysCal output file name (`zvo_cisajs_003.dat`) to its family
/// (`zvo_cisajs`), or `None` when the file is not a compared observable.
fn observable_family(file_name: &str) -> Option<&'static str> {
    let stem = file_name.strip_suffix(".dat")?;
    let (base, index) = stem.rsplit_once('_')?;
    if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    PHYSCAL_OBSERVABLE_FAMILIES
        .iter()
        .copied()
        .find(|family| *family == base)
}

fn observable_files(
    dir: &Path,
) -> Result<BTreeMap<&'static str, BTreeMap<String, PathBuf>>, String> {
    let mut map: BTreeMap<&'static str, BTreeMap<String, PathBuf>> = BTreeMap::new();
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read entry in {}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(family) = observable_family(&name) {
            map.entry(family).or_default().insert(name, entry.path());
        }
    }
    Ok(map)
}

/// Compare one observable file. Integer/token columns must match exactly; every
/// other column is compared as a float. Returns `(numeric values, max abs, max
/// rel, max error ratio)`.
fn compare_observable_file(
    rust: &Path,
    julia: &Path,
    family: &str,
    file: &str,
) -> Result<(usize, f64, f64, f64), String> {
    let rust_text =
        fs::read_to_string(rust).map_err(|e| format!("cannot read {}: {e}", rust.display()))?;
    let julia_text =
        fs::read_to_string(julia).map_err(|e| format!("cannot read {}: {e}", julia.display()))?;
    let rust_lines: Vec<&str> = rust_text.lines().collect();
    let julia_lines: Vec<&str> = julia_text.lines().collect();
    if rust_lines.len() != julia_lines.len() {
        return Err(format!(
            "{family}/{file}: row count differs (rust {} vs julia {})",
            rust_lines.len(),
            julia_lines.len(),
        ));
    }
    let mut values = 0usize;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_error_ratio = 0.0f64;
    for (row, (rust_line, julia_line)) in rust_lines.iter().zip(&julia_lines).enumerate() {
        if julia_line.trim_start().starts_with('#') {
            if rust_line != julia_line {
                return Err(format!("{family}/{file}: header row {row} differs"));
            }
            continue;
        }
        let rust_columns: Vec<&str> = rust_line.split_whitespace().collect();
        let julia_columns: Vec<&str> = julia_line.split_whitespace().collect();
        if rust_columns.len() != julia_columns.len() {
            return Err(format!(
                "{family}/{file}: row {row} column count differs (rust {} vs julia {})",
                rust_columns.len(),
                julia_columns.len(),
            ));
        }
        for (column, (rust_token, julia_token)) in
            rust_columns.iter().zip(&julia_columns).enumerate()
        {
            if let Ok(expected) = julia_token.parse::<i64>() {
                if rust_token.parse::<i64>().ok() != Some(expected) {
                    return Err(format!(
                        "{family}/{file}: row {row} column {column} discrete value differs \
                         ({rust_token} vs {julia_token})"
                    ));
                }
                continue;
            }
            match (rust_token.parse::<f64>(), julia_token.parse::<f64>()) {
                (Ok(actual), Ok(expected)) => {
                    let difference = (actual - expected).abs();
                    let scale = actual.abs().max(expected.abs());
                    let relative = if scale > 0.0 { difference / scale } else { 0.0 };
                    let error_ratio =
                        difference / (PHYSCAL_OBSERVABLE_ATOL + PHYSCAL_OBSERVABLE_RTOL * scale);
                    max_abs = max_abs.max(difference);
                    max_rel = max_rel.max(relative);
                    max_error_ratio = max_error_ratio.max(error_ratio);
                    values += 1;
                }
                _ if rust_token != julia_token => {
                    return Err(format!(
                        "{family}/{file}: row {row} column {column} text differs \
                         ({rust_token} vs {julia_token})"
                    ));
                }
                _ => {}
            }
        }
    }
    Ok((values, max_abs, max_rel, max_error_ratio))
}

/// Compare every PhysCal observable file family present in `rust_dir` against
/// the same-named file in `julia_dir`. Shape and discrete-token mismatches are
/// errors; floating-point differences are summarized per family.
fn compare_physcal_observables(
    rust_dir: &Path,
    julia_dir: &Path,
) -> Result<Vec<ObservableDiff>, String> {
    let rust_files = observable_files(rust_dir)?;
    let julia_files = observable_files(julia_dir)?;
    for family in julia_files.keys() {
        if !rust_files.contains_key(family) {
            return Err(format!(
                "{}: Julia wrote {family} files but the Rust run did not",
                rust_dir.display(),
            ));
        }
    }
    let mut diffs = Vec::new();
    for (family, rust_group) in &rust_files {
        let family = *family;
        let julia_group = julia_files.get(family).ok_or_else(|| {
            format!(
                "{}: Rust wrote {} {family} file(s) but Julia did not",
                rust_dir.display(),
                rust_group.len(),
            )
        })?;
        let mut files = 0usize;
        let mut values = 0usize;
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut max_error_ratio = 0.0f64;
        for (name, rust_path) in rust_group {
            let julia_path = julia_group
                .get(name)
                .ok_or_else(|| format!("{family}/{name}: missing on the Julia side"))?;
            let (file_values, file_abs, file_rel, file_ratio) =
                compare_observable_file(rust_path, julia_path, family, name)?;
            files += 1;
            values += file_values;
            max_abs = max_abs.max(file_abs);
            max_rel = max_rel.max(file_rel);
            max_error_ratio = max_error_ratio.max(file_ratio);
        }
        diffs.push(ObservableDiff {
            family: family.to_string(),
            files,
            values,
            max_abs,
            max_rel,
            max_error_ratio,
        });
    }
    Ok(diffs)
}

fn print_physcal_observables(diffs: &[(String, Vec<ObservableDiff>)]) {
    if diffs.is_empty() {
        return;
    }
    println!("=== PhysCal observables (Rust vs Julia, last rep) ===");
    println!(
        "{:<24} {:<26} {:>5} {:>8} {:>11} {:>11}  status",
        "model", "family", "files", "values", "max|Δ|", "max_rel"
    );
    for (model, rows) in diffs {
        if rows.is_empty() {
            println!("{model:<24} (no observable files found)");
            continue;
        }
        for row in rows {
            println!(
                "{:<24} {:<26} {:>5} {:>8} {:>11.3e} {:>11.3e}  {}",
                model,
                row.family,
                row.files,
                row.values,
                row.max_abs,
                row.max_rel,
                if row.within_tolerance() { "ok" } else { "DIFF" },
            );
        }
    }
}

fn write_julia_physcal_runner(workspace: &Path) -> Result<PathBuf, String> {
    let dir = workspace.join("target").join("bench");
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = dir.join("julia_physcal_bench_runner.jl");
    fs::write(&path, JULIA_PHYSCAL_BENCH_RUNNER)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

// ── bench-physcal-hubbard: fixed-parameter PhysCal at Hubbard-chain scale ───

/// Rust-local PhysCal inputs derived from the commit-checked Hubbard-chain
/// optimization inputs (`benchmark/hubbard_chain/inputs`). See
/// `benchmark/physcal/README.md` for provenance and regeneration.
const HUBBARD_PHYSCAL_MODELS: &[HubbardModel] = &[
    HubbardModel {
        name: "hubbard_chain_L16",
    },
    HubbardModel {
        name: "hubbard_chain_L24",
    },
    HubbardModel {
        name: "hubbard_chain_L32",
    },
];

#[derive(Debug)]
struct PhyscalHubbardBenchConfig {
    reps: usize,
    warmups: usize,
    threads: usize,
    keep_output: bool,
    julia_root: PathBuf,
    julia_bin: PathBuf,
    csv: PathBuf,
    report: PathBuf,
    models: Vec<HubbardModel>,
}

fn bench_physcal_hubbard(args: &[String]) -> Result<(), String> {
    let workspace = workspace_root();
    let config = parse_physcal_hubbard_args(args, &workspace)?;

    println!("=== Rust vs Julia Hubbard-chain PhysCal benchmark ===");
    println!("warmups    : {}", config.warmups);
    println!("reps       : {}", config.reps);
    println!(
        "threads    : {} (pinned on both Rust & Julia)",
        config.threads
    );
    println!("inputs     : benchmark/physcal/inputs");
    println!("julia root : {}", config.julia_root.display());
    println!("julia bin  : {}", config.julia_bin.display());
    println!("csv        : {}", config.csv.display());
    println!("report     : {}", config.report.display());
    println!();

    build_rust_cli(&workspace)?;

    let run_root = workspace
        .join("target")
        .join("bench-output")
        .join(format!("physcal-hubbard-{}", unix_timestamp_millis()));
    fs::create_dir_all(&run_root)
        .map_err(|e| format!("cannot create {}: {e}", run_root.display()))?;

    let julia_runner = write_julia_physcal_runner(&workspace)?;
    let mut measurements = Vec::new();
    let mut observable_diffs: Vec<(String, Vec<ObservableDiff>)> = Vec::new();

    for model in &config.models {
        println!("--- {} ---", model.name);
        let rust = run_rust_physcal_hubbard(&workspace, &run_root, *model, &config)?;
        print_summary("rust", &rust);
        measurements.extend(rust);

        let julia = run_julia_physcal_hubbard(
            &config.julia_root,
            &julia_runner,
            &run_root,
            *model,
            &config,
        )?;
        print_summary("julia", &julia);
        measurements.extend(julia);

        let last_rust = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", config.reps));
        let last_julia = run_root
            .join("julia")
            .join(model.name)
            .join(format!("run_{}", config.warmups + config.reps));
        let diffs = compare_physcal_observables(&last_rust, &last_julia)?;
        observable_diffs.push((model.name.to_string(), diffs));
        println!();
    }

    write_csv(&config.csv, &measurements)?;
    print_physcal_hubbard_comparison(&measurements, &config.models);
    print_physcal_observables(&observable_diffs);
    write_report(
        &config.report,
        &build_physcal_hubbard_report(&config, &measurements, &observable_diffs),
    )?;
    println!("csv    : {}", config.csv.display());
    println!("report : {}", config.report.display());

    if config.keep_output {
        println!("kept outputs: {}", run_root.display());
    } else if let Err(e) = fs::remove_dir_all(&run_root) {
        eprintln!("warning: failed to remove {}: {e}", run_root.display());
    }

    Ok(())
}

fn parse_physcal_hubbard_args(
    args: &[String],
    workspace: &Path,
) -> Result<PhyscalHubbardBenchConfig, String> {
    let mut reps = 3usize;
    let mut warmups = 1usize;
    let mut threads = 1usize;
    let mut model_names: Vec<String> = Vec::new();
    let mut julia_root = workspace.join("extern/Julia-mVMC");
    let mut csv = workspace
        .join("target")
        .join("bench")
        .join("physcal_hubbard.csv");
    let mut report = workspace
        .join("target")
        .join("bench")
        .join("physcal_hubbard_report.md");
    let mut julia_bin = PathBuf::from("julia");
    let mut keep_output = false;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                print_physcal_hubbard_help();
                std::process::exit(0);
            }
            "--reps" => {
                idx += 1;
                reps = parse_value(args.get(idx), "--reps")?;
            }
            "--warmups" => {
                idx += 1;
                warmups = parse_value(args.get(idx), "--warmups")?;
            }
            "--threads" => {
                idx += 1;
                threads = parse_value(args.get(idx), "--threads")?;
            }
            "--model" => {
                idx += 1;
                model_names.push(
                    args.get(idx)
                        .ok_or_else(|| "--model requires a value".to_string())?
                        .clone(),
                );
            }
            "--julia-root" => {
                idx += 1;
                julia_root = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-root requires a value".to_string())?,
                );
            }
            "--julia-bin" => {
                idx += 1;
                julia_bin = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--julia-bin requires a value".to_string())?,
                );
            }
            "--csv" => {
                idx += 1;
                csv = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--csv requires a value".to_string())?,
                );
            }
            "--report" => {
                idx += 1;
                report = PathBuf::from(
                    args.get(idx)
                        .ok_or_else(|| "--report requires a value".to_string())?,
                );
            }
            "--keep-output" => keep_output = true,
            flag => return Err(format!("unknown bench-physcal-hubbard flag `{flag}`")),
        }
        idx += 1;
    }

    if reps == 0 {
        return Err("--reps must be positive".to_string());
    }
    if threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    if warmups + reps == 0 {
        return Err("--warmups + --reps must be positive".to_string());
    }

    let julia_root = canonicalize_existing_dir(&julia_root, "Julia-mVMC root")?;
    let models = if model_names.is_empty() {
        HUBBARD_PHYSCAL_MODELS.to_vec()
    } else {
        model_names
            .iter()
            .map(|name| {
                HUBBARD_PHYSCAL_MODELS
                    .iter()
                    .copied()
                    .find(|model| model.name == name)
                    .ok_or_else(|| format!("unknown model `{name}`"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(PhyscalHubbardBenchConfig {
        reps,
        warmups,
        threads,
        keep_output,
        julia_root,
        julia_bin,
        csv,
        report,
        models,
    })
}

fn physcal_hubbard_input_dir(workspace: &Path, model: HubbardModel) -> PathBuf {
    workspace
        .join("benchmark")
        .join("physcal")
        .join("inputs")
        .join(model.name)
}

fn physcal_hubbard_namelist(workspace: &Path, model: HubbardModel) -> PathBuf {
    physcal_hubbard_input_dir(workspace, model).join("namelist.def")
}

fn physcal_hubbard_opt_params(workspace: &Path, model: HubbardModel) -> PathBuf {
    physcal_hubbard_input_dir(workspace, model).join("zqp_opt.dat")
}

fn physcal_hubbard_samples(workspace: &Path, model: HubbardModel) -> usize {
    modpara_value(&physcal_hubbard_namelist(workspace, model), "NDataQtySmp").unwrap_or(0)
}

fn run_rust_physcal_hubbard(
    workspace: &Path,
    run_root: &Path,
    model: HubbardModel,
    config: &PhyscalHubbardBenchConfig,
) -> Result<Vec<Measurement>, String> {
    let binary = workspace
        .join("target")
        .join("release")
        .join(executable_name("mvmc"));
    if !binary.is_file() {
        return Err(format!("Rust CLI binary not found: {}", binary.display()));
    }

    for warmup in 0..config.warmups {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("warmup-{}", warmup + 1));
        run_rust_physcal_hubbard_once(workspace, &binary, model, &out, config)?;
    }

    let samples = physcal_hubbard_samples(workspace, model);
    let mut measurements = Vec::with_capacity(config.reps);
    for rep in 0..config.reps {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", rep + 1));
        let (seconds, energy, _output) =
            run_rust_physcal_hubbard_once(workspace, &binary, model, &out, config)?;
        measurements.push(Measurement {
            implementation: "rust",
            model: model.name.to_string(),
            rep: rep + 1,
            steps: samples,
            seconds,
            final_energy_per_site: energy,
        });
    }
    Ok(measurements)
}

fn run_rust_physcal_hubbard_once(
    workspace: &Path,
    binary: &Path,
    model: HubbardModel,
    out_root: &Path,
    config: &PhyscalHubbardBenchConfig,
) -> Result<(f64, Option<f64>, Output), String> {
    let namelist = physcal_hubbard_namelist(workspace, model);
    let opt_params = physcal_hubbard_opt_params(workspace, model);
    if !namelist.is_file() {
        return Err(format!(
            "PhysCal namelist not found: {}",
            namelist.display()
        ));
    }
    if !opt_params.is_file() {
        return Err(format!(
            "PhysCal fixed parameters not found: {}",
            opt_params.display()
        ));
    }
    let mut command = Command::new(binary);
    command
        .arg(&namelist)
        .arg("--physcal")
        .arg(&opt_params)
        .arg("--mode")
        .arg("real")
        .arg("--out-dir")
        .arg(out_root);
    apply_thread_env(&mut command, config.threads);
    let output = command
        .output()
        .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;
    ensure_success(&binary.to_string_lossy(), &output)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let seconds = parse_rust_physcal_seconds(&stdout).ok_or_else(|| {
        format!(
            "cannot parse Rust PhysCal timing for {} from:\n{stdout}",
            model.name
        )
    })?;
    let energy = read_rust_physcal_energy_per_site(out_root, &namelist);
    Ok((seconds, energy, output))
}

/// Parse `=== Completed 100 PhysCal samples in 1.98s ===` into `1.98`.
fn parse_rust_physcal_seconds(stdout: &str) -> Option<f64> {
    stdout.lines().find_map(|line| {
        let rest = line.strip_prefix("=== Completed ")?;
        let (_, tail) = rest.split_once(" PhysCal samples in ")?;
        tail.split_whitespace()
            .next()?
            .trim_end_matches('s')
            .parse()
            .ok()
    })
}

/// Read the last indexed `zvo_out_NNN.dat` sample row and divide by `Nsite`.
///
/// PhysCal writes one indexed file per sample, unlike the optimization path's
/// single `zvo_out.dat`.
fn read_rust_physcal_energy_per_site(out_root: &Path, namelist: &Path) -> Option<f64> {
    let mut entries: Vec<PathBuf> = fs::read_dir(out_root)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("zvo_out_") && name.ends_with(".dat"))
        })
        .collect();
    entries.sort();
    let last = entries.last()?;
    let text = fs::read_to_string(last).ok()?;
    let row = text.lines().rfind(|line| !line.trim().is_empty())?;
    let etot = row.split_whitespace().next()?.parse::<f64>().ok()?;
    let nsite = modpara_nsite(namelist)?;
    (nsite != 0).then(|| etot / nsite as f64)
}

fn run_julia_physcal_hubbard(
    julia_root: &Path,
    runner: &Path,
    run_root: &Path,
    model: HubbardModel,
    config: &PhyscalHubbardBenchConfig,
) -> Result<Vec<Measurement>, String> {
    let workspace = workspace_root();
    let namelist = physcal_hubbard_namelist(&workspace, model);
    let opt_params = physcal_hubbard_opt_params(&workspace, model);
    if !namelist.is_file() {
        return Err(format!(
            "PhysCal namelist not found: {}",
            namelist.display()
        ));
    }
    if !opt_params.is_file() {
        return Err(format!(
            "PhysCal fixed parameters not found: {}",
            opt_params.display()
        ));
    }
    let samples = physcal_hubbard_samples(&workspace, model);
    let nsite = modpara_nsite(&namelist).unwrap_or(0);

    let out_root = run_root.join("julia").join(model.name);
    fs::create_dir_all(&out_root)
        .map_err(|e| format!("cannot create {}: {e}", out_root.display()))?;

    let mut command = Command::new(&config.julia_bin);
    if config.julia_bin == Path::new("julia") {
        command.arg("+1.13.1");
    }
    command
        .arg(format!("--project={}", julia_root.display()))
        .arg("--startup-file=no")
        .arg("--history-file=no")
        .arg(format!("--threads={}", config.threads));
    apply_thread_env(&mut command, config.threads);
    let output = command
        .arg(runner)
        .arg(model.name)
        .arg("real")
        .arg(&namelist)
        .arg(&opt_params)
        .arg(config.warmups.to_string())
        .arg(config.reps.to_string())
        .arg(&out_root)
        .arg(nsite.to_string())
        .current_dir(julia_root)
        .output()
        .map_err(|e| format!("failed to spawn Julia: {e}"))?;
    ensure_success("julia bench-physcal-hubbard runner", &output)?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut measurements = Vec::with_capacity(config.reps);
    for line in stdout.lines() {
        if let Some(measurement) = parse_julia_bench_line(line, model.name, samples) {
            measurements.push(measurement);
        }
    }

    if measurements.len() != config.reps {
        return Err(format!(
            "Julia benchmark produced {} measurements, expected {}\nstdout:\n{}\nstderr:\n{}",
            measurements.len(),
            config.reps,
            stdout,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(measurements)
}

fn print_physcal_hubbard_comparison(measurements: &[Measurement], models: &[HubbardModel]) {
    println!("=== Summary ===");
    for model in models {
        let rust = seconds_for(measurements, "rust", model.name);
        let julia = seconds_for(measurements, "julia", model.name);
        if rust.is_empty() || julia.is_empty() {
            continue;
        }
        let rust_median = median(&rust);
        let julia_median = median(&julia);
        let energy_str = match energy_delta(measurements, model.name) {
            Some(d) => format!("|ΔE|={d:.2e}"),
            None => "|ΔE|=   n/a".to_string(),
        };
        println!(
            "{:<24} rust(med)={:>7.3}s julia(med)={:>7.3}s  speedup(julia/rust)={:>5.2}x  {}",
            model.name,
            rust_median,
            julia_median,
            julia_median / rust_median,
            energy_str,
        );
    }
}

fn build_physcal_hubbard_report(
    config: &PhyscalHubbardBenchConfig,
    measurements: &[Measurement],
    observables: &[(String, Vec<ObservableDiff>)],
) -> String {
    let julia_version = if config.julia_bin == Path::new("julia") {
        command_output("julia", &["+1.13.1", "--version"])
    } else {
        command_output(&config.julia_bin.to_string_lossy(), &["--version"])
    };
    let mut report = String::new();
    report.push_str("# Rust vs Julia Hubbard-chain PhysCal benchmark\n\n");
    report.push_str(&format!(
        "- generated: {}\n",
        command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
    ));
    report.push_str(&format!(
        "- platform: {}\n",
        command_output("uname", &["-sm"])
    ));
    report.push_str(&format!(
        "- rustc: {}\n",
        command_output("rustc", &["--version"])
    ));
    report.push_str(&format!("- julia: {julia_version}\n"));
    report.push_str(&format!(
        "- reps/warmups/threads: {}/{}/{}\n",
        config.reps, config.warmups, config.threads
    ));
    report.push_str("- inputs: benchmark/physcal/inputs\n");
    report.push_str(
        "- timing: Rust process wall clock excluding warm-up runs; Julia\n  `run_phys_cal_from_namelist` wall clock excluding JIT and warm-up runs.\n  Both sides are thread-pinned but the measured scopes differ (Rust includes\n  process startup).\n\n",
    );
    report.push_str("| model | Rust median (s) | Julia median (s) | speedup (julia/rust) |\n");
    report.push_str("|---|---:|---:|---:|\n");
    for model in &config.models {
        let rust = seconds_for(measurements, "rust", model.name);
        let julia = seconds_for(measurements, "julia", model.name);
        if rust.is_empty() || julia.is_empty() {
            continue;
        }
        let rust_median = median(&rust);
        let julia_median = median(&julia);
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3}x |\n",
            model.name,
            rust_median,
            julia_median,
            julia_median / rust_median,
        ));
    }
    report.push_str("\n`speedup = julia / rust`; values above `1.0x` mean Rust was faster.\n");
    report.push_str("\n## PhysCal observables (Rust vs Julia, last rep)\n\n");
    if observables.iter().all(|(_, rows)| rows.is_empty()) {
        report.push_str("No observable output files were found.\n");
    } else {
        report.push_str(
            "Maximum absolute and relative difference of the measured Green-function\n\
             files (`zvo_cisajs*`) and, when `NLanczosMode > 0`, the Lanczos files\n\
             (`zvo_ls_*`). `status` uses the documented PhysCal bounds\n\
             (absolute `1e-10`, relative `1e-9`).\n\n",
        );
        report.push_str("| model | family | files | values | max abs | max rel | status |\n");
        report.push_str("|---|---|---:|---:|---:|---:|---|\n");
        for (model, rows) in observables {
            if rows.is_empty() {
                report.push_str(&format!("| {model} | - | 0 | 0 | - | - | - |\n"));
                continue;
            }
            for row in rows {
                report.push_str(&format!(
                    "| {} | {} | {} | {} | {:.3e} | {:.3e} | {} |\n",
                    model,
                    row.family,
                    row.files,
                    row.values,
                    row.max_abs,
                    row.max_rel,
                    if row.within_tolerance() { "ok" } else { "DIFF" },
                ));
            }
        }
    }
    report.push_str(
        "\nRun with `MVMC_C_TIMER=1` and `--keep-output` to keep each side's\n`zvo_CalcTimer.dat` for the section breakdown.\n",
    );
    report
}

fn print_physcal_hubbard_help() {
    println!("USAGE: cargo run -p xtask -- bench-physcal-hubbard [options]");
    println!();
    println!("Runs fixed-parameter PhysCal (NVMCCalMode=1) on the Rust-local");
    println!("Hubbard-chain inputs under benchmark/physcal/inputs and compares");
    println!("Rust (mvmc-cli --physcal) with Julia (run_phys_cal_from_namelist).");
    println!();
    println!("Options:");
    println!("  --model <NAME>       run only NAME (repeatable); default: L16/L24/L32");
    println!("  --reps <N>           measured repetitions (default: 3)");
    println!("  --warmups <N>        warm-up repetitions (default: 1)");
    println!("  --threads <N>        pinned thread count (default: 1)");
    println!("  --julia-root <PATH>  Julia-mVMC root (default: extern/Julia-mVMC)");
    println!("  --julia-bin <PATH>   Julia binary (default: julia)");
    println!("  --csv <PATH>         CSV output (default: target/bench/physcal_hubbard.csv)");
    println!(
        "  --report <PATH>      Markdown report (default: target/bench/physcal_hubbard_report.md)"
    );
    println!("  --keep-output        keep per-run outputs (including zvo_CalcTimer.dat)");
    println!("  --help               show this help");
}

fn write_julia_runner(workspace: &Path) -> Result<PathBuf, String> {
    let dir = workspace.join("target").join("bench");
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = dir.join("julia_bench_runner.jl");
    fs::write(&path, JULIA_BENCH_RUNNER)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

fn canonicalize_existing_dir(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_dir() {
        return Err(format!("{label} is not a directory: {}", path.display()));
    }
    path.canonicalize()
        .map_err(|e| format!("cannot canonicalize {}: {e}", path.display()))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask directory has parent")
        .to_path_buf()
}

fn executable_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mu = mean(values);
    let var = values.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

/// Pin BLAS / OpenMP / Julia thread counts to the same value on both runtimes
/// so the benchmark compares scalar/SIMD performance rather than thread-pool
/// scheduling. Setting all of these is intentional: different upstream BLAS
/// backends (OpenBLAS / MKL / Accelerate) read different env vars, and Julia
/// honors JULIA_NUM_THREADS on top of OpenBLAS.
fn apply_thread_env(command: &mut Command, n: usize) {
    let v = n.to_string();
    command
        .env("OPENBLAS_NUM_THREADS", &v)
        .env("OMP_NUM_THREADS", &v)
        .env("MKL_NUM_THREADS", &v)
        .env("BLIS_NUM_THREADS", &v)
        .env("VECLIB_MAXIMUM_THREADS", &v)
        .env("JULIA_NUM_THREADS", &v)
        .env("RAYON_NUM_THREADS", &v);
}

/// Rust-only inner-kernel controls (`MVMC_RS_INNER_THREADS` / `MVMC_RS_INNER_THRESHOLD`).
/// They never reach the Julia runner, whose thread count stays `--threads`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct InnerEnv {
    workers: Option<usize>,
    threshold: Option<usize>,
    /// Linux CPU list (`taskset -c`) for the Rust process. Woken Rayon workers are
    /// otherwise packed onto the waker's core or hyperthread sibling by the scheduler.
    cpus: Option<String>,
}

impl InnerEnv {
    fn apply(&self, command: &mut Command) {
        if let Some(n) = self.workers {
            command.env("MVMC_RS_INNER_THREADS", n.to_string());
        }
        if let Some(n) = self.threshold {
            command.env("MVMC_RS_INNER_THRESHOLD", n.to_string());
        }
    }

    /// The command that starts a Rust run: `binary`, or `taskset -c <cpus> binary`.
    fn command(&self, binary: &Path) -> Command {
        match &self.cpus {
            Some(cpus) => {
                let mut command = Command::new("taskset");
                command.arg("-c").arg(cpus).arg(binary);
                command
            }
            None => Command::new(binary),
        }
    }

    fn describe(&self) -> String {
        let show = |v: Option<usize>, default: &str| {
            v.map_or_else(|| format!("{default} (default)"), |n| n.to_string())
        };
        format!(
            "inner workers {}, inner threshold {}, cpu list {}",
            show(self.workers, "1"),
            show(self.threshold, "automatic work estimate"),
            self.cpus.as_deref().unwrap_or("<any>")
        )
    }

    /// Consume `--inner-workers` / `--inner-threshold`; returns whether `flag` was one.
    fn parse_flag(&mut self, flag: &str, value: Option<&String>) -> Result<bool, String> {
        if flag == "--inner-cpus" {
            let list = value.ok_or_else(|| format!("{flag} requires a value"))?;
            if list.is_empty() || !list.chars().all(|c| c.is_ascii_digit() || ",-".contains(c)) {
                return Err(format!(
                    "invalid value for {flag} (use a taskset list like 2,4-7)"
                ));
            }
            self.cpus = Some(list.clone());
            return Ok(true);
        }
        let slot = match flag {
            "--inner-workers" => &mut self.workers,
            "--inner-threshold" => &mut self.threshold,
            _ => return Ok(false),
        };
        let n: usize = parse_value(value, flag)?;
        if n == 0 {
            return Err(format!("{flag} must be positive"));
        }
        *slot = Some(n);
        Ok(true)
    }
}

const INNER_HELP: &str = "  --inner-workers <N>  Rust MVMC_RS_INNER_THREADS (Rust only; BLAS stays at --threads)\n  --inner-threshold <N> Rust MVMC_RS_INNER_THRESHOLD: force the item-count gate (default: automatic work estimate)\n  --inner-cpus <LIST>  run the Rust process under `taskset -c LIST` (Linux; e.g. 2,4,6,8)";

fn print_help() {
    println!("xtask -- workspace task runner");
    println!();
    println!("USAGE: cargo run -p xtask -- <task>");
    println!();
    println!("TASKS:");
    println!("  bench-julia     compare Rust examples with ../extern/Julia-mVMC");
    println!("  bench-hubbard   compare Rust vs Julia on the report Hubbard-chain inputs");
    println!("  bench-physcal   compare Rust vs Julia fixed-parameter PhysCal speed");
    println!("  bench-physcal-hubbard   PhysCal on the Hubbard-chain benchmark inputs");
}

fn print_bench_help() {
    println!("USAGE: cargo run -p xtask -- bench-julia [options]");
    println!();
    println!("Options:");
    println!("  --steps <N>          SR steps per run [default: 10]");
    println!("  --reps <N>           measured repetitions [default: 3]");
    println!("  --warmups <N>        warmup repetitions [default: 1]");
    println!("  --threads <N>        pin BLAS / OpenMP / Julia threads on both sides");
    println!("{INNER_HELP}");
    println!("  --model <NAME>       benchmark one model; repeatable");
    println!("  --julia-root <DIR>   Julia-mVMC checkout [default: ../extern/Julia-mVMC]");
    println!("  --julia-bin <PATH>   Julia executable or pinned binary [default: julia +1.13.1]");
    println!("  --csv <PATH>         CSV output [default: target/bench/julia_vs_rust.csv]");
    println!("  --keep-output        keep per-run zvo_out.dat / zqp_opt.dat files");
}

fn print_hubbard_help() {
    println!("USAGE: cargo run -p xtask -- bench-hubbard [options]");
    println!();
    println!("Compares Rust (mvmc-cli) and Julia on the C-vs-Julia report Hubbard-chain");
    println!("inputs (L=16/24/32, half filling) with matching internal timing.");
    println!();
    println!("Options:");
    println!("  --steps <N>          SR steps per run [default: 300]");
    println!("  --reps <N>           measured repetitions [default: 3]");
    println!("  --warmups <N>        warmup repetitions [default: 1]");
    println!("  --threads <N>        pin BLAS / OpenMP / Julia threads [default: 1]");
    println!("{INNER_HELP}");
    println!("  --model <NAME>       benchmark one model; repeatable");
    println!("  --julia-root <DIR>   Julia-mVMC checkout [default: extern/Julia-mVMC]");
    println!("  --julia-bin <PATH>   Julia executable or pinned binary [default: julia +1.13.1]");
    println!("  --csv <PATH>         CSV output [default: target/bench/hubbard_chain.csv]");
    println!(
        "  --report <PATH>      Markdown report [default: target/bench/hubbard_chain_report.md]"
    );
    println!("  --keep-output        keep per-run output files");
}

fn print_physcal_help() {
    println!("USAGE: cargo run -p xtask -- bench-physcal [options]");
    println!();
    println!("Compares Rust (mvmc-cli --physcal) and Julia on fixed-parameter");
    println!("PhysCal integration fixtures.");
    println!();
    println!("Options:");
    println!("  --reps <N>           measured repetitions [default: 3]");
    println!("  --warmups <N>        warmup repetitions [default: 1]");
    println!("  --threads <N>        pin BLAS / OpenMP / Julia threads [default: 1]");
    println!("  --model <NAME>       benchmark one model; repeatable");
    println!("  --julia-root <DIR>   Julia-mVMC checkout [default: extern/Julia-mVMC]");
    println!("  --julia-bin <PATH>   Julia executable or pinned binary [default: julia +1.13.1]");
    println!("  --csv <PATH>         CSV output [default: target/bench/physcal_chain.csv]");
    println!("  --keep-output        keep per-run output files");
}

const JULIA_BENCH_RUNNER: &str = r#"
using MVMCOptimizers
using LinearAlgebra

# Honor whatever the xtask runner pinned via env (OPENBLAS_NUM_THREADS etc.).
# Setting BLAS.set_num_threads is required because the env var only gates
# OpenBLAS at libdl-load time; once Julia is up we must set it explicitly to
# avoid the default "all physical cores" behaviour from biasing the timings.
let nt = get(ENV, "OPENBLAS_NUM_THREADS", "")
    if !isempty(nt)
        try
            BLAS.set_num_threads(parse(Int, nt))
        catch
        end
    end
end

model = ARGS[1]
mode = Symbol(ARGS[2])
namelist = ARGS[3]
steps = parse(Int, ARGS[4])
warmups = parse(Int, ARGS[5])
reps = parse(Int, ARGS[6])
out_root = ARGS[7]

function run_once(iter::Int)
    out_dir = joinpath(out_root, "run_$(iter)")
    rm(out_dir; force = true, recursive = true)
    mkpath(out_dir)
    GC.gc()
    t0 = time_ns()
    result = MVMCOptimizers.run_para_opt_from_namelist(
        namelist;
        nsteps = steps,
        nsmp = steps,
        mode = mode,
        output_dir = out_dir,
    )
    elapsed = (time_ns() - t0) / 1.0e9
    return elapsed, result.final_energy_per_site
end

for iter in 1:(warmups + reps)
    elapsed, final_energy = run_once(iter)
    if iter <= warmups
        println("WARMUP\tjulia\t$(model)\t$(iter)\t$(elapsed)\t$(final_energy)")
    else
        rep = iter - warmups
        println("BENCH\tjulia\t$(model)\t$(rep)\t$(elapsed)\t$(final_energy)")
    end
    flush(stdout)
end
"#;

const JULIA_PHYSCAL_BENCH_RUNNER: &str = r#"
using MVMCOptimizers
using LinearAlgebra

# Honor whatever the xtask runner pinned via env (OPENBLAS_NUM_THREADS etc.).
# Setting BLAS.set_num_threads is required because the env var only gates
# OpenBLAS at libdl-load time; once Julia is up we must set it explicitly to
# avoid the default "all physical cores" behaviour from biasing the timings.
let nt = get(ENV, "OPENBLAS_NUM_THREADS", "")
    if !isempty(nt)
        try
            BLAS.set_num_threads(parse(Int, nt))
        catch
        end
    end
end

model = ARGS[1]
mode = Symbol(ARGS[2])
namelist = ARGS[3]
opt_para = ARGS[4]
warmups = parse(Int, ARGS[5])
reps = parse(Int, ARGS[6])
out_root = ARGS[7]
nsite = parse(Int, ARGS[8])

function energy_per_site(out_dir)::String
    nsite == 0 && return "n/a"
    path = joinpath(out_dir, "zvo_out.dat")
    if !isfile(path)
        # C-indexed PhysCal output (one file per sample): read the last sample,
        # matching the Rust reader. Older Julia wrote a single non-indexed file.
        indexed = sort(filter(n -> startswith(n, "zvo_out_") && endswith(n, ".dat"),
                              readdir(out_dir)))
        isempty(indexed) && return "n/a"
        path = joinpath(out_dir, last(indexed))
    end
    for line in reverse(readlines(path))
        stripped = strip(line)
        if !isempty(stripped)
            etot = parse(Float64, split(stripped)[1])
            return string(etot / nsite)
        end
    end
    return "n/a"
end

function run_once(iter::Int)
    out_dir = joinpath(out_root, "run_$(iter)")
    rm(out_dir; force = true, recursive = true)
    mkpath(out_dir)
    GC.gc()
    t0 = time_ns()
    result = MVMCOptimizers.run_phys_cal_from_namelist(
        namelist;
        opt_para = opt_para,
        mode = mode,
        output_dir = out_dir,
    )
    elapsed = (time_ns() - t0) / 1.0e9
    energy = energy_per_site(out_dir)
    return elapsed, energy
end

for iter in 1:(warmups + reps)
    elapsed, final_energy = run_once(iter)
    if iter <= warmups
        println("WARMUP\tjulia\t$(model)\t$(iter)\t$(elapsed)\t$(final_energy)")
    else
        rep = iter - warmups
        println("BENCH\tjulia\t$(model)\t$(rep)\t$(elapsed)\t$(final_energy)")
    end
    flush(stdout)
end
"#;

#[cfg(test)]
mod tests {
    use super::{
        ctest_failure, parse_julia_bench_line, parse_rust_final_energy, parse_rust_seconds,
        Command, InnerEnv,
    };

    #[test]
    fn ctest_requires_both_statistical_and_absolute_thresholds() {
        assert!(!ctest_failure(9.0e-9, 0.0));
        assert!(!ctest_failure(2.0e-8, 1.0e-8));
        assert!(ctest_failure(3.1e-8, 1.0e-8));
        assert!(!ctest_failure(f64::NAN, 0.0));
    }

    #[test]
    fn inner_env_parses_flags_and_rejects_zero() {
        let mut inner = InnerEnv::default();
        assert!(inner
            .parse_flag("--inner-workers", Some(&"4".to_string()))
            .unwrap());
        assert!(inner
            .parse_flag("--inner-threshold", Some(&"8".to_string()))
            .unwrap());
        assert!(!inner
            .parse_flag("--threads", Some(&"2".to_string()))
            .unwrap());
        assert!(inner
            .parse_flag("--inner-cpus", Some(&"2,4-7".to_string()))
            .unwrap());
        assert!(inner
            .parse_flag("--inner-cpus", Some(&"2;rm".to_string()))
            .is_err());
        assert_eq!(inner.cpus.as_deref(), Some("2,4-7"));
        assert_eq!(inner.workers, Some(4));
        assert_eq!(inner.threshold, Some(8));
        assert!(inner
            .parse_flag("--inner-workers", Some(&"0".to_string()))
            .is_err());
        let mut command = Command::new("true");
        inner.apply(&mut command);
        let envs: Vec<_> = command.get_envs().collect();
        assert_eq!(envs.len(), 2);
    }

    #[test]
    fn parses_rust_internal_timing() {
        let stdout = "some output\n=== Completed 300 SR steps in 4.11s ===\nmore\n";
        assert_eq!(parse_rust_seconds(stdout), Some(4.11));
        assert_eq!(parse_rust_seconds("no timing here"), None);
    }

    #[test]
    fn parses_rust_final_energy() {
        let stdout = "Final energy / site: -0.5418807042\n";
        assert_eq!(parse_rust_final_energy(stdout), Some(-0.5418807042));
        assert_eq!(parse_rust_final_energy("Final energy / site = 0.1"), None);
    }

    #[test]
    fn parses_julia_bench_line() {
        let line = "BENCH\tjulia\thubbard_chain_L16\t2\t2.968304542\t-0.5418807042567225";
        let measurement =
            parse_julia_bench_line(line, "hubbard_chain_L16", 300).expect("valid BENCH line");
        assert_eq!(measurement.implementation, "julia");
        assert_eq!(measurement.rep, 2);
        assert_eq!(measurement.steps, 300);
        assert!((measurement.seconds - 2.968304542).abs() < 1e-12);
        assert!(parse_julia_bench_line("WARMUP\tjulia\tx\t1\t1.0\t0.0", "x", 1).is_none());
        assert!(parse_julia_bench_line("garbage", "x", 1).is_none());
    }

    #[test]
    fn committed_hubbard_inputs_report_nsite() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask has a workspace parent");
        for (model, expected) in [
            ("hubbard_chain_L16", 16usize),
            ("hubbard_chain_L24", 24),
            ("hubbard_chain_L32", 32),
        ] {
            let namelist = workspace
                .join("benchmark/hubbard_chain/inputs")
                .join(model)
                .join("namelist.def");
            assert_eq!(
                super::modpara_nsite(&namelist),
                Some(expected),
                "model {model}"
            );
        }
    }

    #[test]
    fn committed_physcal_inputs_report_nvmc_sample() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask has a workspace parent");
        for (model, expected) in [
            ("heisenberg_chain_real", 100usize),
            ("heisenberg_chain_cmp", 100),
            ("heisenberg_chain_fsz", 200),
            ("hubbard_chain_real", 100),
        ] {
            let namelist = workspace
                .join("extern/Julia-mVMC/test/integration/reference")
                .join(model)
                .join("physcal_ref/inputs/namelist.def");
            assert_eq!(
                super::modpara_nvmc_sample(&namelist),
                Some(expected),
                "model {model}"
            );
        }
    }

    #[test]
    fn observable_family_strips_sample_index() {
        assert_eq!(
            super::observable_family("zvo_cisajs_001.dat"),
            Some("zvo_cisajs")
        );
        assert_eq!(
            super::observable_family("zvo_cisajscktalt_100.dat"),
            Some("zvo_cisajscktalt")
        );
        assert_eq!(
            super::observable_family("zvo_cisajscktaltex_007.dat"),
            Some("zvo_cisajscktaltex")
        );
        assert_eq!(
            super::observable_family("zvo_ls_qqqq_001.dat"),
            Some("zvo_ls_qqqq")
        );
        assert_eq!(super::observable_family("zvo_out_001.dat"), None);
        assert_eq!(super::observable_family("zvo_cisajs.dat"), None);
        assert_eq!(super::observable_family("zvo_cisajs_00x.dat"), None);
    }

    #[test]
    fn observable_comparison_reports_max_difference() {
        let root = std::env::temp_dir().join(format!(
            "xtask-observable-{}-{}",
            std::process::id(),
            "max_difference"
        ));
        let rust = root.join("rust");
        let julia = root.join("julia");
        std::fs::create_dir_all(&rust).unwrap();
        std::fs::create_dir_all(&julia).unwrap();
        std::fs::write(
            rust.join("zvo_cisajs_001.dat"),
            "0 0 0 0  1.0000000000000000e+00  0.0000000000000000e+00\n",
        )
        .unwrap();
        std::fs::write(
            julia.join("zvo_cisajs_001.dat"),
            "0 0 0 0  1.0000000000000002e+00  0.0000000000000000e+00\n",
        )
        .unwrap();
        let diffs = super::compare_physcal_observables(&rust, &julia).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].family, "zvo_cisajs");
        assert_eq!(diffs[0].files, 1);
        assert_eq!(diffs[0].values, 2);
        assert!(diffs[0].max_abs > 0.0 && diffs[0].max_abs < 1.0e-15);
        assert!(diffs[0].within_tolerance());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn observable_comparison_rejects_discrete_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "xtask-observable-{}-{}",
            std::process::id(),
            "discrete_mismatch"
        ));
        let rust = root.join("rust");
        let julia = root.join("julia");
        std::fs::create_dir_all(&rust).unwrap();
        std::fs::create_dir_all(&julia).unwrap();
        std::fs::write(rust.join("zvo_cisajs_001.dat"), "1 0 0 0  0.0  0.0\n").unwrap();
        std::fs::write(julia.join("zvo_cisajs_001.dat"), "0 0 0 0  0.0  0.0\n").unwrap();
        let error = super::compare_physcal_observables(&rust, &julia).unwrap_err();
        assert!(error.contains("discrete value differs"), "{error}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
