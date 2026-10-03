//! Workspace task runner.

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
            flag => return Err(format!("unknown bench-julia flag `{flag}`")),
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
        run_rust_once(&binary, julia_root, &out, config.steps, config.threads)?;
    }

    let mut measurements = Vec::with_capacity(config.reps);
    for rep in 0..config.reps {
        let out = run_root
            .join("rust")
            .join(model.name)
            .join(format!("rep-{}", rep + 1));
        let (duration, output) =
            run_rust_once(&binary, julia_root, &out, config.steps, config.threads)?;
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
    steps: usize,
    threads: Option<usize>,
) -> Result<(Duration, Output), String> {
    let mut command = Command::new(binary);
    command
        .env("JULIA_MVMC_ROOT", julia_root)
        .env("JULIA_MVMC_EXAMPLE_STEPS", steps.to_string())
        .env("MVMC_OUT_DIR", out_root)
        .env("RUST_BACKTRACE", "1");
    if let Some(n) = threads {
        apply_thread_env(&mut command, n);
    }
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

fn print_help() {
    println!("xtask -- workspace task runner");
    println!();
    println!("USAGE: cargo run -p xtask -- <task>");
    println!();
    println!("TASKS:");
    println!("  bench-julia     compare Rust examples with ../extern/Julia-mVMC");
}

fn print_bench_help() {
    println!("USAGE: cargo run -p xtask -- bench-julia [options]");
    println!();
    println!("Options:");
    println!("  --steps <N>          SR steps per run [default: 10]");
    println!("  --reps <N>           measured repetitions [default: 3]");
    println!("  --warmups <N>        warmup repetitions [default: 1]");
    println!("  --threads <N>        pin BLAS / OpenMP / Julia threads on both sides");
    println!("  --model <NAME>       benchmark one model; repeatable");
    println!("  --julia-root <DIR>   Julia-mVMC checkout [default: ../extern/Julia-mVMC]");
    println!("  --julia-bin <PATH>   Julia executable or pinned binary [default: julia +1.13.1]");
    println!("  --csv <PATH>         CSV output [default: target/bench/julia_vs_rust.csv]");
    println!("  --keep-output        keep per-run zvo_out.dat / zqp_opt.dat files");
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

#[cfg(test)]
mod tests {
    use super::ctest_failure;

    #[test]
    fn ctest_requires_both_statistical_and_absolute_thresholds() {
        assert!(!ctest_failure(9.0e-9, 0.0));
        assert!(!ctest_failure(2.0e-8, 1.0e-8));
        assert!(ctest_failure(3.1e-8, 1.0e-8));
        assert!(!ctest_failure(f64::NAN, 0.0));
    }
}
