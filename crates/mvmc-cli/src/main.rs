//! `mvmc` — Julia-mVMC → Rust port command-line driver.
//!
//! Phase 5: wires `mvmc_core::run_para_opt_from_namelist` to a CLI so
//! the binary can reproduce the Julia `examples/*.jl` workflow from the
//! terminal.
//!
//! Usage:
//!   mvmc <namelist.def> [options]
//!
//! Options:
//!   --nsteps <N>      SR optimisation steps  [default: value in modpara.def]
//!   --out-dir <DIR>   Output directory       [default: namelist parent dir]
//!   --seed <N>        RNG seed override      [default: RndSeed in modpara.def]
//!   --help / -h       Print this help text
//!
//! Environment:
//!   MVMC_NSTEPS       Same as --nsteps (CLI flag takes precedence)
//!
//! Mirror of `extern/Julia-mVMC/examples/heisenberg_chain_real.jl` et al.

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

fn print_usage(program: &str) {
    eprintln!("Usage: {program} <namelist.def> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --nsteps <N>    SR optimisation steps [default: NSROptItrStep in modpara.def]");
    eprintln!("  --out-dir <DIR> Output directory      [default: namelist parent dir]");
    eprintln!("  --seed <N>      RNG seed override     [default: RndSeed in modpara.def]");
    eprintln!("  --help, -h      Print this help");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  MVMC_NSTEPS     Same as --nsteps (CLI flag takes precedence)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let program = args.first().map(String::as_str).unwrap_or("mvmc");

    // ── argument parsing (no external crate dependency) ──────────────────────
    let mut namelist: Option<PathBuf> = None;
    let mut nsteps_arg: Option<usize> = None;
    let mut out_dir_arg: Option<PathBuf> = None;
    let mut seed_arg: Option<i64> = None;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                print_usage(program);
                process::exit(0);
            }
            "--nsteps" => {
                idx += 1;
                nsteps_arg = args.get(idx).and_then(|s| s.parse().ok());
            }
            "--out-dir" => {
                idx += 1;
                out_dir_arg = args.get(idx).map(PathBuf::from);
            }
            "--seed" => {
                idx += 1;
                seed_arg = args.get(idx).and_then(|s| s.parse().ok());
            }
            flag if flag.starts_with('-') => {
                eprintln!("error: unknown flag `{flag}`");
                print_usage(program);
                process::exit(2);
            }
            path => {
                if namelist.is_none() {
                    namelist = Some(PathBuf::from(path));
                } else {
                    eprintln!("error: unexpected argument `{path}`");
                    process::exit(2);
                }
            }
        }
        idx += 1;
    }

    let namelist = match namelist {
        Some(p) => p,
        None => {
            eprintln!("error: <namelist.def> argument is required");
            print_usage(program);
            process::exit(2);
        }
    };

    if !namelist.is_file() {
        eprintln!("error: file not found: {}", namelist.display());
        process::exit(1);
    }

    // MVMC_NSTEPS env var (CLI flag takes precedence)
    let nsteps_env: Option<usize> = std::env::var("MVMC_NSTEPS")
        .ok()
        .and_then(|s| s.parse().ok());
    let nsteps_override = nsteps_arg.or(nsteps_env);

    // Default output dir: namelist's parent directory.
    let out_dir: PathBuf =
        out_dir_arg.unwrap_or_else(|| namelist.parent().unwrap_or(Path::new(".")).join("output"));
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("error: cannot create output dir {}: {e}", out_dir.display());
        process::exit(1);
    }

    // ── banner ────────────────────────────────────────────────────────────────
    println!("=== mvmc — Julia-mVMC Rust port ===");
    println!("namelist : {}", namelist.display());
    println!("out-dir  : {}", out_dir.display());
    if let Some(n) = nsteps_override {
        println!("nsteps   : {n} (override)");
    }
    if let Some(s) = seed_arg {
        println!("seed     : {s} (override)");
    }
    println!();

    // ── peek at modpara to determine defaults and show model info ─────────────
    let nsteps = match mvmc_expert_parsers::parse_expert_mode_files(&namelist) {
        Ok(data) => {
            let p = &data.modpara;
            let nsteps_modpara = p.nsr_opt_itr_step.max(0) as usize;
            let nsteps = nsteps_override.unwrap_or(nsteps_modpara);
            println!(
                "model    : Nsite={} Nelec={} NSROptItrStep={}",
                p.nsite, p.nelec, nsteps,
            );
            println!("mode     : NVMCCalMode={}", p.vmc_calc_mode);
            println!(
                "sample   : NVMCSample={} NVMCWarmUp={}",
                p.nvmc_sample, p.nvmc_warmup
            );
            println!();
            nsteps
        }
        Err(e) => {
            // Fall back: if we can't parse, use the override or a safe default.
            eprintln!("warning: could not pre-parse modpara ({e}), using nsteps override or 0");
            nsteps_override.unwrap_or(0)
        }
    };

    if nsteps == 0 {
        eprintln!("error: NSROptItrStep is 0 — nothing to run. Use --nsteps <N>.");
        process::exit(1);
    }

    // ── run ───────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let result = mvmc_core::run_para_opt_from_namelist(&namelist, nsteps, seed_arg, Some(&out_dir));
    let elapsed = t0.elapsed();

    // ── result ────────────────────────────────────────────────────────────────
    match result {
        Ok(summary) => {
            println!();
            println!(
                "=== Completed {} SR steps in {:.2}s ===",
                summary.nsteps,
                elapsed.as_secs_f64()
            );
            if let Some(ref d) = summary.output_dir {
                println!("Output files written to: {}", d.display());
                // List key files that were created.
                for fname in &["zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"] {
                    let p = d.join(fname);
                    if p.exists() {
                        println!("  {fname}");
                    }
                }
            }
            // Print the last energy line for quick sanity-check.
            if let Some(ref d) = summary.output_dir {
                let zvo = d.join("zvo_out.dat");
                if let Ok(content) = std::fs::read_to_string(&zvo) {
                    if let Some(last) = content.lines().filter(|l| !l.trim().is_empty()).last() {
                        let tokens: Vec<&str> = last.split_whitespace().collect();
                        if let Some(e_str) = tokens.first() {
                            if let Ok(e) = e_str.parse::<f64>() {
                                println!();
                                println!("Final energy (last SR step): {e:.10}");
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("error: run failed: {e}");
            process::exit(1);
        }
    }
}
