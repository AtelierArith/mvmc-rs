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
//!   --nsmp <N>        Final averaging window [default: NSROptItrSmp]
//!   --mode <MODE>     Sanity label: real/cmp/fsz [default: inferred]
//!   --initial-def <auto|none|PATH> Starting parameter file [default: auto]
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
    eprintln!("  --nsmp <N>      Final averaging window [default: NSROptItrSmp]");
    eprintln!("  --mode <MODE>   Sanity label: real, cmp or fsz [default: inferred]");
    eprintln!("  --initial-def <auto|none|PATH> Starting parameter file [default: auto]");
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
    let mut nsteps_arg: Option<i64> = None;
    let mut out_dir_arg: Option<PathBuf> = None;
    let mut seed_arg: Option<i64> = None;
    let mut nsmp_arg: Option<i64> = None;
    let mut mode_arg: Option<String> = None;
    let mut initial_def = mvmc_core::InitialDef::Auto;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => {
                print_usage(program);
                process::exit(0);
            }
            "--nsteps" => {
                idx += 1;
                nsteps_arg = Some(args.get(idx).and_then(|s| s.parse().ok()).unwrap_or_else(
                    || {
                        eprintln!("error: --nsteps requires an integer");
                        process::exit(2)
                    },
                ));
            }
            "--out-dir" => {
                idx += 1;
                out_dir_arg = Some(args.get(idx).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("error: --out-dir requires a directory");
                    process::exit(2)
                }));
            }
            "--seed" => {
                idx += 1;
                seed_arg = Some(
                    args.get(idx)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("error: --seed requires an integer");
                            process::exit(2)
                        }),
                );
            }
            "--nsmp" => {
                idx += 1;
                nsmp_arg = Some(
                    args.get(idx)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("error: --nsmp requires an integer");
                            process::exit(2)
                        }),
                );
            }
            "--mode" => {
                idx += 1;
                mode_arg = Some(args.get(idx).cloned().unwrap_or_else(|| {
                    eprintln!("error: --mode requires real, cmp or fsz");
                    process::exit(2)
                }));
            }
            "--initial-def" => {
                idx += 1;
                initial_def = match args.get(idx).map(String::as_str) {
                    Some("auto") => mvmc_core::InitialDef::Auto,
                    Some("none") => mvmc_core::InitialDef::None,
                    Some(path) => mvmc_core::InitialDef::Path(PathBuf::from(path)),
                    None => {
                        eprintln!("error: --initial-def requires auto, none or a path");
                        process::exit(2)
                    }
                };
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
    let nsteps_env: Option<i64> = std::env::var("MVMC_NSTEPS")
        .ok()
        .and_then(|s| s.parse().ok());
    let nsteps_override = nsteps_arg.or(nsteps_env);

    // Default output dir: namelist's parent directory.
    let out_dir: PathBuf =
        out_dir_arg.unwrap_or_else(|| namelist.parent().unwrap_or(Path::new(".")).join("output"));

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
    let (nsteps, inferred_mode) = match mvmc_expert_parsers::parse_expert_mode_files(&namelist) {
        Ok(data) => {
            if let Err(e) = mvmc_core::validation::validate_para_opt(&data) {
                eprintln!("error: {e}");
                process::exit(1);
            }
            let p = &data.modpara;
            let nsteps_modpara = p.nsr_opt_itr_step;
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
            let mode = if data.i_flg_orbital_general != 0 {
                "fsz"
            } else if mvmc_core::get_all_complex_flag(&data) {
                "cmp"
            } else {
                "real"
            };
            (nsteps, mode)
        }
        Err(e) => {
            eprintln!("error: could not parse Expert input: {e}");
            process::exit(1);
        }
    };

    if nsteps <= 0 {
        eprintln!("error: NSROptItrStep is 0 — nothing to run. Use --nsteps <N>.");
        process::exit(1);
    }

    // ── run ───────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let config = mvmc_core::RunConfig {
        nsmp: nsmp_arg,
        seed: seed_arg,
        output_dir: Some(out_dir),
        initial_def,
        ..mvmc_core::RunConfig::new(nsteps, mode_arg.as_deref().unwrap_or(inferred_mode))
    };
    let result = mvmc_core::run_para_opt_from_namelist(&namelist, config);
    let elapsed = t0.elapsed();

    // ── result ────────────────────────────────────────────────────────────────
    match result {
        Ok(summary) => {
            println!();
            println!(
                "=== Completed {} SR steps in {:.2}s ===",
                summary.effective_nsteps,
                elapsed.as_secs_f64()
            );
            println!("Output files written to: {}", summary.output_dir.display());
            println!("Final energy / site: {:.10}", summary.final_energy_per_site);
            println!(
                "Final-window means ({} steps): {:?}",
                summary.effective_nsmp, summary.ctest_values
            );
        }
        Err(e) => {
            eprintln!("error: run failed: {e}");
            process::exit(1);
        }
    }
}
