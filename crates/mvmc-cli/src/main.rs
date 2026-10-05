//! `mvmc` — Julia-mVMC → Rust port command-line driver.
//!
//! Phase 5: wires `mvmc_core::run_para_opt_from_namelist` to a CLI so
//! the binary can reproduce the Julia `examples/*.jl` workflow from the
//! terminal.
//!
//! Usage:
//!   mvmc <namelist.def> [options]
//!   mvmc uhf <namelist.def> [OptParaFile]   (C ComplexUHF initial-orbital tool)
//!
//! Options:
//!   --nsteps <N>      SR optimisation steps  [default: value in modpara.def]
//!   --out-dir <DIR>   Output directory       [default: namelist parent dir]
//!   --seed <N>        RNG seed override      [default: RndSeed in modpara.def]
//!   --nsmp <N>        Final averaging window [default: NSROptItrSmp]
//!   --mode <MODE>     Sanity label: real/cmp/fsz [default: inferred]
//!   --initial-def <auto|none|PATH> Starting parameter file [default: auto]
//!   -o / --opt-trans  Enable C optimized-translation mode [default: disabled]
//!   --physcal <PATH>  Run fixed-parameter PhysCal with this parameter file
//!                     (requires NVMCCalMode=1 in ModPara; NVMCCalMode=1 requires --physcal)
//!   -F, --flush-interval <N>  Flush _time_/_SRinfo files every N steps (C -F)
//!   --help / -h       Print this help text
//!
//! Environment:
//!   MVMC_NSTEPS       Same as --nsteps (CLI flag takes precedence)
//!
//! Mirror of `extern/Julia-mVMC/examples/heisenberg_chain_real.jl` et al.

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

mod physcal_trace;

fn print_usage(program: &str) {
    eprintln!("Usage: {program} <namelist.def> [options]");
    eprintln!("       {program} uhf <namelist.def> [OptParaFile]");
    eprintln!("       {program} -s <stan.in> [options]");
    eprintln!("       {program} --dry-run <stan.in>");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  -s, --standard  Standard mode: generate the Expert files from <stan.in> (StdFace)"
    );
    eprintln!(
        "                  into --out-dir (default: current directory), then run namelist.def"
    );
    eprintln!("  --dry-run       Generate the Expert files from <stan.in> and stop (vmcdry.out)");
    eprintln!("  -e, --expert    Expert mode (default; accepted for C compatibility)");
    eprintln!("  --nsteps <N>    SR optimisation steps [default: NSROptItrStep in modpara.def]");
    eprintln!("  --out-dir <DIR> Output directory      [default: namelist parent dir]");
    eprintln!("  --seed <N>      RNG seed override     [default: RndSeed in modpara.def]");
    eprintln!("  --nsmp <N>      Final averaging window [default: NSROptItrSmp]");
    eprintln!("  --mode <MODE>   Sanity label: real, cmp or fsz [default: inferred]");
    eprintln!("  --initial-def <auto|none|PATH> Starting parameter file [default: auto]");
    eprintln!("  -F, --flush-interval <N> Flush _time_/_SRinfo files every N steps [default: 1]");
    eprintln!("  -o, --opt-trans Enable C OptTrans mode [default: disabled]");
    eprintln!("  --physcal <PATH> Run fixed-parameter PhysCal using PATH");
    eprintln!("  --physcal-trace <NEW_DIR> Nonconsuming serial PhysCal diagnostics");
    eprintln!("  --help, -h      Print this help");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  MVMC_NSTEPS     Same as --nsteps (CLI flag takes precedence)");
}

/// `mvmc uhf namelist.def [OptParaFile]`: port of the C `UHF` executable.
///
/// Like C, definition files and outputs are resolved against the current
/// directory and the optional second argument is accepted but unused. A
/// non-converged run still writes its files and exits with 255 (C `return -1`).
fn run_uhf(args: &[String]) -> i32 {
    if args.is_empty() || args.len() > 2 {
        eprintln!("ED Error: UHF NameListFile [OptParaFile]");
        return 1;
    }
    match mvmc_uhf::run(Path::new(&args[0]), &mvmc_uhf::UhfOptions::default()) {
        Ok(report) if report.converged => 0,
        Ok(_) => 255,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

/// `StdFace_main(input)` writing the Expert files into `gen_dir`; prints what C prints to
/// `stdout` and returns the process exit status (0, or C's `exit(-1)` status on failure).
fn run_stdface(input: &Path, gen_dir: &Path) -> i32 {
    if let Err(error) = std::fs::create_dir_all(gen_dir) {
        eprintln!("error: cannot create {}: {error}", gen_dir.display());
        return 1;
    }
    match mvmc_stdface::stdface_main(input, gen_dir) {
        Ok(report) => {
            print!("{}", report.log);
            0
        }
        Err(failure) => {
            print!("{}", failure.log);
            match failure.error {
                mvmc_stdface::StdFaceError::Exit(code) => {
                    eprintln!("error: StdFace failed (exit status {code})");
                    code
                }
                mvmc_stdface::StdFaceError::Io(message) => {
                    eprintln!("error: {message}");
                    1
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let program = args.first().map(String::as_str).unwrap_or("mvmc");

    if args.get(1).map(String::as_str) == Some("uhf") {
        process::exit(run_uhf(&args[2..]));
    }

    // ── argument parsing (no external crate dependency) ──────────────────────
    let mut namelist: Option<PathBuf> = None;
    let mut nsteps_arg: Option<i64> = None;
    let mut out_dir_arg: Option<PathBuf> = None;
    let mut seed_arg: Option<i64> = None;
    let mut nsmp_arg: Option<i64> = None;
    let mut mode_arg: Option<String> = None;
    let mut initial_def = mvmc_core::InitialDef::Auto;
    let mut opt_trans_arg = false;
    let mut flush_interval_arg: Option<i64> = None;
    let mut physcal_params: Option<PathBuf> = None;
    let mut physcal_trace_dir: Option<PathBuf> = None;
    let mut standard_mode = false;
    let mut dry_run = false;

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
            "-F" | "--flush-interval" => {
                idx += 1;
                flush_interval_arg = Some(
                    args.get(idx)
                        .and_then(|s| s.parse().ok())
                        .filter(|&n| n >= 1)
                        .unwrap_or_else(|| {
                            eprintln!("error: -F: FileFlushInterval should be natural number.");
                            process::exit(2)
                        }),
                );
            }
            "-o" | "--opt-trans" => {
                opt_trans_arg = true;
            }
            "-s" | "--standard" => {
                standard_mode = true;
            }
            "-e" | "--expert" => {
                standard_mode = false;
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--physcal" => {
                idx += 1;
                physcal_params = Some(args.get(idx).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("error: --physcal requires a fixed parameter file");
                    process::exit(2)
                }));
            }
            "--physcal-trace" => {
                idx += 1;
                physcal_trace_dir = Some(args.get(idx).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("error: --physcal-trace requires a NEW directory");
                    process::exit(2);
                }));
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

    // vmcdry.out: generate the Expert files from the Standard-mode input and stop.
    if dry_run {
        let gen_dir = out_dir_arg.clone().unwrap_or_else(|| PathBuf::from("."));
        process::exit(run_stdface(&namelist, &gen_dir));
    }

    if physcal_trace_dir.is_some()
        && (physcal_params.is_none()
            || mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok())
                .is_some_and(|context| context.world_size > 1))
    {
        eprintln!("error: --physcal-trace requires serial --physcal execution");
        process::exit(2);
    }

    #[cfg(feature = "mpi")]
    let mpi_context = {
        let launched = mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok());
        if launched.is_some_and(|context| context.world_size > 1) {
            Some(
                mvmc_core::mpi::MpiContext::initialize().unwrap_or_else(|error| {
                    eprintln!("error: {error}");
                    process::exit(1);
                }),
            )
        } else {
            None
        }
    };

    // Standard mode (vmcmain.c -s): rank 0 runs StdFace_main, then every rank reads namelist.def.
    let mut namelist = namelist;
    if standard_mode {
        let gen_dir = out_dir_arg.clone().unwrap_or_else(|| PathBuf::from("."));
        #[cfg(feature = "mpi")]
        let status = match &mpi_context {
            Some(context) => {
                let mut status = [if context.is_root() {
                    i64::from(run_stdface(&namelist, &gen_dir))
                } else {
                    0
                }];
                context
                    .broadcast_i64(0, &mut status)
                    .unwrap_or_else(|error| {
                        eprintln!("error: {error}");
                        process::exit(1);
                    });
                status[0] as i32
            }
            None => run_stdface(&namelist, &gen_dir),
        };
        #[cfg(not(feature = "mpi"))]
        let status = run_stdface(&namelist, &gen_dir);
        if status != 0 {
            process::exit(status);
        }
        namelist = gen_dir.join("namelist.def");
    }

    // MVMC_NSTEPS env var (CLI flag takes precedence)
    let nsteps_env: Option<i64> = std::env::var("MVMC_NSTEPS")
        .ok()
        .and_then(|s| s.parse().ok());
    let nsteps_override = nsteps_arg.or(nsteps_env);
    #[cfg(feature = "mpi")]
    if let Some(context) = &mpi_context {
        let controls = format!(
            "physcal={};nsteps={nsteps_override:?};mode={mode_arg:?};nsmp={nsmp_arg:?};seed={seed_arg:?};opt_trans={opt_trans_arg};initial={}",
            physcal_params.is_some(),
            match &initial_def {
                mvmc_core::InitialDef::Auto => "auto",
                mvmc_core::InitialDef::None => "none",
                mvmc_core::InitialDef::Path(_) => "path",
            }
        );
        agree_controls(&controls, context).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            process::exit(1);
        });
    }

    // Default output dir: namelist's parent directory.
    let out_dir: PathBuf =
        out_dir_arg.unwrap_or_else(|| namelist.parent().unwrap_or(Path::new(".")).join("output"));

    // ── peek at modpara to determine defaults and show model info ─────────────
    let peek = (|| match mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
        &namelist,
        opt_trans_arg,
    ) {
        Ok(data) => {
            // C vmcmain.c dispatches on NVMCCalMode: 0 optimizes, 1 runs fixed-parameter
            // PhysCal. Make the CLI selection explicit and reject any disagreement
            // between ModPara and the supplied options before initialization or IO.
            select_calculation(data.modpara.vmc_calc_mode, physcal_params.is_some())?;
            let validation = if physcal_params.is_some() {
                mvmc_core::validation::validate_phys_cal(&data)
            } else {
                mvmc_core::validation::validate_para_opt(&data)
            };
            validation?;
            Ok(data)
        }
        Err(e) => Err(format!("could not parse Expert input: {e}")),
    })();
    #[cfg(feature = "mpi")]
    let peek = match &mpi_context {
        Some(context) => agree_result(peek, context, "CLI parse/validation"),
        None => peek,
    };
    let data = peek.unwrap_or_else(|error| {
        eprintln!("error: {error}");
        process::exit(1);
    });
    let p = &data.modpara;
    let nsteps = nsteps_override.unwrap_or(p.nsr_opt_itr_step);
    let inferred_mode = if data.i_flg_orbital_general != 0 {
        "fsz"
    } else if mvmc_core::get_all_complex_flag(&data).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        process::exit(1);
    }) {
        "cmp"
    } else {
        "real"
    };
    #[cfg(feature = "mpi")]
    if let Some(context) = &mpi_context {
        // ModPara determines loop counts and collective buffer shapes. Agree it
        // before dispatch, not merely whether each local value is valid.
        let controls = format!(
            "modpara={p:?};mode={};general={};parameters={};projection={:?};qp_opt_trans={};green_lengths={:?}",
            mode_arg.as_deref().unwrap_or(inferred_mode),
            data.i_flg_orbital_general,
            data.count_variational_parameters(),
            data.projection_layout(),
            data.n_qp_opt_trans,
            (data.green_one_terms.len(), data.green_two_terms.len(), data.green_two_ex_terms.len()),
        );
        agree_controls(&controls, context).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            process::exit(1);
        });
    }
    #[cfg(feature = "mpi")]
    let output_root = mpi_context.as_ref().is_none_or(|context| context.is_root());
    #[cfg(not(feature = "mpi"))]
    let output_root = true;
    // readdef.c:749-752: rank 0 reports the negative-DSROptStepDt mode on stderr.
    if data.modpara.sr_flag && output_root {
        eprintln!("remark: Diagonalization Mode");
    }

    // ── banner ────────────────────────────────────────────────────────────────
    if output_root {
        println!(
            "model    : Nsite={} Nelec={} NSROptItrStep={}",
            p.nsite, p.nelec, nsteps
        );
        println!("mode     : NVMCCalMode={}", p.vmc_calc_mode);
        println!(
            "sample   : NVMCSample={} NVMCWarmUp={}",
            p.nvmc_sample, p.nvmc_warmup
        );
        println!();
        println!("=== mvmc — Julia-mVMC Rust port ===");
        println!("namelist : {}", namelist.display());
        println!("out-dir  : {}", out_dir.display());
        if let Some(path) = &physcal_params {
            println!("physcal  : {}", path.display());
        }
        if let Some(n) = nsteps_override {
            println!("nsteps   : {n} (override)");
        }
        if let Some(s) = seed_arg {
            println!("seed     : {s} (override)");
        }
        println!();
    }

    let step_validation = if physcal_params.is_none() && nsteps <= 0 {
        Err("NSROptItrStep is 0 — nothing to run. Use --nsteps <N>.".to_owned())
    } else {
        Ok(())
    };
    #[cfg(feature = "mpi")]
    let step_validation = match &mpi_context {
        Some(context) => agree_result(step_validation, context, "CLI step validation"),
        None => step_validation,
    };
    if let Err(error) = step_validation {
        eprintln!("error: {error}");
        process::exit(1);
    }

    // ── run ───────────────────────────────────────────────────────────────────
    let t0 = Instant::now();

    if let Some(fixed_params) = physcal_params {
        let trace = physcal_trace_dir
            .as_ref()
            .map(|directory| {
                physcal_trace::Trace::create(
                    directory,
                    seed_arg,
                    mode_arg.as_deref().unwrap_or(inferred_mode),
                    opt_trans_arg,
                )
                .map(std::rc::Rc::new)
            })
            .transpose()
            .unwrap_or_else(|error| {
                eprintln!("error: {error}");
                process::exit(1);
            });
        let _trace_guard = trace.as_ref().map(|trace| {
            mvmc_core::run::install_physcal_green_observer(trace.clone()).unwrap_or_else(|error| {
                eprintln!("error: {error}");
                process::exit(1);
            })
        });
        match run_physcal_with_selected_backend(
            &namelist,
            &fixed_params,
            seed_arg,
            &out_dir,
            mode_arg.as_deref().unwrap_or(inferred_mode),
            opt_trans_arg,
            #[cfg(feature = "mpi")]
            mpi_context.as_ref(),
        ) {
            Ok(result) => {
                if let Some(trace) = &trace {
                    trace.finish(result.iterations).unwrap_or_else(|error| {
                        eprintln!("error: PhysCal trace failed: {error}");
                        process::exit(1);
                    });
                }
                if output_root {
                    println!();
                    println!(
                        "=== Completed {} PhysCal samples in {:.2}s ===",
                        result.iterations,
                        t0.elapsed().as_secs_f64()
                    );
                    println!("Output files written to: {}", out_dir.display());
                }
            }
            Err(e) => {
                eprintln!("error: PhysCal failed: {e}");
                process::exit(1);
            }
        }
    } else {
        let config = mvmc_core::RunConfig {
            nsmp: nsmp_arg,
            seed: seed_arg,
            output_dir: Some(out_dir),
            initial_def,
            enable_opt_trans: Some(opt_trans_arg),
            file_flush_interval: flush_interval_arg,
            ..mvmc_core::RunConfig::new(nsteps, mode_arg.as_deref().unwrap_or(inferred_mode))
        };
        match run_with_selected_backend(
            &namelist,
            config,
            #[cfg(feature = "mpi")]
            mpi_context.as_ref(),
        ) {
            Ok(summary) => {
                if output_root {
                    println!();
                    println!(
                        "=== Completed {} SR steps in {:.2}s ===",
                        summary.effective_nsteps,
                        t0.elapsed().as_secs_f64()
                    );
                    println!("Output files written to: {}", summary.output_dir.display());
                    println!("Final energy / site: {:.10}", summary.final_energy_per_site);
                    println!(
                        "Final-window means ({} steps): {:?}",
                        summary.effective_nsmp, summary.ctest_values
                    );
                }
            }
            Err(e) => {
                eprintln!("error: run failed: {e}");
                process::exit(1);
            }
        }
    }
}

/// Validate that NVMCCalMode and the CLI options select the same calculation.
fn select_calculation(vmc_calc_mode: i64, has_physcal_file: bool) -> Result<(), String> {
    match (vmc_calc_mode, has_physcal_file) {
        (0, false) | (1, true) => Ok(()),
        (0, true) => Err(
            "--physcal requires NVMCCalMode=1 in ModPara (found NVMCCalMode=0); \
             set NVMCCalMode=1 for fixed-parameter PhysCal"
                .into(),
        ),
        (1, false) => Err(
            "NVMCCalMode=1 selects fixed-parameter PhysCal; supply the fixed parameter file \
             with --physcal <PATH>"
                .into(),
        ),
        (mode, _) => Err(format!(
            "unsupported NVMCCalMode={mode}; the CLI supports 0 (optimization) and 1 (PhysCal)"
        )),
    }
}

fn run_physcal_with_selected_backend(
    namelist: &Path,
    fixed_params: &Path,
    seed: Option<i64>,
    output_dir: &Path,
    mode: &str,
    enable_opt_trans: bool,
    #[cfg(feature = "mpi")] mpi_context: Option<&mvmc_core::mpi::MpiContext>,
) -> Result<mvmc_core::PhysCalResult, String> {
    let flags = mvmc_core::c_timer::TimerEnv::from_env();
    if flags.legacy_warning() {
        eprintln!(
            "warning: MVMC_TIMER is deprecated; use MVMC_C_TIMER=1 for the C-compatible zvo_CalcTimer.dat timer."
        );
    }
    if flags.enabled() {
        run_physcal_with_selected_backend_timed::<true>(
            namelist,
            fixed_params,
            seed,
            output_dir,
            mode,
            enable_opt_trans,
            #[cfg(feature = "mpi")]
            mpi_context,
            &mut mvmc_core::c_timer::CTimer::<true>::new(),
            flags,
        )
    } else {
        run_physcal_with_selected_backend_timed::<false>(
            namelist,
            fixed_params,
            seed,
            output_dir,
            mode,
            enable_opt_trans,
            #[cfg(feature = "mpi")]
            mpi_context,
            &mut mvmc_core::c_timer::CTimer::<false>::new(),
            flags,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_physcal_with_selected_backend_timed<const TIMED: bool>(
    namelist: &Path,
    fixed_params: &Path,
    seed: Option<i64>,
    output_dir: &Path,
    mode: &str,
    enable_opt_trans: bool,
    #[cfg(feature = "mpi")] mpi_context: Option<&mvmc_core::mpi::MpiContext>,
    timer: &mut mvmc_core::c_timer::CTimer<TIMED>,
    flags: mvmc_core::c_timer::TimerEnv,
) -> Result<mvmc_core::PhysCalResult, String> {
    // [0] All / [1] Initialization / [11] ReadDefFile bracket the CLI-side
    // parse. The in-place core records [20]/[3]/[4]/[21]/[22]; [2] VMCPhysCal
    // wraps the measurement call below. Mirrors run_para_opt_from_namelist.
    timer.reset();
    timer.diagnostics = flags;
    timer.start(0);
    timer.start(1);
    timer.start(11);
    let parsed = (|| {
        let parsed = mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
            namelist,
            enable_opt_trans,
        )
        .map_err(|error| error.to_string())?;
        mvmc_core::validation::validate_phys_cal(&parsed)?;
        if !fixed_params.is_file() {
            return Err(format!(
                "fixed parameter file not found: {}",
                fixed_params.display()
            ));
        }
        Ok(parsed)
    })();
    timer.stop(11);
    #[cfg(feature = "mpi")]
    let parsed = match mpi_context {
        Some(context) => agree_result(parsed, context, "CLI PhysCal parse/load/validation"),
        None => parsed,
    };
    let parsed = parsed?;

    let launched = mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok());
    if launched.is_some_and(|context| context.world_size > 1) {
        #[cfg(feature = "mpi")]
        {
            let context = mpi_context.ok_or("MPI context missing")?;
            agree_split_size(parsed.modpara.nsplit_size, context)?;
            let group;
            let reducer: &dyn mvmc_core::Reducer = if parsed.modpara.nsplit_size > 1 {
                group = context.split_groups(parsed.modpara.nsplit_size as usize)?;
                &group
            } else {
                context
            };
            // [13] InitParameter: fixed load + overlays + sync.
            timer.start(13);
            let preparation = mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
                namelist,
                fixed_params,
                mode,
                seed,
                reducer,
                enable_opt_trans,
            )?;
            timer.stop(13);
            timer.stop(1);
            timer.start(2);
            let result = mvmc_core::vmc_phys_cal_with_reducer_timed::<TIMED, _>(
                preparation,
                Some(output_dir),
                reducer,
                timer,
            )?;
            timer.stop(2);
            timer.stop(0);
            write_physcal_timer::<TIMED, _>(timer, output_dir, reducer, flags)?;
            return Ok(result);
        }
        #[cfg(not(feature = "mpi"))]
        {
            return Err(
                "MPI launcher detected; rebuild mvmc-cli with --features mpi to enable MPI execution"
                    .into(),
            );
        }
    }

    mvmc_core::validation::validate_reducer_rank(&parsed, &mvmc_core::SingleProcessReducer)?;
    // [13] InitParameter: fixed load + overlays + sync.
    timer.start(13);
    let preparation = mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
        namelist,
        fixed_params,
        mode,
        seed,
        &mvmc_core::SingleProcessReducer,
        enable_opt_trans,
    )?;
    timer.stop(13);
    timer.stop(1);
    timer.start(2);
    let result = mvmc_core::vmc_phys_cal_with_reducer_timed::<TIMED, _>(
        preparation,
        Some(output_dir),
        &mvmc_core::SingleProcessReducer,
        timer,
    )?;
    timer.stop(2);
    timer.stop(0);
    write_physcal_timer::<TIMED, _>(timer, output_dir, &mvmc_core::SingleProcessReducer, flags)?;
    Ok(result)
}

/// Write the C-compatible PhysCal timer report(s) on the output root.
fn write_physcal_timer<const TIMED: bool, R: mvmc_core::Reducer + ?Sized>(
    timer: &mvmc_core::c_timer::CTimer<TIMED>,
    output_dir: &Path,
    reducer: &R,
    flags: mvmc_core::c_timer::TimerEnv,
) -> Result<(), String> {
    if TIMED && reducer.is_output_root() {
        timer
            .write_phys_cal(output_dir, "zvo")
            .map_err(|error| error.to_string())?;
        if flags.any_diag() {
            timer
                .write_diag(output_dir, "zvo")
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn run_with_selected_backend(
    namelist: &Path,
    config: mvmc_core::RunConfig,
    #[cfg(feature = "mpi")] mpi_context: Option<&mvmc_core::mpi::MpiContext>,
) -> Result<mvmc_core::RunSummary, String> {
    let launched = mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok());
    if launched.is_some_and(|context| context.world_size > 1) {
        #[cfg(feature = "mpi")]
        {
            let context = mpi_context.ok_or("MPI context missing")?;
            let parsed = agree_result(
                mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(
                    namelist,
                    config.enable_opt_trans.unwrap_or(false),
                )
                .map_err(|error| error.to_string()),
                context,
                "CLI optimization parse",
            )?;
            agree_split_size(parsed.modpara.nsplit_size, context)?;
            if parsed.modpara.nsplit_size > 1 {
                let group = context.split_groups(parsed.modpara.nsplit_size as usize)?;
                return mvmc_core::run_para_opt_from_namelist_with_reducer(
                    namelist, config, &group,
                );
            }
            return mvmc_core::run_para_opt_from_namelist_with_reducer(namelist, config, context);
        }
        #[cfg(not(feature = "mpi"))]
        {
            return Err(
                "MPI launcher detected; rebuild mvmc-cli with --features mpi to enable MPI execution"
                    .into(),
            );
        }
    }
    mvmc_core::run_para_opt_from_namelist(namelist, config)
}

#[cfg(feature = "mpi")]
fn agree_result<T>(
    result: Result<T, String>,
    reducer: &dyn mvmc_core::Reducer,
    operation: &str,
) -> Result<T, String> {
    if reducer.any_failure(result.is_err()) {
        Err(result
            .err()
            .unwrap_or_else(|| format!("{operation} failed on another MPI rank")))
    } else {
        result
    }
}

#[cfg(feature = "mpi")]
fn agree_split_size(value: i64, context: &mvmc_core::mpi::MpiContext) -> Result<(), String> {
    let reducer: &dyn mvmc_core::Reducer = context;
    let mut root_value = [value];
    reducer.broadcast_i64(0, &mut root_value)?;
    agree_result(
        if value == root_value[0] {
            Ok(())
        } else {
            Err("NSplitSize differs between MPI ranks".into())
        },
        reducer,
        "CLI communicator selection",
    )
}

#[cfg(feature = "mpi")]
fn agree_controls(value: &str, context: &mvmc_core::mpi::MpiContext) -> Result<(), String> {
    let reducer: &dyn mvmc_core::Reducer = context;
    let mut length = [i64::try_from(value.len()).map_err(|error| error.to_string())?];
    reducer.broadcast_i64(0, &mut length)?;
    let mut root_value = if context.is_root() {
        value.bytes().map(i64::from).collect::<Vec<_>>()
    } else {
        vec![0; usize::try_from(length[0]).map_err(|error| error.to_string())?]
    };
    reducer.broadcast_i64(0, &mut root_value)?;
    agree_result(
        if root_value.iter().copied().eq(value.bytes().map(i64::from)) {
            Ok(())
        } else {
            Err("CLI run controls differ between MPI ranks".into())
        },
        reducer,
        "CLI control agreement",
    )
}
