//! `mvmc` — Julia-mVMC → Rust port command-line driver.
//!
//! Phase 5: wires `mvmc_core::run_para_opt_from_namelist` to a CLI so
//! the binary can reproduce the Julia `examples/*.jl` workflow from the
//! terminal.
//!
//! Usage (C `vmc.out \[option\] NameListFile \[OptParaFile\]`):
//!   mvmc \[options\] <namelist.def> \[initpara\]
//!   mvmc -s \[options\] <stan.in>             (Standard mode: StdFace, then run)
//!   mvmc --dry-run <stan.in>                (vmcdry.out: generate the Expert files)
//!   mvmc uhf <namelist.def> \[OptParaFile\]   (C ComplexUHF initial-orbital tool)
//!
//! C options (`vmcmain.c:83-165`, getopt `"bhm:oF:esv"`):
//!   -b                Binary parameter output (`_varbin_` files instead of `_var_`)
//!   -F <N>            Flush _time_/_SRinfo files every N steps (N >= 1)
//!   -o                Enable C optimized-translation mode
//!   -e                Expert mode (accepted; the only mode until StdFace, #353)
//!   -v                Print the version and exit
//!   -h                Print usage and exit
//!   -s                Standard mode: StdFace generates the Expert files from <stan.in>
//!   -m <N>            MultiDef mode: `mvmc -m N <dirlist> <namelist.def> [initpara]` splits the
//!                     MPI world into N groups (div/mod rule), each running <namelist.def> in the
//!                     directory named by its line of <dirlist> (#348). `-e` and `-s` after `-m`
//!                     cancel it, as in C.
//!
//! Positional `initpara`: the fixed parameter file for NVMCCalMode=1 (optional, as in
//! C) and the initial parameter file for NVMCCalMode=0.
//!
//! Rust extensions:
//!   --nsteps <N>      SR optimisation steps  [default: value in modpara.def]
//!   --out-dir <DIR>   Output directory       [default: <namelist parent dir>/output]
//!   --seed <N>        RNG seed override      [default: RndSeed in modpara.def]
//!   --nsmp <N>        Final averaging window [default: NSROptItrSmp]
//!   --mode <MODE>     Sanity check real/cmp/fsz; a mismatch with the input is an error
//!   --initial-def <auto|none|PATH> Starting parameter file [default: auto]
//!   --physcal <PATH>  Alias of the positional fixed parameter file (NVMCCalMode=1)
//!   --flush-interval, --opt-trans, --help, --version: long forms of -F, -o, -h, -v
//!
//! Environment:
//!   MVMC_NSTEPS       Same as --nsteps (CLI flag takes precedence)
//!
//! Mirror of `extern/Julia-mVMC/examples/heisenberg_chain_real.jl` et al.

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

mod physcal_trace;

const USAGE_LINE: &str = "Usage: {program} [option] NameListFile [OptParaFile]";

fn print_usage(program: &str) {
    eprintln!("{}", USAGE_LINE.replace("{program}", program));
    eprintln!("  -b     binary mode (write _varbin_ files instead of _var_ text files)");
    eprintln!("  -m N   multiDef mode: -m N DirListFile NameListFile [OptParaFile]");
    eprintln!("  -o     optTrans mode");
    eprintln!("  -F N   set interval of file flush");
    eprintln!(
        "  -s     Standard mode: generate the Expert files from <stan.in> (StdFace), then run"
    );
    eprintln!("  -e     Expert mode");
    eprintln!("  -v     print version");
    eprintln!("  -h     show this message");
    eprintln!();
    eprintln!("OptParaFile: fixed parameters for NVMCCalMode=1 (optional), initial parameters");
    eprintln!("             for NVMCCalMode=0.");
    eprintln!();
    eprintln!("Rust extensions:");
    eprintln!("  uhf <namelist.def> [OptParaFile]  C ComplexUHF initial-orbital tool");
    eprintln!(
        "  --dry-run <stan.in>  Generate the Expert files from <stan.in> and stop (vmcdry.out)"
    );
    eprintln!("  --standard, --expert  Long forms of -s, -e");
    eprintln!("  --nsteps <N>    SR optimisation steps [default: NSROptItrStep in modpara.def]");
    eprintln!("  --out-dir <DIR> Output directory      [default: <namelist parent dir>/output]");
    eprintln!("  --seed <N>      RNG seed override     [default: RndSeed in modpara.def]");
    eprintln!("  --nsmp <N>      Final averaging window [default: NSROptItrSmp]");
    eprintln!("  --mode <MODE>   Check real, cmp or fsz against the input; mismatch is an error");
    eprintln!("  --initial-def <auto|none|PATH> Starting parameter file [default: auto]");
    eprintln!("  --physcal <PATH> Alias of the positional fixed parameter file (NVMCCalMode=1)");
    eprintln!("  --physcal-trace <NEW_DIR> Nonconsuming serial PhysCal diagnostics");
    eprintln!("  --flush-interval, --opt-trans, --help, --version  Long forms of -F, -o, -h, -v");
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
            eprint!("{}", report.stderr);
            0
        }
        Err(failure) => {
            print!("{}", failure.log);
            eprint!("{}", failure.stderr);
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

/// C `printVersion` analogue. The Rust port reports its own crate version and the
/// authoritative C release it follows.
fn print_version() {
    println!(
        "mvmc-rs version {} (follows C mVMC 1.3.0)",
        env!("CARGO_PKG_VERSION")
    );
}

/// C `strtol` handling shared by `-F` (`vmcmain.c:124-150`): no digits and
/// out-of-`int` values are errors, trailing characters only warn.
fn parse_c_int(option: char, text: &str) -> Result<i64, String> {
    let trimmed = text.trim_start();
    let bytes = trimmed.as_bytes();
    let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let digits_start = end;
    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    if end == digits_start {
        return Err(format!("-{option}: No digits were found"));
    }
    let value: i64 = trimmed[..end]
        .parse()
        .map_err(|_| format!("-{option}: Numerical result out of range"))?;
    if i32::try_from(value).is_err() {
        return Err(format!("-{option}: Numerical result out of range"));
    }
    if end != trimmed.len() {
        eprintln!(
            "warning: -{option}: Futher characters after number: {}",
            &trimmed[end..]
        );
    }
    Ok(value)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let program = args.first().map(String::as_str).unwrap_or("mvmc");

    if args.get(1).map(String::as_str) == Some("uhf") {
        process::exit(run_uhf(&args[2..]));
    }

    // ── argument parsing (no external crate dependency) ──────────────────────
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut nsteps_arg: Option<i64> = None;
    let mut out_dir_arg: Option<PathBuf> = None;
    let mut seed_arg: Option<i64> = None;
    let mut nsmp_arg: Option<i64> = None;
    let mut mode_arg: Option<String> = None;
    let mut initial_def = mvmc_core::InitialDef::Auto;
    let mut initial_def_explicit = false;
    let mut opt_trans_arg = false;
    let mut binary_arg = false;
    let mut flush_interval_arg: Option<i64> = None;
    let mut physcal_flag: Option<PathBuf> = None;
    let mut physcal_trace_dir: Option<PathBuf> = None;
    let mut standard_mode = false;
    let mut dry_run = false;
    // C `flagMultiDef`/`nMultiDef`: `-e` and `-s` clear the flag (vmcmain.c:151-157).
    let mut multi_def: Option<i64> = None;

    let usage_error = |message: &str| -> ! {
        eprintln!("error: {message}");
        process::exit(2)
    };
    let mut idx = 1;
    let mut options_done = false;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if options_done || arg == "-" || !arg.starts_with('-') {
            positional.push(PathBuf::from(arg));
            idx += 1;
            continue;
        }
        if arg == "--" {
            options_done = true;
            idx += 1;
            continue;
        }
        if !arg.starts_with("--") {
            // C getopt: clustered short options (`-bo`), arguments attached (`-F2`)
            // or in the next word (`-F 2`).
            let cluster: Vec<char> = arg[1..].chars().collect();
            let mut pos = 0;
            while pos < cluster.len() {
                let option = cluster[pos];
                pos += 1;
                match option {
                    'b' => binary_arg = true,
                    'e' => {
                        standard_mode = false;
                        multi_def = None;
                    }
                    'o' => opt_trans_arg = true,
                    'h' => {
                        print_usage(program);
                        process::exit(0);
                    }
                    'v' => {
                        print_version();
                        process::exit(0);
                    }
                    's' => {
                        standard_mode = true;
                        multi_def = None;
                    }
                    'm' | 'F' => {
                        let value: String = if pos < cluster.len() {
                            let rest: String = cluster[pos..].iter().collect();
                            pos = cluster.len();
                            rest
                        } else {
                            idx += 1;
                            match args.get(idx) {
                                Some(value) => value.clone(),
                                None => {
                                    eprintln!("error: option requires an argument -- '{option}'");
                                    print_usage(program);
                                    process::exit(2)
                                }
                            }
                        };
                        if option == 'm' {
                            match parse_c_int('m', &value) {
                                Ok(n) => multi_def = Some(n),
                                Err(error) => usage_error(&error),
                            }
                            continue;
                        }
                        match parse_c_int('F', &value) {
                            Ok(n) if n >= 1 => flush_interval_arg = Some(n),
                            Ok(_) => usage_error("-F: FileFlushInterval should be natural number."),
                            Err(error) => usage_error(&error),
                        }
                    }
                    other => {
                        eprintln!("error: invalid option -- '{other}'");
                        print_usage(program);
                        process::exit(2)
                    }
                }
            }
            idx += 1;
            continue;
        }
        match arg {
            "--help" => {
                print_usage(program);
                process::exit(0);
            }
            "--version" => {
                print_version();
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
                let value = args.get(idx).cloned().unwrap_or_else(|| {
                    eprintln!("error: --mode requires real, cmp or fsz");
                    process::exit(2)
                });
                if !matches!(value.as_str(), "real" | "cmp" | "fsz") {
                    usage_error(&format!("--mode requires real, cmp or fsz; got `{value}`"));
                }
                mode_arg = Some(value);
            }
            "--initial-def" => {
                idx += 1;
                initial_def_explicit = true;
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
            "--flush-interval" => {
                idx += 1;
                match args.get(idx).map(|value| parse_c_int('F', value)) {
                    Some(Ok(n)) if n >= 1 => flush_interval_arg = Some(n),
                    Some(Ok(_)) => usage_error("-F: FileFlushInterval should be natural number."),
                    Some(Err(error)) => usage_error(&error),
                    None => usage_error("--flush-interval requires an integer"),
                }
            }
            "--opt-trans" => {
                opt_trans_arg = true;
            }
            "--standard" => {
                standard_mode = true;
                multi_def = None;
            }
            "--expert" => {
                standard_mode = false;
                multi_def = None;
            }
            "--dry-run" => dry_run = true,
            "--physcal" => {
                idx += 1;
                physcal_flag = Some(args.get(idx).map(PathBuf::from).unwrap_or_else(|| {
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
            flag => {
                eprintln!("error: unknown flag `{flag}`");
                print_usage(program);
                process::exit(2);
            }
        }
        idx += 1;
    }

    // MultiDef mode (vmcmain.c:172-190): the first positional is the directory-list file.
    let dir_list: Option<PathBuf> = if multi_def.is_some() {
        if positional.len() < 2 {
            eprintln!("error: Argument count mismatch");
            print_usage(program);
            process::exit(2);
        }
        Some(positional.remove(0))
    } else {
        None
    };
    if positional.len() > 2 {
        eprintln!("error: Argument count mismatch");
        print_usage(program);
        process::exit(2);
    }
    let positional_initpara = positional.get(1).cloned();
    if positional_initpara.is_some() && physcal_flag.is_some() {
        usage_error(
            "both a positional initpara and --physcal were given; use one \
             (--physcal is an alias of the positional fixed parameter file)",
        );
    }
    if positional_initpara.is_some() && initial_def_explicit {
        usage_error(
            "both a positional initpara and --initial-def were given; the positional \
             file is the C initial parameter file",
        );
    }
    // Positional initpara and --physcal name the same file; its role (fixed parameters or
    // initial parameters) is decided by NVMCCalMode after ModPara is read.
    let parameter_file: Option<PathBuf> = positional_initpara.or(physcal_flag.clone());

    let namelist = match positional.first() {
        Some(p) => p.clone(),
        None => {
            eprintln!("error: Argument count mismatch: <namelist.def> is required");
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
        && mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok())
            .is_some_and(|context| context.world_size > 1)
    {
        eprintln!("error: --physcal-trace requires serial --physcal execution");
        process::exit(2);
    }

    #[cfg(feature = "mpi")]
    let mpi_world = {
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

    // MultiDef mode (vmcmain.c:193-198): split the world, enter this group's directory.
    // The group communicator then plays the role of MPI_COMM_WORLD for the whole run.
    #[cfg(feature = "mpi")]
    let mpi_group = match (multi_def, &dir_list) {
        (Some(n), Some(list)) => init_multi_def(n, list, mpi_world.as_ref()),
        _ => None,
    };
    #[cfg(feature = "mpi")]
    let mpi_context: Option<&mvmc_core::mpi::MpiContext> =
        mpi_group.as_ref().or(mpi_world.as_ref());
    #[cfg(not(feature = "mpi"))]
    if let (Some(n), Some(list)) = (multi_def, &dir_list) {
        init_multi_def(n, list);
    }

    // Standard mode (vmcmain.c -s): rank 0 runs StdFace_main, then every rank reads namelist.def.
    let mut namelist = namelist;
    if standard_mode {
        let gen_dir = out_dir_arg.clone().unwrap_or_else(|| PathBuf::from("."));
        #[cfg(feature = "mpi")]
        let status = match mpi_context {
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
        // StdFace may write `initial.def` (a UHF initial guess for Wannier90 double counting),
        // which is not an mVMC initial-parameter file; C reads one only when it is given
        // explicitly as the second argument.
        if matches!(initial_def, mvmc_core::InitialDef::Auto) {
            initial_def = mvmc_core::InitialDef::None;
        }
    }

    // MVMC_NSTEPS env var (CLI flag takes precedence)
    let nsteps_env: Option<i64> = std::env::var("MVMC_NSTEPS")
        .ok()
        .and_then(|s| s.parse().ok());
    let nsteps_override = nsteps_arg.or(nsteps_env);
    #[cfg(feature = "mpi")]
    if let Some(context) = mpi_context {
        let controls = format!(
            "physcal={};nsteps={nsteps_override:?};mode={mode_arg:?};nsmp={nsmp_arg:?};seed={seed_arg:?};opt_trans={opt_trans_arg};initial={}",
            parameter_file.is_some(),
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
            select_calculation(data.modpara.vmc_calc_mode, physcal_flag.is_some())?;
            let validation = if data.modpara.vmc_calc_mode == 1 {
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
    let peek = match mpi_context {
        Some(context) => agree_result(peek, context, "CLI parse/validation"),
        None => peek,
    };
    let data = peek.unwrap_or_else(|error| {
        eprintln!("error: {error}");
        process::exit(1);
    });
    let p = &data.modpara;
    let is_physcal = p.vmc_calc_mode == 1;
    // The trace observes the fixed-file lifecycle stages, so it needs the explicit file.
    if physcal_trace_dir.is_some() && !(is_physcal && parameter_file.is_some()) {
        eprintln!("error: --physcal-trace requires serial --physcal execution");
        process::exit(2);
    }
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
    if let Some(requested) = mode_arg.as_deref() {
        if requested != inferred_mode {
            eprintln!(
                "error: --mode {requested} contradicts the input files, which declare a \
                 {inferred_mode} calculation; fix --mode or the definition files"
            );
            process::exit(2);
        }
    }
    #[cfg(feature = "mpi")]
    if let Some(context) = mpi_context {
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
    let output_root = mpi_context.is_none_or(|context| context.is_root());
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
        match (&parameter_file, is_physcal) {
            (Some(path), true) => println!("physcal  : {}", path.display()),
            (None, true) => println!("physcal  : (no parameter file; C InitParameter draws)"),
            (Some(path), false) => println!("initpara : {}", path.display()),
            (None, false) => {}
        }
        if binary_arg {
            println!("binary   : varbin output (-b)");
        }
        if let Some(n) = nsteps_override {
            println!("nsteps   : {n} (override)");
        }
        if let Some(s) = seed_arg {
            println!("seed     : {s} (override)");
        }
        println!();
    }

    let step_validation = if !is_physcal && nsteps <= 0 {
        Err("NSROptItrStep is 0 — nothing to run. Use --nsteps <N>.".to_owned())
    } else {
        Ok(())
    };
    #[cfg(feature = "mpi")]
    let step_validation = match mpi_context {
        Some(context) => agree_result(step_validation, context, "CLI step validation"),
        None => step_validation,
    };
    if let Err(error) = step_validation {
        eprintln!("error: {error}");
        process::exit(1);
    }

    // ── run ───────────────────────────────────────────────────────────────────
    let t0 = Instant::now();

    if is_physcal {
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
            parameter_file.as_deref(),
            seed_arg,
            &out_dir,
            mode_arg.as_deref().unwrap_or(inferred_mode),
            opt_trans_arg,
            binary_arg,
            #[cfg(feature = "mpi")]
            mpi_context,
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
        // C `ReadInitParameter(fileInitPara)` for a positional initpara (mode 0).
        if let Some(path) = &parameter_file {
            initial_def = mvmc_core::InitialDef::Path(path.clone());
        }
        let config = mvmc_core::RunConfig {
            binary_output: binary_arg,
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
            mpi_context,
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
        report_inner_profile();
    }
}

/// `MVMC_RS_INNER_PROFILE=1`: print per-call-site inner-kernel counts and time.
fn report_inner_profile() {
    let sites = mvmc_core::threading::dispatch_profile();
    if sites.is_empty() {
        return;
    }
    eprintln!("inner-kernel profile (site, mode, calls, items, total ms, us/call, ns/item):");
    for site in sites {
        let millis = site.nanos as f64 / 1e6;
        let per_item = if site.items > 0 {
            site.nanos as f64 / site.items as f64
        } else {
            0.0
        };
        eprintln!(
            "  {:<44} {:<8} {:>9} {:>11} {:>9.1} {:>8.2} {:>8.1}",
            site.site,
            if site.parallel { "pool" } else { "serial" },
            site.calls,
            site.items,
            millis,
            millis * 1e3 / site.calls as f64,
            per_item
        );
    }
}

/// C `initMultiDefMode` (`vmcmain.c:727-800`) for a serial launch: the world has one rank,
/// so only `-m 1` is valid. Exits with C's status on every error.
#[cfg(not(feature = "mpi"))]
fn init_multi_def(n: i64, dir_list: &Path) {
    if mvmc_core::parallel::LaunchContext::from_env(|key| std::env::var(key).ok())
        .is_some_and(|context| context.world_size > 1)
    {
        eprintln!(
            "error: MPI launcher detected; rebuild mvmc-cli with --features mpi to enable MPI execution"
        );
        process::exit(1);
    }
    if let Err(message) = mvmc_core::multidef::check_world(1, n) {
        eprintln!("{message}");
        process::exit(1);
    }
    let names = mvmc_core::multidef::read_dir_list(dir_list, 1).unwrap_or_else(|message| {
        eprintln!("{message}");
        process::exit(1)
    });
    if let Err(message) = mvmc_core::multidef::change_directory(&names[0]) {
        eprintln!("{message}");
        process::exit(1);
    }
}

/// C `initMultiDefMode` (`vmcmain.c:727-800`): split the world into `n` groups, read the
/// directory list on rank 0, give group `g` the `g`-th name and change into it. Returns
/// the group communicator (`comm0`), or `None` for a serial launch (a one-rank world,
/// where only `-m 1` is valid). Failures exit with status 1 on every rank, as C's
/// `exit(EXIT_FAILURE)` / `MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)`.
#[cfg(feature = "mpi")]
fn init_multi_def(
    n: i64,
    dir_list: &Path,
    world: Option<&mvmc_core::mpi::MpiContext>,
) -> Option<mvmc_core::mpi::MpiContext> {
    use mvmc_core::multidef;
    let (rank, size) = world.map_or((0, 1), |context| (context.rank(), context.world_size()));
    match multidef::check_world(size, n) {
        Ok(check) => {
            if let (0, Some(warning)) = (rank, check.warning) {
                eprintln!("{warning}");
            }
        }
        Err(message) => {
            if rank == 0 {
                eprintln!("{message}");
            }
            process::exit(1);
        }
    }
    let n = usize::try_from(n).expect("validated by check_world");
    let group = world.map(|context| {
        context.split_multi_def(n).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            process::exit(1)
        })
    });

    // Rank 0 reads the list; every rank learns the names (C scatters them to the group
    // leaders only, see `change_directory`).
    let mut names = Vec::new();
    let mut failed = false;
    if rank == 0 {
        match multidef::read_dir_list(dir_list, n) {
            Ok(list) => names = list,
            Err(message) => {
                eprintln!("{message}");
                failed = true;
            }
        }
    }
    if let Some(context) = world {
        let reducer: &dyn mvmc_core::Reducer = context;
        let mut status = [i64::from(failed)];
        let mut text = names.join("\n").into_bytes();
        let mut length = [i64::try_from(text.len()).expect("directory list length")];
        let broadcast = (|| -> Result<(), String> {
            reducer.broadcast_i64(0, &mut status)?;
            reducer.broadcast_i64(0, &mut length)?;
            let mut bytes: Vec<i64> = text.iter().map(|&byte| i64::from(byte)).collect();
            bytes.resize(
                usize::try_from(length[0]).map_err(|error| error.to_string())?,
                0,
            );
            reducer.broadcast_i64(0, &mut bytes)?;
            text = bytes.into_iter().map(|byte| byte as u8).collect();
            Ok(())
        })();
        if let Err(error) = broadcast {
            eprintln!("error: {error}");
            process::exit(1);
        }
        failed = status[0] != 0;
        if !failed {
            names = String::from_utf8_lossy(&text)
                .split('\n')
                .map(str::to_owned)
                .collect();
        }
    }
    if failed {
        process::exit(1);
    }

    let group_index = multidef::group_of_rank(rank, size, n);
    let changed = multidef::change_directory(&names[group_index]);
    if let Err(message) = &changed {
        if group.as_ref().is_none_or(|context| context.rank() == 0) {
            eprintln!("{message}");
        }
    }
    let any_failed = match world {
        Some(context) => mvmc_core::Reducer::any_failure(context, changed.is_err()),
        None => changed.is_err(),
    };
    if any_failed {
        process::exit(1);
    }
    group
}

/// Validate the NVMCCalMode dispatch (`vmcmain.c:304-319`). The positional `initpara` is
/// optional in both modes, so only the explicit `--physcal` alias can disagree.
fn select_calculation(vmc_calc_mode: i64, has_physcal_flag: bool) -> Result<(), String> {
    match (vmc_calc_mode, has_physcal_flag) {
        (0, false) | (1, _) => Ok(()),
        (0, true) => Err(
            "--physcal requires NVMCCalMode=1 in ModPara (found NVMCCalMode=0); \
             set NVMCCalMode=1 for fixed-parameter PhysCal, or pass the file positionally \
             as the initial parameter file"
                .into(),
        ),
        (mode, _) => Err(format!(
            "unsupported NVMCCalMode={mode}; the CLI supports 0 (optimization) and 1 (PhysCal)"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_physcal_with_selected_backend(
    namelist: &Path,
    fixed_params: Option<&Path>,
    seed: Option<i64>,
    output_dir: &Path,
    mode: &str,
    enable_opt_trans: bool,
    binary_output: bool,
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
            binary_output,
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
            binary_output,
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
    fixed_params: Option<&Path>,
    seed: Option<i64>,
    output_dir: &Path,
    mode: &str,
    enable_opt_trans: bool,
    binary_output: bool,
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
        if let Some(fixed_params) = fixed_params {
            if !fixed_params.is_file() {
                return Err(format!(
                    "fixed parameter file not found: {}",
                    fixed_params.display()
                ));
            }
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
            let mut preparation = prepare_physcal(
                namelist,
                fixed_params,
                mode,
                seed,
                reducer,
                enable_opt_trans,
            )?;
            preparation.binary_output = binary_output;
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
    let mut preparation = prepare_physcal(
        namelist,
        fixed_params,
        mode,
        seed,
        &mvmc_core::SingleProcessReducer,
        enable_opt_trans,
    )?;
    preparation.binary_output = binary_output;
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

/// C `InitParameter` + optional `ReadInitParameter(fileInitPara)` for NVMCCalMode=1:
/// a given file fixes the parameters, otherwise the initialization draws are used.
fn prepare_physcal<R: mvmc_core::Reducer + ?Sized>(
    namelist: &Path,
    fixed_params: Option<&Path>,
    mode: &str,
    seed: Option<i64>,
    reducer: &R,
    enable_opt_trans: bool,
) -> Result<mvmc_core::PhysCalPreparation, String> {
    match fixed_params {
        Some(path) => mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
            namelist,
            path,
            mode,
            seed,
            reducer,
            enable_opt_trans,
        ),
        None => mvmc_core::prepare_phys_cal_without_parameter_file_with_reducer_and_opt_trans(
            namelist,
            mode,
            seed,
            reducer,
            enable_opt_trans,
        ),
    }
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
