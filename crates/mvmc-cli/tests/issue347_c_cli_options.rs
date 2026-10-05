//! Issue #347: C `vmc.out` option set (`bhm:oF:esv`) and positional contract
//! `vmc.out [opts] namelist.def [initpara]` (`vmcmain.c:83-193`).
//!
//! Expected binary files come from the unmodified C `vmc.out` (see
//! `tests/fixtures/issue347_varbin/PROVENANCE.md`); no C runs here. All work happens in
//! per-test temporary directories that are removed on drop.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "issue347-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
}

/// Optimizer inputs (NVMCCalMode=0, 4 steps, NPara=14).
fn opt_inputs(work: &Work) -> PathBuf {
    let dir = work.path("opt");
    copy_dir(&fixtures().join("issue347_varbin/opt_inputs"), &dir);
    dir
}

/// PhysCal inputs (NVMCCalMode=1, NDataIdxStart=7, two samples) and the fixed file.
fn phys_inputs(work: &Work) -> (PathBuf, PathBuf) {
    let root = fixtures().join("physcal_181/two-samples/heisenberg_chain_real");
    let dir = work.path("phys");
    copy_dir(&root.join("inputs"), &dir);
    fs::copy(root.join("zqp_opt.dat"), dir.join("zqp_opt.dat")).unwrap();
    (dir.clone(), dir.join("zqp_opt.dat"))
}

fn mvmc(cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .args(args)
        .current_dir(cwd)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|iter| {
            iter.flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// Decoded `_varbin_` file: `(NPara, step count, blocks)`. `full` selects Rust's
/// `2*NPara`-double blocks; C's blocks hold `NPara` doubles (vmcmain.c:658).
fn read_varbin(path: &Path, full: bool) -> (i32, i32, Vec<Vec<f64>>) {
    let bytes = fs::read(path).unwrap();
    let n_para = i32::from_ne_bytes(bytes[0..4].try_into().unwrap());
    let steps = i32::from_ne_bytes(bytes[4..8].try_into().unwrap());
    let width = n_para as usize * if full { 2 } else { 1 };
    let doubles = bytes[8..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect::<Vec<_>>();
    assert_eq!(bytes[8..].len() % 8, 0, "trailing partial double");
    let blocks = doubles
        .chunks(width)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();
    (n_para, steps, blocks)
}

fn c_expected(name: &str) -> PathBuf {
    fixtures().join("issue347_varbin/c_expected").join(name)
}

/// First divergence from C is the SR solve (BLAS operation order): about 5e-10 absolute on
/// O(1) parameters. Step 0 must agree exactly; later steps use this explicit bound.
const LATER_STEP_ATOL: f64 = 1e-8;

fn assert_blocks_match_c(rust: &Path, c_name: &str, expect_n_para: i32) {
    let (n_para, steps, blocks) = read_varbin(rust, true);
    let (c_n_para, c_steps, c_blocks) = read_varbin(&c_expected(c_name), false);
    assert_eq!((n_para, steps), (c_n_para, c_steps), "header ints");
    assert_eq!(n_para, expect_n_para);
    // Header bytes exactly.
    assert_eq!(
        fs::read(rust).unwrap()[..8],
        fs::read(c_expected(c_name)).unwrap()[..8]
    );
    assert_eq!(blocks.len(), c_blocks.len());
    // C's block is the first NPara doubles of Rust's complete 2*NPara block.
    assert!(blocks
        .iter()
        .all(|block| block.len() == 2 * n_para as usize));
    assert_eq!(
        blocks[0][..n_para as usize],
        c_blocks[0][..],
        "step 0 is exact"
    );
    for (step, (r, c)) in blocks.iter().zip(&c_blocks).enumerate().skip(1) {
        for (slot, (a, b)) in r[..n_para as usize].iter().zip(c).enumerate() {
            assert!(
                (a - b).abs() <= LATER_STEP_ATOL,
                "step {step} slot {slot}: rust {a} vs C {b}"
            );
        }
    }
}

#[test]
fn version_flag_prints_crate_and_c_reference_version_without_files() {
    for flag in ["-v", "--version"] {
        let work = Work::new();
        let output = mvmc(&work.0, &[flag]);
        assert!(output.status.success(), "{}", stderr(&output));
        let text = stdout(&output);
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        assert!(text.contains("C mVMC 1.3.0"), "{text}");
        assert!(entries(&work.0).is_empty());
    }
}

#[test]
fn help_flag_lists_c_options_and_states_the_real_default_output_directory() {
    for flag in ["-h", "--help"] {
        let work = Work::new();
        let output = mvmc(&work.0, &[flag]);
        assert!(output.status.success());
        let text = stderr(&output);
        for needle in [
            "NameListFile [OptParaFile]",
            "-b     binary mode",
            "-m N   multiDef mode (not yet implemented, #348)",
            "-o     optTrans mode",
            "-F N   set interval of file flush",
            "-s     Standard mode",
            "-e     Expert mode",
            "-v     print version",
            "-h     show this message",
            "<namelist parent dir>/output",
        ] {
            assert!(text.contains(needle), "missing `{needle}` in:\n{text}");
        }
        assert!(!text.contains("[default: namelist parent dir]"));
        assert!(entries(&work.0).is_empty());
    }
}

#[test]
fn expert_flag_is_an_accepted_no_op() {
    let work = Work::new();
    let (dir, fixed) = phys_inputs(&work);
    let namelist = dir.join("namelist.def");
    let run = |extra: &[&str], out: &str| {
        let out = work.path(out);
        let mut args: Vec<&str> = extra.to_vec();
        let namelist = namelist.to_str().unwrap();
        let fixed = fixed.to_str().unwrap();
        let out_str = out.to_str().unwrap();
        args.extend([namelist, fixed, "--out-dir", out_str]);
        let output = mvmc(&work.0, &args);
        assert!(output.status.success(), "{}", stderr(&output));
        out
    };
    let plain = run(&[], "plain");
    let expert = run(&["-e"], "expert");
    assert_eq!(entries(&plain), entries(&expert));
    assert_eq!(
        fs::read(plain.join("zvo_out_007.dat")).unwrap(),
        fs::read(expert.join("zvo_out_007.dat")).unwrap()
    );
}

#[test]
fn multidef_flag_is_rejected_before_any_io() {
    let work = Work::new();
    let opt = opt_inputs(&work);
    let namelist = opt.join("namelist.def");
    let namelist = namelist.to_str().unwrap();
    let out = work.path("out");
    let out = out.to_str().unwrap();
    for (flags, needle) in [(vec!["-m", "2"], "#348"), (vec!["-m2"], "#348")] {
        let mut args = flags;
        args.extend([namelist, "--out-dir", out]);
        let output = mvmc(&work.0, &args);
        assert!(!output.status.success());
        assert!(stderr(&output).contains(needle), "{}", stderr(&output));
        assert!(!work.path("out").exists());
    }
    let output = mvmc(&work.0, &["-m", "x", namelist]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("-m: No digits were found"));
}

#[test]
fn flush_interval_follows_c_strtol_rules() {
    let work = Work::new();
    let opt = opt_inputs(&work);
    let namelist = opt.join("namelist.def");
    let namelist = namelist.to_str().unwrap();
    for (value, needle) in [
        ("0", "FileFlushInterval should be natural number"),
        ("-3", "FileFlushInterval should be natural number"),
        ("x", "-F: No digits were found"),
        ("99999999999", "-F: Numerical result out of range"),
    ] {
        let output = mvmc(&work.0, &["-F", value, namelist]);
        assert!(!output.status.success(), "-F {value}");
        assert!(stderr(&output).contains(needle), "{}", stderr(&output));
    }
    // Attached value and trailing characters (warning only), through the C getopt forms.
    for (args, out, warns) in [
        (vec!["-F2"], "attached", false),
        (vec!["-F", "2abc"], "trailing", true),
        (vec!["-bF", "2"], "cluster", false),
    ] {
        let out_dir = work.path(out);
        let mut full = args;
        full.extend([namelist, "--out-dir", out_dir.to_str().unwrap()]);
        let output = mvmc(&work.0, &full);
        assert!(output.status.success(), "{}", stderr(&output));
        assert_eq!(
            stderr(&output).contains("Futher characters after number"),
            warns
        );
        assert!(out_dir.join("zvo_time_001.dat").is_file());
    }
}

#[test]
fn binary_mode_optimizer_matches_c_varbin_layout() {
    for (namelist_name, c_name, n_para) in [
        ("namelist.def", "opt_even_zvo_varbin_001.dat", 14),
        ("namelist_odd.def", "opt_odd_zvo_varbin_001.dat", 13),
    ] {
        let work = Work::new();
        let opt = opt_inputs(&work);
        let out = work.path("out");
        let namelist = opt.join(namelist_name);
        let output = mvmc(
            &work.0,
            &[
                "-b",
                namelist.to_str().unwrap(),
                "--out-dir",
                out.to_str().unwrap(),
            ],
        );
        assert!(output.status.success(), "{}", stderr(&output));
        // C skips the `_var_` text file in binary mode and keeps every other file.
        let names = entries(&out);
        assert!(
            names.contains(&"zvo_varbin_001.dat".to_owned()),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|name| name.starts_with("zvo_var_")),
            "{names:?}"
        );
        assert!(!names.contains(&"zvo_var.dat".to_owned()));
        assert!(names.contains(&"zvo_out.dat".to_owned()));
        assert!(names.contains(&"zvo_SRinfo.dat".to_owned()));
        assert!(names.contains(&"zvo_time_001.dat".to_owned()));
        assert_blocks_match_c(&out.join("zvo_varbin_001.dat"), c_name, n_para);
        // Size: 8-byte header and one 2*NPara-double block per step (C stores NPara doubles).
        let size = fs::metadata(out.join("zvo_varbin_001.dat")).unwrap().len();
        assert_eq!(size, 8 + 4 * 16 * n_para as u64);
    }
}

#[test]
fn binary_mode_blocks_contain_every_text_parameter() {
    let work = Work::new();
    let opt = opt_inputs(&work);
    let namelist = opt.join("namelist.def");
    let namelist = namelist.to_str().unwrap();
    let text_out = work.path("text");
    let bin_out = work.path("bin");
    let text = mvmc(
        &work.0,
        &[namelist, "--out-dir", text_out.to_str().unwrap()],
    );
    let bin = mvmc(
        &work.0,
        &["-b", namelist, "--out-dir", bin_out.to_str().unwrap()],
    );
    assert!(text.status.success() && bin.status.success());
    assert!(!entries(&text_out).iter().any(|n| n.contains("varbin")));
    let (n_para, _, blocks) = read_varbin(&bin_out.join("zvo_varbin_001.dat"), true);
    let var = fs::read_to_string(text_out.join("zvo_var.dat")).unwrap();
    for (step, line) in var.lines().enumerate() {
        // Six energy diagnostics, then (re, im, 0.0) per parameter.
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|token| token.parse().unwrap())
            .collect();
        let params = &values[6..];
        let interleaved: Vec<f64> = params
            .chunks(3)
            .flat_map(|triple| [triple[0], triple[1]])
            .collect();
        assert_eq!(
            interleaved.len(),
            2 * n_para as usize,
            "every parameter present"
        );
        assert_eq!(blocks[step], interleaved, "step {step}");
    }
}

#[test]
fn binary_mode_physcal_matches_c_varbin_files() {
    let work = Work::new();
    let (dir, fixed) = phys_inputs(&work);
    let out = work.path("out");
    let output = mvmc(
        &work.0,
        &[
            "-b",
            dir.join("namelist.def").to_str().unwrap(),
            fixed.to_str().unwrap(),
            "--out-dir",
            out.to_str().unwrap(),
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let names = entries(&out);
    assert!(
        !names.iter().any(|name| name.starts_with("zvo_var_")),
        "{names:?}"
    );
    assert!(names.contains(&"zvo_out_007.dat".to_owned()));
    for index in ["007", "008"] {
        let rust = out.join(format!("zvo_varbin_{index}.dat"));
        let c_name = format!("physcal_zvo_varbin_{index}.dat");
        let (n_para, steps, rust_blocks) = read_varbin(&rust, true);
        let (_, _, c_blocks) = read_varbin(&c_expected(&c_name), false);
        assert_eq!((n_para, steps), (14, 1));
        assert_eq!(
            fs::read(&rust).unwrap()[..8],
            fs::read(c_expected(&c_name)).unwrap()[..8]
        );
        // Fixed parameters come from the file, not from a computation: C's NPara doubles
        // equal the leading NPara doubles of Rust's complete block, exactly.
        assert_eq!(rust_blocks[0].len(), 2 * n_para as usize);
        assert_eq!(rust_blocks[0][..n_para as usize], c_blocks[0][..]);
    }
}

#[test]
fn positional_initpara_equals_physcal_alias_for_nvmccalmode_one() {
    let work = Work::new();
    let (dir, fixed) = phys_inputs(&work);
    let namelist = dir.join("namelist.def");
    let namelist = namelist.to_str().unwrap();
    let fixed = fixed.to_str().unwrap();
    let positional = work.path("positional");
    let alias = work.path("alias");
    let first = mvmc(
        &work.0,
        &[namelist, fixed, "--out-dir", positional.to_str().unwrap()],
    );
    let second = mvmc(
        &work.0,
        &[
            namelist,
            "--physcal",
            fixed,
            "--out-dir",
            alias.to_str().unwrap(),
        ],
    );
    assert!(first.status.success(), "{}", stderr(&first));
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(entries(&positional), entries(&alias));
    for name in ["zvo_out_007.dat", "zvo_var_007.dat", "zvo_cisajs_007.dat"] {
        assert_eq!(
            fs::read(positional.join(name)).unwrap(),
            fs::read(alias.join(name)).unwrap(),
            "{name}"
        );
    }
    // Both spellings together are ambiguous.
    let both = mvmc(&work.0, &[namelist, fixed, "--physcal", fixed]);
    assert!(!both.status.success());
    assert!(stderr(&both).contains("both a positional initpara and --physcal"));
}

#[test]
fn nvmccalmode_one_without_parameter_file_follows_c_initialization() {
    let work = Work::new();
    let (dir, _) = phys_inputs(&work);
    let out = work.path("out");
    let output = mvmc(
        &work.0,
        &[
            "-b",
            dir.join("namelist.def").to_str().unwrap(),
            "--out-dir",
            out.to_str().unwrap(),
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    for index in ["007", "008"] {
        let name = format!("physcal_noinit_zvo_varbin_{index}.dat");
        let (n_para, _, rust_blocks) =
            read_varbin(&out.join(format!("zvo_varbin_{index}.dat")), true);
        let (_, _, c_blocks) = read_varbin(&c_expected(&name), false);
        // C InitParameter draws (RNG-exact, not an accumulated computation); C's block is
        // the leading NPara doubles of Rust's complete block.
        assert_eq!(rust_blocks[0][..n_para as usize], c_blocks[0][..]);
        assert_eq!(
            fs::read(out.join(format!("zvo_varbin_{index}.dat"))).unwrap()[..8],
            fs::read(c_expected(&name)).unwrap()[..8]
        );
    }
    // Energy row of the C run: same RNG path, so only roundoff-level differences remain.
    let rust: Vec<f64> = fs::read_to_string(out.join("zvo_out_007.dat"))
        .unwrap()
        .split_whitespace()
        .map(|token| token.parse().unwrap())
        .collect();
    let c: Vec<f64> = fs::read_to_string(c_expected("physcal_noinit_zvo_out_007.dat"))
        .unwrap()
        .split_whitespace()
        .map(|token| token.parse().unwrap())
        .collect();
    assert_eq!(rust.len(), c.len());
    for (a, b) in rust.iter().zip(&c) {
        // Observed difference 9e-16 (summation order); bound 1e-12 relative to scale 5.6.
        assert!((a - b).abs() <= 1e-12, "rust {a} vs C {b}");
    }
}

#[test]
fn positional_initpara_is_the_initial_parameter_file_for_nvmccalmode_zero() {
    let work = Work::new();
    let opt = opt_inputs(&work);
    let out = work.path("out");
    let output = mvmc(
        &work.0,
        &[
            "-b",
            opt.join("namelist.def").to_str().unwrap(),
            opt.join("initpara.dat").to_str().unwrap(),
            "--out-dir",
            out.to_str().unwrap(),
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    assert_blocks_match_c(
        &out.join("zvo_varbin_001.dat"),
        "opt_initpara_zvo_varbin_001.dat",
        14,
    );
    // The explicit alias is invalid in optimization mode, and --initial-def conflicts.
    let alias = mvmc(
        &work.0,
        &[
            opt.join("namelist.def").to_str().unwrap(),
            "--physcal",
            opt.join("initpara.dat").to_str().unwrap(),
        ],
    );
    assert!(!alias.status.success());
    assert!(stderr(&alias).contains("--physcal requires NVMCCalMode=1"));
    let conflict = mvmc(
        &work.0,
        &[
            opt.join("namelist.def").to_str().unwrap(),
            opt.join("initpara.dat").to_str().unwrap(),
            "--initial-def",
            "none",
        ],
    );
    assert!(!conflict.status.success());
    assert!(stderr(&conflict).contains("both a positional initpara and --initial-def"));
}

#[test]
fn mode_label_must_agree_with_the_input_files() {
    let work = Work::new();
    let opt = opt_inputs(&work);
    let namelist = opt.join("namelist.def");
    let namelist = namelist.to_str().unwrap();
    let out = work.path("out");
    let mismatch = mvmc(
        &work.0,
        &[
            "--mode",
            "cmp",
            namelist,
            "--out-dir",
            out.to_str().unwrap(),
        ],
    );
    assert!(!mismatch.status.success());
    assert!(stderr(&mismatch).contains("--mode cmp contradicts the input files"));
    assert!(!out.exists());
    let unknown = mvmc(&work.0, &["--mode", "bogus", namelist]);
    assert!(!unknown.status.success());
    assert!(stderr(&unknown).contains("--mode requires real, cmp or fsz"));
    let matching = mvmc(
        &work.0,
        &[
            "--mode",
            "real",
            namelist,
            "--out-dir",
            out.to_str().unwrap(),
        ],
    );
    assert!(matching.status.success(), "{}", stderr(&matching));
}

#[test]
fn argument_count_and_unknown_option_errors_follow_c_usage() {
    let work = Work::new();
    let missing = mvmc(&work.0, &["-b"]);
    assert!(!missing.status.success());
    assert!(stderr(&missing).contains("Argument count mismatch"));
    let extra = mvmc(&work.0, &["a.def", "b.dat", "c.dat"]);
    assert!(!extra.status.success());
    assert!(stderr(&extra).contains("Argument count mismatch"));
    let unknown = mvmc(&work.0, &["-z", "a.def"]);
    assert!(!unknown.status.success());
    assert!(stderr(&unknown).contains("invalid option -- 'z'"));
    assert!(entries(&work.0).is_empty());
}
