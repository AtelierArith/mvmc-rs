//! Portable function-level GPU/CPU benchmark and numerical-validation suite (issue #450).
//!
//! ```text
//! cd gpu/mvmc-gpu-cuda
//! cargo run --release --example function_suite -- <family> [--quick|--full] --out FILE.csv
//!     family: pfaffian | sr | sampler | transfers | na
//! ```
//!
//! The primary output of every family is a numerical verdict (PASS/FAIL per function and
//! size) against the C-order CPU oracle with an explicit, justified bound; timings are
//! recorded as secondary reference columns. The process exits non-zero when any row is FAIL
//! or ERROR (all rows are still written). Driven by `scripts/bench/run_all.sh`; the CSV schema
//! is documented in `benchmark/function_suite/README.md`.

#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

mod csv;
mod pf;
mod sampler;
mod sr;
mod transfers;

use csv::{Csv, Row, Verdict};

pub struct Cfg {
    pub full: bool,
    pub cpu_only: bool,
    pub cpu_label: String,
    pub inputs: std::path::PathBuf,
    pub max_gb: Option<f64>,
    pub max_bytes: usize,
    /// Optional overrides of the Pfaffian grid (`--ns 16,64 --batches 1,8`).
    /// Sampler timing grid cap on `W * L^2` (full profile budget).
    pub work_cap: f64,
    pub ns: Option<Vec<usize>>,
    pub batches: Option<Vec<usize>>,
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|x| x.trim().parse().expect("number"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let family = args.first().cloned().unwrap_or_default();
    let full = args.iter().any(|a| a == "--full");
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cfg = Cfg {
        full,
        cpu_only: args.iter().any(|a| a == "--cpu-only"),
        cpu_label: arg(&args, "--cpu-label").unwrap_or("cpu").to_string(),
        inputs: arg(&args, "--inputs").map_or_else(
            || repo.join("benchmark/hubbard_chain/inputs"),
            std::path::PathBuf::from,
        ),
        max_gb: arg(&args, "--max-gb").map(|s| s.parse().expect("--max-gb")),
        max_bytes: arg(&args, "--max-bytes")
            .map_or(if full { 2usize << 30 } else { 256 << 20 }, |s| {
                s.parse().expect("--max-bytes")
            }),
        work_cap: arg(&args, "--work-cap").map_or(2.1e6, |s| s.parse().expect("--work-cap")),
        ns: arg(&args, "--ns").map(list),
        batches: arg(&args, "--batches").map(list),
    };
    if family == "meta" {
        // device/driver/library facts (key=value lines) for the metadata block
        mvmc_gpu_cuda::install();
        match mvmc_core::backend::device_report(mvmc_core::backend::BackendKind::Cuda(0)) {
            Ok(r) => print!("{}", r.render()),
            Err(e) => println!("device_report_error={e}"),
        }
        return;
    }
    let out = arg(&args, "--out").expect("--out FILE.csv is required");
    let mut csv = Csv::create(out, &family);
    eprintln!(
        "# function_suite family={family} profile={} cpu_only={} label={}",
        if full { "full" } else { "quick" },
        cfg.cpu_only,
        cfg.cpu_label
    );
    if !cfg.cpu_only {
        mvmc_gpu_cuda::install();
    }
    match family.as_str() {
        "pfaffian" => pf::run(&cfg, &mut csv),
        "sr" => sr::run(&cfg, &mut csv),
        "sampler" => sampler::run(&cfg, &mut csv),
        "transfers" => transfers::run(&cfg, &mut csv),
        "na" => {
            let reason = arg(&args, "--reason").unwrap_or("not available");
            let name = arg(&args, "--name").unwrap_or("unknown");
            csv.push(Row::new(name, "all", "-", "-", "").verdict(Verdict::NotAvailable, reason));
        }
        other => {
            eprintln!("unknown family '{other}' (pfaffian|sr|sampler|transfers|na)");
            std::process::exit(2);
        }
    }
    let (fails, total) = csv.summary();
    eprintln!("# {family}: {total} rows, {fails} FAIL/ERROR");
    if fails > 0 {
        std::process::exit(1);
    }
}
