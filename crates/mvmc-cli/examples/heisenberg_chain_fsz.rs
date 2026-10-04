// examples/heisenberg_chain_fsz.rs
//
// Run VMCParaOpt on a 16-site Heisenberg chain (fsz / generalised-orbital mode).
// Inputs are in extern/Julia-mVMC/examples/inputs/heisenberg_chain_fsz/.
// Step count is overridable via JULIA_MVMC_EXAMPLE_STEPS (default 50).
//
// Port of: extern/Julia-mVMC/examples/heisenberg_chain_fsz.jl
//
// Run:
//   cargo run --example heisenberg_chain_fsz
//   JULIA_MVMC_EXAMPLE_STEPS=10 cargo run --example heisenberg_chain_fsz

#[path = "support/mod.rs"]
mod support;

fn main() {
    let julia = support::julia_mvmc_root().unwrap_or_else(|| {
        eprintln!(
            "error: Julia-mVMC checkout not found.\n\
             Set JULIA_MVMC_ROOT to the path of the Julia-mVMC directory."
        );
        std::process::exit(1);
    });

    let input_dir = julia
        .join("examples")
        .join("inputs")
        .join("heisenberg_chain_fsz");
    let namelist = input_dir.join("namelist.def");

    if !namelist.is_file() {
        eprintln!("error: namelist.def not found at {}", namelist.display());
        std::process::exit(1);
    }

    let nsteps = support::example_nsteps();
    println!("=== Heisenberg chain (fsz) — {nsteps} SR steps ===");

    let out_dir = support::make_output_dir("heisenberg_chain_fsz");

    let summary = mvmc_core::run_para_opt_from_namelist(
        &namelist,
        mvmc_core::RunConfig {
            nsmp: Some(nsteps as i64),
            seed: None,
            output_dir: Some(out_dir.clone()),
            ..mvmc_core::RunConfig::new(nsteps as i64, "fsz")
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("error: run failed: {e}");
        std::process::exit(1);
    });

    println!("Final energy / site = {}", summary.final_energy_per_site);
}
