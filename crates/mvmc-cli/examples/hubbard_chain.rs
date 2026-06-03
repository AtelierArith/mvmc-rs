// examples/hubbard_chain.rs
//
// Run VMCParaOpt on a Hubbard chain (real mode).
// Inputs are in extern/Julia-mVMC/examples/inputs/hubbard_chain_real/.
// Step count is overridable via JULIA_MVMC_EXAMPLE_STEPS (default 50).
//
// Port of: extern/Julia-mVMC/examples/hubbard_chain.jl
//
// Run:
//   cargo run --example hubbard_chain
//   JULIA_MVMC_EXAMPLE_STEPS=10 cargo run --example hubbard_chain

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
        .join("hubbard_chain_real");
    let namelist = input_dir.join("namelist.def");

    if !namelist.is_file() {
        eprintln!("error: namelist.def not found at {}", namelist.display());
        std::process::exit(1);
    }

    let nsteps = support::example_nsteps();
    println!("=== Hubbard chain — {nsteps} SR steps ===");

    let out_dir = support::make_output_dir("hubbard_chain");

    let summary = mvmc_core::run_para_opt_from_namelist(&namelist, nsteps, None, Some(&out_dir))
        .unwrap_or_else(|e| {
            eprintln!("error: run failed: {e}");
            std::process::exit(1);
        });

    match summary.final_energy_per_site {
        Some(e) => println!("Final energy / site = {e}"),
        None => eprintln!("warning: could not read final energy from zvo_out.dat"),
    }
}
