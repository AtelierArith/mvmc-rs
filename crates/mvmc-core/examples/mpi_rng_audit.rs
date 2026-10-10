//! Optional developer audit: production run and exact rank-local RNG snapshots.
//! These diagnostic executions never supply benchmark timings.
use mvmc_core::{mpi::MpiContext, run::run_para_opt_from_namelist_observed, RunConfig};
use std::path::PathBuf;

fn audit(world: &MpiContext) -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: mpi_rng_audit NAMELIST STEPS OUTPUT RANKS".into());
    }
    let steps = args[1].parse::<i64>().map_err(|e| e.to_string())?;
    let ranks = args[3].parse::<usize>().map_err(|e| e.to_string())?;
    if steps <= 0 || ranks != world.world_size() {
        return Err("positive steps and matching actual MPI world required".into());
    }
    let output = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let mut config = RunConfig::new(steps, "real");
    config.nsmp = Some(steps);
    config.output_dir = Some(output.join("production"));
    let (summary, state, rng) = run_para_opt_from_namelist_observed(&args[0], config, world)?;
    let (words, index) = rng.state_snapshot();
    let electron = &state.electron_config;
    let snapshot = serde_json::json!({
        "rank": world.rank(), "world_size": world.world_size(),
        "steps": summary.effective_nsteps, "final_energy_per_site": summary.final_energy_per_site,
        "rng_words": words.as_slice(), "rng_index": index,
        "rng_words_consumed": rng.words_consumed(),
        "ele_idx": electron.ele_idx, "ele_cfg": electron.ele_cfg,
        "ele_num": electron.ele_num, "ele_spn": electron.ele_spn,
        "ele_proj_cnt": electron.ele_proj_cnt, "counter": electron.counter,
    });
    std::fs::write(
        output.join(format!("rng-rank-{}.json", world.rank())),
        serde_json::to_vec_pretty(&snapshot).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn run() {
    let world = MpiContext::initialize().expect("MPI initialization");
    if let Err(error) = audit(&world) {
        eprintln!("MPI RNG audit failed: {error}");
        use mpi::traits::Communicator;
        mpi::topology::SimpleCommunicator::world().abort(1);
    }
}

fn main() {
    if mvmc_core::threading::inner_thread_config().threads > 1 {
        mvmc_core::threading::install(run);
    } else {
        run();
    }
}
