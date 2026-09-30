# Regenerate the Phase-5 50-step `zvo_out.dat` reference goldens consumed by
# crates/mvmc-core/tests/phase5_regression_50step.rs.
#
# Each case runs Julia-mVMC for 50 SR steps with nsmp=50 and RndSeed=1 (from
# modpara.def), and writes the first 50 raw `zvo_out.dat` lines to:
#   reference/<model>/zvo_out_first50.dat
#
# Run against the bundled Julia-mVMC checkout (requires Julia 1.11):
#   julia +1.11 --project=extern/Julia-mVMC scripts/dump_zvo_50step_reference.jl

using MVMCOptimizers

const ROOT = abspath(joinpath(@__DIR__, ".."))
const JULIA_MVMC = joinpath(ROOT, "extern", "Julia-mVMC")
const N_STEPS = 50

const MODELS = (
    (name = "heisenberg_chain_real", input = "heisenberg_chain_real", example = "heisenberg_chain_real.jl", mode = :real),
    (name = "heisenberg_chain_cmp", input = "heisenberg_chain_cmp", example = "heisenberg_chain_cmp.jl", mode = :cmp),
    (name = "heisenberg_chain_fsz", input = "heisenberg_chain_fsz", example = "heisenberg_chain_fsz.jl", mode = :fsz),
    (name = "hubbard_chain_real", input = "hubbard_chain_real", example = "hubbard_chain.jl", mode = :real),
)

for model in MODELS
    namelist = joinpath(JULIA_MVMC, "examples", "inputs", model.input, "namelist.def")
    isfile(namelist) || error("namelist missing: $namelist")

    println("=== $(model.name) — $(N_STEPS) SR steps (mode=$(model.mode)) ===")
    result = MVMCOptimizers.run_para_opt_from_namelist(
        namelist;
        nsteps = N_STEPS,
        nsmp = N_STEPS,
        mode = model.mode,
    )

    lines = result.zvo_first_n
    length(lines) == N_STEPS || error("$(model.name): expected $(N_STEPS) rows, got $(length(lines))")

    out_dir = joinpath(ROOT, "reference", model.name)
    mkpath(out_dir)
    out_path = joinpath(out_dir, "zvo_out_first50.dat")
    open(out_path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    println("wrote $(out_path)  (final energy / site = $(result.final_energy_per_site))")
end
