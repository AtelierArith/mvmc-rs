# End-to-end exact-bit SR-CG fixtures. Prefix runs record complete RNG
# blocks without copying SFMT.jl's process-global C RNG or perturbing a run.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
VERSION == v"1.13.1" || error("CG runner fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const CASE = let opts = filter(a -> startswith(a, "--case="), ARGS)
    isempty(opts) ? "real" : split(only(opts), "="; limit=2)[2]
end
CASE in ("real", "cmp", "fsz", "hubbard") || error("Unknown case: $CASE")
const PREFIXES = let opts = filter(a -> startswith(a, "--steps="), ARGS)
    isempty(opts) ? [1, 2, 3, 50] : [parse(Int, split(only(opts), "="; limit=2)[2])]
end
const FIXTURE_ROOT = joinpath(@__DIR__, "..", "tests", "fixtures", "sr_cg", CASE * "_runner")
const SNAPSHOTS = Ref{Any}()
function capture_source_step!(step, data, state)
    params = vcat([t.value for t in data.gutzwiller_terms],
                  [t.value for t in data.jastrow_terms], [t.value for t in data.orbital_terms])
    SNAPSHOTS[] = (copy(params), state.energy.etot, deepcopy(state.electron_config))
end
# Add only an observation hook to a copy of the authoritative optimizer.
# All numerical kernels, sampling, SR, synchronization, and output are original.
src = read(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "src", "vmc_para_opt.jl"), String)
a = first(findfirst("function vmc_para_opt!(", src))
b = last(findnext("\nend\n", src, a))
body = replace(src[a:b], "function vmc_para_opt!(" => "function source_cg_oracle!("; count=1)
body = replace(body, "        # Callback" => "        Main.capture_source_step!(step, data, state)\n        # Callback"; count=1)
Base.include_string(MVMCOptimizers, body)
hex(v) = join(string.(reinterpret.(UInt64, v); base=16, pad=16), " ")
function verify(name, actual)
    path = joinpath(FIXTURE_ROOT, name)
    if "--write" in ARGS
        mkpath(FIXTURE_ROOT); write(path, actual)
    else
        @test actual == read(path, String)
    end
end
@testset "SR-CG $CASE deterministic prefix runs" begin
    for steps in PREFIXES
        input = CASE == "hubbard" ? "hubbard_chain_real" : "heisenberg_chain_" * CASE
        namelist = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "examples", "inputs", input, "namelist.def")
        data = parse_expert_mode_files(namelist)
        data.modpara.nsr_opt_itr_step = steps
        data.modpara.nsr_opt_itr_smp = steps
        data.modpara.nsrcg = 1; data.modpara.nstore_o = 0
        rng = SFMT19937RNG(); Random.seed!(rng, 1)
        MVMCExpertModeParsers.init_parameter!(data; rng)
        MVMCExpertModeParsers.sync_modified_parameter!(data)
        MVMCExpertModeParsers.init_qp_weight!(data)
        mktempdir() do dir
            @test MVMCOptimizers.source_cg_oracle!(data; rng, output_dir=dir) == 0
            params, energy, configs = SNAPSHOTS[]
            doubles = Float64[]
            for z in params; append!(doubles, (real(z), imag(z))); end
            verify("step-$steps-parameters.txt", hex(doubles)*"\n")
            verify("step-$steps-energy.txt", hex([real(energy), imag(energy)])*"\n")
            io = IOBuffer()
            for vals in (configs.ele_idx, configs.ele_cfg, configs.ele_num, configs.ele_proj_cnt)
                println(io, join(vals, " "))
            end
            verify("step-$steps-configs.txt", String(take!(io)))
            verify("step-$steps-rng.txt", join([rand(rng, UInt32) for _ in 1:624], " ")*"\n")
            verify("step-$steps-SRinfo.txt", read(joinpath(dir, "zvo_SRinfo.dat"), String))
        end
    end
end
