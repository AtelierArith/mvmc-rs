# End-to-end exact-bit SR-CG fixtures. Prefix runs record complete RNG
# blocks without copying SFMT.jl's process-global C RNG or perturbing a run.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
VERSION == v"1.13.1" || error("CG runner fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const CASE = let opts = filter(a -> startswith(a, "--case="), ARGS)
    isempty(opts) ? "real" : split(only(opts), "="; limit=2)[2]
end
CASE in ("real", "cmp", "fsz", "hubbard", "interall", "pairhop_real", "pairhop_fsz") || error("Unknown case: $CASE")
pairhop_namelist() = joinpath(@__DIR__,"..","extern","Julia-mVMC","test","integration","reference","hubbard_chain_"*CASE,"inputs","namelist.def")
if "--general" in ARGS
    CASE == "fsz" || error("--general requires --case=fsz")
    "--write" in ARGS && error("General must verify the existing AP/P fixtures")
end
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
if CASE == "interall" || startswith(CASE,"pairhop_")
    @testset "$CASE input and initialization boundary" begin
        data = parse_expert_mode_files(CASE == "interall" ? joinpath(@__DIR__, "..", "tests", "fixtures", "interall", "spin_chain", "namelist.def") : pairhop_namelist())
        if CASE == "interall"
            @test length(data.inter_all_terms) == 26
            @test data.i_flg_orbital_general == 1 && MVMCOptimizers.get_all_complex_flag(data)
        else
            @test length(data.pair_hop_terms) == 2
            @test data.i_flg_orbital_general == Int(CASE == "pairhop_fsz")
            @test MVMCOptimizers.get_all_complex_flag(data) == (CASE == "pairhop_fsz")
        end
        rng = SFMT19937RNG(); Random.seed!(rng,1)
        MVMCExpertModeParsers.init_parameter!(data; rng)
        MVMCExpertModeParsers.sync_modified_parameter!(data)
        MVMCExpertModeParsers.init_qp_weight!(data)
        values = vcat([t.value for t in data.gutzwiller_terms],
                      [t.value for t in data.jastrow_terms], [t.value for t in data.orbital_terms])
        verify("initial-flags.txt",join(Int.(data.optimization_flags)," ")*"\n")
        verify("initial-parameters.txt",hex(collect(reinterpret(Float64,values)))*"\n")
        verify("initial-rng.txt",join([rand(rng,UInt32) for _ in 1:624]," ")*"\n")
        model = CASE == "interall" ? "six-site XYZ spin chain with imaginary Hermitian InterAll coefficients" : "canonical hubbard_chain_"*CASE*" inputs, unchanged sampling settings"
        verify("reference.txt","# Julia $VERSION; $(BLAS.get_config()); threads=1\n# Julia-mVMC 8bb1b9e; numerical sources c2ea432; seed=1; $model\n")
    end
end
@testset "SR-CG $CASE deterministic prefix runs" begin
    for steps in PREFIXES
        input = CASE == "hubbard" ? "hubbard_chain_real" : "heisenberg_chain_" * CASE
        namelist = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "examples", "inputs", input, "namelist.def")
        if "--general" in ARGS
            namelist = joinpath(@__DIR__, "..", "tests", "fixtures", "orbital_general", "heisenberg", "namelist.def")
        end
        if CASE == "interall"
            namelist = joinpath(@__DIR__, "..", "tests", "fixtures", "interall", "spin_chain", "namelist.def")
        end
        if startswith(CASE,"pairhop_")
            namelist = pairhop_namelist()
        end
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
            if CASE in ("interall","pairhop_fsz")
                println(io, join(configs.ele_spn, " "))
                println(io, join(configs.burn_ele_idx, " "))
                println(io, join(vcat(configs.counter[1:9],configs.counter[11]), " "))
            end
            verify("step-$steps-configs.txt", String(take!(io)))
            verify("step-$steps-rng.txt", join([rand(rng, UInt32) for _ in 1:624], " ")*"\n")
            verify("step-$steps-SRinfo.txt", read(joinpath(dir, "zvo_SRinfo.dat"), String))
        end
    end
end
