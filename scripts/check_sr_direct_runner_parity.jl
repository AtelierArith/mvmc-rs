# End-to-end exact-bit Direct SR fixtures. Prefix runs record complete RNG
# blocks without copying SFMT.jl's process-global C RNG or perturbing a run.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
VERSION == v"1.13.1" || error("Direct SR runner fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const CASE = let opts = filter(a -> startswith(a, "--case="), ARGS)
    isempty(opts) ? "real" : split(only(opts), "="; limit=2)[2]
end
CASE in ("real", "cmp", "fsz", "hubbard", "interall", "pairhop_real", "pairhop_fsz", "dh2_real", "dh2_cmp", "dh2_fsz", "dh4_real", "dh4_cmp", "dh4_fsz", "dh24_real", "dh24_cmp", "dh24_fsz", "rbm_real", "rbm_cmp", "rbm_general_cmp", "rbm_dh24_cmp", "rbm_fsz", "rbm_reference_cmp", "opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp") || error("Unknown case: $CASE")
const DH_CASE = startswith(CASE,"dh2_") || startswith(CASE,"dh4_") || startswith(CASE,"dh24_")
const RBM_CASE = startswith(CASE,"rbm_")
const OPT_CASE = startswith(CASE,"opt_")
const LOADED_CASE = DH_CASE || RBM_CASE || OPT_CASE
opt_namelist() = joinpath(@__DIR__,"..","tests","fixtures","opttrans","run_"*CASE,"namelist.def")
rbm_namelist() = CASE == "rbm_reference_cmp" ? joinpath(@__DIR__,"..","extern","Julia-mVMC","test","integration","reference","general_rbm_cmp","inputs","namelist.def") : joinpath(@__DIR__,"..","tests","fixtures","rbm","run_"*CASE,"namelist.def")
const SEED = CASE == "rbm_reference_cmp" ? 12395 : 1
rbm_values(data) = ComplexF64[t.value for section in MVMCOptimizers._rbm_parameter_sections(data) for t in section]
function dh_namelist()
    family, mode = split(CASE,"_";limit=2)
    directory = family == "dh2" ? "production_"*mode : "production_"*CASE
    return joinpath(@__DIR__,"..","tests","fixtures", family == "dh2" ? "dh2" : "dh4",directory,"namelist.def")
end
if "--general" in ARGS
    CASE == "fsz" || error("--general requires --case=fsz")
    "--write" in ARGS && error("General must verify the existing AP/P fixtures")
end
const PREFIXES = let opts = filter(a -> startswith(a, "--steps="), ARGS)
    isempty(opts) ? [1, 2, 3, 50] : parse.(Int, split(split(only(opts), "="; limit=2)[2], ","))
end
const STORE = let opts = filter(a -> startswith(a, "--store="), ARGS)
    isempty(opts) ? 0 : parse(Int, split(only(opts), "="; limit=2)[2])
end
STORE in (0, 1) || error("Unsupported NStore: $STORE")
const FIXTURE_ROOT = joinpath(@__DIR__, "..", "tests", "fixtures", "sr_direct", CASE * (STORE == 0 ? "_runner" : "_store_runner"))
const SNAPSHOTS = Ref{Any}()
const FAILURE_STEP = Ref(-1)
const INPUTS = Ref{Any}()
function capture_inputs!(step, data, state)
    INPUTS[] = (data, deepcopy(state.sr_opt))
end
const SOLVE = Dict{String,Vector{Float64}}()
function capture_matrix!(S, g, mapping)
    SOLVE["matrix"] = vec(copy(S)); SOLVE["gradient"] = copy(g)
    SOLVE["mapping"] = Float64.(mapping)
end
capture_factor!(S) = SOLVE["factor"] = vec(copy(S))
capture_solution!(g) = SOLVE["solution"] = copy(g)
# Observe the original direct solver without altering arithmetic or status.
opt_src = read(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "src", "stochastic_opt.jl"), String)
opt_a = first(findfirst("function stochastic_opt!(", opt_src))
opt_b = last(findnext("\nend\n", opt_src, opt_a))
opt_body = replace(opt_src[opt_a:opt_b], "function stochastic_opt!(" => "function source_direct_solver!("; count=1)
opt_body = replace(opt_body, "    ctimer_stop!(c_timer, 56)" => "    Main.capture_matrix!(S, g, smat_to_para_idx)\n    ctimer_stop!(c_timer, 56)"; count=1)
opt_body = replace(opt_body, "        potrf!('U', S)" => "        potrf!('U', S)\n        Main.capture_factor!(S)"; count=1)
opt_body = replace(opt_body, "        potrs!('U', S, g)" => "        potrs!('U', S, g)\n        Main.capture_solution!(g)"; count=1)
Base.include_string(MVMCOptimizers, opt_body)
function capture_source_step!(step, data, state; failed=false)
    failed && (FAILURE_STEP[] = step)
    @assert failed || real(state.energy.wc)>0 && isfinite(state.energy.etot) "Original source produced no finite weighted samples"
    params = vcat([t.value for t in data.gutzwiller_terms],
                  [t.value for t in data.jastrow_terms], data.doublon_holon_2site_params, data.doublon_holon_4site_params, rbm_values(data), [t.value for t in data.orbital_terms], data.opt_trans)
    SNAPSHOTS[] = (copy(params), state.energy.etot, deepcopy(state.electron_config))
end
# Add only an observation hook to a copy of the authoritative optimizer.
# All numerical kernels, sampling, SR, synchronization, and output are original.
src = read(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "src", "vmc_para_opt.jl"), String)
a = first(findfirst("function vmc_para_opt!(", src))
b = last(findnext("\nend\n", src, a))
body = replace(src[a:b], "function vmc_para_opt!(" => "function source_direct_oracle!("; count=1)
body = replace(body, "        # Callback" => "        Main.capture_source_step!(step, data, state)\n        # Callback"; count=1)
body = replace(body, "        # 8. Stochastic optimization" => "        Main.capture_inputs!(step, data, state)\n        # 8. Stochastic optimization"; count=1)
body = replace(body, "info = stochastic_opt!(data, state, timer)" => "info = source_direct_solver!(data, state, timer)"; count=1)
body = replace(body, "        if info != 0" => "        if info != 0\n            Main.capture_source_step!(step, data, state; failed=true)"; count=1)
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
@testset "Direct SR $CASE deterministic prefix runs" begin
    for steps in PREFIXES
        namelist = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "examples", "inputs", (CASE == "hubbard" ? "hubbard_chain_real" : "heisenberg_chain_" * CASE), "namelist.def")
        if "--general" in ARGS
            namelist = joinpath(@__DIR__, "..", "tests", "fixtures", "orbital_general", "heisenberg", "namelist.def")
        end
        if CASE == "interall"
            namelist = joinpath(@__DIR__, "..", "tests", "fixtures", "interall", "spin_chain", "namelist.def")
        end
        if startswith(CASE,"pairhop_")
            namelist = joinpath(@__DIR__,"..","extern","Julia-mVMC","test","integration","reference","hubbard_chain_"*CASE,"inputs","namelist.def")
        end
        if DH_CASE
            namelist = dh_namelist()
        end
        if RBM_CASE
            namelist = rbm_namelist()
        end
        OPT_CASE && (namelist = opt_namelist())
        data = parse_expert_mode_files(namelist)
        data.modpara.nsr_opt_itr_step = steps
        data.modpara.nsr_opt_itr_smp = steps
        data.modpara.nsrcg = 0; data.modpara.nstore_o = STORE
        rng = SFMT19937RNG(); Random.seed!(rng, SEED)
        MVMCExpertModeParsers.init_parameter!(data; rng)
        !LOADED_CASE && MVMCExpertModeParsers.sync_modified_parameter!(data)
        if CASE == "rbm_reference_cmp"
            @test MVMCOptimizers.read_initial_def!(data,joinpath(dirname(namelist),"initial.def"))
        end
        if LOADED_CASE
            MVMCExpertModeParsers.read_input_parameters!(data,namelist)
            MVMCOptimizers.sync_modified_parameter!(data)
        end
        MVMCExpertModeParsers.init_qp_weight!(data)
        mktempdir() do dir
            FAILURE_STEP[] = -1
            info = MVMCOptimizers.source_direct_oracle!(data; rng, output_dir=dir)
            @test info == 0 || (CASE == "rbm_fsz" && steps == 50 && info == 1) || (OPT_CASE && info == 1)
            (RBM_CASE || OPT_CASE) && verify("step-$steps-status.txt", "$info $(FAILURE_STEP[])\n")
            params, energy, configs = SNAPSHOTS[]
            input_data, sr = INPUTS[]
            if steps == 1
                io = IOBuffer()
                println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
                println(io, "# Julia-mVMC c2ea432785bc14364a3cd5e9eef44db464289cc9")
                println(io, sr.sr_opt_size, " ", length(SOLVE["gradient"]))
                println(io, join(Int.(SOLVE["mapping"]), " "))
                complex = MVMCOptimizers.get_all_complex_flag(input_data)
                oo = complex ? reinterpret(Float64, sr.sr_opt_oo) : sr.sr_opt_oo_real
                ho = complex ? reinterpret(Float64, sr.sr_opt_ho) : sr.sr_opt_ho_real
                println(io, hex(oo)); println(io, hex(ho))
                for name in ("matrix", "gradient", "factor", "solution")
                    println(io, hex(SOLVE[name]))
                end
                verify("fixed-input.txt", String(take!(io)))
                if STORE == 1
                    gram = zeros(eltype(complex ? sr.sr_opt_oo : sr.sr_opt_oo_real),
                                 length(complex ? sr.sr_opt_oo : sr.sr_opt_oo_real))
                    raw_store = complex ? sr.sr_opt_o_store : sr.sr_opt_o_store_real
                    finalize = complex ? MVMCOptimizers.finalize_oo_store! : MVMCOptimizers.finalize_oo_store_real!
                    finalize(gram, raw_store, sr.sr_opt_size, input_data.modpara.nvmc_sample)
                    io = IOBuffer()
                    println(io, sr.sr_opt_size, " ", input_data.modpara.nvmc_sample)
                    println(io, hex(complex ? reinterpret(Float64, raw_store) : raw_store))
                    println(io, hex(complex ? reinterpret(Float64, gram) : gram))
                    verify("gram.txt", String(take!(io)))
                end
            end
            doubles = Float64[]
            for z in params; append!(doubles, (real(z), imag(z))); end
            verify("step-$steps-parameters.txt", hex(doubles)*"\n")
            verify("step-$steps-energy.txt", hex([real(energy), imag(energy)])*"\n")
            io = IOBuffer()
            for vals in (configs.ele_idx, configs.ele_cfg, configs.ele_num, configs.ele_proj_cnt)
                println(io, join(vals, " "))
            end
            if CASE in ("interall","pairhop_fsz","dh2_fsz","dh4_fsz","dh24_fsz","rbm_fsz","opt_fsz")
                println(io, join(configs.ele_spn, " "))
                println(io, join(configs.burn_ele_idx, " "))
                println(io, join(vcat(configs.counter[1:9],configs.counter[11]), " "))
            elseif RBM_CASE || OPT_CASE
                println(io, join(configs.burn_ele_idx, " "))
                println(io, join(vcat(configs.counter[1:9],configs.counter[11]), " "))
            end
            verify("step-$steps-configs.txt", String(take!(io)))
            verify("step-$steps-rng.txt", join([rand(rng, UInt32) for _ in 1:624], " ")*"\n")
            if LOADED_CASE
                for name in ("zvo_out.dat","zvo_var.dat","zqp_opt.dat","zqp_gutzwiller_opt.dat","zqp_jastrow_opt.dat","zqp_orbital_opt.dat")
                    if isfile(joinpath(dir,name))
                        verify("step-$steps-"*name,read(joinpath(dir,name),String))
                    else
                        @test info != 0 && startswith(name,"zqp_")
                        verify("step-$steps-"*name, "# absent after source SR failure\n")
                    end
                end
                @test !isfile(joinpath(dir,"zqp_dh2_opt.dat")) # Canonical writer omits DH coefficients.
                @test !isfile(joinpath(dir,"zqp_dh4_opt.dat"))
                if OPT_CASE
                    verify("reference.txt","# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=$SEED\n# Julia-mVMC 8bb1b9e; numerical sources c2ea432; nonidentity three-sector OptTrans; original base sampling settings; "*CASE*"\n")
                    @test !isfile(joinpath(dir,"zqp_opttrans_opt.dat"))
                elseif RBM_CASE
                    verify("reference.txt","# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=$SEED\n# Julia-mVMC 8bb1b9e; numerical sources c2ea432; "*(CASE == "rbm_reference_cmp" ? "canonical GeneralRBM initial.def and unchanged input/sampling settings" : "nonzero InRBM overlays")*"; "*CASE*"\n")
                    @test !isfile(joinpath(dir,"zqp_rbm_opt.dat"))
                else
                verify("reference.txt","# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=1\n# Nonzero "*(startswith(CASE,"dh2_") ? "InDH2" : startswith(CASE,"dh24_") ? "InDH2+InDH4" : "InDH4")*"; normal modes use canonical Hubbard sample=100, FSZ uses canonical PairHop sample=2000 without PairHop; warmup=10; "*CASE*"\n")
                end
            end
        end
    end
end
