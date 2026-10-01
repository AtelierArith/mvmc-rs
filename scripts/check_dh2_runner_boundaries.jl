# Original-source loaded initialization, full RNG block and post-sync history.
using Test, Random, SFMT, LinearAlgebra, MVMCOptimizers, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("DH2 fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const ROOT=joinpath(@__DIR__,"..","tests","fixtures","dh2")
hex(v)=join(string.(reinterpret(UInt64,collect(reinterpret(Float64,v)));base=16,pad=16)," ")
const HISTORY=Ref{Any}()
capture_history!(state)=(HISTORY[]=deepcopy(state.opt_data))
src=read(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCOptimizers.jl","src","vmc_para_opt.jl"),String)
a=first(findfirst("function vmc_para_opt!(",src)); b=last(findnext("\nend\n",src,a))
body=replace(src[a:b],"function vmc_para_opt!("=>"function source_dh2_boundary_oracle!(";count=1)
body=replace(body,"        # Callback"=>"        Main.capture_history!(state)\n        # Callback";count=1)
Base.include_string(MVMCOptimizers,body)
function verify(name,actual)
    if "--write" in ARGS;write(joinpath(ROOT,name),actual)
    else;@test actual==read(joinpath(ROOT,name),String);end
end
function prepared(mode)
    namelist=joinpath(ROOT,"production_$mode","namelist.def")
    d=parse_expert_mode_files(namelist)
    rng=SFMT19937RNG(); Random.seed!(rng,1)
    MVMCExpertModeParsers.init_parameter!(d;rng)
    MVMCExpertModeParsers.read_input_parameters!(d,namelist)
    MVMCOptimizers.sync_modified_parameter!(d)
    MVMCExpertModeParsers.init_qp_weight!(d)
    d,rng
end
@testset "DH2 loaded initialization and history source boundaries" begin
    for mode in ("real","cmp","fsz")
        d,rng=prepared(mode)
        io=IOBuffer();println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=1")
        println(io,join(Int.(d.optimization_flags)," "))
        println(io,hex(vcat(MVMCExpertModeParsers.projection_parameters(d),[t.value for t in d.orbital_terms])))
        println(io,join([rand(rng,UInt32) for _=1:624]," "))
        verify("loaded-$mode.txt",String(take!(io)))
        d,rng=prepared(mode) # SFMT process-global generator is intentionally reseeded for a separate run.
        d.modpara.nsr_opt_itr_step=3;d.modpara.nsr_opt_itr_smp=3
        mktempdir() do dir
            @test MVMCOptimizers.source_dh2_boundary_oracle!(d;rng,output_dir=dir)==0
            history=HISTORY[]
            @test length(history)==3
            io=IOBuffer();println(io,"# Original history omits DH2, retaining Gutzwiller/Jastrow/orbital term order")
            for point in history
                @test isfinite(point.energy)
                @test length(point.parameters)==length(d.gutzwiller_terms)+length(d.jastrow_terms)+length(d.orbital_terms)
                println(io,hex(ComplexF64[point.energy]));println(io,hex(point.parameters))
            end
            verify("history-$mode.txt",String(take!(io)))
        end
    end
end
