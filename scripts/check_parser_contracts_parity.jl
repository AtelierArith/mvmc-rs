include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Julia v0.5.0 reference for Rust validation and optimization flag tests.
# Run under Julia 1.13.1 with the pinned Julia-mVMC workspace packages.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("Parser contract verification requires Julia 1.13.1")
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCExpertModeParsers.jl", "test", "test_validation.jl"))

@testset "validation severity controls" begin
    p = ModParaParameters(nsite=4, nelec=2, ncond=3, nblock_size_rbm_ratio=202)
    r = validate_modpara_params(p)
    @test r.errors == ["NCond must be even"]
    @test r.warnings == ["NElec (2) differs from expected value (1) based on NCond", "NBlockSize_RBMRatio should be multiple of 8"]
    p.dsr_opt_step_dt = NaN; p.ncond = -1
    @test validate_modpara_params(p).is_valid
    t = TransferTerm(0, 1, NaN+Inf*im, :up)
    r = validate_transfer_terms([t,t], 4)
    @test r.is_valid && isempty(r.errors) && isempty(r.warnings)
end

const fixed_steps = IOBuffer()
println(fixed_steps,"# Julia $VERSION; $(BLAS.get_config()); threads=1; canonical parsed flags, three direct SR/sync steps; seed=1")
hex(values) = join([string(reinterpret(UInt64,x);base=16,pad=16) for v in values for x in (real(v),imag(v))]," ")
definition(name, width, complex, rows) = "===\n$name $width\nComplexType $complex\n===\n===\n$rows"
@testset "parsed component flags, draws, and gauge shifts" begin
    for (jastrow_complex, parallel) in ((1,true),(0,false),(1,false))
        mktempdir() do dir
            write(joinpath(dir,"modpara.def"), "Nsite 3\nNElec 1\n")
            write(joinpath(dir,"g.def"), definition("NGutzwillerIdx",2,0,"0 0\n1 0\n2 1\n0 1\n1 0\n"))
            write(joinpath(dir,"j.def"), definition("NJastrowIdx",2,jastrow_complex,"0 1 0\n1 2 1\n0 0\n1 1\n"))
            write(joinpath(dir,"o.def"), definition("NOrbitalIdx",parallel ? 7 : 2,0,"0 1 0\n1 0 1\n0 0\n1 1\n"))
            write(joinpath(dir,"p.def"), definition("NOrbitalParallel",2,0,"0 1 0\n1 2 1\n0 1\n1 0\n"))
            namelist = "ModPara modpara.def\nOrbitalAntiParallel o.def\nGutzwiller g.def\nJastrow j.def\n" * (parallel ? "OrbitalParallel p.def\n" : "")
            write(joinpath(dir,"namelist.def"), namelist)
            d = parse_expert_mode_files(joinpath(dir,"namelist.def"))
            projection = Bool[1,0,0,0,0,0,1,jastrow_complex!=0]
            orbitals = parallel ? repeat(Bool[0,1,1,1,1,1,1,1,1,0,0], inner=2) :
                (jastrow_complex==1 ? Bool[0,0,1,1] : Bool[0,1,1,1])
            @test d.optimization_flags == vcat(projection, orbitals)
            println("flags complexJ=", jastrow_complex, " parallel=", parallel, " ", Int.(d.optimization_flags))
            rng = SFMT19937RNG(); Random.seed!(rng,1)
            MVMCExpertModeParsers.init_parameter!(d;rng)
            if parallel
                @test all(t.value == 0.0 for t in d.orbital_terms if t.idx in (0,9,10))
                hash = UInt64(0xcbf29ce484222325)
                for _ in 1:624
                    hash = (hash ⊻ UInt64(rand(rng,UInt32))) * UInt64(0x100000001b3)
                end
                @test hash == UInt64(16445735861883055124)
                expected = ComplexF64[
                    0.0+0.0im, -0.22854561532191836+0.14249578128268756im,
                    -0.299485566106154-0.1347245207905717im,
                    -0.28930135836557447+0.17361291906410883im,
                    0.0+0.0im, 0.0+0.0im,
                ]
                @test ReferenceNumericalComparison.close_values(reinterpret(Float64,[t.value for t in d.orbital_terms]),reinterpret(Float64,expected),32*eps(Float64),32*eps(Float64))
                println("post-init SFMT block hash=", hash)
                println("Slater=", [t.value for t in d.orbital_terms])
            else
                for (i,t) in enumerate(d.gutzwiller_terms); t.value = Float64(i)+0.2im; end
                for (i,t) in enumerate(d.jastrow_terms); t.value = Float64(i+2)+0.5im; end
                d.orbital_terms[1].value = 1.0+0.0im
                d.orbital_terms[2].value = 2.0+0.0im
                g = [t.value for t in d.gutzwiller_terms]; j = [t.value for t in d.jastrow_terms]
                for _ in 1:3
                    MVMCOptimizers.sync_modified_parameter!(d)
                    @test [t.value for t in d.gutzwiller_terms] == g
                    @test [t.value for t in d.jastrow_terms] == j
                end
                @test [t.value for t in d.orbital_terms] == ComplexF64[2,4]
                d.modpara.dsr_opt_sta_del = 0.0
                d.modpara.dsr_opt_step_dt = 0.125
                n = MVMCExpertModeParsers.count_variational_parameters(d)
                complex = jastrow_complex != 0
                # Pin the runtime mode to the supplied real/complex SR buffers.
                d.complex_flags = [Int(complex)]
                println(fixed_steps,"$jastrow_complex $n")
                println(fixed_steps,join(Int.(d.optimization_flags)," "))
                for step in 1:3
                    state=MVMCOptimizers.VMCOptimizationState(3,1,4,n,1,1,complex,false)
                    state.energy.wc=1.0+0.0im
                    off=complex ? 2 : 1
                    size=off*(n+1)
                    oo=complex ? state.sr_opt.sr_opt_oo : state.sr_opt.sr_opt_oo_real
                    ho=complex ? state.sr_opt.sr_opt_ho : state.sr_opt.sr_opt_ho_real
                    for component in off:size-1
                        oo[component*size+component+1]=1.0+step/4
                        ho[component+1]=(component+1)/16
                    end
                    @test MVMCOptimizers.stochastic_opt!(d,state)==0
                    MVMCOptimizers.sync_modified_parameter!(d)
                    @test d.gutzwiller_terms[2].value == g[2]
                    @test d.jastrow_terms[1].value == j[1]
                    println(fixed_steps,hex(vcat([t.value for t in d.gutzwiller_terms],[t.value for t in d.jastrow_terms],[t.value for t in d.orbital_terms])))
                end
                println(fixed_steps,join([rand(rng,UInt32) for _ in 1:624]," "))
            end
        end
    end
end

@testset "Parsed fixed-component SR step fixture" begin
    path=joinpath(@__DIR__,"..","tests","fixtures","sr_failure","fixed_flag_steps.txt")
    actual=String(take!(fixed_steps))
    if "--write" in ARGS
        write(path,actual)
    else
        @test compare_hex_text(actual,read(path,String),(r,c,t)->mod1(r,6) in (3,4,5) ? (1e-13,1e-13) : nothing)
    end
end
