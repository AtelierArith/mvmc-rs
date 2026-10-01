# Exact RBM parser/layout/init contract from the unmodified canonical sources.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("RBM fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const P = MVMCExpertModeParsers
root = joinpath(@__DIR__, "..", "tests", "fixtures", "rbm")
names = ["$(f)RBM_$(s)" for s in ("PhysLayer", "HiddenLayer", "PhysHidden") for f in ("Charge", "Spin", "General")]
fields = [Symbol(lowercase(f), "_rbm_", s, "_terms") for s in ("phys_layer", "hidden_layer", "phys_hidden") for f in ("Charge", "Spin", "General")]
parsers = [getfield(P, Symbol("parse_", string(field)[1:end-6], "_content_extended")) for field in fields]
cols = [2,2,3,2,2,2,3,3,4]
hex(values) = join([string(reinterpret(UInt64,x); base=16,pad=16) for v in values for x in (real(v),imag(v))]," ")
function verify(file,actual)
    path = joinpath(root,file)
    "--write" in ARGS ? write(path,actual) : @test(actual == read(path,String))
end
@testset "RBM original parser and initialization" begin
    # Canonical unit contracts remain in use alongside exact SFMT fixtures.
    include(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCExpertModeParsers.jl","test","test_parameter_init_complexflag_rbm.jl"))
    include(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCExpertModeParsers.jl","test","test_read_input_parameters_rbm_layout.jl"))
    io = IOBuffer()
    println(io,"# Julia 1.13.1 Float64 sin/cos; SFMT seed 20251002; exact RBM phase values")
    rng=SFMT19937RNG(); Random.seed!(rng,20251002)
    phases=vcat([0.0,-0.0], [f(x) for x in Float64.((π/4,π/2,3π/4,π,5π/4,3π/2,7π/4,2π)) for f in (prevfloat,identity,nextfloat) if f(x)<=2π], [2π*rand(rng) for _ in 1:2048])
    for x in phases
        for phase in (x,-x)
            println(io,join([string(reinterpret(UInt64,y);base=16,pad=16) for y in (phase,sin(phase),cos(phase))]," "))
        end
    end
    verify("phase.txt",String(take!(io)))
    io = IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; Julia-mVMC 8bb1b9e; parser source c2ea432")
    for (name, parser, nc) in zip(names,parsers,cols)
        cases = sort(filter(n -> startswith(n,"$(nc)_") && endswith(n,".def"), readdir(root)))
        for case in cases
            r = parser(read(joinpath(root,case),String))
            println(io,name," ",case," ",Int(r.success)," ",r.n_rbm_idx," ",Int(r.is_complex_flag)," ",r.line_number)
            println(io,r.error_message)
            println(io,join(["$idx:$flag" for (idx,flag) in sort(collect(r.opt_flags))]," "))
            if r.terms === nothing; println(io,"-1")
            else
                println(io,length(r.terms))
                for t in r.terms
                    coords = [getfield(t,field) for field in fieldnames(typeof(t)) if !(field in (:value,:is_complex,:idx))]
                    println(io,join(vcat(coords,[t.idx,Int(t.is_complex)])," ")," ",hex([t.value]))
                end
            end
        end
    end
    verify("parser.txt",String(take!(io)))
    io=IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=11272; exact bits + next 624 UInt32 words")
    for case in ("all","reverse","rbm_first","no_flags","empty","no_orbital_flags","failed_replacement","empty_replacement","cross_section_flags","complex_gutz","dh24","tied","mixed","failed_first")
        for mode in ("parsed","complex","missing_flags","inactive","truncated","zero_neuron","negative_neuron")
            d = parse_expert_mode_files(joinpath(root,"namelist_$case.def"))
            mode == "complex" && (d.modpara.complex_flag=1)
            mode == "missing_flags" && empty!(d.optimization_flags)
            mode == "inactive" && fill!(d.optimization_flags,false)
            mode == "truncated" && resize!(d.optimization_flags,2)
            if mode in ("zero_neuron","negative_neuron")
                d.modpara.nneuron = mode == "zero_neuron" ? 0 : -7
                d.modpara.nneuron_charge=0; d.modpara.nneuron_spin=0; d.modpara.nneuron_general=0
            end
            # Seed existing values to prove init resets inactive parameters.
            for field in fields, t in getfield(d,field); t.value=7-9im; end
            sizes = [P._rbm_section_nparam(getfield(d,field)) for field in fields]
            @test P.count_rbm_parameters(d) == sum(sizes)
            println(io,case," ",mode)
            println(io,join(sizes," "))
            println(io,join(Int.(d.optimization_flags)," "))
            rng=SFMT19937RNG(); Random.seed!(rng,11272)
            P.init_parameter!(d; rng)
            vals = vcat(P.projection_parameters(d),[t.value for field in fields for t in getfield(d,field)],[t.value for t in d.orbital_terms])
            println(io,hex(vals))
            println(io,join([rand(rng,UInt32) for _ in 1:624]," "))
        end
    end
    verify("initial.txt",String(take!(io)))
end
