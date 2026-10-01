# Canonical strict DH4 definition/layout tests plus exact initialization boundaries.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("DH4 fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCExpertModeParsers.jl","test","test_doublon_holon_parser.jl"))
root = joinpath(@__DIR__,"..","tests","fixtures","dh4")
hex(v) = join(string.(reinterpret(UInt64,collect(reinterpret(Float64,v)));base=16,pad=16)," ")
function verify(name, actual)
    path = joinpath(root,name)
    if "--write" in ARGS; write(path,actual)
    else; @test actual == read(path,String); end
end
@testset "Strict DH4 original parser and initialization" begin
    io = IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1")
    println(io,"# Julia-mVMC 8bb1b9e; parser sources c2ea432; strict runtime parsers; legacy source tests are reference-only")
    names = filter(n -> endswith(n,".def") && !startswith(n,"namelist_") && !(n in ("modpara.def","gutz.def","jast.def","orbital.def","ap_orbital.def","parallel_orbital.def","general_orbital.def")),readdir(root))
    valid = ("complex","real","nonbinary_complex","empty","empty_real","multi","comments","crlf","signed_ignored_indices","arbitrary_header_labels")
    cases = vcat([(replace(n,".def"=>""),3) for n in names],[("empty",typemax(Int)),("complex",0),("complex",-1)])
    for (name,nsite) in cases
        result = MVMCExpertModeParsers.parse_doublon_holon_4site_content(read(joinpath(root,"$name.def"),String),nsite)
        @test result.success == (name in valid && nsite > 0)
        println(io,name," ",nsite," ",Int(result.success)," ",result.line_number)
        println(io,result.error_message)
        if result.success
            d = result.data
            println(io,length(d.indices)," ",Int(d.is_complex))
            for t in d.indices; println(io,join(vec(permutedims(t.neighbors))," ")); end
            println(io,join(Int.(d.opt_flags)," "))
            @test length(d.opt_flags) == 10length(d.indices)
        else
            @test result.data === nothing
        end
    end
    verify("parser.txt",String(take!(io)))
    @test !MVMCExpertModeParsers.parse_doublon_holon_4site_def(joinpath(root,"absent.def"),3).success
    for name in ("invalid","missing","before_modpara","invalid_alias","missing_alias")
        @test_throws ErrorException parse_expert_mode_files(joinpath(root,"namelist_$name.def"))
    end
    io = IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=11272")
    println(io,"# Parsed final layout/flags/mode, projection packing, original initialization and next 624 SFMT words")
    for name in ("orbital_first","dh_first","alias","real","empty","replacement","reverse_replacement","combined_real","only_dh4_complex","empty_last","empty_real","ap_parallel","general")
        data = parse_expert_mode_files(joinpath(root,"namelist_$name.def"))
        layout = projection_layout(data)
        println(io,name)
        println(io,join([getfield(layout,i) for i in 1:fieldcount(typeof(layout))]," "))
        println(io,join(Int.(data.optimization_flags)," "))
        println(io,Int(data.doublon_holon_2site_complex)," ",Int(data.doublon_holon_4site_complex)," ",Int(MVMCOptimizers.get_all_complex_flag(data))," ",data.i_flg_orbital_general," ",data.n_orbital_anti_parallel," ",data.modpara.n_orbital_idx)
        # Parameter storage is independent of neighbor-definition ordering.
        for i in eachindex(data.doublon_holon_2site_params)
            data.doublon_holon_2site_params[i] = ComplexF64(i/8,-i/16)
        end
        for i in eachindex(data.doublon_holon_4site_params)
            data.doublon_holon_4site_params[i] = ComplexF64(i/32,-i/64)
        end
        println(io,hex(projection_parameters(data)))
        rng = SFMT19937RNG(); Random.seed!(rng,11272)
        MVMCExpertModeParsers.init_parameter!(data; rng)
        @test all(iszero,data.doublon_holon_2site_params)
        @test all(iszero,data.doublon_holon_4site_params)
        println(io,hex(vcat(projection_parameters(data),[t.value for t in data.orbital_terms])))
        println(io,join([rand(rng,UInt32) for _ in 1:624]," "))
        if name in ("orbital_first","dh_first","alias")
            @test layout.n_proj == 37 && layout.dh2_offset == 5 && layout.dh4_offset == 17
        end
    end
    verify("initial.txt",String(take!(io)))
    io = IOBuffer()
    println(io,"# Original get_all_complex_flag: DH4 declaration, parameter imaginary value*4, explicit runtime flag (-1=absent), result")
    for declared in (0,1), imaginary in (0,1), flag in (-1,0,2)
        data = MVMCExpertModeParsers.ExpertModeData()
        data.doublon_holon_4site_complex = declared != 0
        data.doublon_holon_4site_params = ComplexF64[ComplexF64(.125,imaginary/4)]
        flag != -1 && (data.complex_flags = [flag])
        mode = MVMCOptimizers.get_all_complex_flag(data)
        @test mode == (flag == -1 ? declared != 0 || imaginary != 0 : flag != 0)
        println(io,"$declared $imaginary $flag ",Int(mode))
    end
    verify("mode.txt",String(take!(io)))
end
