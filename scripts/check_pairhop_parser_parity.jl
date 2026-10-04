# Extends canonical test_parsers.jl's PairHop type/parser tests.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("PairHop fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
root = joinpath(@__DIR__, "..", "tests", "fixtures", "pairhop")
hex(x::Float64) = string(reinterpret(UInt64, x); base=16, pad=16)
function verify(name, actual)
    path = joinpath(root,name)
    if "--write" in ARGS
        write(path,actual)
    else
        @test read(path,String) == actual
    end
end
@testset "PairHop directed parser and atomic orchestration" begin
    io = IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1")
    println(io,"# Julia-mVMC 8bb1b9e; parser sources c2ea432; Float64 bits")
    for name in ("parser_cases","raw","replace","invalid")
        result = MVMCExpertModeParsers.parse_pairhop_def(joinpath(root,"$name.def"))
        @test result.success == (name != "invalid")
        println(io,name," ",Int(result.success)," ",length(result.data))
        for t in result.data
            println(io,t.site1," ",t.site2," ",hex(t.value))
        end
        println(io,result.error_message)
        if name == "parser_cases"
            @test length(result.data) == 14
            @test [(t.site1,t.site2) for t in result.data[1:6]] == [(0,1),(1,0),(2,3),(3,2),(2,2),(2,2)]
            @test all(signbit(t.value) for t in result.data[5:6])
            @test result.data[1] == result.data[7]
            @test result.data[9].value == .0125
            @test result.data[end].site1 == 99
        elseif name == "raw"
            @test length(result.data) == 12
            @test [t.value for t in result.data[1:2:end]] == 1:6
        elseif name == "invalid"
            @test length(result.data) == 4
            @test length(split(result.error_message,"; ")) == 6
        end
    end
    verify("parser.txt",String(take!(io)))
    @test isempty(MVMCExpertModeParsers.parse_pairhop_content("# comment\n// comment\n").data)
    @test !MVMCExpertModeParsers.parse_pairhop_def(joinpath(root,"absent.def")).success
    data = parse_expert_mode_files(joinpath(root,"namelist.def"))
    original = MVMCExpertModeParsers.parse_pairhop_def(joinpath(root,"parser_cases.def")).data
    replacement = MVMCExpertModeParsers.parse_pairhop_def(joinpath(root,"replace.def")).data
    @test data.pair_hop_terms == original
    @test parse_expert_mode_files(joinpath(root,"namelist_replace.def")).pair_hop_terms == replacement
    @test parse_expert_mode_files(joinpath(root,"namelist_invalid.def")).pair_hop_terms == replacement
    @test isempty(parse_expert_mode_files(joinpath(root,"namelist_missing.def")).pair_hop_terms)
    @test !MVMCOptimizers.get_all_complex_flag(data)
    plain = deepcopy(data); empty!(plain.pair_hop_terms)
    flags = copy(data.optimization_flags)
    rng = SFMT19937RNG(); Random.seed!(rng,11272)
    MVMCExpertModeParsers.init_parameter!(data; rng)
    words = [rand(rng,UInt32) for _ in 1:624]
    plain_rng = SFMT19937RNG(); Random.seed!(plain_rng,11272)
    MVMCExpertModeParsers.init_parameter!(plain; rng=plain_rng)
    @test data.pair_hop_terms == original
    @test data.optimization_flags == flags
    @test [t.value for t in data.orbital_terms] == [t.value for t in plain.orbital_terms]
    @test words == [rand(plain_rng,UInt32) for _ in 1:624]
    io = IOBuffer()
    println(io,"# seed=11272; real orbital initialization; PairHop does not consume initialization draws")
    println(io,join(Int.(flags)," "))
    println(io,join([hex(x) for t in data.orbital_terms for x in (real(t.value),imag(t.value))]," "))
    println(io,join(words," "))
    verify("initial.txt",String(take!(io)))
end
