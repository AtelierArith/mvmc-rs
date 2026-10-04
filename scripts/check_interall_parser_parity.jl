# Extends upstream test_parsers.jl's term/parser tests and
# test_parse_expert_mode_files.jl's orchestration tests for InterAll.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("InterAll fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
root = joinpath(@__DIR__, "..", "tests", "fixtures", "interall")
hex(x::Float64) = string(reinterpret(UInt64, x); base=16, pad=16)
indices(t) = (t.site0, t.spin0, t.site1, t.spin1, t.site2, t.spin2, t.site3, t.spin3)

function check_fixture(name, result)
    path = joinpath(root, name)
    if "--write" in ARGS
        write(path, result)
    else
        @test result == read(path, String)
    end
end

@testset "InterAll parser and orchestration" begin
    io = IOBuffer()
    println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
    println(io, "# Julia-mVMC 8bb1b9e; parser/numerical sources c2ea432; complex Float64 bits")
    for name in ("parser_cases", "raw", "kitaev", "replace")
        parsed = MVMCExpertModeParsers.parse_interall_def(joinpath(root, "$name.def"))
        @test parsed.success
        terms = parsed.data
        println(io, name, " ", length(terms))
        for t in terms
            println(io, join(indices(t), " "), " ", hex(real(t.value)), " ", hex(imag(t.value)), " ", Int(t.is_complex))
        end
        if name == "parser_cases"
            @test length(terms) == 10
            @test indices(terms[1]) == (3,1,0,0,2,1,1,0)
            @test terms[1].value == -0.375 + 0.625im
            @test terms[1] == terms[3]
            @test signbit(imag(terms[2].value))
            @test !terms[5].is_complex && !terms[6].is_complex
            @test terms[7].is_complex && terms[8].is_complex
            @test indices(terms[9]) == (-1,2,4,-1,0,0,1,1)
            @test signbit(real(terms[9].value))
            @test terms[10].value == 0.125 - 0.5im
        elseif name == "raw"
            @test length(terms) == 6
            @test real.([t.value for t in terms]) == 1:6
        elseif name == "kitaev"
            @test length(terms) == 48
            original = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test", "samples", "Standard", "Spin", "Kitaev", "interall.def")
            normalized = join(rstrip.(split(read(original, String), '\n')), "\n")
            @test read(joinpath(root, "$name.def"), String) == normalized
        end
    end
    check_fixture("parser.txt", String(take!(io)))
    @test isempty(MVMCExpertModeParsers.parse_interall_content("# comment\n===\nNInterAll 1\n").data)
    @test !MVMCExpertModeParsers.parse_interall_def(joinpath(root, "absent.def")).success
    data = parse_expert_mode_files(joinpath(root, "namelist.def"))
    @test data.inter_all_terms == MVMCExpertModeParsers.parse_interall_def(joinpath(root, "parser_cases.def")).data
    replaced = parse_expert_mode_files(joinpath(root, "namelist_replace.def"))
    @test replaced.inter_all_terms == MVMCExpertModeParsers.parse_interall_def(joinpath(root, "replace.def")).data

    @test !MVMCOptimizers.get_all_complex_flag(data)
    original = copy(data.inter_all_terms)
    plain = deepcopy(data); empty!(plain.inter_all_terms)
    rng = SFMT19937RNG(); Random.seed!(rng,11272)
    flags = copy(data.optimization_flags)
    MVMCExpertModeParsers.init_parameter!(data; rng)
    words = [rand(rng,UInt32) for _ in 1:624]
    # SFMT.jl uses a global stream; take the first snapshot before reseeding.
    plain_rng = SFMT19937RNG(); Random.seed!(plain_rng,11272)
    MVMCExpertModeParsers.init_parameter!(plain; rng=plain_rng)
    @test data.inter_all_terms == original
    @test [t.value for t in data.orbital_terms] == [t.value for t in plain.orbital_terms]
    @test data.optimization_flags == flags
    @test words == [rand(plain_rng,UInt32) for _ in 1:624]
    io = IOBuffer()
    println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
    println(io, "# seed=11272; real orbital initialization; InterAll coefficients do not select wavefunction mode")
    println(io, join(Int.(flags), " "))
    println(io, join([hex(x) for t in data.orbital_terms for x in (real(t.value),imag(t.value))], " "))
    println(io, join(words, " "))
    check_fixture("initial.txt", String(take!(io)))
end
