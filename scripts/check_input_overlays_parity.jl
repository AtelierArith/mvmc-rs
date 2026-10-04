# Julia v0.5.0 oracle for Rust input_overlays.rs.
using Test, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("Parameter contract verification requires Julia 1.13.1")
include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCExpertModeParsers.jl", "test", "test_read_input_parameters.jl"))

@testset "permissive fallback, duplicates and ordered shared mappings" begin
    mktempdir() do dir
        definition(count, rows) = "===\nNOrbitalIdx $count\nComplexType 1\n===\n===\n$rows"
        path = joinpath(dir, "in.def")
        write(path, definition(3, "0 1 2\n-1 9 9\nbad 3 4\n1 broken 5\n0 6 7 # overwrite\n2 8\n"))
        params = MVMCExpertModeParsers.parse_input_parameter_file(path)
        @test params == Dict(0=>6.0+7.0im,1=>0.0+5.0im)
        @test isempty(MVMCExpertModeParsers.parse_input_parameter_file(joinpath(dir,"missing.def")))
        d = ExpertModeData()
        d.modpara.n_orbital_idx = 5
        d.n_orbital_anti_parallel = 3
        d.i_flg_orbital_anti_parallel = 1
        d.i_flg_orbital_parallel = 1
        d.orbital_terms = [OrbitalTerm(0,1,idx,7.0+0.0im,true,1) for idx in (0,1,3,4,3)]
        write(joinpath(dir,"namelist.def"), "InOrbitalParallel p.def\nInOrbitalGeneral g.def\nInOrbital missing.def\n")
        write(joinpath(dir,"p.def"),definition(2,"1 20 0.2\n0 10 0.1\n"))
        write(joinpath(dir,"g.def"),definition(1,"3 30 0.3\n"))
        MVMCExpertModeParsers.read_input_parameters!(d,joinpath(dir,"namelist.def"))
        @test [t.value for t in d.orbital_terms] == ComplexF64[7,7,30+0.3im,20+0.2im,30+0.3im]
    end
end
