# Run with Julia 1.13.1 and the pinned Julia-mVMC workspace environment.
using Test, MVMCOptimizers, MVMCExpertModeParsers
include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test_unit", "test_unit_read_opt_para.jl"))

@testset "declared sparse projection and shared orbital loading" begin
    mktempdir() do dir
        d = ExpertModeData()
        d.n_gutzwiller_idx = 3
        d.n_jastrow_idx = 2
        d.modpara.n_orbital_idx = 4
        d.gutzwiller_terms = [GutzwillerTerm(i, 7.0+8.0im, true) for i in 0:1]
        d.jastrow_terms = [JastrowTerm(0, 1, 7.0+8.0im, true)]
        d.orbital_terms = [OrbitalTerm(0, 1, i, 7.0+8.0im, true) for i in (0,1,0)]
        path = joinpath(dir, "initial.def")
        record = "1 2 3 4 5 6 " * join(["$(i+10) $(-(i+10)) 99" for i in 0:8], " ")
        write(path, record)
        @test MVMCOptimizers.read_opt_para_file!(d, path) == 9
        loaded = vcat([t.value for t in d.gutzwiller_terms], [t.value for t in d.jastrow_terms], [t.value for t in d.orbital_terms])
        @test loaded == ComplexF64[10-10im,11-11im,13-13im,15-15im,16-16im,15-15im]
        @test MVMCOptimizers.read_initial_def!(d, path)
        @test vcat([t.value for t in d.gutzwiller_terms], [t.value for t in d.jastrow_terms], [t.value for t in d.orbital_terms]) == loaded
    end
end
