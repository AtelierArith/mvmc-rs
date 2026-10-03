# Reference for Rust runner_config.rs. Julia 1.13.1, pinned workspace packages.
using Test, MVMCOptimizers
include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
namelist = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "examples", "inputs", "heisenberg_chain_real", "namelist.def")
@testset "runner options and summaries" begin
    for options in ((nsteps=0,mode=:real), (nsteps=-1,mode=:real), (nsteps=1,mode=:wrong), (nsteps=1,mode=:real,nsmp=0), (nsteps=1,mode=:real,nsmp=-2))
        @test_throws ArgumentError run_para_opt_from_namelist("absent.def"; options...)
    end
    @test_throws ArgumentError run_para_opt_from_namelist(namelist;nsteps=1,mode=:real)
    @test_throws ArgumentError run_para_opt_from_namelist(namelist;nsteps=1,mode=:real,nsmp=2)
    @test_throws ArgumentError run_para_opt_from_namelist(namelist;nsteps=1,mode=:real,nsmp=1,initial_def="definitely-missing-initial.def")
    result = run_para_opt_from_namelist(namelist;nsteps=2,mode=:cmp,nsmp=2,seed=1,initial_def=:none)
    try
        @test result.status == 0
        @test result.effective_nsteps == result.effective_nsmp == 2
        @test isabspath(result.output_dir)
        @test length(result.zvo_first_n) == 2
        rows = [parse.(Float64, split(row)) for row in result.zvo_first_n]
        # A two-value sum and division require only a short rounding budget.
        @test ReferenceNumericalComparison.close_values(result.ctest_values,
            [(rows[1][1]+rows[2][1])/2,(rows[1][2]+rows[2][2])/2],
            8*eps(Float64),8*eps(Float64);context="two-row runner summary")
        @test within(result.final_energy_per_site,rows[2][1]/6,8*eps(Float64),8*eps(Float64))
        println("Julia runner rows=", result.zvo_first_n)
        println("Julia ctest=", result.ctest_values, " final/site=", result.final_energy_per_site)
    finally
        rm(result.output_dir;recursive=true)
    end
end
