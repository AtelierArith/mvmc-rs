# Julia v0.5.0 oracle for c_timer.rs and the committed zero-time report fixtures.
using Test, MVMCOptimizers
const M = MVMCOptimizers
@testset "C timer environment and disabled operations" begin
    @test M.CTIMER_N == 1000
    t = M.CTimer(false)
    M.ctimer_start!(t,typemax(Int)); M.ctimer_stop!(t,typemax(Int))
    @test all(iszero,t.elapsed_ns) && all(iszero,t.start_ns)
    key = "MVMC_C_TIMER"
    previous = get(ENV,key,nothing)
    try
        pop!(ENV,key,nothing)
        @test !M.ctimer_env_enabled(key)
        for value in ("0","","false","1","00")
            ENV[key] = value
            @test M.ctimer_env_enabled(key) == (value != "0")
        end
    finally
        previous === nothing ? pop!(ENV,key,nothing) : (ENV[key]=previous)
    end
end
@testset "inclusive timing and reset" begin
    t = M.CTimer(true)
    M.ctimer_start!(t,0)
    M.ctimer_start!(t,3)
    M.ctimer_stop!(t,3)
    M.ctimer_stop!(t,0)
    @test t.elapsed_ns[1] >= t.elapsed_ns[4]
    @test M.ctimer_reset!(t) === t
    @test all(iszero,t.elapsed_ns) && all(iszero,t.start_ns)
end
@testset "C timer report bytes" begin
    mktempdir() do dir
        t = M.CTimer(false)
        path = M.write_ctimer_para_opt(t,dir;prefix="custom")
        golden = joinpath(@__DIR__,"..","tests","fixtures","timers","julia_para_opt_zero.dat")
        @test path == joinpath(dir,"custom_CalcTimer.dat")
        @test read(path,String) == read(golden,String)
        path = M.write_ctimer_diag(t,dir;prefix="custom")
        @test read(path,String) == read(joinpath(dirname(golden),"julia_diag_zero.dat"),String)
        t.elapsed_ns[1] = UInt64(9123456789)
        path = M.write_ctimer_para_opt(t,dir;prefix="nonzero")
        @test first(readlines(path)) == "All                         [0]      9.12346"
    end
end
