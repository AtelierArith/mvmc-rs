# Julia source unit tests and three-step runner diagnostics.
# Rust/Julia fixed-input CG bit parity is checked by check_sr_cg_fixed_parity.jl
# and crates/mvmc-core/tests/sr_cg.rs. This source-only runner does not establish
# end-to-end Rust numerical parity. check_sr_cg_runner_parity.jl and the
# exact-bit Rust prefix tests cover parameters, samples, full RNG blocks,
# energies, and SRinfo for the real, complex, and FSZ chains.
using Test, MVMCOptimizers
using Random, SFMT, MVMCExpertModeParsers
include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl",
                 "test_unit", "test_unit_stochastic_opt.jl"))

@testset "Rust sampled operator and CG fixtures" begin
    ws = MVMCOptimizers.CGWorkspace(2, 2, true)
    ws.stcOs_real .= [1.0 2.0; 3.0 4.0]
    ws.stcOs_imag .= [2.0 0.0; -1.0 3.0]
    ws.stcO .= [0.25, -0.5]
    ws.sdiag .= [2.0, 3.0]
    z = zeros(2)
    MVMCOptimizers.operate_by_s!(z, [0.5,-1.0], ws, 2, 2, 0.25, 0.1, true)
    @test z ≈ [-1.18125, -7.6125] atol=1e-14

    ws = MVMCOptimizers.CGWorkspace(2, 2, false)
    ws.stcOs_real .= [2.0 0.0; 1.0 1.0]
    ws.g .= [6.0, 4.0]
    @test MVMCOptimizers.stochastic_opt_cg_main!(ws, 2, 2, 1.0, 0.0, 1e-14, 10, false) == 2
    @test ws.x ≈ [1.0, 1.0] atol=1e-14

    ws = MVMCOptimizers.CGWorkspace(2, 2, false)
    @test MVMCOptimizers.stochastic_opt_cg_main!(ws, 2, 2, 1.0, 0.0, 1e-6, 10, false) == 0
    @test ws.x == [0.0,0.0]
    ws.g .= [1.0,0.0]
    @test MVMCOptimizers.stochastic_opt_cg_main!(ws, 2, 2, 1.0, 0.0, 1e-6, 10, false) == 1
    @test ws.x == [0.0,0.0]
    @test MVMCOptimizers.stochastic_opt_cg_main!(ws, 2, 2, 1.0, 0.0, 1e-6, 0, false) == 0
end

@testset "standard CG three-step runner oracle" begin
    for name in ("heisenberg_chain_real","heisenberg_chain_cmp","heisenberg_chain_fsz")
        namelist = joinpath(@__DIR__,"..","extern","Julia-mVMC","examples","inputs",name,"namelist.def")
        data = parse_expert_mode_files(namelist)
        data.modpara.nsr_opt_itr_step = 3
        data.modpara.nsr_opt_itr_smp = 3
        data.modpara.nsrcg = 1
        data.modpara.nstore_o = 0
        rng = SFMT19937RNG(); Random.seed!(rng,1)
        MVMCExpertModeParsers.init_parameter!(data;rng)
        MVMCExpertModeParsers.sync_modified_parameter!(data)
        MVMCExpertModeParsers.init_qp_weight!(data)
        mktempdir() do dir
            @test vmc_para_opt!(data;rng,output_dir=dir) == 0
            @test length(readlines(joinpath(dir,"zvo_SRinfo.dat"))) == 4
            println(name," CG output=",read(joinpath(dir,"zvo_out.dat"),String))
            println(name," CG SRinfo=",read(joinpath(dir,"zvo_SRinfo.dat"),String))
        end
        h = UInt64(0xcbf29ce484222325)
        for _ in 1:624
            h = (h ⊻ UInt64(rand(rng,UInt32))) * UInt64(0x100000001b3)
        end
        println(name," CG hash624=",h)
    end
end
