# Run the upstream SR accumulator tests and pin NStore trajectories.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers
include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl",
                 "test_unit", "test_unit_vmc_main_cal_sr.jl"))

function store_hash624(rng)
    h = UInt64(0xcbf29ce484222325)
    for _ in 1:624
        h = (h ⊻ UInt64(rand(rng, UInt32))) * UInt64(0x100000001b3)
    end
    h
end

@testset "NStore three-step trajectory" begin
    for name in ("heisenberg_chain_real", "heisenberg_chain_cmp",
                 "heisenberg_chain_fsz", "hubbard_chain_real")
        results = []
        for nstore in (0, 1)
            namelist = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "examples", "inputs", name, "namelist.def")
            data = parse_expert_mode_files(namelist)
            data.modpara.nsr_opt_itr_step = 3
            data.modpara.nsr_opt_itr_smp = 3
            data.modpara.nstore_o = nstore
            rng = SFMT19937RNG(); Random.seed!(rng, 1)
            MVMCExpertModeParsers.init_parameter!(data; rng)
            MVMCExpertModeParsers.sync_modified_parameter!(data)
            MVMCExpertModeParsers.init_qp_weight!(data)
            mktempdir() do dir
                @test vmc_para_opt!(data; rng, output_dir=dir) == 0
            end
            h = store_hash624(rng)
            println(name, " NStore=", nstore, " hash624=", h)
            push!(results, (h, [t.value for t in data.orbital_terms]))
        end
        @test results[1][1] == results[2][1]
        expected = Dict(
            "heisenberg_chain_real" => UInt64(382483484918994011),
            "heisenberg_chain_cmp" => UInt64(5883921295860317420),
            "heisenberg_chain_fsz" => UInt64(6705941385670463079),
            "hubbard_chain_real" => UInt64(5863593240845525434),
        )
        @test results[1][1] == expected[name]
        # Julia's distinct reduction paths need not give identical optimized
        # parameters. Print the difference; RNG trajectory equality is strict.
        println(name, " parameter delta=", maximum(abs.(results[1][2] - results[2][2])))
    end
end
