# Oracle for callback order, sampling-only output and full post-run RNG blocks.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers
import MVMCExpertModeParsers: ExpertModeData, GutzwillerTerm, JastrowTerm, OrbitalTerm
namelist = joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "examples", "inputs", "heisenberg_chain_real", "namelist.def")
function prepared(steps)
    data = parse_expert_mode_files(namelist)
    data.modpara.nsr_opt_itr_step = steps
    data.modpara.nsr_opt_itr_smp = steps
    rng = SFMT19937RNG(); Random.seed!(rng,1)
    MVMCExpertModeParsers.init_parameter!(data;rng)
    MVMCExpertModeParsers.sync_modified_parameter!(data)
    MVMCExpertModeParsers.init_qp_weight!(data)
    return data,rng
end
function hash624(rng)
    h = UInt64(0xcbf29ce484222325)
    for _ in 1:624
        h = (h ⊻ UInt64(rand(rng,UInt32))) * UInt64(0x100000001b3)
    end
    return h
end
@testset "callback and skip_sr contracts" begin
    mktempdir() do dir
        d,rng = prepared(3)
        records = []
        callback = (step,data,energy,info) -> push!(records,(step,copy([t.value for t in data.orbital_terms]),energy,info))
        @test vmc_para_opt!(d;rng,output_dir=dir,callback) == 0
        @test first.(records) == [0,1,2]
        @test records[end][2] == [t.value for t in d.orbital_terms]
        @test all(record[4] == 0 for record in records)
        println("Julia callback three-step hash624=",hash624(rng))
    end
    mktempdir() do dir
        d,rng = prepared(3)
        d.modpara.dsr_opt_step_dt = NaN
        before = [t.value for t in d.orbital_terms]
        records = []
        callback = (step,data,energy,info) -> push!(records,(step,energy,info))
        @test vmc_para_opt!(d;rng,output_dir=dir,callback,skip_sr=true) == 0
        @test length(records) == 1 && records[1][1] == 0 && records[1][3] == 0
        @test [t.value for t in d.orbital_terms] == before
        @test length(readlines(joinpath(dir,"zvo_out.dat"))) == 1
        @test !isfile(joinpath(dir,"zqp_opt.dat"))
        println("Julia sampling-only hash624=",hash624(rng))
    end
    mktempdir() do dir
        d,rng = prepared(3)
        calls = Ref(0)
        callback = (step,data,energy,info) -> begin calls[] += 1; error("callback failed") end
        @test_throws ErrorException vmc_para_opt!(d;rng,output_dir=dir,callback)
        @test calls[] == 1
        @test !isfile(joinpath(dir,"zqp_opt.dat"))
    end
end

@testset "history mapping and snapshot ownership" begin
    data = ExpertModeData()
    data.gutzwiller_terms = [GutzwillerTerm(0,1.0+2.0im,true)]
    data.jastrow_terms = [JastrowTerm(0,1,3.0+4.0im,true)]
    data.orbital_terms = [OrbitalTerm(0,1,idx,ComplexF64(idx+5),true,1) for idx in (1,0,1)]
    state = MVMCOptimizers.VMCOptimizationState(2,1,2,4,1,1,true,false)
    state.energy.etot = -1.0+0.25im
    MVMCOptimizers.store_opt_data!(data,state,2)
    @test length(state.opt_data) == 3
    @test isempty(state.opt_data[1].parameters)
    @test state.opt_data[3].energy == state.energy.etot
    expected = ComplexF64[1+2im,3+4im,6,5,6]
    @test state.opt_data[3].parameters == expected
    data.orbital_terms[1].value = 9.0+0.0im
    MVMCOptimizers.store_opt_data!(data,state,3)
    @test state.opt_data[3].parameters == expected
    @test state.opt_data[4].parameters[3] == 9.0+0.0im
    MVMCOptimizers.store_opt_data!(data,state,0)
    @test length(state.opt_data) == 4
end
