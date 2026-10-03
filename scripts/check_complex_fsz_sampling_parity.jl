include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Original complex FSZ sampler, including conduction-spin branch and burn reuse.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("Complex FSZ fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
hex(v) = join(string.(reinterpret(UInt64,collect(reinterpret(Float64,v)));base=16,pad=16)," ")
io = IOBuffer()
println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1")
println(io,"# Original complex FSZ sampler; seed=11272; Julia-mVMC 8bb1b9e, numerical sources c2ea432")
@testset "Complex FSZ complete sampling state" begin
    for (name,local_spin,two_sz,path) in (("conduction",false,-1,0),("fixed_sz",false,0,0),("local",true,-1,2),("cancelled",false,-1,0),("failure",false,-1,0)), calls in (name == "failure" ? (1,) : (1,2,3,10))
        data = MVMCExpertModeParsers.ExpertModeData()
        data.modpara.nsite = 4; data.modpara.nelec = 2; data.modpara.nmp_trans = 2
        data.modpara.nvmc_warmup = 0; data.modpara.nvmc_sample = 1; data.modpara.nvmc_interval = 1
        data.modpara.two_sz = two_sz; data.modpara.nex_update_path = path
        data.i_flg_orbital_general = 1; data.complex_flags = [1]
        data.locspin_terms = [MVMCExpertModeParsers.LocSpinTerm(i,Int(local_spin)) for i in 0:3]
        data.n_qp_trans = 2; data.para_qp_trans = ComplexF64[1,-0.375]
        name == "cancelled" && (data.para_qp_trans = ComplexF64[1,-1])
        data.n_gutzwiller_idx = 1; data.gutzwiller_idx = zeros(Int,4)
        data.gutzwiller_terms = [MVMCExpertModeParsers.GutzwillerTerm(0,0.125+0im,false)]
        data.n_jastrow_idx = 1; data.jastrow_idx = [i==j ? -1 : 0 for i in 1:4,j in 1:4]
        data.jastrow_terms = [MVMCExpertModeParsers.JastrowTerm(0,1,-0.2+0im,false)]
        MVMCExpertModeParsers.init_qp_weight!(data)
        state = MVMCOptimizers.VMCOptimizationState(4,2,2,0,2,1,true,true)
        mat = state.slater_matrix
        for qp in 0:1,i in 0:7,j in (i+1):7
            plane = name == "cancelled" ? 0 : qp
            z = name == "failure" ? ComplexF64(NaN,NaN) : ComplexF64(((17i+13j+7plane)%31-15)/7+0.125,((11i+3j+plane)%19-9)/13)
            mat.slater_elm[qp*64+i*8+j+1] = z
            mat.slater_elm[qp*64+j*8+i+1] = -z
        end
        println(io,name," ",calls)
        if calls == 1
            # An independent initial-boundary probe; capture the global SFMT
            # stream before seeding the separate complete prefix run below.
            probe = deepcopy(state); c = probe.electron_config
            probe_rng = SFMT19937RNG(); Random.seed!(probe_rng,11272)
            @test MVMCOptimizers.make_initial_sample_fsz!(c.tmp_ele_idx,c.tmp_ele_cfg,c.tmp_ele_num,c.tmp_ele_proj_cnt,c.tmp_ele_spn,1,3,data,probe,probe_rng) == Int(name == "failure")
            for values in (c.tmp_ele_idx,c.tmp_ele_cfg,c.tmp_ele_num,c.tmp_ele_proj_cnt,c.tmp_ele_spn)
                println(io,join(values," "))
            end
            println(io,hex(probe.slater_matrix.pf_m)); println(io,hex(probe.slater_matrix.inv_m[1:32]))
            println(io,join([rand(probe_rng,UInt32) for _ in 1:624]," "))
        end
        rng = SFMT19937RNG(); Random.seed!(rng,11272)
        for _ in 1:calls; MVMCOptimizers.vmc_make_sample_fsz!(data,state,rng); end
        # The original failure return precedes the driver's counter extension.
        burn_flag = length(state.electron_config.counter) >= 11 ? state.electron_config.counter[11] : 0
        @test burn_flag == Int(name != "failure")
        c = state.electron_config
        for values in (c.ele_idx,c.ele_cfg,c.ele_num,c.ele_proj_cnt,c.ele_spn,
                       c.tmp_ele_idx,c.tmp_ele_cfg,c.tmp_ele_num,c.tmp_ele_proj_cnt,c.tmp_ele_spn,c.burn_ele_idx)
            println(io,join(values," "))
        end
        println(io,join(vcat(c.counter[1:9],burn_flag)," "))
        println(io,hex(mat.pf_m)); println(io,hex(mat.inv_m[1:32]))
        println(io,join([rand(rng,UInt32) for _ in 1:624]," "))
    end
    actual = String(take!(io)); root = joinpath(@__DIR__,"..","tests","fixtures","complex_fsz")
    if "--write" in ARGS; mkpath(root); write(joinpath(root,"sampling.txt"),actual)
    else; @test compare_sampling_text(actual,read(joinpath(root,"sampling.txt"),String)); end
end
