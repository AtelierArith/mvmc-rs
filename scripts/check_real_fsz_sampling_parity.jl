# Real FSZ sampler oracle. Kernels and RNG code are the checked-out sources.
# The reference has one nonnumerical API mismatch: it passes a Float64 log
# to save_ele_config_fsz!, which accepts only ComplexF64 and never reads it.
# A source copy converts that unused argument at the save boundary. No
# numerical or RNG operation is replaced; vendored files remain untouched.
using Test, Random, SFMT, LinearAlgebra, MVMCOptimizers, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("Real FSZ sampling fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures", "real_fsz")
hex(v) = join(string.(reinterpret(UInt64, collect(reinterpret(Float64, v))); base=16, pad=16), " ")
function sample_data(case)
    data = MVMCExpertModeParsers.ExpertModeData()
    p = data.modpara
    p.nsite = 4; p.nelec = 2; p.two_sz = case == "fixed" ? 0 : -1
    p.nex_update_path = case == "localspin" ? 2 : (case == "mixed" ? 3 : 1)
    p.nvmc_warmup = 2; p.nvmc_sample = 3; p.nvmc_interval = 2
    data.i_flg_orbital_general = 1
    data.qp_weights = MVMCExpertModeParsers.QuantumProjectionWeights()
    data.qp_weights.qp_full_weight = ComplexF64[0.625, 0.375]
    data.n_gutzwiller_idx = 1; data.n_jastrow_idx = 1
    data.gutzwiller_idx = zeros(Int, 4); data.jastrow_idx = zeros(Int, 4, 4)
    data.gutzwiller_terms = [MVMCExpertModeParsers.GutzwillerTerm(i, 0.13+0im, false) for i in 0:3]
    data.jastrow_terms = [MVMCExpertModeParsers.JastrowTerm(i, j, -0.17+0im, false) for i in 0:3 for j in (i+1):3]
    if case == "localspin"
        data.locspin_terms = [MVMCExpertModeParsers.LocSpinTerm(i, 1) for i in 0:3]
    elseif case == "mixed"
        data.locspin_terms = [MVMCExpertModeParsers.LocSpinTerm(0, 1)]
    end
    return data
end
function sample_state()
    state = MVMCOptimizers.VMCOptimizationState(4, 2, 2, 0, 2, 3, false, true)
    for qp in 0:1, i in 0:7, j in (i+1):7
        z = ComplexF64(((17i + 13j + 7qp) % 31 - 15) / 7 + 0.125, 0)
        state.slater_matrix.slater_elm[qp*64 + i*8+j+1] = z
        state.slater_matrix.slater_elm[qp*64 + j*8+i+1] = -z
    end
    return state
end
src = read(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "src", "vmc_sampling.jl"), String)
a = first(findfirst("function vmc_make_sample_fsz_real!(", src))
b = last(findnext("\nend\n", src, a))
body = replace(src[a:b], "function vmc_make_sample_fsz_real!(" => "function source_real_fsz_sampler!("; count=1)
save = "save_ele_config_fsz!(\n                sample,\n                log_ip_old,"
@test occursin(save, body)
body = replace(body, save => "save_ele_config_fsz!(\n                sample,\n                ComplexF64(log_ip_old),"; count=1)
Base.include_string(MVMCOptimizers, body)
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
println(io, "# Real FSZ source sampler; unused save-log argument converted to ComplexF64")
@testset "Real FSZ sampling prefixes" begin
    for case in ("conduction", "localspin", "mixed", "fixed"), steps in (1, 2, 3, 50)
        data = sample_data(case); state = sample_state()
        rng = SFMT19937RNG(); Random.seed!(rng, 1)
        for step in 1:steps
            MVMCOptimizers.source_real_fsz_sampler!(data, state, rng)
        end
        c = state.electron_config; s = state.slater_matrix
        @test c.counter[11] == 1
        @test sum(c.counter[[2, 4, 6]]) > 0
        @test all(isfinite, s.pf_m_real)
        println(io, case, " ", steps)
        # Julia slot 10 is reserved; Rust's 10th slot stores the burn marker.
        println(io, join(vcat(c.counter[1:9], c.counter[11]), " "))
        for v in (c.ele_idx, c.ele_cfg, c.ele_num, c.ele_proj_cnt, c.ele_spn)
            println(io, join(v, " "))
        end
        println(io, join(c.burn_ele_idx[1:26], " "))
        println(io, hex(s.pf_m_real)); println(io, hex(s.inv_m_real[1:32]))
        println(io, join([rand(rng, UInt32) for _ in 1:624], " "))
    end
end
actual = String(take!(io)); path = joinpath(ROOT, "sampling.txt")
if "--write" in ARGS
    write(path, actual)
else
    @test actual == read(path, String)
end
