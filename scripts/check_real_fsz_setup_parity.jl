# FSZ setup parity: explicit spin indexing, QP slices and initial-sample retries.
# Range/sentinel checks follow test_unit_vmc_sampling_qp_split.jl; matrix
# construction follows test_unit_slater_update.jl. Runtime kernels are unmodified.
using Test, Random, SFMT, LinearAlgebra, MVMCOptimizers, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("Real FSZ fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
root = joinpath(@__DIR__, "..", "tests", "fixtures", "real_fsz")
hex(v) = join(string.(reinterpret(UInt64, collect(reinterpret(Float64, v))); base=16, pad=16), " ")
function oracle_data(ns, ne)
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = ns; data.modpara.nelec = ne; data.modpara.two_sz = -1
    return data
end
function sentinels!(state)
    s = state.slater_matrix
    fill!(s.pf_m, 37 + 11im); fill!(s.inv_m, 37 + 11im)
    fill!(s.pf_m_real, 23); fill!(s.inv_m_real, 23)
end
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
println(io, "# Julia-mVMC c2ea432785bc14364a3cd5e9eef44db464289cc9 numerical sources")
@testset "Real FSZ full calculation" begin
    for ne in 1:4, polarized in (false, true), complex_input in (false, true)
        ns = 2ne; nq = 4
        data = oracle_data(ns, ne)
        state = MVMCOptimizers.VMCOptimizationState(ns, ne, 0, 0, nq, 1, false, true)
        sentinels!(state)
        idx = collect(0:(2ne-1))
        spins = polarized ? ones(Int, 2ne) : [isodd(i) ? 1 : 0 for i in 0:(2ne-1)]
        slt = state.slater_matrix.slater_elm
        for qp in 0:(nq-1), i in 0:(2ns-1), j in (i+1):(2ns-1)
            z = ComplexF64(((17i + 13j + 7qp) % 31 - 15) / 7 + 0.125,
                           complex_input ? ((11i + 19j + 3qp) % 23 - 11) / 13 : 0)
            slt[qp*(2ns)^2 + i*2ns + j + 1] = z
            slt[qp*(2ns)^2 + j*2ns + i + 1] = -z
        end
        @test MVMCOptimizers.calculate_m_all_fsz_real!(idx, spins, 2, 4, data, state) == 0
        s = state.slater_matrix
        @test s.pf_m_real[[1, 4]] == [23, 23]
        @test s.pf_m[[1, 4]] == [37+11im, 37+11im]
        @test all(isfinite, s.inv_m_real)
        before = deepcopy(s)
        @test MVMCOptimizers.calculate_m_all_fsz_real!(idx, spins, 3, 3, data, state) == 0
        @test s.pf_m == before.pf_m && s.inv_m == before.inv_m
        println(io, ns, " ", ne, " ", Int(polarized), " ", Int(complex_input))
        println(io, join(idx, " ")); println(io, join(spins, " ")); println(io, hex(slt))
        println(io, hex(s.pf_m)); println(io, hex(s.inv_m[1:nq*(2ne)^2]))
        println(io, hex(s.pf_m_real)); println(io, hex(s.inv_m_real[1:nq*(2ne)^2]))
    end
end
@testset "Real FSZ failure publishes no partial QP results" begin
    data = oracle_data(3, 1)
    state = MVMCOptimizers.VMCOptimizationState(3, 1, 0, 0, 4, 1, false, true)
    sentinels!(state)
    s = state.slater_matrix
    for qp in 0:3
        s.slater_elm[qp*36+6] = 1.25; s.slater_elm[qp*36+31] = -1.25
    end
    s.slater_elm[73:108] .= ComplexF64(NaN, 0)
    before = deepcopy(s)
    @test MVMCOptimizers.calculate_m_all_fsz_real!([0, 2], [0, 1], 2, 4, data, state) == 3
    @test s.pf_m == before.pf_m && s.inv_m == before.inv_m
    @test s.pf_m_real == before.pf_m_real && s.inv_m_real == before.inv_m_real
end
@testset "Real FSZ initial-sample failure and retry" begin
    for kind in ("retry", "failure", "zero", "localspin", "magnetized")
        data = oracle_data(3, 1)
        state = MVMCOptimizers.VMCOptimizationState(3, 1, 0, 0, 1, 1, false, true)
        sentinels!(state)
        s = state.slater_matrix
        fill!(s.slater_elm, kind == "zero" ? 0im : ComplexF64(NaN, 0))
        if kind == "retry"
            s.slater_elm[0*6+4+1] = 1.25; s.slater_elm[4*6+0+1] = -1.25
            # Initial electrons always have opposite spin; clear their diagonals.
            for i in 0:5; s.slater_elm[i*6+i+1] = 0; end
        end
        if kind in ("localspin", "magnetized")
            for i in 0:5, j in (i+1):5
                z = ComplexF64((i + 2j + 1) / 7, 0)
                s.slater_elm[i*6+j+1] = z; s.slater_elm[j*6+i+1] = -z
            end
            for i in 0:5; s.slater_elm[i*6+i+1] = 0; end
            if kind == "localspin"
                data.locspin_terms = [MVMCExpertModeParsers.LocSpinTerm(0, 1)]
            else
                data.modpara.two_sz = 2
            end
        end
        rng = SFMT19937RNG(); Random.seed!(rng, 1)
        c = state.electron_config
        info = MVMCOptimizers.make_initial_sample_fsz_real!(c.tmp_ele_idx, c.tmp_ele_cfg,
            c.tmp_ele_num, c.tmp_ele_proj_cnt, c.tmp_ele_spn, 1, 2, data, state, rng)
        @test info == (kind == "failure" ? 1 : 0)
        if kind == "retry"
            @test c.tmp_ele_idx == [0, 1]
        elseif kind == "failure"
            @test s.pf_m_real == [23] && all(==(23), s.inv_m_real)
            @test s.pf_m == [37+11im] && all(==(37+11im), s.inv_m)
        elseif kind == "zero"
            @test s.pf_m_real == [0]
            @test all(isnan, s.inv_m_real[1:4])
        end
        println(io, kind, " ", info)
        for v in (c.tmp_ele_idx, c.tmp_ele_cfg, c.tmp_ele_num, c.tmp_ele_proj_cnt, c.tmp_ele_spn)
            println(io, join(v, " "))
        end
        println(io, join([rand(rng, UInt32) for _ in 1:624], " "))
    end
end
actual = String(take!(io)); path = joinpath(root, "setup.txt")
if "--write" in ARGS
    mkpath(root); write(path, actual)
else
    @test actual == read(path, String)
end
