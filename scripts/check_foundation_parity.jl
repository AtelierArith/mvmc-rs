# Reference checks for the Rust foundation parity tests.
# Run with Julia 1.13.1 and a project containing the pinned Julia-mVMC v0.5.0
# workspace packages (c2ea432785bc14364a3cd5e9eef44db464289cc9).
# No fixture regeneration or upstream source mutation is performed.
# References: test_unit_types.jl, test_unit_parallel.jl, and
# MVMCExpertModeParsers.jl/test/test_orbital_qptrans_utils.jl.

using MVMCExpertModeParsers: ExpertModeData, OrbitalTerm, GutzwillerTerm, JastrowTerm
using MVMCOptimizers
function model()
    d = ExpertModeData()
    d.orbital_terms = [OrbitalTerm(0, 1, 0, 1.0 + 0.0im, false, 1)]
    d
end
cases = [
    ("real", d -> nothing, false),
    ("gutz_flag", d -> push!(d.gutzwiller_terms, GutzwillerTerm(0, 0.0 + 0.0im, true)), true),
    ("jast_flag", d -> push!(d.jastrow_terms, JastrowTerm(0, 1, 0.0 + 0.0im, true)), true),
    ("orb_imag", d -> (d.orbital_terms[1].value = 1.0 + 0.5im), true),
    ("gutz_imag", d -> push!(d.gutzwiller_terms, GutzwillerTerm(0, 0.0 + 0.5im, false)), true),
    ("jast_imag", d -> push!(d.jastrow_terms, JastrowTerm(0, 1, 0.0 + 0.5im, false)), true),
    ("explicit_zero", d -> (d.complex_flags = [0,0]; d.orbital_terms[1].value = 1.0 + 0.5im), false),
    ("explicit_nonzero", d -> (d.complex_flags = [0,-1]), true),
    ("modpara_only", d -> (d.modpara.complex_flag = 1), false),
]
for (name, setup!, expected) in cases
    d = model(); setup!(d)
    complex = MVMCOptimizers.get_all_complex_flag(d)
    @assert complex == expected
    size = 1 + 1 + length(d.gutzwiller_terms) + length(d.jastrow_terms)
    sr = MVMCOptimizers.SROptData(size, 1, complex)
    @assert length(sr.sr_opt_oo_real) == (complex ? 0 : size*(size+2))
    println(name, " complex=", complex, " OO=", length(sr.sr_opt_oo), " OO_real=", length(sr.sr_opt_oo_real))
end


using MVMCOptimizers, SFMT, Random
ctx = MVMCOptimizers.serial_context()
@assert MVMCOptimizers.resolve_rnd_seed(ctx, 11272, nothing) == 11272
@assert MVMCOptimizers.resolve_rnd_seed(ctx, 0, nothing) == 0
@assert MVMCOptimizers.resolve_rnd_seed(ctx, 123, nothing) == 123
@assert MVMCOptimizers.resolve_rnd_seed(ctx, -1, 777) == 777
before = floor(Int, time())
timed = MVMCOptimizers.resolve_rnd_seed(ctx, -1, nothing)
@assert before <= timed <= floor(Int, time())
for seed in (-1, Int(typemax(UInt32)) + 1)
    rejected = false
    try
        Random.seed!(SFMT19937RNG(), seed)
    catch error
        @assert error isa InexactError
        rejected = true
    end
    @assert rejected
end
for seed in (11272, 0, 123, 1700000000, 777, 103, 1700000003)
    rng = SFMT19937RNG(); Random.seed!(rng, seed)
    words = [rand(rng, UInt32) for _ in 1:624]
    hash = UInt64(0xcbf29ce484222325)
    for word in words
        hash = (hash ⊻ UInt64(word)) * UInt64(0x100000001b3)
    end
    println(seed, " first8=", words[1:8], " hash624=", hash)
end


using MVMCExpertModeParsers: ExpertModeData, OrbitalTerm, build_orbital_sgn_matrix!, init_qp_weight!
using MVMCOptimizers
for count in (4, -4)
    d = ExpertModeData(); d.modpara.nsite = 4; d.modpara.nmp_trans = count
    d.orbital_terms = [OrbitalTerm(0, 1, 2, 0.3+0.0im, false, -1)]
    build_orbital_sgn_matrix!(d)
    @assert d.orbital_idx_matrix[1,2] == 2
    @assert d.orbital_idx_matrix[2,1] == 0
    @assert d.orbital_sgn[2,1] == (count > 0 ? 1 : 0)
    @assert d.orbital_sgn[1,2] == (count > 0 ? 1 : -1)
    println("normal nmp=", count, " idx=", d.orbital_idx_matrix, " signs=", d.orbital_sgn)
end
for count in (1, -1)
    d = ExpertModeData(); d.modpara.nsite = 2; d.modpara.nelec = 1
    d.modpara.nmp_trans = count; d.modpara.nsp_gauss_leg = 1
    d.modpara.n_orbital_idx = 2; d.i_flg_orbital_general = 1
    d.orbital_terms = [OrbitalTerm(2,3,1,0.3+0.2im,true,-1)]
    build_orbital_sgn_matrix!(d)
    @assert d.orbital_idx_matrix[3,4] == 1
    @assert d.orbital_idx_matrix[4,3] == 1
    @assert d.orbital_idx_matrix[1,4] == 0
    @assert d.orbital_sgn[3,4] == (count > 0 ? 1 : -1)
    @assert d.orbital_sgn[1,4] == (count > 0 ? 1 : 0)
    println("general nmp=", count, " idx=", d.orbital_idx_matrix, " signs=", d.orbital_sgn)
    # Retain the sign-built matrices while performing Julia's runtime normalization.
    d.modpara.nmp_trans = abs(count)
    d.para_qp_trans = [1.0+0.0im]; init_qp_weight!(d)
    d.qp_trans = [[0,1]]; d.qp_trans_inv = [[0,1]]; d.qp_trans_sgn = [[1,1]]
    d.qp_opt_trans = [[0,1]]; d.qp_opt_trans_sgn = [[1,1]]
    state = MVMCOptimizers.VMCOptimizationState(2,1,0,2,1,1,true,true)
    MVMCOptimizers.update_slater_elm_fsz!(d,state)
    println("general slater=", state.slater_matrix.slater_elm)
end
