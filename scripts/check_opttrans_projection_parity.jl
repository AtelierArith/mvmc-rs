# OptTrans mapping order and derivative contracts from canonical Julia sources.
using Test, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("OptTrans fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const P = MVMCExpertModeParsers
const O = MVMCOptimizers
const root = normpath(joinpath(@__DIR__, "..", "tests", "fixtures", "opttrans"))
hex(values) = join([string(reinterpret(UInt64, x); base=16, pad=16) for v in values for x in (real(v), imag(v))], " ")
function verify(file, actual)
    path = joinpath(root, file)
    "--write" in ARGS ? write(path, actual) : @test(actual == read(path, String))
end
function model(mode, leg, boundary, mapping)
    d = P.ExpertModeData()
    fsz = mode == "fsz"
    d.modpara = P.ModParaParameters(nsite=4, nelec=1, nmp_trans=2*boundary, nsp_gauss_leg=leg)
    d.i_flg_orbital_general = Int(fsz)
    d.n_qp_trans = 2
    d.n_qp_opt_trans = 3
    dim = fsz ? 8 : 4
    d.modpara.n_orbital_idx = dim * dim
    d.orbital_terms = [P.OrbitalTerm(i, j, i*dim+j, ComplexF64((i*dim+j+1)/16, mode == "real" ? 0.0 : -(i*dim+j)/32), mode != "real", isodd(i+j) ? -1 : 1) for i in 0:dim-1 for j in 0:dim-1]
    P.build_orbital_sgn_matrix!(d)
    # vmc_para_opt! normalizes the marker after parser caches boundary signs.
    # Rust retains the marker and uses its absolute value for dimensions.
    d.modpara.nmp_trans = 2
    d.qp_trans = [[0, 1, 2, 3], [3, 2, 1, 0]]
    d.qp_trans_inv = copy.(d.qp_trans)
    d.qp_trans_sgn = boundary == -1 ? [[1, -1, -1, 1], [-1, 1, -1, 1]] : [ones(Int, 4), ones(Int, 4)]
    d.qp_opt_trans = [[0, 1, 2, 3], [1, 2, 3, 0], [3, 2, 1, 0]]
    d.qp_opt_trans_sgn = boundary == -1 ? [[1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]] : [ones(Int, 4) for _ in 1:3]
    d.para_qp_trans = [1.0 + 0.25im, -0.5 - 0.125im]
    d.opt_trans = [0.25 - 0.125im, -0.75 + 0.5im, 0.5 + 0.25im]
    mapping == "absent" && (empty!(d.qp_opt_trans); empty!(d.qp_opt_trans_sgn))
    mapping == "short_maps" && resize!(d.qp_opt_trans, 1)
    mapping == "short_signs" && resize!(d.qp_opt_trans_sgn, 1)
    mapping == "short_maps" && (d.qp_opt_trans[1] = [2, 3, 0, 1])
    mapping == "short_signs" && (d.qp_opt_trans_sgn[1] = [-1, 1, 1, -1])
    mapping == "missing_maps" && empty!(d.qp_opt_trans)
    mapping == "missing_signs" && empty!(d.qp_opt_trans_sgn)
    P.init_qp_weight!(d)
    return d
end
@testset "Canonical OptTrans projection and derivative contracts" begin
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test", "test_slater_update.jl"))
    io = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; Julia-mVMC 8bb1b9e / numerical c2ea432; exact matrix and derivative bits")
    for (mode, leg) in (("real", 1), ("real", 3), ("complex", 1), ("complex", 3), ("fsz", 1)), boundary in (-1, 1), mapping in ("complete", "absent", "short_maps", "short_signs", "missing_maps", "missing_signs")
        d = model(mode, leg, boundary, mapping)
        nq = length(d.qp_weights.qp_full_weight)
        n = d.modpara.n_orbital_idx
        state = O.VMCOptimizationState(4, 1, 0, n+3, nq, 1, mode != "real", mode == "fsz")
        for phase in ("initial", "changed")
            if phase == "changed"
                for t in d.orbital_terms; t.value *= 0.5; end
                d.opt_trans .= [0.5 + 0.25im, -0.125 - 0.75im, 1.5 - 0.5im]
                O.update_qp_weight!(d)
            end
            mode == "fsz" ? O.update_slater_elm_fsz!(d, state) : O.update_slater_elm_fcmp!(d, state)
            println(io, "$mode $leg $boundary $mapping $phase")
            println(io, hex(state.slater_matrix.slater_elm))
            # Deterministic supplied Pfaffians/inverses isolate derivative mapping
            # and reduction order from separate Pfaffian-factorization coverage.
            for qp in 0:nq-1
                state.slater_matrix.pf_m[qp+1] = ComplexF64((qp+1)/8, (2-qp)/16)
                z = ComplexF64((qp+3)/16, (1-qp)/32)
                state.slater_matrix.inv_m[qp*4+1:qp*4+4] .= [0.0+0.0im, z, -z, 0.0+0.0im]
            end
            sr = fill(ComplexF64(7, -9), 2*n)
            ip = 1.25 - 0.75im
            mode == "fsz" ? O.slater_elm_diff_fsz!(sr, ip, [0, 3], [0, 1], d, state) : O.slater_elm_diff_fcmp!(sr, ip, [0, 3], d, state)
            println(io, hex(sr))
            opt = fill(ComplexF64(7, -9), 6)
            O.opt_trans_diff!(opt, ip, d, state)
            println(io, hex(opt))
        end
    end
    verify("projection.txt", String(take!(io)))
    io = IOBuffer()
    println(io, "# Julia $VERSION; bounded OptTrans derivative views and partial Pfaffian vectors")
    for count in (0, 1, 3), fixed in (-1, 0, 2), pfcount in (0, 1, 6), width in (0, 1, 2, 3, 6, 7)
        d = P.ExpertModeData()
        d.opt_trans = fill(1.0 + 0.0im, count)
        if fixed >= 0
            d.qp_weights = P.QuantumProjectionWeights()
            fixed > 0 && (d.qp_weights.qp_fix_weight = [1.0 + 0.25im, -0.5 - 0.125im])
        end
        state = O.VMCOptimizationState(4, 1, 0, 3, 6, 1, true, false)
        state.slater_matrix.pf_m = [ComplexF64((q+1)/8, (2-q)/16) for q in 0:pfcount-1]
        opt = fill(ComplexF64(7, -9), width)
        O.opt_trans_diff!(opt, 1.25 - 0.75im, d, state)
        println(io, "$count $fixed $pfcount $width")
        println(io, hex(opt))
    end
    verify("opt_derivatives.txt", String(take!(io)))
    io = IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; canonical Slater amplitude cutoff and duplicate values")
    for mode in ("real","complex","fsz"), boundary in (-1,1), case in ("zero","negative_zero","below","equal","above","complex_below","complex_above","duplicate_zero","duplicate_tiny")
        d = model(mode,1,boundary,"complete")
        value = case == "negative_zero" ? ComplexF64(-0.0,-0.0) : case == "below" ? ComplexF64(prevfloat(1e-14),0) : case == "equal" ? ComplexF64(1e-14,0) : case == "above" ? ComplexF64(nextfloat(1e-14),0) : case == "complex_below" ? ComplexF64(1e-15,-1e-15) : case == "complex_above" ? ComplexF64(1e-14,-1e-14) : startswith(case,"duplicate") ? ComplexF64(2e-14,0) : ComplexF64(0,0)
        for t in d.orbital_terms; t.value = value; end
        if startswith(case,"duplicate")
            t=deepcopy(d.orbital_terms[2]);t.value = case == "duplicate_zero" ? ComplexF64(0,0) : ComplexF64(1e-15,0)
            push!(d.orbital_terms,t)
        end
        state=O.VMCOptimizationState(4,1,0,d.modpara.n_orbital_idx+3,6,1,mode!="real",mode=="fsz")
        mode=="fsz" ? O.update_slater_elm_fsz!(d,state) : O.update_slater_elm_fcmp!(d,state)
        println(io,"$mode $boundary $case")
        println(io,hex(state.slater_matrix.slater_elm))
    end
    verify("slater_threshold.txt",String(take!(io)))
end
