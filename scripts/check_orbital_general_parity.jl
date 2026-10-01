# Pure General matrix/Slater/derivative fixtures, based on upstream
# test_orbital_qptrans_utils.jl and test_unit_slater_update.jl.
using Test, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
using MVMCExpertModeParsers: build_orbital_sgn_matrix!
VERSION == v"1.13.1" || error("General fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
root = joinpath(@__DIR__, "..", "tests", "fixtures", "orbital_general")
hex(v) = join(string.(reinterpret(UInt64, collect(reinterpret(Float64, v))); base=16, pad=16), " ")
rowmajor(m) = vec(permutedims(m))
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
println(io, "# Julia-mVMC 8bb1b9e (numerical sources c2ea432), row-major orbital matrices")
@testset "Pure General orbital layout" begin
    for boundary in (-1, 1), kind in ("general", "ap_parallel", "sparse", "cached")
        input = kind == "cached" ? "general" : kind
        data = parse_expert_mode_files(joinpath(root, "namelist_$(input).def"))
        @test size(data.orbital_idx_matrix) == (6, 6)
        @test data.modpara.nelec == 2
        data.modpara.nmp_trans = 2boundary
        build_orbital_sgn_matrix!(data)
        if kind == "cached"
            data.orbital_idx_matrix[1, 2] = data.orbital_idx_matrix[2, 1] = 4
            data.orbital_sgn[1, 2] = -1; data.orbital_sgn[2, 1] = 1
        end
        for term in data.orbital_terms
            term.value = ComplexF64((term.idx + 1) / 7, (term.idx % 3 - 1) / 5)
        end
        data.para_qp_trans = ComplexF64[1, -0.375]
        data.n_qp_trans = 2
        data.qp_trans = [[0, 1, 2], [1, 2, 0]]
        data.qp_trans_inv = [[0, 1, 2], [2, 0, 1]]
        data.qp_trans_sgn = boundary > 0 ? [ones(Int, 3), ones(Int, 3)] : [[1, 1, 1], [-1, 1, -1]]
        data.qp_opt_trans = [[0, 1, 2]]; data.qp_opt_trans_sgn = [ones(Int, 3)]
        MVMCExpertModeParsers.init_qp_weight!(data)
        # Source normalizes the count but retains the parsed boundary signs.
        data.modpara.nmp_trans = 2
        n = data.modpara.n_orbital_idx
        state = MVMCOptimizers.VMCOptimizationState(3, 2, 0, n, 2, 1, true, true)
        pf = ComplexF64[0.75 - 0.2im, -0.125 + 0.375im]
        inverse = ComplexF64[ComplexF64((i % 7 - 3) / 11, (i % 5 - 2) / 13) for i in 0:31]
        state.slater_matrix.pf_m .= pf
        state.slater_matrix.inv_m[1:32] .= inverse
        MVMCOptimizers.update_slater_elm_fsz!(data, state)
        o = zeros(ComplexF64, 2n)
        ip = 1.25 - 0.75im
        MVMCOptimizers.slater_elm_diff_fsz!(o, ip, [0, 1, 1, 2], [0, 0, 1, 1], data, state)
        println(io, kind, " ", boundary, " ", n)
        println(io, join(rowmajor(data.orbital_idx_matrix), " "))
        println(io, join(rowmajor(data.orbital_sgn), " "))
        println(io, hex(pf)); println(io, hex(inverse)); println(io, hex(ComplexF64[ip]))
        println(io, hex(state.slater_matrix.slater_elm)); println(io, hex(o))
        if kind == "general"
            global general_idx = copy(data.orbital_idx_matrix)
            global general_sgn = copy(data.orbital_sgn)
            global general_slater = copy(state.slater_matrix.slater_elm)
            global general_o = copy(o)
        elseif kind == "ap_parallel"
            @test data.orbital_idx_matrix == general_idx
            @test data.orbital_sgn == general_sgn
            @test reinterpret(UInt64, state.slater_matrix.slater_elm) == reinterpret(UInt64, general_slater)
            @test reinterpret(UInt64, o) == reinterpret(UInt64, general_o)
        end
    end
    path = joinpath(root, "matrices.txt")
    result = String(take!(io))
    if "--write" in ARGS
        write(path, result)
    else
        @test result == read(path, String)
    end
end
