# Exact-bit scalar FSZ two-electron proposals and accepted inverse updates.
using Test, LinearAlgebra, MVMCOptimizers, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("Real FSZ move fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
hex(v) = join(string.(reinterpret(UInt64, collect(v)); base=16, pad=16), " ")
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
@testset "Real FSZ two-electron moves" begin
    for ne in 1:4, synthetic in (false, true)
        ns = 2ne; size = 2ne; nq = 2
        data = MVMCExpertModeParsers.ExpertModeData()
        data.modpara.nsite = ns; data.modpara.nelec = ne
        state = MVMCOptimizers.VMCOptimizationState(ns, ne, 0, 0, nq, 1, false, true)
        s = state.slater_matrix
        for qp in 0:1, i in 0:(2ns-1), j in (i+1):(2ns-1)
            z = ((17i + 13j + 7qp) % 31 - 15) / 7 + 0.125
            s.slater_elm[qp*(2ns)^2+i*2ns+j+1] = z
            s.slater_elm[qp*(2ns)^2+j*2ns+i+1] = -z
        end
        s.slater_elm_real .= real.(s.slater_elm)
        idx = collect(0:(size-1)); spins = [i % 2 for i in 0:(size-1)]
        @test MVMCOptimizers.calculate_m_all_fsz_real!(idx, spins, 1, 3, data, state) == 0
        if synthetic
            for qp in 0:1, i in 0:(size-1), j in (i+1):(size-1)
                v = ((17i + 19j + 3qp) % 37 - 18) / 11
                s.inv_m_real[qp*size^2+i*size+j+1] = v
                s.inv_m_real[qp*size^2+j*size+i+1] = -v
            end
            s.pf_m_real .= [0.75, -0.25]
        end
        println(io, ns, " ", ne, " ", Int(synthetic))
        println(io, hex(s.slater_elm_real)); println(io, hex(s.pf_m_real))
        println(io, hex(s.inv_m_real[1:nq*size^2]))
        idx[1], idx[2] = idx[2], idx[1]
        new_pf = zeros(2)
        MVMCOptimizers.calculate_new_pf_m_two2_fsz_real!(0, 0, 1, 1, new_pf, idx, spins, 1, 3, data, state)
        println(io, hex(new_pf))
        MVMCOptimizers.update_m_all_two_fsz_real!(0, 0, 1, 1, 0, 1, idx, spins, 1, 3, data, state)
        println(io, hex(s.pf_m_real)); println(io, hex(s.inv_m_real[1:nq*size^2]))
    end
end
actual = String(take!(io)); path = joinpath(@__DIR__, "..", "tests", "fixtures", "real_fsz", "moves.txt")
if "--write" in ARGS; write(path, actual); else; @test actual == read(path, String); end
# Evaluate the original scalar FSZ bilinear loop on the existing normal-mode
# vectorized fixtures. Sharing inputs distinguishes the operation orders.
src = read(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "src", "vmc_sampling.jl"), String)
a = first(findfirst("function calculate_new_pf_m_two2_fsz_real!(", src))
a = first(findnext("        @inbounds for msi = 0:(n_size-1)\n            tmp = 0.0", src, a))
b = first(findnext("        inv_m_ab =", src, a))
Base.include_string(MVMCOptimizers, """
function source_real_fsz_bilinear(inv_m_real, vec_a, vec_b, n_size)
    inv_offset = 0
    bMa = 0.0
$(src[a:b-1])
    bMa
end
""")
lines = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(joinpath(@__DIR__, "..", "tests", "fixtures", "pfaffian_cg", "two_hop_bilinear.txt")))
io = IOBuffer(); println(io, "# Julia 1.13.1 original FSZ scalar bilinear; inputs in pfaffian_cg/two_hop_bilinear.txt")
@testset "Real FSZ scalar reduction differs from normal vectorized reduction" begin
    differences = 0
    for i in 1:2:length(lines)
        n = parse(Int, lines[i])
        values = reinterpret.(Float64, parse.(UInt64, split(lines[i+1]); base=16))
        actual = MVMCOptimizers.source_real_fsz_bilinear(values[2:1+n*n], values[2+n*n:1+n*n+n], values[2+n*n+n:end], n)
        differences += reinterpret(UInt64, actual) != reinterpret(UInt64, values[1])
        println(io, n, " ", hex([actual]))
    end
    @test differences > 0
end
actual = String(take!(io)); path = joinpath(@__DIR__, "..", "tests", "fixtures", "real_fsz", "bilinear.txt")
if "--write" in ARGS; write(path, actual); else; @test actual == read(path, String); end
# Full proposal-kernel arithmetic cases on the same shared inputs. These
# synthetic tables isolate operation order independently of a walker model.
io = IOBuffer(); println(io, "# Julia 1.13.1 calculate_new_pf_m_two2_fsz_real!; shared bilinear inputs")
for i in 1:2:length(lines)
    n = parse(Int, lines[i]); ns = n; ne = n ÷ 2
    values = reinterpret.(Float64, parse.(UInt64, split(lines[i+1]); base=16))
    data = MVMCExpertModeParsers.ExpertModeData(); data.modpara.nsite = ns; data.modpara.nelec = ne
    state = MVMCOptimizers.VMCOptimizationState(ns, ne, 0, 0, 1, 1, false, true)
    s = state.slater_matrix
    s.inv_m_real[1:n*n] .= values[2:1+n*n]; s.pf_m_real .= 1
    s.slater_elm_real[1:n] .= values[2+n*n:1+n*n+n]
    s.slater_elm_real[2n+1:3n] .= values[2+n*n+n:end]
    new_pf = zeros(1)
    MVMCOptimizers.calculate_new_pf_m_two2_fsz_real!(0, 0, 1, 0, new_pf, collect(0:(n-1)), zeros(Int,n), 1, 2, data, state)
    println(io, n, " ", hex(new_pf))
end
actual = String(take!(io)); path = joinpath(@__DIR__, "..", "tests", "fixtures", "real_fsz", "proposals.txt")
if "--write" in ARGS; write(path, actual); else; @test actual == read(path, String); end
