include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Julia 1.13.1 projection coefficients, including the trig kernels in Base.
# Run with --write to regenerate the losslessly encoded fixture.
using MVMCExpertModeParsers, Test, LinearAlgebra
VERSION == v"1.13.1" || error("Projection fixture requires Julia 1.13.1")
path = joinpath(@__DIR__, "..", "tests", "fixtures", "projection_math.txt")
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config())
println(io, "# n index beta quadrature_weight cos_half sin_half cos_sin cos_cos sin_sin qp_weight")
hex(x) = string(reinterpret(UInt64, x); base=16, pad=16)
for n in (1, 2, 3, 4, 8, 16, 32, 64, 128)
    p = MVMCExpertModeParsers.ModParaParameters()
    p.nsp_gauss_leg = n; p.nsp_stot = 0; p.nmp_trans = 1
    w = MVMCExpertModeParsers.QuantumProjectionWeights()
    MVMCExpertModeParsers.init_qp_weight!(w, p, [1.0 + 0.0im])
    beta, quadrature = MVMCExpertModeParsers.gauss_legendre(0.0, Float64(pi), n)
    for i in 1:n
        vals = (beta[i], quadrature[i], real(w.spgl_cos[i]), real(w.spgl_sin[i]),
                real(w.spgl_cos_sin[i]), real(w.spgl_cos_cos[i]), real(w.spgl_sin_sin[i]),
                real(w.qp_full_weight[i]))
        println(io, n, " ", i-1, " ", join(hex.(vals), " "))
    end
end
actual = String(take!(io))
if "--write" in ARGS
    write(path, actual)
else
    numerical(s) = join(filter(l -> !startswith(l, "#"), split(s, '\n')), '\n')
    @test compare_hex_text(actual,read(path,String),(r,c,t)->c>=3 ? (1e-13,1e-13) : nothing)
end

# Reciprocal-then-multiply in the upstream Legendre recurrence is observable.
io = IOBuffer()
println(io, "# Julia 1.13.1 legendre_poly; exponent n, input bits, output bits")
for n in (2, 3, 4, 5, 8, 12), x in (-0.9, -0.3, 0.3, 0.9)
    y = MVMCExpertModeParsers.legendre_poly(x, n)
    println(io, n, " ", hex(x), " ", hex(y))
end
actual = String(take!(io))
path = joinpath(@__DIR__, "..", "tests", "fixtures", "legendre_poly.txt")
if "--write" in ARGS
    write(path, actual)
else
    numerical(s) = join(filter(l -> !startswith(l, "#"), split(s, '\n')), '\n')
    @test compare_hex_text(actual,read(path,String),(r,c,t)->c>=3 ? (1e-13,1e-13) : nothing)
end
