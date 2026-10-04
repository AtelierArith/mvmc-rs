include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Fixed real/complex Slater matrices from the seed=1 reference chain.
# Run with Julia 1.13.1 --project=extern/Julia-mVMC; --write regenerates outputs.
using Test, PfaPack, LinearAlgebra
VERSION == v"1.13.1" || error("Pfaffian fixture requires Julia 1.13.1")
BLAS.set_num_threads(1)
complex_mode = "--complex" in ARGS
path = joinpath(@__DIR__, "..", "tests", "fixtures", "pfaffian_cg", complex_mode ? "complex.txt" : "real.txt")
lines = filter(l -> !startswith(l, "#"), readlines(path))
input_only = length(lines) == 16
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
println(io, complex_mode ? "# First complex Slater matrices, seed=1, 8 QP planes" : "# First real Slater matrices, heisenberg_chain_real seed=1, 8 QP planes")
parse_bits(l) = [reinterpret(Float64, parse(UInt64, s; base=16)) for s in split(l)]
hexes(v) = join(string.(reinterpret(UInt64, vec(v)); base=16, pad=16), " ")
for start in 1:(input_only ? 2 : 6):length(lines)
    n = parse(Int, lines[start])
    input = parse_bits(lines[start+1])
    A = reshape(complex_mode ? reinterpret(ComplexF64, copy(input)) : copy(input), n, n)
    piv = zeros(Int, n)
    @test (complex_mode ? PfaPack.julia_zsktf2_turbo!(A, piv) : PfaPack.julia_dsktf2!(A, piv)) == 0
    println(io, n)
    println(io, hexes(input))
    println(io, join(piv, " "))
    println(io, hexes(vec(A)))
    println(io, hexes([PfaPack.utu2pfa(n, A, n, piv)]))
    PfaPack.utu2inv!(n, A, n, piv, zeros(eltype(A),n-1), zeros(eltype(A),n,n), n)
    println(io, hexes(vec(A)))
end
actual = String(take!(io))
if "--write" in ARGS
    write(path, actual)
else
    numerical(s) = join(filter(l -> !startswith(l, "#"), split(s, '\n')), '\n')
    @test compare_hex_text(actual,read(path,String),(r,c,t)->mod1(r,6)>=4 ? (1e-13,1e-13) : nothing)
end
