# Evaluate the original @turbo bilinear kernel on saved and additional inputs.
using Test, MVMCOptimizers
VERSION == v"1.13.1" || error("Two-hop fixture requires Julia 1.13.1")
path = joinpath(@__DIR__, "..", "tests", "fixtures", "pfaffian_cg", "two_hop_bilinear.txt")
src = read(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "src", "vmc_sampling.jl"), String)
a = first(findfirst("function calculate_new_pf_m_two2_real!(", src))
a = first(findnext("        bMa = 0.0", src, a))
b = first(findnext("        # Calculate ratio =", src, a))
Base.include_string(MVMCOptimizers, """
function source_two_hop_bilinear(inv_m_real::Vector{Float64}, vec_a::Vector{Float64}, vec_b::Vector{Float64}, n_size::Int)
    inv_offset = 0
$(src[a:b-1])
    bMa
end
""")
lines = filter(l -> !isempty(l) && !startswith(l, "#"), readlines(path))
rows = Any[]
for i in 1:2:length(lines)
    n = parse(Int, lines[i])
    row = reinterpret.(Float64, parse.(UInt64, split(lines[i+1]); base=16))
    push!(rows, (n, row[2:1+n*n], row[2+n*n:1+n*n+n], row[2+n*n+n:end]))
end
if "--write" in ARGS
    filter!(row -> row[1] == 6, rows)
    # Nonbinary inputs exercise all outer tails and several inner SIMD blocks.
    for n in (2, 4, 8, 10, 12, 14, 16, 18, 20, 24, 32, 64), case in 1:3
        inv = [((i*37+case*11)%101 - 50) / (i%7+3) for i in 1:n*n]
        va = [((i*13+case*7)%31 - 15) / (i%5+3) for i in 1:n]
        vb = [((i*23+case*3)%41 - 20) / (i%3+5) for i in 1:n]
        push!(rows, (n, inv, va, vb))
    end
end
io = IOBuffer()
println(io, "# Julia 1.13.1 original calculate_new_pf_m_two2_real!; AVX2, four inner lanes")
for (n, inv, va, vb) in rows
    result = MVMCOptimizers.source_two_hop_bilinear(inv, va, vb, n)
    println(io, n)
    values = vcat(result, inv, va, vb)
    println(io, join(string.(reinterpret.(UInt64, values); base=16, pad=16), " "))
end
actual = String(take!(io))
if "--write" in ARGS
    write(path, actual)
else
    @test actual == read(path, String)
end
