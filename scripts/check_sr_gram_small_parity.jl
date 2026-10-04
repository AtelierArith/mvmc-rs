include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
using Test, LinearAlgebra, MVMCOptimizers
VERSION == v"1.13.1" || error("Gram fixture requires Julia 1.13.1")
BLAS.set_num_threads(1)
path = joinpath(@__DIR__, "..", "tests", "fixtures", "sr_direct", "small_gram.txt")
io = IOBuffer()
println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
hex(v) = join(string.(reinterpret.(UInt64,v);base=16,pad=16), " ")
for (n,k) in vcat([(n,k) for n in 1:3 for k in 1:3], [(1,4),(4,1),(4,3)])
    store = [((i % 7) - 3) * 0.17320508075688773 + (i % 3) * 1e-8 for i in 1:n*k]
    oo = fill(777.0,n*(n+2))
    MVMCOptimizers.finalize_oo_store_real!(oo,store,n,k)
    println(io,n," ",k); println(io,hex(store)); println(io,hex(oo))
end
actual=String(take!(io))
if "--write" in ARGS
    write(path,actual)
else
    @test compare_hex_text(actual,read(path,String),(r,c,t)->mod1(r,3)==3 ? (256*eps(Float64),256*eps(Float64)) : nothing)
end
