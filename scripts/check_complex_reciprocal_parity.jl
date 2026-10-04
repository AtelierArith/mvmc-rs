include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
using Test
VERSION == v"1.13.1" || error("Reciprocal fixture requires Julia 1.13.1")
path=joinpath(@__DIR__,"..","tests","fixtures","complex_reciprocal.txt")
hex(v)=join(string.(reinterpret.(UInt64,v);base=16,pad=16)," ")
io=IOBuffer()
for z in (1.3+0.0im, 1.3+2.7im, -4.2+0.7im, 1e300+2e299im, 1e-300+2e-299im, Inf-Inf*im, 1e-160+2e-160im)
    v=inv(z); println(io,hex([real(z),imag(z),real(v),imag(v)]))
end
actual=String(take!(io))
if "--write" in ARGS;write(path,actual);else;@test compare_hex_text(actual, read(path,String), (row,col,fields)->col>=3 ? (4*nextfloat(0.0),64*eps(Float64)) : nothing);end
