include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Julia 1.13.1 Base complex division, used by Green ratios.
using Test
VERSION == v"1.13.1" || error("Complex division fixture requires Julia 1.13.1")
io = IOBuffer()
for (z,w) in ((ComplexF64(83.70445651679867,0),ComplexF64(92.70224199043041,0)),
             (3.0+4im,2.0-7im),(1e200+1e199im,1e200-1e199im),
             (1e-200+2e-200im,3e-200-4e-200im),(1.0+2im,1e-300+1e-310im))
    v = z/w
    println(io, join(string.(reinterpret.(UInt64,[real(z),imag(z),real(w),imag(w),real(v),imag(v)]);base=16,pad=16)," "))
end
actual = String(take!(io))
path = joinpath(@__DIR__,"..","tests","fixtures","complex_division.txt")
if "--write" in ARGS
    write(path,actual)
else
    @test compare_hex_text(actual, read(path,String), (row,col,fields)->col>=5 ? (4*nextfloat(0.0),64*eps(Float64)) : nothing)
end
