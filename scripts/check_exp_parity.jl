include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
using Test
VERSION == v"1.13.1" || error("exp fixture requires Julia 1.13.1")
path = joinpath(@__DIR__, "..", "tests", "fixtures", "exp.txt")
values = vcat(collect(-1000:1000) ./ 1000,
              collect(-1500:1500) ./ 2,
              [0.0, -0.0, Inf, -Inf, NaN, 1e300, -1e300,
               prevfloat(709.7827128933841), 709.7827128933841,
               nextfloat(-745.1332191019412), -745.1332191019412])
io = IOBuffer()
println(io, "# Julia 1.13.1 Base exp; Intel macOS 15.8.1; native FMA")
for x in values
    println(io, string(reinterpret(UInt64, x); base=16, pad=16), " ",
            string(reinterpret(UInt64, exp(x)); base=16, pad=16))
end
actual = String(take!(io))
if "--write" in ARGS
    write(path, actual)
else
    @test compare_hex_text(actual, read(path,String), (row,col,fields)->col>=2 ? (4*nextfloat(0.0),64*eps(Float64)) : nothing)
end
