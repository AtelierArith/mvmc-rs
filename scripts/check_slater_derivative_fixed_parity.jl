include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Fixed first-sample QP buffer from the real Heisenberg chain, Julia 1.13.1.
using Test, MVMCOptimizers
VERSION == v"1.13.1" || error("Slater derivative fixture requires Julia 1.13.1")
path = joinpath(@__DIR__,"..","tests","fixtures","pfaffian_cg","slater_derivative.txt")
lines = filter(l -> !startswith(l,"#"),readlines(path))
n, nqp = parse.(Int,split(lines[1]))
parse_complex(l) = collect(reinterpret(ComplexF64,[reinterpret(Float64,parse(UInt64,s;base=16)) for s in split(l)]))
buffer, weights, ip = parse_complex(lines[2]), parse_complex(lines[3]), only(parse_complex(lines[4]))
actual = zeros(ComplexF64,2n)
MVMCOptimizers._store_slater_sr_opt_o_fast!(actual,buffer,weights,1.0/ip,nqp,n)
expected = parse_complex(lines[5])
@test ReferenceNumericalComparison.close_values(reinterpret(Float64,actual),reinterpret(Float64,expected),1e-13,1e-13;context="Slater derivative")
