# RBM numerical primitives from Julia 1.13.1 Base, without source modifications.
using Test, Random, SFMT, LinearAlgebra
VERSION == v"1.13.1" || error("RBM math fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
hex(x)=string(reinterpret(UInt64,x);base=16,pad=16)
hex(z::Complex)=hex(real(z))*" "*hex(imag(z))
io=IOBuffer()
println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=11272; original Base exp/log/log1p/tanh")
real_values=[-709.0,-178.0,-177.0,-3.0,-2.1,-1.0,-0.75,-0.25,-1e-100,-0.0,0.0,1e-100,0.125,0.5,1.0,2.1,3.0,177.0,178.0,709.0]
phases=[-1e300,-1e100,-1e10,-1e6,-7.0,-2pi,-pi,-pi/2,prevfloat(-pi/2),nextfloat(-pi/2),-pi/4,-0.25,-1e-100,-0.0,0.0,1e-100,0.125,pi/4,prevfloat(pi/2),pi/2,nextfloat(pi/2),pi,2pi,7.0,1e6,1e10,1e100,1e300]
values=[ComplexF64(x,y) for x in real_values for y in phases]
append!(values,ComplexF64[Complex(Inf,0.5),Complex(-Inf,0.5),Complex(NaN,0.0),Complex(NaN,-0.0),
    Complex(floatmax(Float64),0.5),Complex(floatmin(Float64),0.0),Complex(nextfloat(0.0),0.0),Complex(0.0,nextfloat(0.0))])
rng=SFMT19937RNG();Random.seed!(rng,11272)
append!(values,[ComplexF64(8rand(rng)-4,16rand(rng)-8) for _ in 1:2048])
for z in values
    println(io,hex(z)," ",join(hex.((exp(z),log(z),log1p(z),tanh(z)))," "))
end
path=joinpath(@__DIR__,"..","tests","fixtures","rbm","production","math.txt")
actual=String(take!(io))
@testset "Original RBM complex primitives" begin
    if "--write" in ARGS;write(path,actual);else;@test actual==read(path,String);end
end
