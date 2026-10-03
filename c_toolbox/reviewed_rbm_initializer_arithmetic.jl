# Optional first-operation diagnostic; no numerical method replacements.
using SFMT, Random, Printf, InteractiveUtils
VERSION==v"1.13.1" || error("Julia 1.13.1 required")
function scalar(name,value)
    @printf("%s %.18e %016x\n",name,value,reinterpret(UInt64,value))
end
rng=SFMT.SFMT19937RNG()
Random.seed!(rng,12395) # Independent diagnostic process, original SFMT calls.
r1=rand(rng); r2=rand(rng)
radius=1e-2*r1
phase=2.0im*π*r2
unit=exp(phase)
decomposed=radius*unit
actual_expression=ComplexF64(1e-2*r1*exp(2.0im*π*r2))
for (name,value) in (("r1",r1),("r2",r2),("radius",radius),
    ("phase-real",real(phase)),("phase-imag",imag(phase)),
    ("unit-real",real(unit)),("unit-imag",imag(unit)),
    ("decomposed-real",real(decomposed)),("decomposed-imag",imag(decomposed)),
    ("actual-expression-real",real(actual_expression)),("actual-expression-imag",imag(actual_expression)))
    scalar(name,value)
end
s,c=sincos(imag(phase))
scalar("sincos-sin",s); scalar("sincos-cos",c)
println("complex_exp_method=",which(exp,(ComplexF64,)))
println("sincos_method=",which(sincos,(Float64,)))
println("DIAGNOSTIC_ONLY first_operation no_numerical_budget no_full_initialization_RNG_claim")
