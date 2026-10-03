# Explicit optional archive audit; no Cargo dependency or Rust oracle.
using LinearAlgebra
include(joinpath(@__DIR__, "ctest_direct_sr_capture.jl"))
length(ARGS) in (1,2) || error("usage: standalone-metrics-archive [prefixes; default1,2,3,20]")
root = abspath(ARGS[1])
prefixes = length(ARGS)==2 ? parse.(Int,split(ARGS[2],',')) : [1,2,3,20]
all(>(0),prefixes) && allunique(prefixes) || error("invalid prefixes")
numbers(path) = parse.(Float64, split(read(path, String)))
cases = 0
maximum_eta = BigFloat(0)
maximum_omega = BigFloat(0)
maximum_condition = 0.0
for model in sort(readdir(root))
    isdir(joinpath(root, model)) || continue
    for steps in prefixes
        case = joinpath(root, model, "step-$steps")
        b = numbers(joinpath(case, "direct-sr-rhs.txt"))
        x = numbers(joinpath(case, "direct-sr-increment.txt"))
        raw = numbers(joinpath(case, "direct-sr-matrix.txt"))
        n = length(b)
        n > 0 && length(x) == n && length(raw) == n*n || error("shape $case")
        all(isfinite, vcat(raw, b, x)) || error("nonfinite $case")
        indices = parse.(Int, split(read(joinpath(case, "direct-sr-active-indices.txt"), String)))
        flags = split(read(joinpath(case, "direct-sr-flags.txt"), String))
        length(indices) == n && issorted(indices) && allunique(indices) || error("mapping $case")
        all(i -> 0 <= i < length(flags) && flags[i+1] in ("1", "true"), indices) || error("flags $case")
        metadata = read(joinpath(case, "direct-sr-metrics.txt"), String)
        occursin("prefix=$steps iteration=$steps dimension=$n status=0", metadata) || error("capture $case")
        A = Matrix(Symmetric(reshape(raw, n, n), :U))
        diagnostic = CTestDirectSRCapture.diagnose((matrix=reshape(raw, n, n), rhs=b),
            (increment=x, status=0))
        eta, omega = diagnostic.normwise_backward_error, diagnostic.componentwise_backward_error
        eta <= 128*n*eps(Float64) || error("existing #190 backward budget failed: $case eta=$eta")
        diagnostic.lambda_min > 0 || error("not positive definite: $case")
        global maximum_eta = max(maximum_eta, eta)
        global maximum_omega = max(maximum_omega, omega)
        global maximum_condition = max(maximum_condition, diagnostic.condition2_estimate)
        global cases += 1
    end
end
cases == 52 || error("incomplete canonical direct captures: $cases")
println("OFFLINE DIRECT SR AUDIT cases=$cases max_normwise_backward_error=$maximum_eta max_componentwise_backward_error=$maximum_omega max_condition2_estimate=$maximum_condition; not Rust model execution")
