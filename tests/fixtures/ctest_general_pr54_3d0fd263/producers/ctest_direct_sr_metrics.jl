# Optional independent capture: never loaded by Cargo or the vendored reference.
using MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra, SHA, Printf
include(joinpath(@__DIR__, "ctest_direct_sr_capture.jl"))
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 3 || error("usage: external-stage models prefixes")
const METRICS_REPO = normpath(joinpath(@__DIR__, ".."))
const METRICS_STAGE = abspath(ARGS[1])
startswith(METRICS_STAGE * "/", METRICS_REPO * "/") && error("stage outside repository")
const METRICS_MODELS = split(ARGS[2], ',')
const METRICS_PREFIXES = parse.(Int, split(ARGS[3], ','))
const METRICS_CASES = [(model, steps) for model in METRICS_MODELS for steps in METRICS_PREFIXES]
const METRICS_CASE_INDEX = Ref(1)
const METRICS_ITERATION = Ref(0)
const METRICS_BEFORE = Ref{Any}()

function metrics_before(S, g, mapping, data)
    METRICS_ITERATION[] += 1
    METRICS_BEFORE[] = (copy(S), copy(g), copy(mapping), copy(data.optimization_flags))
    nothing
end

function metrics_after(x, info)
    model, steps = METRICS_CASES[METRICS_CASE_INDEX[]]
    iteration = METRICS_ITERATION[]
    info == 0 || error("actual reference solve failed: $model iteration=$iteration info=$info")
    if iteration == steps
        raw, b, mapping, flags = METRICS_BEFORE[]
        # POTRF/POTRS U solve the symmetric matrix defined by its upper triangle.
        A = Matrix(Symmetric(raw, :U))
        solution = copy(x) # Actual overwritten g, before parameter update/sync.
        case = joinpath(METRICS_STAGE, model, "step-$steps")
        mkpath(case)
        for (file, values) in (("direct-sr-matrix.txt", vec(raw)),
                ("direct-sr-rhs.txt", b), ("direct-sr-increment.txt", solution),
                ("direct-sr-active-indices.txt", mapping), ("direct-sr-flags.txt", flags))
            open(joinpath(case, file), "w") do io
                println(io, join(repr.(values), " "))
            end
        end
        # Diagnostics only: never fed back into the reference trajectory.
        diagnostic = CTestDirectSRCapture.diagnose((matrix=raw, rhs=b),
            (increment=solution, status=info))
        open(joinpath(case, "direct-sr-metrics.txt"), "w") do io
            println(io, "model=$model prefix=$steps iteration=$iteration dimension=$(length(b)) status=$info")
            println(io, "capture=actual_Julia_pre_factorization_and_post_substitution_before_update")
            println(io, "matrix=column_major_raw upper_triangle=authoritative symmetric_completion=U")
            println(io, "residual_precision_bits=256 condition=Float64_symmetric_eigenvalue_estimate_not_certificate")
            for name in (:lambda_min, :lambda_max, :condition2_estimate, :symmetry_discrepancy,
                    :matrix_norm_inf, :rhs_norm_inf, :increment_norm_inf,
                    :residual_norm_inf, :normwise_backward_error, :componentwise_backward_error)
                println(io, name, "=", getproperty(diagnostic, name))
            end
        end
        METRICS_CASE_INDEX[] += 1
        METRICS_ITERATION[] = 0
        println("DIRECT SR CAPTURED $model prefix=$steps dimension=$(length(b))")
    end
    nothing
end

source = joinpath(METRICS_REPO, "extern/Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl")
body_match = match(r"(?ms)^function stochastic_opt!\(.*?^end\b", read(source, String))
body_match === nothing && error("direct solve extraction boundary missing")
body = body_match.match
for (needle, replacement) in (
        "        info = _solve_direct_sr!(S, g)" => "        Main.metrics_before(S, g, smat_to_para_idx, data)\n        info = _solve_direct_sr!(S, g)",
        "    ctimer_stop!(c_timer, 57)" => "    Main.metrics_after(g, info)\n    ctimer_stop!(c_timer, 57)")
    length(findall(needle, body)) == 1 || error("capture boundary changed: $needle")
    global body = replace(body, needle => replacement; count=1)
end
Base.include_string(MVMCOptimizers, body, "ctest_direct_sr_observed")
# Reuse the exact canonical runner, C coefficient/counter shim, and required
# native FSZ energy bridge path. This includes all existing trajectory checks.
include(joinpath(@__DIR__, "ctest_prefix_oracle.jl"))
METRICS_CASE_INDEX[] == length(METRICS_CASES) + 1 || error("missing actual direct solves")
open(joinpath(METRICS_STAGE, "direct-sr-provenance.txt"), "w") do io
    println(io, "Actual Julia direct-SR captures; no Rust expectations; not full native-C solver validation")
    println(io, "Julia=$VERSION platform=$(Sys.MACHINE) BLAS=$(BLAS.get_config()) threads=$(BLAS.get_num_threads())")
    for path in (source, @__FILE__, joinpath(@__DIR__, "ctest_prefix_oracle.jl"),
            joinpath(@__DIR__, "ctest_direct_sr_capture.jl"),
            joinpath(METRICS_REPO, "extern/mVMC-1.3.0/src/mVMC/stcopt.c"),
            joinpath(METRICS_REPO, "extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c"))
        println(io, bytes2hex(sha256(read(path))), "  ", relpath(path, METRICS_REPO))
    end
    println(io, "Extraction: complete stochastic_opt! through top-level end; hooks immediately before POTRF U and after POTRS/catch; originals unchanged")
    println(io, "Residuals use stored Float64 A,b,actual x promoted to 256-bit BigFloat; diagnostics do not alter state or RNG")
    println(io, "Original/effective counts, seed, overlays and input hashes: per-case model-settings.txt and inputs.sha256; bridge/backend: provenance.txt")
end
