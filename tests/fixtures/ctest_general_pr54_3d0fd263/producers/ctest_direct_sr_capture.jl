# Reusable dev-only hooks for serial or active MPI Julia reference runs.
# Loading this module never runs a model, invokes Rust, or writes a fixture.
module CTestDirectSRCapture
using MVMCOptimizers, LinearAlgebra
const BEFORE = Ref{Any}(nothing)
const AFTER = Ref{Any}(nothing)

capture_settings(data) = (seed=data.modpara.rnd_seed,
    steps=data.modpara.nsr_opt_itr_step, window=data.modpara.nsr_opt_itr_smp,
    diagonal_shift=data.modpara.dsr_opt_sta_del, redundant_cut=data.modpara.dsr_opt_red_cut,
    step_dt=data.modpara.dsr_opt_step_dt, nsrcg=data.modpara.nsrcg, nstore=data.modpara.nstore_o)

function before_capture(S, g, mapping, data)
    # Consumers receive owned copies, not mutable reference solver arrays.
    Base.invokelatest(BEFORE[], (
        matrix=copy(S), rhs=copy(g), active_indices=copy(mapping),
        flags=copy(data.optimization_flags), settings=capture_settings(data),
        triangle='U', not_solved=nothing))
    nothing
end
function after_capture(g, info)
    Base.invokelatest(AFTER[], (increment=copy(g), status=info, not_solved=nothing))
    nothing
end

function no_active_capture(data, mapping)
    # Original Julia branch returns success without calling LAPACK. Absence of
    # numerical operands is explicit; do not synthesize an empty solved matrix.
    Base.invokelatest(BEFORE[], (matrix=nothing, rhs=nothing,
        active_indices=copy(mapping), flags=copy(data.optimization_flags),
        settings=capture_settings(data), triangle='U', not_solved="NoActiveComponents"))
    Base.invokelatest(AFTER[], (increment=nothing, status=0, not_solved="NoActiveComponents"))
    nothing
end

"""Diagnose retained actual systems after the reference run; never feed results back."""
function diagnose(before, after; precision_bits=256)
    before.matrix === nothing && error("NotSolved observation has no numerical system to diagnose")
    after.status == 0 || error("actual solve status $(after.status)")
    diagnose_operands(before.matrix, before.rhs, after.increment; precision_bits)
end

"""Diagnose retained operands only; makes no solver-status or convergence claim."""
function diagnose_operands(raw, b, x; precision_bits=256)
    n = length(b)
    n > 0 && size(raw) == (n, n) && length(x) == n || error("incomplete actual system")
    all(isfinite, raw) && all(isfinite, b) && all(isfinite, x) || error("nonfinite actual solve")
    A = Matrix(Symmetric(raw, :U))
    # A may be a high-precision materialization of retained sampled operands.
    # Eigenvalues remain Float64 diagnostics, not certified spectral bounds;
    # residuals below retain A's input precision before BigFloat arithmetic.
    eigenvalues = eigvals(Symmetric(Float64.(A), :U))
    lambda_min, lambda_max = extrema(eigenvalues)
    condition2 = maximum(abs, eigenvalues) / minimum(abs, eigenvalues)
    residual, eta, omega, anorm, bnorm, xnorm = setprecision(precision_bits) do
        bigA, bigb, bigx = BigFloat.(A), BigFloat.(b), BigFloat.(x)
        r = bigA * bigx - bigb
        matrix_norm = maximum(sum(abs, bigA; dims=2))
        rhs_norm, increment_norm = maximum(abs, bigb), maximum(abs, bigx)
        residual_norm = maximum(abs, r)
        denominator = matrix_norm * increment_norm + rhs_norm
        backward = denominator == 0 ? (residual_norm == 0 ? BigFloat(0) : BigFloat(Inf)) : residual_norm/denominator
        denominators = abs.(bigA) * abs.(bigx) + abs.(bigb)
        componentwise = maximum([denominators[i] == 0 ?
            (r[i] == 0 ? BigFloat(0) : BigFloat(Inf)) : abs(r[i])/denominators[i] for i in eachindex(r)])
        (residual_norm, backward, componentwise, matrix_norm, rhs_norm, increment_norm)
    end
    (dimension=n, lambda_min=lambda_min, lambda_max=lambda_max,
        condition2_estimate=condition2, symmetry_discrepancy=maximum(abs, raw-transpose(raw)),
        residual_precision_bits=precision_bits, residual_norm_inf=residual,
        normwise_backward_error=eta, componentwise_backward_error=omega,
        matrix_norm_inf=anorm, rhs_norm_inf=bnorm, increment_norm_inf=xnorm)
end

function install!(before, after;
        source=normpath(joinpath(@__DIR__, "../extern/Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl")))
    VERSION == v"1.13.1" || error("Julia 1.13.1 required")
    body_match = match(r"(?ms)^function stochastic_opt!\(.*?^end\b", read(source, String))
    body_match === nothing && error("direct solve extraction boundary missing")
    body = body_match.match
    for (needle, replacement) in (
            "        potrf!('U', S)" =>
                "        Main.CTestDirectSRCapture.before_capture(S, g, smat_to_para_idx, data)\n        potrf!('U', S)",
            "    ctimer_stop!(c_timer, 57)" =>
                "    Main.CTestDirectSRCapture.after_capture(g, info)\n    ctimer_stop!(c_timer, 57)",
            "    if n_smat == 0\n" =>
                "    if n_smat == 0\n        Main.CTestDirectSRCapture.no_active_capture(data, smat_to_para_idx)\n")
        length(findall(needle, body)) == 1 || error("capture boundary changed: $needle")
        body = replace(body, needle => replacement; count=1)
    end
    BEFORE[], AFTER[] = before, after
    Base.include_string(MVMCOptimizers, body, "ctest_direct_sr_readonly_capture")
    nothing
end
end
