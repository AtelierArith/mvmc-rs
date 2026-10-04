# Optional diagnostics of retained real CG stopping, not an acceptance bound.
using LinearAlgebra, SHA
base = joinpath(@__DIR__, "ctest_audit_mpi_cg_operator.jl")
# Reuse the frozen operand extraction without invoking its cross-backend audit.
include_string(Main, first(split(read(base, String), "\nfunction audit()")), base)
length(ARGS) == 2 || error("usage: JULIA_CAPTURE_DIR RUST_CAPTURE_DIR")
@assert all(path -> occursin("r4-real-s1-cg1-store0/prefix3", path), ARGS)
setprecision(256) do
    for (label, directory) in zip(("Julia", "Rust"), ARGS)
        s = retained_system(directory, 1e-5, 2)
        rec = records(joinpath(directory, "cg-rank-0.txt"))
        ids = findall(i -> eventkind(rec, i) == 0,
            collect(0:parse(Int, only(rec["d:events"]))-1)) .- 1
        @assert length(ids) == 3
        final = eventkey(ids[3]-1)
        recursive = BigFloat.(values(rec, "n:$final-residual"))
        b = BigFloat.(s.gradient)
        r = b - s.A * BigFloat.(s.solution)
        threshold = (1e-10 * 1e-10) * 10.0 * 10.0
        println("backend=$label step=2 iterations=$(s.iterations) default_max_iter=10 stopping_threshold_squared=$threshold")
        println("recursive_norm_squared=$(sum(abs2, recursive)) actual_norm2=$(norm(r)) gradient_norm2=$(norm(b)) relative_residual2=$(norm(r)/norm(b))")
        println("actual_norm_inf=$(maximum(abs,r)) gradient_norm_inf=$(maximum(abs,b)) relative_residual_inf=$(maximum(abs,r)/maximum(abs,b))")
        println("condition2_estimate=$(s.metrics.condition2_estimate) condition_times_relative_residual2=$(BigFloat(s.metrics.condition2_estimate)*norm(r)/norm(b))")
        println("backward_error=$(s.metrics.normwise_backward_error) condition_times_backward_error=$(BigFloat(s.metrics.condition2_estimate)*s.metrics.normwise_backward_error)")
    end
end
println("DIAGNOSTIC_ONLY iteration_limit_not_convergence no_certified_forward_bound no_new_tolerance")
println("script_sha256=", bytes2hex(sha256(read(@__FILE__))))
println("operand_auditor_sha256=", bytes2hex(sha256(read(base))))
