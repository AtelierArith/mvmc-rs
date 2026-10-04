# Scoped complex world-4/prefix-2 retained-operand diagnostic; no expected output.
using LinearAlgebra, SHA
include(joinpath(@__DIR__, "ctest_direct_sr_capture.jl"))

function records(path)
    result = Dict{String,Vector{String}}()
    for (line_number, line) in enumerate(eachline(path))
        fields = String.(split(line))
        isempty(fields) && error("empty record $path:$line_number")
        haskey(result, fields[1]) && error("duplicate key $path:$line_number")
        result[fields[1]] = fields[2:end]
    end
    result
end
values(rec, key) = parse.(Float64, rec[key])
scalar(rec, key) = only(values(rec, key))
eventkey(i) = "event-" * lpad(string(i), 6, "0")
eventkind(rec, i) = parse(Int, only(rec["d:$(eventkey(i))-kind"]))

function retained_system(directory, input_shift, step)
    ranks = [records(joinpath(directory, "cg-rank-$rank.txt")) for rank in 0:3]
    states = [records(joinpath(directory, "rank-$rank.txt")) for rank in 0:3]
    prepared = [findall(i -> eventkind(rec, i) == 0,
        collect(0:parse(Int, only(rec["d:events"]))-1)) .- 1 for rec in ranks]
    @assert all(length(ids) == 2 for ids in prepared)
    first_event = prepared[1][step]
    key = eventkey(first_event)
    mapping = ranks[1]["d:$key-mapping"]
    mean = values(ranks[1], "n:$key-mean")
    diagonal = values(ranks[1], "n:$key-diagonal")
    gradient = values(ranks[1], "n:$key-gradient")
    n = length(mapping)
    @assert n == 20 && length(unique(mapping)) == n && length(mean) == length(diagonal) == length(gradient) == n
    energy_key = step == 1 ? "n:reduced-energy" : "n:step-$(step-1)-reduced-energy"
    weights = [values(rec, energy_key)[1] for rec in states]
    @assert all(w -> isfinite(w) && w > 0 && w == weights[1], weights)
    @assert all(isfinite, mean) && all(isfinite, diagonal) && all(isfinite, gradient)
    raw_gram = zeros(BigFloat, n, n)
    sample_count = 0
    for (rank, rec) in enumerate(ranks)
        rank_key = eventkey(prepared[rank][step])
        @assert rec["d:$rank_key-mapping"] == mapping
        for field in ("mean", "diagonal", "gradient")
            @assert rec["n:$rank_key-$field"] == ranks[1]["n:$key-$field"]
        end
        real = values(rec, "n:$rank_key-real-samples")
        # Complex C operator requires both retained sample planes.
        imag = values(rec, "n:$rank_key-imag-samples")
        @assert length(real) % n == 0 && length(real) > 0
        @assert length(imag) == length(real)
        @assert all(isfinite, real) && all(isfinite, imag)
        samples = length(real) ÷ n
        @assert samples == 3
        sample_count += samples
        O = reshape(BigFloat.(real), n, samples)
        raw_gram += O * transpose(O)
        if !isempty(imag)
            I = reshape(BigFloat.(imag), n, samples)
            raw_gram += I * transpose(I)
        end
    end
    @assert sample_count == weights[1] == 12 # This retained complex, unit-weight cell.
    mu = BigFloat.(mean)
    covariance = raw_gram / BigFloat(weights[1]) - mu * transpose(mu)
    A = covariance + Diagonal(BigFloat(input_shift) .* BigFloat.(diagonal))
    last_event = step == 2 ? parse(Int, only(ranks[1]["d:events"]))-1 : prepared[1][step+1]-1
    @assert eventkind(ranks[1], last_event) == 3
    final_key = eventkey(last_event)
    solution = values(ranks[1], "n:$final_key-solution")
    recorded_residual = values(ranks[1], "n:$final_key-residual")
    @assert length(solution) == length(recorded_residual) == n
    @assert all(isfinite, solution) && all(isfinite, recorded_residual)
    # This helper reports operands only: no invented CG status/convergence field.
    metrics = CTestDirectSRCapture.diagnose_operands(A, gradient, solution)
    residual = BigFloat.(gradient) - A * BigFloat.(solution)
    residual_gap = maximum(abs, residual - BigFloat.(recorded_residual))
    alpha = Float64[]
    dq = BigFloat[]
    for i in first_event:last_event
        kind = eventkind(ranks[1], i)
        ek = eventkey(i)
        if kind == 2
            if haskey(ranks[1], "n:$ek-alpha")
                append!(alpha, values(ranks[1], "n:$ek-alpha"))
            else
                @assert only(ranks[1]["d:$ek-iteration"]) == "0"
            end
        elseif kind == 1 && only(ranks[1]["d:$ek-phase"]) == "3"
            d = values(ranks[1], "n:$ek-search")
            q = values(ranks[1], "n:$ek-product")
            @assert length(d) == length(q) == n
            @assert all(isfinite, d) && all(isfinite, q)
            push!(dq, sum(BigFloat.(d) .* BigFloat.(q)))
        end
    end
    @assert !isempty(alpha) && !isempty(dq) && all(isfinite, alpha)
    spectrum = eigen(Symmetric(Float64.(A), :U))
    cov_spectrum = eigvals(Symmetric(Float64.(covariance), :U))
    rank_scale = n * eps(Float64) * maximum(abs, cov_spectrum)
    estimated_cov_rank = count(v -> abs(v) > rank_scale, cov_spectrum)
    (A=A, gradient=gradient, solution=solution, metrics=metrics,
        spectrum=spectrum, estimated_cov_rank=estimated_cov_rank,
        covariance_rank_diagnostic_scale=rank_scale, residual_gap=residual_gap,
        max_alpha=maximum(abs, alpha), min_abs_dq=minimum(abs, dq),
        iterations=parse(Int, only(ranks[1]["d:$final_key-iterations"])))
end

function audit()
    length(ARGS) == 3 || error("usage: JULIA_CAPTURE_DIR RUST_CAPTURE_DIR MODPARA")
    julia_directory, rust_directory, modpara = ARGS
    @assert all(path -> occursin("r4-cmp-s1-cg1-store0", path), ARGS)
    for name in ("orbitalidx.def", "gutzwilleridx.def", "jastrowidx.def")
        fields = split.(readlines(joinpath(dirname(modpara), name)))
        complex = only(row[2] for row in fields if length(row) == 2 && row[1] == "ComplexType")
        # Actual canonical input: complex orbitals, real Gutzwiller/Jastrow.
        @assert complex == (name == "orbitalidx.def" ? "1" : "0")
    end
    input = Dict(fields[1] => fields[2] for fields in split.(readlines(modpara)) if length(fields) == 2)
    @assert parse(Int, input["NSRCG"]) == 1 && parse(Int, input["NStore"]) == 0
    @assert parse(Int, input["NVMCSample"]) == 3
    shift = parse(Float64, input["DSROptStaDel"])
    @assert isfinite(shift) && shift == 1e-5
    @assert parse(Float64, input["DSROptStepDt"]) == 0.01
    @assert parse(Int, input["RndSeed"]) == 1
    setprecision(256) do
        for step in 1:2
            J = retained_system(julia_directory, shift, step)
            R = retained_system(rust_directory, shift, step)
            for (label, s) in (("Julia", J), ("Rust", R))
                println("step=$step backend=$label dimension=$(s.metrics.dimension) iterations=$(s.iterations) covariance_rank_estimate=$(s.estimated_cov_rank) rank_scale=$(s.covariance_rank_diagnostic_scale)")
                println("lambda_min=$(s.metrics.lambda_min) lambda_max=$(s.metrics.lambda_max) condition_estimate=$(s.metrics.condition2_estimate) backward_error=$(s.metrics.normwise_backward_error) componentwise_error=$(s.metrics.componentwise_backward_error)")
                println("actual_residual_inf=$(s.metrics.residual_norm_inf) recursive_residual_gap=$(s.residual_gap) max_abs_alpha=$(s.max_alpha) min_abs_recorded_dq=$(s.min_abs_dq)")
            end
            dx = BigFloat.(R.solution) - BigFloat.(J.solution)
            weak = J.spectrum.vectors[:, argmin(abs.(J.spectrum.values))]
            println("step=$step operator_delta_inf=$(maximum(abs, R.A-J.A)) gradient_delta_inf=$(maximum(abs, R.gradient-J.gradient)) solution_delta_inf=$(maximum(abs, dx)) weak_direction_projection=$(abs(sum(BigFloat.(weak).*dx)))")
        end
    end
    println("DIAGNOSTIC_ONLY no_solver_acceptance no_new_tolerance no_trajectory_claim")
    println("script_sha256=", bytes2hex(sha256(read(@__FILE__))))
    println("input_sha256=", bytes2hex(sha256(read(modpara))))
    println("helper_sha256=", bytes2hex(sha256(read(joinpath(@__DIR__, "ctest_direct_sr_capture.jl")))))
end
audit()
