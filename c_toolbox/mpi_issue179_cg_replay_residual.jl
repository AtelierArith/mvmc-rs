# Optional retained-C-output operand audit, not a model run or acceptance bound.
using LinearAlgebra, SHA
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: <C-rank-output-directory> <frozen-diagnostic-helper>")
include(ARGS[2])
function records(path)
    result = Dict{String,Vector{String}}()
    for line in eachline(path)
        fields = String.(split(line))
        length(fields) > 1 && !haskey(result, first(fields)) || error("malformed C record")
        result[first(fields)] = fields[2:end]
    end
    result
end
values(rec, key) = parse.(Float64, rec[key])
ranks = [records(joinpath(ARGS[1],"c-rank-$rank.txt")) for rank in 0:3]
n = parse(Int,only(ranks[1]["d:dimension"]))
imaginary = parse(Int,only(ranks[1]["d:imag"]))
n == (imaginary == 1 ? 20 : 10) || error("unexpected active dimension")
for (rank, rec) in enumerate(ranks)
    rec["d:rank"] == [string(rank-1)] && rec["d:world"] == ["4"] &&
        rec["d:width"] == ["1"] && rec["d:samples"] == ["3"] || error("wrong actual C domain")
    for key in ("d:dimension", "d:imag", "d:iterations", "n:weight", "n:shift", "n:tolerance",
                "n:mean", "n:diagonal", "n:gradient", "n:solution", "n:residual", "n:direction")
        rec[key] == ranks[1][key] || error("inconsistent actual global C state $key")
    end
    values(rec,"n:weight") == [12.0] && values(rec,"n:shift") == [1e-5] &&
        values(rec,"n:tolerance") == [1e-10] || error("wrong actual C settings")
end
setprecision(256) do
    gram = zeros(BigFloat,n,n)
    for rec in ranks
        for field in (imaginary == 1 ? ("real-samples","imag-samples") : ("real-samples",))
            sample = values(rec,"n:$field")
            length(sample) == n*3 && all(isfinite,sample) || error("missing/nonfinite C sample plane")
            O = reshape(BigFloat.(sample),n,3)
            gram += O*transpose(O)
        end
    end
    mean, diagonal, gradient, solution, recursive =
        [values(ranks[1],"n:$field") for field in ("mean","diagonal","gradient","solution","residual")]
    all(x -> length(x)==n && all(isfinite,x),(mean,diagonal,gradient,solution,recursive)) || error("malformed C operands")
    covariance = gram/BigFloat(12) - BigFloat.(mean)*transpose(BigFloat.(mean))
    A = covariance + Diagonal(BigFloat(1e-5).*BigFloat.(diagonal))
    metrics = CTestDirectSRCapture.diagnose_operands(A,gradient,solution)
    actual = BigFloat.(gradient) - A*BigFloat.(solution)
    println("scope=ACTUAL_C_FIXED_JULIA_OPERANDS_DIAGNOSTIC_ONLY")
    println("iterations=",only(ranks[1]["d:iterations"])," maxiter=",n," threshold=",1e-10*1e-10*n*n)
    for (name,value) in pairs(metrics)
        println(name,"=",value)
    end
    println("actual_residual_l2=",norm(actual))
    println("recursive_residual_l2=",norm(BigFloat.(recursive)))
    println("recursive_residual_gap_inf=",maximum(abs,actual-BigFloat.(recursive)))
    println("DIAGNOSTIC_ONLY no_solver_status_invented no_convergence_or_forward_allowance")
end
for rank in 0:3
    path = joinpath(ARGS[1],"c-rank-$rank.txt")
    println("input_sha256=",bytes2hex(sha256(read(path)))," ",path)
end
println("helper_sha256=",bytes2hex(sha256(read(ARGS[2]))))
println("auditor_sha256=",bytes2hex(sha256(read(@__FILE__))))
