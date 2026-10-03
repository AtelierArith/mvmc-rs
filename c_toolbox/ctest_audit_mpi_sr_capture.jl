# Offline diagnostics only; never creates expectations or modifies captures.
using LinearAlgebra, SHA
include(joinpath(@__DIR__, "ctest_direct_sr_capture.jl"))
function records(path)
    result=Dict{String,Vector{String}}()
    for (number,line) in enumerate(eachline(path))
        fields=String.(split(line))
        isempty(fields) && error("empty record at $path:$number")
        key=first(fields)
        haskey(result,key) && error("duplicate record key $key at $path:$number")
        result[key]=fields[2:end]
    end
    result
end
function audit()
root=get(ENV, "CTEST_MPI_CAPTURE_ROOT", "/home/vscode/.cache/mvmc/issue179-sr-independent")
total=0; maxeta=BigFloat(0); maxcond=0.0
for world in (2,4), steps in 1:3, rank in 0:world-1
cell=joinpath(root,"r$(world)-width3-prefix$(steps)")
j=records(joinpath(cell,"julia","rank-$(rank).txt"))
for workers in (1,2,4)
r=records(joinpath(cell,"w$(workers)","rust","rank-$(rank).txt"))
@assert j["d:sr-systems"]==r["d:sr-systems"]==[string(steps)]
for idx in 0:steps-1
key="sr-system-"*lpad(string(idx),6,"0")
for field in ("dimension","not-solved","status","active","settings")
@assert j["d:$(key)-$(field)"]==r["d:$(key)-$(field)"]
end
n=parse(Int,only(r["d:$(key)-dimension"]))
@assert n>0
metrics=[]
for rec in (j,r)
status=parse(Int,only(rec["d:$(key)-status"]))
@assert status==0
@assert only(rec["d:$(key)-not-solved"])=="0"
matrix=parse.(Float64,rec["n:$(key)-matrix"])
@assert length(matrix)==n*n
A=reshape(matrix,n,n)
b=parse.(Float64,rec["n:$(key)-rhs"]); x=parse.(Float64,rec["n:$(key)-increment"])
@assert size(A)==(n,n) && length(b)==n && length(x)==n
@assert all(isfinite,A) && all(isfinite,b) && all(isfinite,x)
m=CTestDirectSRCapture.diagnose((matrix=A,rhs=b),(increment=x,status=status))
@assert m.lambda_min>0 && m.normwise_backward_error<=128*n*eps(Float64)
push!(metrics,m)
end
total+=1
maxeta=max(maxeta,metrics[1].normwise_backward_error,metrics[2].normwise_backward_error)
maxcond=max(maxcond,metrics[1].condition2_estimate,metrics[2].condition2_estimate)
if rank==0 && workers==1
deltas=[maximum(abs.(parse.(Float64,j["n:$(key)-$(field)"])-parse.(Float64,r["n:$(key)-$(field)"]))) for field in ("matrix","rhs","increment")]
println("world=",world," prefix=",steps," solve=",idx+1," deltas(A,b,x)=",deltas," condJ/R=",[m.condition2_estimate for m in metrics]," etaJ/R=",[Float64(m.normwise_backward_error) for m in metrics])
end
end
end
end
@assert total==108
println("AUDIT_PASS actual_pairs=",total," max_cond_estimate=",maxcond," max_backward_error=",maxeta)
println("helper_sha256=",bytes2hex(sha256(read(joinpath(@__DIR__, "ctest_direct_sr_capture.jl")))))
end
audit()
