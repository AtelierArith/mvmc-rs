# Optional replay of the archived Julia AVX2 reduction contract.
# Validate against the independent, pre-existing Intel Julia oracle first.
using SHA
function archived_avx2_bilinear(inv, a, b, offset, n)
    accum=zeros(6)
    for i in 0:n-1
        lanes=zeros(4)
        for j in 0:n-1
            lane=mod(j,4)+1
            lanes[lane]=fma(inv[offset+i*n+j+1],a[j+1],lanes[lane])
        end
        dot=(lanes[1]+lanes[3])+(lanes[2]+lanes[4])
        lane=mod(i,6)+1
        accum[lane]=fma(b[i+1],dot,accum[lane])
    end
    (accum[5]+(accum[1]+accum[3]))+(accum[6]+(accum[4]+accum[2]))
end
source=read(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCOptimizers.jl","src","vmc_sampling.jl"),String)
a=first(findfirst("function calculate_new_pf_m_two2_real!(",source)); b=last(findnext("\nend\n",source,a))
body=source[a:b]
start=first(findfirst("        bMa = 0.0\n",body)); stop=first(findnext("        # Calculate ratio",body,start))-1
body=body[1:start-1]*"        bMa = Main.archived_avx2_bilinear(inv_m_real,vec_a,vec_b,inv_offset,n_size)\n\n"*body[stop+1:end]
Base.include_string(MVMCOptimizers,body)

lines=filter(l -> !isempty(l) && !startswith(l,"#"),readlines(joinpath(@__DIR__,"..","tests","fixtures","pfaffian_cg","two_hop_bilinear.txt")))
for i in 1:2:length(lines)
    n=parse(Int,lines[i]); row=reinterpret.(Float64,parse.(UInt64,split(lines[i+1]);base=16))
    actual=archived_avx2_bilinear(row[2:1+n*n],row[2+n*n:1+n*n+n],row[2+n*n+n:end],0,n)
    @assert isapprox(actual,row[1];atol=8*n*n*eps(),rtol=8*n*n*eps()) "Archived AVX2 reduction size=$n"
end
println("Verified archived AVX2 reduction against $(length(lines)÷2) independent Intel Julia cases")


