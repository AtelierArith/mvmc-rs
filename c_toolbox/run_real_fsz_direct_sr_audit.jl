# Optional independent C solve of retained reference buffers; never Cargo.
using SHA, LinearAlgebra
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS)==2 || error("usage: reference-stage NEW-external-audit-stage")
root=realpath(ARGS[1]); dest=abspath(ARGS[2]); repo=realpath(dirname(@__DIR__))
startswith(dest*"/",repo*"/") && error("external destination required")
ispath(dest) && error("fresh destination required")
source=joinpath(repo,"extern/mVMC-1.3.0/src/mVMC")
for (file,hash) in (("stcopt.c","43ed8790cff2715284849f0f8906e4645179dcbb100b51f819918b279d6a36f2"),
                    ("stcopt_dposv.c","2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca"))
    bytes2hex(sha256(read(joinpath(source,file))))==hash || error("C source changed")
end
words(path)=split(read(path,String))
function pairs(path)
    v=parse.(Float64,words(path)); iseven(length(v)) && all(isfinite,v) || error("invalid complex buffer")
    ComplexF64.(v[1:2:end],v[2:2:end])
end
pre=joinpath(root,"pre-sr"); para=pairs(joinpath(pre,"parameters.txt")); n=length(para); s=n+1
oo=pairs(joinpath(pre,"sr_oo.txt")); ho=pairs(joinpath(pre,"sr_ho.txt"))
length(oo)==s*(s+2) && length(ho)==s || error("real buffer shape")
all(isreal,oo) && all(isreal,ho) || error("real buffer contains imaginary component")
mask=parse.(Int,words(joinpath(pre,"c-written-mask.txt")))
flags=parse.(Int,words(joinpath(pre,"defined-flags.txt")))
length(mask)==length(flags)==2n && all(x->x in (0,1),mask) || error("flag shape")
modpara=read(joinpath(root,"inputs/modpara.def"),String)
function setting(name)
    matches=collect(eachmatch(Regex("(?m)^"*name*raw"\s+(\S+)\s*$"),modpara))
    length(matches)==1 || error("ambiguous setting $name")
    parse(Float64,only(matches).captures[1])
end
setting("NSRCG")==0 || error("direct SR required")
delta=setting("DSROptStaDel"); dt=setting("DSROptStepDt"); cut=setting("DSROptRedCut")
mkdir(dest)
function header(file,key)
    a=collect(eachmatch(Regex("(?m)^"*key*raw"\s+(\d+)\s*$"),read(joinpath(root,"inputs",file),String)))
    length(a)==1 || error("independent header $file")
    parse(Int,only(a).captures[1])
end
ng=header("gutzwilleridx.def","NGutzwillerIdx"); nj=header("jastrowidx.def","NJastrowIdx")
# Mixed AP/P C layout adds same-spin up/down blocks separately.
ns=header("orbitalidx.def","NOrbitalIdx")+2header("orbitalidxpara.def","NOrbitalIdx")
ng+nj+ns==n || error("independent C parameter declarations")
parameter_source=read(joinpath(source,"parameter.c"),String)
bytes2hex(sha256(parameter_source))=="46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0" || error("parameter source changed")
start=first(findfirst("void SyncModifiedParameter(",parameter_source))
stop=first(findlast("\n#endif",parameter_source))-1
license=parameter_source[1:last(findfirst("*/",parameter_source))]
excerpt=joinpath(dest,"real_fsz_sync_upstream.inc")
write(excerpt,license*"\n"*parameter_source[start:stop]*"\n")
input=joinpath(dest,"input.txt")
open(input,"w") do f
    println(f,"$n $delta $dt $cut")
    println(f,"$ng $nj $ns")
    for i in eachindex(flags); println(f,"$(mask[i]) $(flags[i])"); end
    for a in (para,oo,ho), z in a; println(f,"$(repr(real(z))) $(repr(imag(z)))"); end
end
binary=joinpath(dest,"real-sr"); adapter=joinpath(@__DIR__,"real_fsz_direct_sr_audit.c")
compiler=`gcc -std=gnu11 -O0 -ffp-contract=off -I$dest -I$source -I$(joinpath(source,"include")) $adapter -llapack -lblas -lm -o $binary`
run(compiler)
outputs=String[]
for unknown in (0,1)
    result=read(addenv(`$binary $input $unknown`,"OPENBLAS_NUM_THREADS"=>"1"),String)
    write(joinpath(dest,"unknown-$unknown.stdout.txt"),result); push!(outputs,result)
end
numerical_lines(output)=filter(line->!startswith(line,"sr_info "),split(output,'\n'))
numerical_lines(outputs[1])==numerical_lines(outputs[2]) || error("unknown imaginary flags affect actual C solve")
records=Dict{String,Vector{String}}()
for line in split(outputs[1],'\n';keepempty=false)
    a=split(line); haskey(records,a[1]) && error("duplicate C record")
    records[a[1]]=a[2:end]
end
only(records["status"])=="0" && only(records["lapack_info"])=="0" || error("C solve failed")
function vector(name)
    a=records[name]; k=parse(Int,a[1]); v=parse.(Float64,a[2:end])
    length(v)==k && all(isfinite,v) || error("C vector $name")
    v
end
b=vector("rhs"); x=vector("increment"); A=reshape(vector("matrix"),length(b),length(b))
# Defined real components; original C diagonal threshold selection, for diagnostics.
diag=[real(oo[(i+1)*s+i+2])-real(oo[i+2])^2 for i in 0:n-1]
threshold=maximum(vcat(diag,zeros(n)))*cut
active=[2i for i in 0:n-1 if flags[2i+1]==1 && diag[i+1]>=threshold]
length(active)==length(b) || error("active C selection does not match defined real components")
actual_active=parse.(Int,records["active"][2:end])
parse(Int,records["active"][1])==length(active) && actual_active==active || error("actual C active mapping differs")
include(joinpath(@__DIR__,"ctest_direct_sr_capture.jl"))
metrics=CTestDirectSRCapture.diagnose_operands(A,b,x)
jfinal=pairs(joinpath(root,"final/parameters.txt"))
cvalues=parse.(Float64,records["parameters"][2:end]); cfinal=ComplexF64.(cvalues[1:2:end],cvalues[2:2:end])
length(cfinal)==length(jfinal)==n || error("final width")
summary=IOBuffer()
println(summary,"active_components0=",join(active,","))
println(summary,"actual_c_info=0 unknown_flag_0_1_identical=true")
println(summary,"unknown_flag_metadata_opt_cut_counts_differ=true (original C diagnostics retained)")
println(summary,"metrics=",metrics)
println(summary,"c_vs_reference_final_max_abs=",maximum(abs.(cfinal.-jfinal)))
open(joinpath(dest,"provenance.txt"),"w") do f
    println(f,"reference_stage=$root\ncommand=$compiler\njulia=$(VERSION)\nblas=",BLAS.get_config())
    println(f,read(`gcc --version`,String)); println(f,read(`ldd $binary`,String))
    for line in split(read(`ldd $binary`,String),'\n')
        a=split(line); length(a)>=3 && a[2]=="=>" && isfile(a[3]) || continue
        println(f,"linked_library_sha256 ",bytes2hex(sha256(read(a[3])))," ",realpath(a[3]))
    end
    for path in (adapter,@__FILE__,input,binary,excerpt,joinpath(source,"parameter.c"),joinpath(source,"stcopt.c"),joinpath(source,"stcopt_dposv.c"),
                 joinpath(pre,"sr_oo.txt"),joinpath(pre,"sr_ho.txt"),joinpath(pre,"parameters.txt"),
                 joinpath(pre,"c-written-mask.txt"),joinpath(pre,"defined-flags.txt"),joinpath(root,"inputs/modpara.def"),
                 joinpath(root,"final/parameters.txt"),joinpath(@__DIR__,"ctest_direct_sr_capture.jl"))
        println(f,"sha256 ",bytes2hex(sha256(read(path)))," ",path)
    end
end
println(summary,"artifact_stage=$dest")
stdout=String(take!(summary)); write(joinpath(dest,"audit.stdout.txt"),stdout); print(stdout)
