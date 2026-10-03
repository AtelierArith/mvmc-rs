# Optional developer acquisition. Normal Cargo consumes only checked-in data.
using LinearAlgebra, PfaPack, SHA, Printf, Libdl
VERSION == v"1.13.1" || error("requires Julia1.13.1")
length(ARGS)==1 || error("usage: NEW_EXTERNAL_STAGE")
root=normpath(joinpath(@__DIR__,"../.."))
stage=abspath(only(ARGS))
startswith(stage*"/",root*"/") && error("external stage required")
ispath(stage) && error("fresh output required")
mkpath(stage)
BLAS.set_num_threads(1); ENV["OPENBLAS_NUM_THREADS"]="1"
src=joinpath(root,"extern/mVMC-1.3.0/src")
commands=String[]; sources=String[]; objects=String[]
preflight=String[joinpath(src,"pfapack/fortran",f) for f in
    ("zsktrf.f","zsktf2.f","zlasktrf.f","zskr2.f","zskr2k.f")]
for folder in (joinpath(src,"common"),joinpath(src,"ltl2inv"))
    for (dir,_,files) in walkdir(folder), file in files
        push!(preflight,joinpath(dir,file))
    end
end
append!(preflight,[joinpath(@__DIR__,"factorization_probe.cc"),@__FILE__,
    joinpath(root,"extern/Julia-mVMC/Manifest-v1.13.toml"),
    joinpath(root,"extern/Julia-mVMC/PfaPack.jl/src/ltl_decomposition.jl"),
    joinpath(root,"extern/Julia-mVMC/PfaPack.jl/src/utu2.jl")])
source_snapshot=Dict(p=>bytes2hex(sha256(read(p))) for p in unique(preflight))
execute(c)=(push!(commands,string(c));run(c))
for name in ("zsktrf.f","zsktf2.f","zlasktrf.f","zskr2.f","zskr2k.f")
    path=joinpath(src,"pfapack/fortran",name); object=joinpath(stage,name*".o")
    push!(sources,path); push!(objects,object)
    execute(`gfortran -O0 -ffp-contract=off -c $path -o $object`)
end
wrap=joinpath(src,"ltl2inv/ilaenv_wrap.f90")
object=joinpath(stage,"ilaenv.o"); push!(objects,object);push!(sources,wrap)
execute(`gfortran -O0 -ffp-contract=off -J$stage -c $wrap -o $object`)
common=joinpath(src,"common"); ltl=joinpath(src,"ltl2inv")
driver=joinpath(@__DIR__,"factorization_probe.cc")
probe=joinpath(stage,"factorization-probe")
execute(`c++ -std=c++17 -O0 -ffp-contract=off -DBLAS_EXTERNAL -I$common -I$(joinpath(common,"deps")) -I$ltl $driver $(joinpath(ltl,"ilaenv_lauum.cc")) $objects -lopenblas -lgfortran -o $probe`)
for folder in (common,ltl)
    for (dir,_,files) in walkdir(folder), file in files
        push!(sources,joinpath(dir,file))
    end
end
manifest=joinpath(root,"extern/Julia-mVMC/Manifest-v1.13.toml")
isfile(manifest) || error("pinned manifest absent")
append!(sources,[driver,@__FILE__,manifest,
    joinpath(root,"extern/Julia-mVMC/PfaPack.jl/src/ltl_decomposition.jl"),
    joinpath(root,"extern/Julia-mVMC/PfaPack.jl/src/utu2.jl")])
before=source_snapshot # taken before any compiler invocation
n=6; original=zeros(ComplexF64,n,n)
for j in 2:n, i in 1:j-1
    original[i,j]=complex((3i+2j)/16,(i-j)/32)
    original[j,i]=-original[i,j]
end
original[1,6]=4+2im; original[6,1]=-original[1,6]
stem="complex6_factorized"
writepairs(io,v)=foreach(z->@printf(io,"%.17g %.17g\n",real(z),imag(z)),v)
open(joinpath(stage,stem*".raw.txt"),"w") do io
    println(io,n);writepairs(io,original)
end
raw=read(pipeline(`$probe`,stdin=joinpath(stage,stem*".raw.txt")),String)
write(joinpath(stage,stem*".native.txt"),raw)
tokens=split(raw); native_info=parse(Int,popfirst!(tokens))
native_info==0 || error("native factor failed")
function consume(count)
    z=[complex(parse(Float64,tokens[2i-1]),parse(Float64,tokens[2i])) for i in 1:count]
    deleteat!(tokens,1:2count);z
end
ltlvalues=consume(n*n)
piv=parse.(Int,tokens[1:n]);deleteat!(tokens,1:n)
piv!=collect(1:n) || error("actual nonidentity pivot required")
cout=consume(2n*n+n-1)
parse.(Int,tokens)==piv || error("C inverse pivot mutation/footer")
open(joinpath(stage,stem*".input.txt"),"w") do io
    println(io,"c $n");writepairs(io,original);println(io,join(piv," "))
end
open(joinpath(stage,stem*".factor.txt"),"w") do io
    println(io,native_info);writepairs(io,ltlvalues);foreach(p->println(io,p),piv)
end
open(joinpath(stage,stem*".c.txt"),"w") do io
    writepairs(io,cout);foreach(p->println(io,p),piv)
end
jfactor=copy(original);jp=zeros(Int,n)
info=julia_zsktf2!(jfactor,jp)
info==0 || error("Julia factor failed")
jp==piv || error("Julia/C factor pivots differ")
println("factor_info_C/J=0 pivots=",piv," factor_max_abs=",maximum(abs.(vec(jfactor)-ltlvalues)))
# Same retained native LTL/pivots for the independent Julia inverse comparison.
ja=reshape(copy(ltlvalues),n,n);jm=fill(17+0im,n,n).*1.;jv=fill(19+0im,n-1).*1.;jp=copy(piv)
utu2inv!(n,ja,n,jp,jv,jm,n);jp==piv || error("Julia inverse changed pivots")
jout=vcat(vec(ja),vec(jm),jv)
open(joinpath(stage,stem*".j.txt"),"w") do io
    writepairs(io,jout);foreach(p->println(io,p),piv)
end
open(joinpath(stage,stem*".operator.txt"),"w") do io;writepairs(io,original);end
println("inverse_all_A_M_vT_max_abs_C_J=",maximum(abs.(cout-jout)))
setprecision(256) do
    b=Complex{BigFloat}.(original);a=reshape(Complex{BigFloat}.(cout[1:n*n]),n,n)
    residual=opnorm(b*a-Matrix{Complex{BigFloat}}(I,n,n),Inf)
    println("residual_inf=",residual," backward_eta=",residual/(opnorm(b,Inf)*opnorm(a,Inf)+1),
        " condition_inf_estimate=",opnorm(b,Inf)*opnorm(inv(b),Inf))
end
open(joinpath(stage,"provenance.txt"),"w") do io
    println(io,"Julia=",VERSION," arch=",Sys.ARCH," kernel=",Sys.KERNEL," project=",Base.active_project())
    println(io,"BLAS=",BLAS.get_config()," threads=",BLAS.get_num_threads())
    for lib in BLAS.get_config().loaded_libs
        println(io,lib.libname," SHA256=",bytes2hex(sha256(read(lib.libname))))
        println(io,unsafe_string(ccall(Libdl.dlsym(Libdl.dlopen(lib.libname),:openblas_get_config64_),Cstring,())))
    end
    println(io,read(`c++ --version`,String),read(`gfortran --version`,String),read(`ldd $probe`,String))
    println(io,"native OpenBLAS package=",read(`dpkg-query -W libopenblas0-pthread`,String))
    println(io,"native OpenBLAS SHA256=",bytes2hex(sha256(read("/lib/x86_64-linux-gnu/libopenblas.so.0"))))
    println(io,"native probe SHA256=",bytes2hex(sha256(read(probe))))
    foreach(c->println(io,c),commands)
    for p in sort(collect(keys(before)))
        bytes2hex(sha256(read(p)))==before[p] || error("source changed during acquisition")
        println(io,relpath(p,root)," SHA256=",before[p])
    end
    for file in sort(readdir(stage))
        endswith(file,".txt") && file!="provenance.txt" || continue
        println(io,file," SHA256=",bytes2hex(sha256(read(joinpath(stage,file)))))
    end
end
