# Optional developer acquisition; ordinary Cargo never executes this script.
using PfaPack, LinearAlgebra, SHA, Printf, Libdl
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: generate.jl PROBE NEW_STAGE")
probe, stage = abspath.(ARGS)
ispath(stage) && error("output must be new")
mkpath(stage)
BLAS.set_num_threads(1)
root = normpath(joinpath(@__DIR__, "../.."))
manifest = joinpath(root, "extern/Julia-mVMC/Manifest-v1.13.toml")
isfile(manifest) || error("pinned Manifest absent")
open(joinpath(stage, "environment.txt"), "w") do io
    println(io, "Julia=", VERSION, " arch=", Sys.ARCH, " kernel=", Sys.KERNEL)
    println(io, "project=", Base.active_project(), " BLAS=", BLAS.get_config())
    println(io, "Manifest-v1.13.toml sha256=", bytes2hex(sha256(read(manifest))))
    println(io, "probe sha256=", bytes2hex(sha256(read(probe))))
    println(io,"BLAS threads=",BLAS.get_num_threads())
    for lib in BLAS.get_config().loaded_libs
        println(io,"BLAS library=",lib.libname," sha256=",bytes2hex(sha256(read(lib.libname))))
        symbol = lib.interface == :ilp64 ? :openblas_get_config64_ : :openblas_get_config
        println(io,"BLAS version=",unsafe_string(ccall(Libdl.dlsym(Libdl.dlopen(lib.libname),symbol),Cstring,())))
    end
    for source in ("extern/mVMC-1.3.0/src/ltl2inv/invert.tcc", "extern/Julia-mVMC/PfaPack.jl/src/utu2.jl", "c_toolbox/issue184_inverse/probe.cc", "c_toolbox/issue184_inverse/generate.jl")
        println(io,source," sha256=",bytes2hex(sha256(read(joinpath(root,source)))))
    end
    println(io,"compiler=",read(`c++ --version`,String))
    println(io,"linked libraries=",read(`ldd $probe`,String))
end
for (name, n) in (("real4_identity", 4), ("complex6_pair_pivots", 6))
    a = zeros(ComplexF64, n, n)
    if n == 4
        a[1,2]=1; a[1,3]=.5; a[1,4]=.3
        a[2,3]=2; a[2,4]=.7; a[3,4]=3
    else
        for j in 2:n, i in 1:j-1
            a[i,j] = complex((i+j)/16, (i-j)/32)
        end
        a[1,2]=2; a[3,4]=3; a[5,6]=4
    end
    piv = n == 4 ? collect(1:n) : [2,1,4,3,6,5]
    input = joinpath(stage, name*".input.txt")
    open(input, "w") do io
        println(io, n == 4 ? "r 4" : "c 6")
        for z in a; @printf(io, "%.17g %.17g\n", real(z), imag(z)); end
        println(io, join(piv, " "))
    end
    output = read(pipeline(`$probe`, stdin=input), String)
    write(joinpath(stage, name*".c.txt"), output)
    vals = parse.(Float64, split(output))
    count = 2*n*n+n-1
    cvalues = complex.(vals[1:2:2*count], vals[2:2:2*count])
    vals[2*count+1:end] == piv || error("C changed pivots")
    ja = n == 4 ? real.(a) : copy(a)
    jm = fill(eltype(ja)(17), n,n); jv=fill(eltype(ja)(19), n-1)
    jp=copy(piv)
    utu2inv!(n, ja, n, jp, jv, jm, n)
    jp == piv || error("Julia changed pivots")
    jvalues=vcat(vec(ja), vec(jm), jv)
    open(joinpath(stage, name*".j.txt"), "w") do io
        for z in jvalues; @printf(io, "%.17g %.17g\n", real(z), imag(z)); end
        for p in jp; println(io,p); end
    end
    println(name, " max_abs_C_J=", maximum(abs.(cvalues-jvalues)),
            " max_scale=", maximum(abs.(cvalues)))
    # Independently reconstruct U*T*transpose(U) from the input factor.
    # No oracle output is used to construct the original dense operator.
    setprecision(256) do
        B = Complex{BigFloat}
        u = Matrix{B}(I,n,n); t=zeros(B,n,n)
        for j in 2:n-1, i in 1:j-1; u[i,j]=B(a[i,j+1]); end
        for i in 1:n-1; t[i,i+1]=B(a[i,i+1]); t[i+1,i]=-t[i,i+1]; end
        operator=u*t*transpose(u)
        # Inverse output is P*inverse(operator)*transpose(P). Recover the
        # corresponding operator with the same sequential swap convention.
        p=Matrix{B}(I,n,n)
        for i in 1:n; p[[i,piv[i]],:]=p[[piv[i],i],:]; end
        operator=p*operator*transpose(p)
        inverse=reshape(B.(cvalues[1:n*n]),n,n)
        residual=opnorm(operator*inverse-Matrix{B}(I,n,n),Inf)
        eta=residual/(opnorm(operator,Inf)*opnorm(inverse,Inf)+1)
        println(name," residual_inf=",residual," backward_eta=",eta,
                " condition_inf_estimate=",opnorm(operator,Inf)*opnorm(inv(operator),Inf))
        open(joinpath(stage,name*".operator.txt"),"w") do io
            for z in operator; @printf(io,"%.17g %.17g\n",Float64(real(z)),Float64(imag(z))); end
        end
    end
end
