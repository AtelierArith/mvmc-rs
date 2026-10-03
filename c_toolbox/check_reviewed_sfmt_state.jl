# Optional observer on/off diagnostic. No Cargo test invokes this script.
using Libdl, SHA
length(ARGS)==2 || error("usage: BASELINE_LIB OBSERVED_LIB")
function api(path)
    handle=Libdl.dlopen(realpath(path),Libdl.RTLD_LOCAL|Libdl.RTLD_DEEPBIND)
    Dict(name=>Libdl.dlsym(handle,name) for name in
        (:init_gen_rand,:gen_rand32,:genrand_real2,:sfmt_dump_rand32,:reviewed_sfmt_state))
end
function state(a)
    words=Vector{UInt32}(undef,624); index=Ref{Cint}(); count=Ref{UInt64}()
    ccall(a[:reviewed_sfmt_state],Cint,(Ptr{UInt32},Ref{Cint},Ref{UInt64}),words,index,count)==0 || error("getter")
    (words=words,index=index[],count=count[])
end
function peek(a)
    words=Vector{UInt32}(undef,624)
    ccall(a[:sfmt_dump_rand32],Cvoid,(Ptr{UInt32},Cint),words,624)
    words
end
baseline,observed=api.(ARGS)
for seed in UInt32[1,12395,0xffffffff]
    for a in (baseline,observed)
        ccall(a[:init_gen_rand],Cvoid,(UInt32,),seed)
    end
    @assert state(observed).count==0 && state(observed).index==624
    for draws in 0:2048
        b,r=state(baseline),state(observed)
        @assert b.words==r.words && b.index==r.index && r.count==draws
        @assert state(observed)==r # raw getter is non-consuming
        if draws in (0,1,623,624,625,2048)
            @assert peek(baseline)==peek(observed)
            @assert state(baseline)==b && state(observed)==r # peek preserves state AND count
        end
        draws==2048 && break
        if iseven(draws)
            @assert ccall(baseline[:gen_rand32],UInt32,())==ccall(observed[:gen_rand32],UInt32,())
        else
            # Exact RNG conversion contract, not computed Monte Carlo floats.
            @assert ccall(baseline[:genrand_real2],Cdouble,())==ccall(observed[:genrand_real2],Cdouble,())
        end
    end
end
println("PASS observer on/off: 3 seeds, 6144 original mixed primitive draws, raw state/index exact at every boundary; peek/state getters non-consuming; observed count exact")
for path in ARGS
    println(bytes2hex(sha256(read(path)))," ",realpath(path))
end
