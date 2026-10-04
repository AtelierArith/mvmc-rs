# Optional scoped reference diagnostic API. Install before the original seed.
module ReviewedSFMTState
using SFMT, Libdl
const library = Ref{Ptr{Cvoid}}(C_NULL)
const symbols = Dict{Symbol,Ptr{Cvoid}}()
function install!(path)
    library[] == C_NULL || error("SFMT diagnostic already installed")
    library[] = Libdl.dlopen(realpath(path), Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
    for name in (:init_gen_rand,:gen_rand32,:genrand_real2,:sfmt_dump_rand32,:reviewed_sfmt_state)
        symbols[name]=Libdl.dlsym(library[],name)
    end
    # Only primitive draw/seed/peek entry points; original C bodies remain intact.
    seed= symbols[:init_gen_rand]; word=symbols[:gen_rand32]
    real2=symbols[:genrand_real2]; peek=symbols[:sfmt_dump_rand32]
    @eval SFMT.C_API begin
        init_gen_rand(seed) = ccall($seed,Cvoid,(UInt32,),seed)
        gen_rand32() = ccall($word,UInt32,())
        genrand_real2() = ccall($real2,Cdouble,())
        sfmt_dump_rand32(array,size) = ccall($peek,Cvoid,(Ptr{UInt32},Cint),array,size)
        gen_rand64() = error("uncounted SFMT64 entry point")
        genrand_real1() = error("unreviewed SFMT real1 entry point")
        genrand_real3() = error("unreviewed SFMT real3 entry point")
        genrand_res53() = error("uncounted SFMT res53 entry point")
        genrand_res53_mix() = error("uncounted SFMT mixed res53 entry point")
        fill_array32(array,size) = error("uncounted SFMT bulk entry point")
        fill_array64(array,size) = error("uncounted SFMT bulk entry point")
        init_by_array(key,size) = error("unreviewed SFMT array-seed entry point")
    end
    nothing
end
function snapshot()
    library[] != C_NULL || error("SFMT diagnostic not installed")
    words=Vector{UInt32}(undef,624); index=Ref{Cint}(); count=Ref{UInt64}()
    ccall(symbols[:reviewed_sfmt_state],Cint,
        (Ptr{UInt32},Ref{Cint},Ref{UInt64}),words,index,count)==0 || error("state getter failed")
    (words=words,index=Int(index[]),count=count[])
end
function capture!(path)
    before=snapshot()
    open(path,"w") do io
        println(io,join(before.words," "))
        println(io,before.index)
        println(io,before.count)
    end
    snapshot()==before || error("read-only state capture changed the RNG")
    before
end
end
