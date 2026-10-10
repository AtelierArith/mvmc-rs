# Optional Linux ELF probe of the already-built SFMT library, outside production timing.
# No native source or RNG state is changed. State integers are exact comparisons.
using MPI, MVMCOptimizers, SFMT, LinearAlgebra, Libdl, SHA
BLAS.set_num_threads(1)
MPI.Init_thread(MPI.THREAD_FUNNELED)
rank=MPI.Comm_rank(MPI.COMM_WORLD)
input,out,steps_s=ARGS
count_draws = get(ENV,"MVMC_RNG_AUDIT_COUNT","0") == "1"
if count_draws
    # These entry points each consume one word (SFMT.c and SFMT-real.c).
    # Count the current seeded stream, matching Rust words_consumed(); retain
    # counts before each reseed separately so initialization is also visible.
    @eval SFMT.C_API begin
        const audit_words = Ref{UInt64}(0)
        const audit_reseeds = Tuple{UInt32,UInt64}[]
        function init_gen_rand(seed)
            push!(audit_reseeds,(UInt32(seed),audit_words[]))
            audit_words[]=0
            ccall((:init_gen_rand,libsfmt),Cvoid,(UInt32,),seed)
        end
        function gen_rand32()
            audit_words[]+=1
            ccall((:gen_rand32,libsfmt),UInt32,())
        end
        function genrand_real2()
            audit_words[]+=1
            ccall((:genrand_real2,libsfmt),Cdouble,())
        end
    end
    # Fail explicitly if a future runner uses an uncounted native primitive.
    for primitive in (:gen_rand64,:genrand_real1,:genrand_real3,
                      :genrand_res53,:genrand_res53_mix)
        @eval SFMT.C_API function $primitive()
            error("uncounted SFMT primitive in optional RNG audit")
        end
    end
    for primitive in (:fill_array32,:fill_array64,:sfmt_dump_rand32,:init_by_array)
        @eval SFMT.C_API function $primitive(array,size)
            error("uncounted SFMT bulk primitive in optional RNG audit")
        end
    end
end
try
    mkpath(out)
    result=MVMCOptimizers.run_para_opt_from_namelist(input;nsteps=parse(Int,steps_s),nsmp=parse(Int,steps_s),mode=:real,output_dir=joinpath(out,"production"))
    @assert result.status==0
    library=SFMT.C_API.libsfmt
    symbols=Dict{String,UInt}()
    for line in split(read(`nm -S $library`,String),'\n')
        fields=split(line)
        length(fields)==4 || continue
        fields[4] in ("gen_rand32","idx","initialized","sfmt") || continue
        symbol_size=parse(UInt,fields[2];base=16)
        fields[4]=="sfmt" && @assert symbol_size==624*sizeof(UInt32)
        fields[4] in ("idx","initialized") && @assert symbol_size==sizeof(Cint)
        symbols[fields[4]]=parse(UInt,fields[1];base=16)
    end
    @assert all(haskey(symbols,k) for k in ("gen_rand32","idx","initialized","sfmt"))
    handle=Libdl.dlopen(library)
    base=UInt(Libdl.dlsym(handle,:gen_rand32))-symbols["gen_rand32"]
    idx=unsafe_load(Ptr{Cint}(base+symbols["idx"]))
    initialized=unsafe_load(Ptr{Cint}(base+symbols["initialized"]))
    @assert 0<=idx<=624 && initialized==1
    words=copy(unsafe_wrap(Vector{UInt32},Ptr{UInt32}(base+symbols["sfmt"]),624;own=false))
    open(joinpath(out,"rng-rank-$rank.txt"),"w") do io
        println(io,"library_sha256=",bytes2hex(sha256(read(library))))
        println(io,"offsets=",join(("$key=$(symbols[key])" for key in sort!(collect(keys(symbols)))),','))
        println(io,"idx=",idx," initialized=",initialized)
        if count_draws
            println(io,"words_consumed=",SFMT.C_API.audit_words[])
            println(io,"reseeds=",join(("$seed:$words" for (seed,words) in SFMT.C_API.audit_reseeds),','))
        end
        println(io,join(words,','))
    end
finally
    MPI.Finalize()
end
