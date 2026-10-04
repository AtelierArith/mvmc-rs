# Explicit optional native C stage probe; Cargo never invokes this command.
# julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/generate_physcal_181_parameters.jl
using SHA, Dates
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
const repo = normpath(joinpath(@__DIR__, ".."))
const vendor = joinpath(repo, "extern", "mVMC-1.3.0", "src")
const fixtures = joinpath(repo, "tests", "fixtures", "physcal_181")
const parameter_path = joinpath(vendor, "mVMC", "parameter.c")
const reader_path = joinpath(vendor, "mVMC", "readdef.c")
const parameter = read(parameter_path, String)
const reader = read(reader_path, String)
function extract(text, start_marker, stop_marker)
    start = findfirst(start_marker, text); stop = findnext(stop_marker, text, last(start)+1)
    text[first(start):prevind(text, first(stop))]
end
const excerpt = extract(parameter, "void InitParameter()", "void shiftGJ()") *
    extract(reader, "int ReadInputParameters(char", "/**********************************************************************/")
const excerpt_path = joinpath(@__DIR__, "physcal_181_parameters_upstream.inc")
write(excerpt_path, "/* Verbatim parameter.c InitParameter/ReadInitParameter/SyncModifiedParameter; readdef.c ReadInputParameters.\n" *
    " * parameter_sha256=$(bytes2hex(sha256(read(parameter_path))))\n" *
    " * readdef_sha256=$(bytes2hex(sha256(read(reader_path)))) */\n" * excerpt)

# Reuse the independent C definition-record enumeration, not Rust state.
generator = read(joinpath(repo, "scripts", "generate_physcal_181.jl"), String)
Base.include_string(Main, extract(generator, "function c_definition_flags(inputs)", "function checkpoint("))
header = read(joinpath(vendor, "mVMC", "include", "readdef.h"), String)
keywords = [m.captures[1] for m in eachmatch(r"\"([^\"]+)\"", extract(header,
    "static char cKWListOfFileNameList", "/**"))]
width_keys = ["Gutzwiller", "Jastrow", "DH2", "DH4",
    "ChargeRBM_PhysLayer", "SpinRBM_PhysLayer", "GeneralRBM_PhysLayer",
    "ChargeRBM_HiddenLayer", "SpinRBM_HiddenLayer", "GeneralRBM_HiddenLayer",
    "ChargeRBM_PhysHidden", "SpinRBM_PhysHidden", "GeneralRBM_PhysHidden", "Orbital", "OptTrans"]

reserved = joinpath(fixtures, "unnormalized-reserved-slater")
mkpath(joinpath(reserved, "inputs"))
base_inputs = joinpath(fixtures, "heisenberg_chain_real", "inputs")
for file in readdir(base_inputs)
    cp(joinpath(base_inputs, file), joinpath(reserved, "inputs", file); force=true)
end
open(joinpath(reserved, "inputs", "orbitalidx.def"), "w") do io
    println(io, "===\nNOrbitalIdx 4\nComplexType 0\n===\n===")
    for i in 0:5, j in 0:5; println(io, "$i $j $((i+j)%2)"); end
    for i in 0:3; println(io, "$i 1"); end
end
open(joinpath(reserved, "inputs", "inorbital.def"), "w") do io
    println(io, "===\nNOrbitalIdx 4\nComplexType 0\n===\n===\n3 16 0\n1 2 0\n0 -4 0\n2 1 0")
end
open(joinpath(reserved, "inputs", "namelist.def"), "a") do io
    println(io, "\nInOrbital inorbital.def")
end
write(joinpath(reserved, "zqp_opt.dat"), "1 2 3 4 5 6 2 0 0 -1 0 0 20 0 0 10 0 0 2 0 0 8 0 0\n")

mktempdir() do work
    compiler = get(ENV, "CC", "gcc")
    binary = joinpath(work, "physcal-parameters")
    options = ["-std=gnu11", "-O0", "-ffp-contract=off", "-DMEXP=19937",
        "-I$(joinpath(vendor, "sfmt"))", "-I$(joinpath(vendor, "mVMC", "include"))", "-I$(@__DIR__)"]
    run(`$compiler $options $(joinpath(@__DIR__, "physcal_181_parameters.c")) -lm -o $binary`)
    for name in ["hubbard_chain_dh_overlays", "hubbard_chain_dh_opttrans", "hubbard_chain_dh_rbm_opttrans", "unnormalized-reserved-slater"]
        dir = joinpath(fixtures, name); inputs = joinpath(dir, "inputs")
        defs = Dict(split(line)[1]=>split(line)[2] for line in split(read(joinpath(inputs,"namelist.def"),String),'\n') if length(split(line))==2)
        widths = [haskey(defs,key) ? parse(Int,split(split(read(joinpath(inputs,defs[key]),String),'\n')[2])[2]) : 0 for key in width_keys]
        complex = any(key->haskey(defs,key) && parse(Int,split(split(read(joinpath(inputs,defs[key]),String),'\n')[3])[2])>0,
            ["Gutzwiller", "Jastrow", "DH2", "DH4", "Orbital"])
        flags, _ = c_definition_flags(inputs)
        spec = joinpath(work, "spec.txt")
        open(spec,"w") do io
            println(io, join(vcat(widths,[Int(complex),6])," "))
            println(io, join(flags," "))
            if widths[end] > 0
                # C definition records populate ParaQPOptTrans before InitParameter.
                records = split(read(joinpath(inputs, defs["OptTrans"]), String), '\n')[6:end]
                weights = zeros(widths[end])
                for line in records[1:widths[end]]
                    fields = split(line)
                    weights[parse(Int, fields[1]) + 1] = parse(Float64, fields[2])
                end
                println(io, join(weights, " "))
            end
            for (key,path) in sort!(collect(defs); by=first)
                startswith(key,"In") || continue
                index = findfirst(==(key),keywords); index === nothing && error("unknown C keyword $key")
                println(io,index-1," ",joinpath(inputs,path))
            end
        end
        out = joinpath(dir,"native-c-stages"); mkpath(out)
        run(`$binary $spec $(joinpath(dir,"zqp_opt.dat")) $out`)
        open(joinpath(out,"provenance.txt"),"w") do io
            println(io,"generated_utc=$(now(UTC))\narchitecture=$(Sys.MACHINE)\ncompiler=$(strip(read(`$compiler --version`,String)))")
            println(io,"compiler_options=$(join(options, " ")) -lm\nseed=1\nBLAS=none; scalar parameter-stage probe")
            println(io,"source_parameter_sha256=$(bytes2hex(sha256(read(parameter_path))))\nsource_readdef_sha256=$(bytes2hex(sha256(read(reader_path))))")
            println(io,"driver_sha256=$(bytes2hex(sha256(read(joinpath(@__DIR__,"physcal_181_parameters.c")))))\nexcerpt_sha256=$(bytes2hex(sha256(read(excerpt_path))))")
            println(io,"scope=actual C initialization/fixed-record/overlay/synchronization functions, explicit dimensions/flags/keyword slots; NOT complete C reader/executable/MPI/sampling acceptance")
            println(io,"stage_order=InitParameter; ReadInitParameter; ReadInputParameters; SyncModifiedParameter, correlation shifts off as C Mode1")
            println(io,"rng=next624 non-consuming sfmt+idx save/restore; draw count derived from actual C InitParameter active-loop clauses and checked by stream position")
            for path in vcat([joinpath(dir,"zqp_opt.dat")],[joinpath(inputs,file) for file in sort(readdir(inputs))])
                println(io,"input_sha256=$(bytes2hex(sha256(read(path)))) $(relpath(path,dir))")
            end
        end
    end
end
