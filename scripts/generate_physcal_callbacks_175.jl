# Optional developer generator; ordinary Cargo tests never run Julia/C or this file.
# JULIA_NUM_THREADS=1 JULIA_MVMC_MPI=0 julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_callbacks_175.jl
using MVMCOptimizers, MVMCExpertModeParsers, SFMT, Random, LinearAlgebra, SHA, Dates, Libdl
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
const repo = normpath(joinpath(@__DIR__, ".."))
const reference = joinpath(repo, "extern", "Julia-mVMC")
const destination = joinpath(repo, "tests", "fixtures", "physcal_callbacks_175")
Base.active_project() == joinpath(reference, "Project.toml") || error("wrong project")
Threads.nthreads() == 1 || error("single Julia thread required")
get(ENV, "JULIA_MVMC_MPI", "0") == "0" || error("serial only")
isempty(strip(read(`git -C $reference status --porcelain`, String))) || error("dirty reference")
BLAS.set_num_threads(1)
const draws = Ref(0)
const directory = Ref("")
const loaded = Ref(ComplexF64[])

# Count original primitive calls without changing their conversions or draws.
@eval SFMT.C_API begin
    function gen_rand32()
        Main.draws[] += 1
        ccall((:gen_rand32, libsfmt), UInt32, ())
    end
    function genrand_real2()
        Main.draws[] += 1
        ccall((:genrand_real2, libsfmt), Cdouble, ())
    end
    gen_rand64() = error("unaccounted 64-bit draw")
    genrand_real1() = error("unaccounted real1 draw")
    genrand_real3() = error("unaccounted real3 draw")
    fill_array32(array, size) = error("unaccounted bulk draw")
    fill_array64(array, size) = error("unaccounted bulk draw")
end

function integers(path, values)
    open(path, "w") do io; println(io, join(values, " ")); end
end
function complex_values(path, values)
    open(path, "w") do io
        for value in values; println(io, repr(real(value)), " ", repr(imag(value))); end
    end
end
function checkpoint(stage, state=nothing)
    dir = joinpath(directory[], stage)
    mkpath(dir)
    integers(joinpath(dir, "draw-count.txt"), [draws[]])
    integers(joinpath(dir, "next624.txt"), SFMT.sfmt_dump_rand32(624))
    if state !== nothing
        ec = state.electron_config
        for field in (:ele_idx, :ele_cfg, :ele_num, :ele_proj_cnt, :ele_spn)
            integers(joinpath(dir, "$field.txt"), getproperty(ec, field))
        end
        integers(joinpath(dir, "counter.txt"), vcat(ec.counter[1:9], ec.counter[11]))
        complex_values(joinpath(dir, "energy.txt"),
            [getproperty(state.energy, field) for field in (:wc, :etot, :etot2, :sztot, :sztot2)])
    end
end
function save_loaded(data)
    loaded[] = copy(MVMCOptimizers.pack_parameters(data))
    complex_values(joinpath(directory[], "fixed-parameters.txt"), loaded[])
end
function restored(data)
    MVMCOptimizers.pack_parameters(data) == loaded[] || error("fixed parameters changed")
end

# Observation-only boundaries; no altered numerical algorithms or injected
# expectations. Actual callback remains the unmodified Julia callback dispatch.
const source_path = joinpath(reference, "MVMCOptimizers.jl", "src", "vmc_phys_cal.jl")
source = read(source_path, String)
for (needle, replacement) in [
    "    # Save current parameter values" => "    Main.save_loaded(data)\n    # Save current parameter values",
    "    # Restore the original parameter values" => "    Main.checkpoint(\"initialized\")\n    # Restore the original parameter values",
    "    # Get parameters" => "    Main.restored(data)\n    # Get parameters",
    "        # Main calculation (energy + Green's functions)" => "        Main.checkpoint(\"sample-\$(ismp)\", state)\n        # Main calculation (energy + Green's functions)",
    "        # Callback" => "        Main.checkpoint(\"averaged-\$(ismp)\", state)\n        # Callback",
]
    count(needle, source) == 1 || error("reference boundary changed: $needle")
    global source = replace(source, needle => replacement)
end
Base.include_string(MVMCOptimizers, source, source_path)

# Restricted to the existing C expert workloads: Gutzwiller/Jastrow/Orbital,
# real components. Enumerate C-defined flag writes, masking unwritten imaginary
# cells (readdef.c ReadDefFileIdxPara); never substitute Julia Boolean flags.
function flags(inputs)
    definitions = Dict(split(line)[1] => split(line)[2] for line in
        split(read(joinpath(inputs, "namelist.def"), String), '\n') if length(split(line)) == 2)
    result = Int[]
    written = Int[]
    for key in ("Gutzwiller", "Jastrow", "Orbital")
        haskey(definitions, key) || continue
        lines = filter(!isempty, strip.(split(read(joinpath(inputs, definitions[key]), String), '\n')))
        width = parse(Int, split(lines[2])[2])
        parse(Int, split(lines[3])[2]) == 0 || error("real-only C fixture expected")
        for record in lines[end-width+1:end]
            fields = split(record)
            length(fields) == 2 || error("invalid optimization flag record")
            append!(result, [parse(Int, fields[2]), 0])
            append!(written, [1, 0])
        end
    end
    integers(joinpath(directory[], "optimization-flags.txt"), result)
    integers(joinpath(directory[], "optimization-flags-written.txt"), written)
end

for model in ("hubbard_chain_lanczos", "spin_chain_lanczos"), mode in (1, 2)
    directory[] = joinpath(destination, "$model-mode$mode")
    mkpath(directory[])
    original = joinpath(reference, "test", "integration", "reference", model, "physcal_ref")
    inputs = joinpath(directory[], "inputs")
    cp(joinpath(original, "inputs"), inputs; force=true)
    cp(joinpath(original, "zqp_opt.dat"), joinpath(directory[], "zqp_opt.dat"); force=true)
    modpara = joinpath(inputs, "modpara.def")
    text = read(modpara, String)
    for (key, value) in (("NLanczosMode", mode), ("NDataQtySmp", 2), ("NDataIdxStart", 7))
        pattern = Regex("(?m)^$key\\s+[^\\n]+")
        count(pattern, text) == 1 || error("missing/duplicate $key")
        text = replace(text, pattern => "$key $value")
    end
    write(modpara, text)
    data = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(inputs, "namelist.def"))
    draws[] = 0
    rng = SFMT.SFMT19937RNG()
    Random.seed!(rng, 1)
    checkpoint("seeded")
    consumed = MVMCOptimizers.read_opt_para_file!(data, joinpath(directory[], "zqp_opt.dat"))
    integers(joinpath(directory[], "consumed-count.txt"), [consumed])
    MVMCOptimizers.read_input_parameters!(data, joinpath(inputs, "namelist.def"))
    MVMCOptimizers.sync_modified_parameter!(MVMCOptimizers.serial_context(), data; shift_correlations=false)
    flags(inputs)
    records = String[]
    callback = function(index, fixed, energy, status)
        MVMCOptimizers.pack_parameters(fixed) == loaded[] || error("callback fixed data changed")
        push!(records, "$index $status $(repr(real(energy))) $(repr(imag(energy)))")
        checkpoint("callback-$index")
    end
    mktempdir() do work
        cd(work) do
            status = Base.invokelatest(MVMCOptimizers.vmc_phys_cal!, data;
                rng=rng, callback=callback, output_dir=work)
            status == 0 || error("PhysCal status $status")
        end
    end
    length(records) == 2 || error("missing callbacks")
    write(joinpath(directory[], "callbacks.txt"), join(records, "\n") * "\n")
    library = first(BLAS.get_config().loaded_libs)
    blas_version = unsafe_string(ccall(Libdl.dlsym(library.handle,
        Symbol("openblas_get_config", library.suffix)), Cstring, ()))
    open(joinpath(directory[], "provenance.txt"), "w") do io
        println(io, "generator=scripts/generate_physcal_callbacks_175.jl\ngenerator_sha256=$(bytes2hex(sha256(read(@__FILE__))))")
        println(io, "generated_utc=$(now(UTC))\njulia=$VERSION\njulia_commit=$(Base.GIT_VERSION_INFO.commit)")
        println(io, "reference_revision=$(strip(read(`git -C $reference rev-parse HEAD`, String)))")
        println(io, "manifest_sha256=$(bytes2hex(sha256(read(joinpath(reference, "Manifest-v1.13.toml")))))")
        println(io, "architecture=$(Sys.MACHINE)\ncpu=$(Sys.CPU_NAME)\nblas=$blas_version\nblas_threads=1\njulia_threads=1")
        println(io, "mode=real\nlanczos_mode=$mode\nseed=1\nsample_quantity=2\nsample_index_start=7")
        println(io, "physcal_source_sha256=$(bytes2hex(sha256(read(source_path))))")
        println(io, "authority=Julia actual callback and exact trajectory; C numerical contract, NOT full native-C trajectory verification")
        println(io, "C_contract=vmcmain.c VMCPhysCal/outputData; average.c WeightAverageWE/WeightAverageGreenFunc; supported non-InterAll hopping+intra or exchange+spin Hamiltonian")
        println(io, "rng=next624 non-consuming SFMT peek; cumulative primitive UInt32 draws; no sampling counts changed")
        println(io, "stages=initialized before fixed restoration; sample-i before measurement; averaged-i after output/counter reduction; callback-i actual callback")
        for (dir, _, files) in walkdir(inputs), file in sort(files)
            path = joinpath(dir, file)
            println(io, "input_sha256=$(bytes2hex(sha256(read(path)))) $(relpath(path, directory[]))")
        end
        println(io, "zqp_sha256=$(bytes2hex(sha256(read(joinpath(directory[], "zqp_opt.dat")))))")
    end
end
