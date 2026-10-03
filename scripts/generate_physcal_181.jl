# Optional developer command; normal Cargo tests never invoke this generator.
# julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_181.jl
using MVMCOptimizers, MVMCExpertModeParsers, SFMT, Random, LinearAlgebra, SHA

VERSION == v"1.13.1" || error("Julia 1.13.1 is required")
const repo = normpath(joinpath(@__DIR__, ".."))
const reference = joinpath(repo, "extern", "Julia-mVMC")
const destination = isempty(ARGS) ? joinpath(repo, "tests", "fixtures", "physcal_181") : abspath(ARGS[1])
const models = [("heisenberg_chain_real", :real), ("heisenberg_chain_cmp", :cmp),
    ("heisenberg_chain_fsz", :fsz), ("hubbard_chain_real", :real),
    ("hubbard_chain_dh_real", :real), ("kondo_chain_real", :real)]
const draws = Ref(0)
const model_directory = Ref("")
BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with JULIA_NUM_THREADS=1")
get(ENV, "JULIA_MVMC_MPI", "0") == "0" || error("serial reference required")
manifest = joinpath(reference, "Manifest-v1.13.toml")
isfile(manifest) || error("missing Manifest-v1.13.toml")
Base.active_project() == joinpath(reference, "Project.toml") || error("wrong reference project")

# Count primitive 32-bit words without changing the C calls or conversions.
# The six input sets use gen_rand32/genrand_real2 exclusively. Reject use of
# other draw entry points rather than silently undercounting them.
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
    open(path, "w") do io
        println(io, join(values, " "))
    end
end

function checkpoint(stage, state = nothing)
    dir = joinpath(model_directory[], stage)
    mkpath(dir)
    integers(joinpath(dir, "draw-count.txt"), [draws[]])
    # C helper saves/restores SFMT state; peeking does not advance the stream.
    integers(joinpath(dir, "next624.txt"), SFMT.sfmt_dump_rand32(624))
    if state !== nothing
        ec = state.electron_config
        for field in (:ele_idx, :ele_cfg, :ele_num, :ele_proj_cnt, :ele_spn)
            integers(joinpath(dir, "$field.txt"), getproperty(ec, field))
        end
        integers(joinpath(dir, "counter.txt"), vcat(ec.counter[1:9], ec.counter[11]))
    end
end

function loaded_parameters(data)
    values = MVMCOptimizers.pack_parameters(data)
    open(joinpath(model_directory[], "fixed-parameters.txt"), "w") do io
        for value in values
            println(io, repr(real(value)), " ", repr(imag(value)))
        end
    end
end

const physcal_path = joinpath(pkgdir(MVMCOptimizers), "src", "vmc_phys_cal.jl")
source = read(physcal_path, String)
for (needle, replacement) in [
    "    # Save current parameter values" => "    Main.loaded_parameters(data)\n    # Save current parameter values",
    "    # Restore the original parameter values" => "    Main.checkpoint(\"initialized\")\n    # Restore the original parameter values",
    "        # Main calculation (energy + Green's functions)" => "        Main.checkpoint(\"sample-\$(ismp)\", state)\n        # Main calculation (energy + Green's functions)",
]
    count(needle, source) == 1 || error("observation boundary changed: $needle")
    global source = replace(source, needle => replacement)
end
Base.include_string(MVMCOptimizers, source, physcal_path)
include(joinpath(reference, "test", "integration", "tools", "green_compare.jl"))

mkpath(destination)
revision = strip(read(`git -C $reference rev-parse HEAD`, String))
isempty(strip(read(`git -C $reference status --porcelain`, String))) || error("dirty Julia reference")
for (name, mode) in models
    fixture = joinpath(reference, "test", "integration", "reference", name, "physcal_ref")
    model_directory[] = joinpath(destination, name)
    mkpath(model_directory[])
    inputs = joinpath(model_directory[], "inputs")
    # Copy only on an explicit developer regeneration. No Cargo dependency on
    # vendor/runtime. The namelist and all sampling counts are unchanged.
    cp(joinpath(fixture, "inputs"), inputs; force = true)
    cp(joinpath(fixture, "zqp_opt.dat"), joinpath(model_directory[], "zqp_opt.dat"); force = true)
    draws[] = 0
    rng = SFMT.SFMT19937RNG()
    Random.seed!(rng, 1)
    checkpoint("seeded")
    comparisons = String[]
    mktempdir() do work
        cd(work) do
            result = Base.invokelatest(MVMCOptimizers.run_phys_cal_from_namelist,
                joinpath(inputs, "namelist.def"); opt_para = joinpath(model_directory[], "zqp_opt.dat"),
                mode = mode, seed = 1, output_dir = work)
            result.status == 0 || error("$name: status $(result.status)")
            for (file, compare) in [("zvo_cisajs_001.dat", compare_green_one),
                ("zvo_cisajscktalt_001.dat", compare_green_two_dc),
                ("zvo_cisajscktaltex_001.dat", compare_green_factored)]
                r = compare(joinpath(work, file), joinpath(fixture, "expected", file))
                push!(comparisons, "$file: ok=$(r.ok), max_abs=$(r.max_abs_err), $(r.detail)")
            end
        end
    end
    open(joinpath(model_directory[], "provenance.txt"), "w") do io
        println(io, "generator=scripts/generate_physcal_181.jl")
        println(io, "julia=$VERSION\nreference_revision=$revision\narchitecture=$(Sys.MACHINE)")
        println(io, "blas=$(BLAS.get_config())\nblas_threads=$(BLAS.get_num_threads())\njulia_threads=$(Threads.nthreads())")
        println(io, "seed=1\nmode=$mode\nmanifest_sha256=$(bytes2hex(sha256(read(manifest))))")
        println(io, "physcal_source_sha256=$(bytes2hex(sha256(read(physcal_path))))")
        println(io, "stages=seeded; initialized (single InitParameter, before fixed restoration); sample-0 (after all saved samples, before measurement)")
        println(io, "rng=next 624 UInt32 outputs, non-consuming C SFMT peek; draw-count=primitive UInt32 words since seed")
        println(io, "input_contract=explicit zqp; no automatic initial.def; unchanged sampling counts; serial; correlation shifts disabled")
        println(io, "trajectory_authority=Julia independently generated; C compatibility of trajectory requires a separate C trace")
        for line in comparisons; println(io, "C_output_check=$line"); end
        for (dir, _, files) in walkdir(inputs), file in sort(files)
            path = joinpath(dir, file)
            println(io, "input_sha256=$(bytes2hex(sha256(read(path)))) $(relpath(path, model_directory[]))")
        end
        println(io, "input_sha256=$(bytes2hex(sha256(read(joinpath(model_directory[], "zqp_opt.dat"))))) zqp_opt.dat")
    end
    println(name, ": saved trajectory; total UInt32 draws=", draws[], "; ", join(comparisons, "; "))
end
