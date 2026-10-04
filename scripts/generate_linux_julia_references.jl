# Staged independent Julia jobs. Only verification/observer instrumentation is
# adapted for module isolation; the reference numerical kernels are unchanged.
module LinuxReferenceBatch
using LinearAlgebra
VERSION == v"1.13.1" || error("Julia 1.13.1 is required")
Sys.islinux() && Sys.ARCH == :x86_64 || error("Linux x86_64 is required")
BLAS.set_num_threads(1)
const CASES = ("fsz", "interall", "pairhop_fsz", "dh2_fsz", "dh4_fsz", "dh24_fsz", "rbm_fsz", "opt_fsz")
const GENERATED = Set{String}()
const FIXTURES = normpath(joinpath(@__DIR__, "..", "tests", "fixtures"))
function record(path, actual)
    push!(GENERATED, relpath(path, FIXTURES))
    return nothing
end
function job(suite)
    if suite == "julia-linux-setup"
        return ("check_real_fsz_setup_parity.jl", "real_fsz", String[])
    elseif suite == "julia-linux-real-sampling"
        return ("check_real_fsz_sampling_parity.jl", "real_fsz", String[])
    elseif suite == "julia-linux-complex-sampling"
        return ("check_complex_fsz_sampling_parity.jl", "complex_fsz", String[])
    elseif suite == "julia-linux-dh2-history"
        return ("check_dh2_runner_boundaries.jl", "dh2", String[])
    elseif suite == "julia-linux-dh4-history"
        return ("check_dh4_runner_boundaries.jl", "dh4", String[])
    end
    match_result = match(r"^julia-linux-(direct|store|cg)-(.+)$", suite)
    match_result === nothing && error("Unknown Linux Julia suite: $suite")
    mode, case = match_result.captures
    case in CASES || error("Unknown runner case: $case")
    cg = mode == "cg"
    filename = cg ? "check_sr_cg_runner_parity.jl" : "check_sr_direct_runner_parity.jl"
    directory = "sr_" * (cg ? "cg" : "direct") * "/" * case * (mode == "store" ? "_store_runner" : "_runner")
    flags = ["--case=$case"]
    !cg && push!(flags, "--store=" * (mode == "store" ? "1" : "0"))
    return (filename, directory, flags)
end
prefixes, log, suites = ARGS[1], ARGS[2], ARGS[3:end]
if "julia-linux-all" in suites
    suites = vcat(["julia-linux-setup", "julia-linux-real-sampling", "julia-linux-complex-sampling", "julia-linux-dh2-history", "julia-linux-dh4-history"],
                  ["julia-linux-$mode-$case" for mode in ("direct", "store", "cg") for case in CASES])
end
for (index, suite) in enumerate(suites)
    filename, directory, flags = job(suite)
    startswith(directory, "sr_") && push!(flags, "--steps=$prefixes")
    push!(flags, "--write")
    empty!(ARGS); append!(ARGS, flags)
    name = Symbol("LinuxHistoricalJob", index)
    scope = Module(name)
    Core.eval(Main, :(const $name = $scope))
    path = joinpath(@__DIR__, filename)
    source = read(path, String)
    # The upstream source copies call Main.capture_*; qualify those observer
    # targets to the isolated job, never change runtime arithmetic or RNG.
    source = replace(source, "Main.capture_" => "Main.$name.capture_")
    output_root = repr(joinpath(FIXTURES, directory))
    # All existing generators serialize their independent expected values via
    # verify(name,actual). Record every produced pathname, even unchanged data.
    instrumentation = "\n    Main.LinuxReferenceBatch.record(joinpath($output_root,name),actual)\n"
    count = length(collect(eachmatch(r"function verify\(name,\s*actual\)", source)))
    source = replace(source, r"function verify\(name,\s*actual\)" => s -> s * instrumentation)
    if count == 0
        # The three small generators write directly rather than through verify.
        direct = directory == "complex_fsz" ? "sampling.txt" : filename == "check_real_fsz_setup_parity.jl" ? "setup.txt" : "sampling.txt"
        push!(GENERATED, joinpath(directory, direct))
    end
    println("Independent Julia job: $suite ", join(flags, " "))
    Base.include_string(scope, source, path)
end
open(log, "w") do io
    for path in sort!(collect(GENERATED)); println(io, path); end
end
end
