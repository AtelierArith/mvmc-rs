# Optional independent mixed-reference generation. Historical Julia fixtures
# remain untouched; outputs go to c_kernel_order/native_fsz/<solver>/<case>.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
VERSION == v"1.13.1" || error("Native FSZ runner fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
include("reference_native_fsz_energy.jl")
include("reference_native_fsz_fixture_inheritance.jl")

option(key, default) = let matches = filter(a -> startswith(a, key * "="), ARGS)
    isempty(matches) ? default : split(only(matches), "="; limit=2)[2]
end
directory = option("--bridge-dir", "")
isempty(directory) && error("First validate/build scripts/check_native_fsz_runner_bridge.py --build-dir DIR, then pass --bridge-dir=DIR")
steps = option("--steps", "1,2,3,50")
selected = option("--job", "all")
output_root = option("--output-root", joinpath(@__DIR__, "..", "tests", "fixtures", "c_kernel_order", "native_fsz"))
write_outputs = "--write" in ARGS
general = "--general" in ARGS
general && write_outputs && error("General verifies the native AP/P FSZ references; first generate those without --general")
NativeFSZEnergyReference.install!(directory)
cases = ("fsz", "interall", "pairhop_fsz", "opt_fsz", "dh2_fsz", "dh4_fsz", "dh24_fsz")
jobs = [(solver, case, store) for case in cases for (solver, store) in
        (("sr_direct", 0), ("sr_direct", 1), ("sr_cg", 0))]
if selected != "all"
    filter!(j -> selected == "$(j[1])/$(j[2])/$(j[3])", jobs)
    isempty(jobs) && error("Unknown --job=$selected")
end
general && filter!(j -> j[2] == "fsz", jobs)
for (index, (solver, case, store)) in enumerate(jobs)
    name = Symbol("NativeFSZReferenceCase", index)
    namespace = Module(name)
    Core.eval(Main, :(const $name = $namespace))
    Core.eval(namespace, :(include(path) = Base.include($namespace, path)))
    path = joinpath(@__DIR__, "check_$(solver)_runner_parity.jl")
    source = replace(read(path, String), "Main.capture_" => "Main.$name.capture_")
    target = joinpath(output_root, solver,
                      case * (solver == "sr_direct" && store == 1 ? "_store_runner" : "_runner"))
    root_line = only(filter(l -> startswith(l, "const FIXTURE_ROOT = "), split(source, '\n')))
    source = replace(source, root_line => "const FIXTURE_ROOT = " * repr(target); count=1)
    source = replace(source, "function verify(name, actual)\n" =>
        "function verify(name, actual)\n    if name in (\"reference.txt\", \"fixed-input.txt\")\n        actual = Main.NativeFSZEnergyReference.PROVENANCE[] * actual\n    end\n    if !(\"--write\" in ARGS) && Main.NativeFSZFixtureInheritance.verify_unused(FIXTURE_ROOT, name, actual)\n        return nothing\n    end\n"; count=1)
    source = replace(source, "@test actual == read(path, String)" =>
        "@test actual == read(Main.NativeFSZFixtureInheritance.resolve(FIXTURE_ROOT, name), String)"; count=1)
    empty!(ARGS)
    append!(ARGS, ["--case=$case", "--steps=$steps"])
    solver == "sr_direct" && push!(ARGS, "--store=$store")
    general && push!(ARGS, "--general")
    write_outputs && push!(ARGS, "--write")
    println("Native C energy / original Julia runner: $solver/$case store=$store steps=$steps")
    Base.include_string(namespace, source, path)
    # Compare source draw/configuration traces explicitly. Differences must
    # remain visible; new C arithmetic must never be relabelled historical Julia.
    original = joinpath(@__DIR__, "..", "tests", "fixtures", solver,
                        case * (solver == "sr_direct" && store == 1 ? "_store_runner" : "_runner"))
    for filename in sort(readdir(original))
        if occursin(r"(?:rng|configs)\.txt$", filename) && isfile(joinpath(original, filename))
            same = read(NativeFSZFixtureInheritance.resolve(target, filename)) == read(joinpath(original, filename))
            println("Historical trace comparison $solver/$case/$filename: ", same ? "exact" : "DIFFERS")
        end
    end
end
