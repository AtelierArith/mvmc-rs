# Optional native C local-energy / original Julia runner DH boundary references.
# The historical source, inputs and fixtures remain unchanged.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
VERSION == v"1.13.1" || error("Native FSZ DH fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
include("reference_native_fsz_energy.jl")
include("reference_native_fsz_fixture_inheritance.jl")
option(key, default) = let matches = filter(a -> startswith(a, key * "="), ARGS)
    isempty(matches) ? default : split(only(matches), "="; limit=2)[2]
end
directory = option("--bridge-dir", "")
output_root = option("--output-root", joinpath(@__DIR__, "..", "tests", "fixtures", "c_kernel_order", "native_fsz"))
isempty(directory) && error("First validate/build the native bridge, then pass --bridge-dir=DIR")
NativeFSZEnergyReference.install!(directory)
for (index, family) in enumerate(("dh2", "dh4"))
    name = Symbol("NativeFSZDHBoundary", index)
    namespace = Module(name)
    Core.eval(Main, :(const $name = $namespace))
    Core.eval(namespace, :(include(path) = Base.include($namespace, path)))
    path = joinpath(@__DIR__, "check_$(family)_runner_boundaries.jl")
    source = replace(read(path, String), "Main.capture_history!" => "Main.$name.capture_history!")
    input_root = joinpath(@__DIR__, "..", "tests", "fixtures", family)
    target = joinpath(output_root, family)
    mkpath(target)
    root_line = only(filter(l -> startswith(l, "const ROOT="), split(source, '\n')))
    source = replace(source, root_line => "const ROOT=" * repr(target); count=1)
    source = replace(source, "namelist=joinpath(ROOT," => "namelist=joinpath(" * repr(input_root) * ","; count=1)
    source = replace(source, "for mode in (\"real\",\"cmp\",\"fsz\")" => "for mode in (\"fsz\",)",
        "for mode in (\"dh4_real\",\"dh4_cmp\",\"dh4_fsz\",\"dh24_real\",\"dh24_cmp\",\"dh24_fsz\")" =>
        "for mode in (\"dh4_fsz\",\"dh24_fsz\")")
    source = replace(source, "function verify(name,actual)\n" =>
        "function verify(name,actual)\n    actual = Main.NativeFSZEnergyReference.PROVENANCE[] * actual\n"; count=1)
    source = replace(source, "read(joinpath(ROOT,name),String)" =>
        "read(Main.NativeFSZFixtureInheritance.resolve(ROOT,name),String)"; count=1)
    println("Native C energy / original Julia DH boundary runner: $family")
    Base.include_string(namespace, source, path)
    for filename in sort(readdir(target))
        if startswith(filename, "loaded-")
            historical = read(joinpath(input_root, filename), String)
            current = read(joinpath(target, filename), String)
            # Metadata records the current provider/platform separately. The
            # deterministic comparison concerns flags, parameters and RNG.
            rows(text) = filter(l -> !startswith(l, "#") && !isempty(l), split(text, '\n'))
            println("Historical loaded parameters/flags/RNG comparison $family/$filename: ",
                    rows(current) == rows(historical) ? "exact" : "DIFFERS")
        end
    end
end
