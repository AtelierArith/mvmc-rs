# Optional regeneration of explicitly labelled mixed-reference regressions.
# Reuse one Julia process; isolate each script's constants and observer hooks.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra

jobs = [
    ("sr_direct", "rbm_reference_cmp", 1, "1,2,3,50"),
    ("sr_cg", "rbm_reference_cmp", 0, "1,2,3,50"),
    ("sr_direct", "opt_real", 0, "1,2,3,27,28,29,50"),
    ("sr_direct", "opt_real", 1, "1,2,3,50"),
]
for (index, (solver, case, store, steps)) in enumerate(jobs)
    name = Symbol("CKernelReferenceCase", index)
    namespace = Module(name)
    Core.eval(Main, :(const $name = $namespace))
    Core.eval(namespace, :(include(path) = Base.include($namespace, path)))
    path = joinpath(@__DIR__, "check_$(solver)_runner_parity.jl")
    source = replace(read(path, String), "Main.capture_" => "Main.$name.capture_")
    empty!(ARGS)
    append!(ARGS, ["--case=$case", "--steps=$steps", "--c-kernel-order", "--write"])
    solver == "sr_direct" && push!(ARGS, "--store=$store")
    println("Generating $solver/$case store=$store steps=$steps")
    Base.include_string(namespace, source, path)
end
