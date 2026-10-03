# Optional, independent ARM oracle. Cargo never runs this script or its bridges.
# Adapt only the BLAS integer ABI; numerical routines remain independent.
using LinearAlgebra, Libdl, SHA, Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("Julia 1.13.1 is required")
Sys.isapple() && Sys.ARCH == :aarch64 || error("macOS aarch64 is required")
BLAS.set_num_threads(1)
option(key, default="") = let values = filter(a -> startswith(a, key * "="), ARGS)
    isempty(values) ? default : split(only(values), "="; limit=2)[2]
end
const SCRIPTS = @__DIR__
const FIXTURES = normpath(joinpath(SCRIPTS, "..", "tests", "fixtures"))
const BRIDGE = option("--blas-bridge")
const FSZ_BRIDGE = option("--fsz-bridge-dir")
const SELECTED = option("--job", "all")
isempty(BRIDGE) && error("--blas-bridge is required (see docs/APPLE_SILICON_PARITY.md)")
include("reference_lp64_blas.jl")
const BLAS_PROVENANCE = ReferenceLP64BLAS.install!(BRIDGE)
include("reference_archived_avx2.jl")
const CORE = unsafe_string(ccall(Libdl.dlsym(ReferenceLP64BLAS.HANDLE[], :mvmc_reference_blas_core), Cstring, ()))
const OUTPUT = option("--output-root", joinpath(FIXTURES, "macos_arm_julia", CORE))
const PROVENANCE = "# Mixed reference: Julia 1.13.1 runner/solver, archived Julia AVX2 bilinear reduction and system LP64 OpenBLAS via an ABI adapter; not unmodified Julia or full C executable/MPI parity.\n" *
    "# macOS $(readchomp(`sw_vers -productVersion`)); arch=$(Sys.ARCH); Julia CPU=$(Sys.CPU_NAME); Manifest-v1.13.toml sha256=$(bytes2hex(sha256(read(joinpath(SCRIPTS,"..","extern","Julia-mVMC","Manifest-v1.13.toml")))))\n" * BLAS_PROVENANCE
const CASES = ("real", "cmp", "fsz", "hubbard", "interall", "pairhop_real", "pairhop_fsz", "dh2_real", "dh2_cmp", "dh2_fsz", "dh4_real", "dh4_cmp", "dh4_fsz", "dh24_real", "dh24_cmp", "dh24_fsz", "rbm_real", "rbm_cmp", "rbm_general_cmp", "rbm_dh24_cmp", "rbm_fsz", "opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp")
const NATIVE = ("fsz", "interall", "pairhop_fsz", "opt_fsz", "dh2_fsz", "dh4_fsz", "dh24_fsz")
const JOBS = vcat(
    [("fixed", "", 0, "")],
    [(solver, case, store, "") for case in CASES for (solver, store) in (("sr_direct", 0), ("sr_direct", 1), ("sr_cg", 0))],
    [(solver, case, store, "c_kernel_order/native_fsz") for case in NATIVE for (solver, store) in (("sr_direct", 0), ("sr_direct", 1), ("sr_cg", 0))],
    [("sr_direct", "rbm_reference_cmp", 1, "c_kernel_order"), ("sr_cg", "rbm_reference_cmp", 0, "c_kernel_order"),
     ("sr_direct", "opt_real", 0, "c_kernel_order"), ("sr_direct", "opt_real", 1, "c_kernel_order")])
job_name((solver, case, store, prefix)) = solver == "fixed" ? "fixed" : joinpath(prefix, solver, case, string(store))
selected = SELECTED == "all" ? JOBS : SELECTED == "remaining" ? filter(j -> !isempty(j[4]), JOBS) : filter(j -> job_name(j) == SELECTED, JOBS)
isempty(selected) && error("Unknown --job=$SELECTED")
const WRITTEN = Set{String}()
function record(path)
    push!(WRITTEN, path)
end
native_installed = false
for (index, (solver, case, store, prefix)) in enumerate(selected)
    if prefix == "c_kernel_order/native_fsz" && !native_installed
        isempty(FSZ_BRIDGE) && error("Native FSZ jobs require --fsz-bridge-dir")
        include("reference_native_fsz_energy.jl")
        NativeFSZEnergyReference.install!(FSZ_BRIDGE)
        global native_installed = true
    end
    name = Symbol("ArmReferenceJob", index)
    scope = Module(name)
    Core.eval(Main, :(const $name = $scope))
    Core.eval(scope, :(include(path) = Base.include($scope, joinpath($SCRIPTS, path))))
    if solver == "fixed"
        target = joinpath(OUTPUT, "sr_cg")
        mkpath(target)
        # The sampled input is independent, archived Julia data, not Rust data.
        cp(joinpath(FIXTURES, "sr_cg", "sampled_complex.txt"), joinpath(target, "sampled_complex.txt"); force=true)
        path = joinpath(SCRIPTS, "check_sr_cg_fixed_parity.jl")
        body = read(path, String)
        root_line = only(filter(l -> startswith(l, "const FIXTURE_ROOT = "), split(body, '\n')))
        body = replace(body, root_line => "const FIXTURE_ROOT = " * repr(target))
        for sampled in (false, true)
            fixed_name = Symbol(name, sampled ? "Sampled" : "Synthetic")
            fixed_scope = Module(fixed_name)
            Core.eval(Main, :(const $fixed_name = $fixed_scope))
            Core.eval(fixed_scope, :(include(path) = Base.include($fixed_scope, joinpath($SCRIPTS, path))))
            empty!(ARGS); push!(ARGS, "--write"); sampled && push!(ARGS, "--sampled")
            Base.include_string(fixed_scope, body, path)
        end
        for file in ("real.txt", "complex.txt", "sampled_complex.txt")
            path = joinpath(target, file)
            write(path, PROVENANCE * read(path, String)); record(path)
        end
        continue
    end
    target = joinpath(OUTPUT, prefix, solver, case * (solver == "sr_direct" && store == 1 ? "_store_runner" : "_runner"))
    path = joinpath(SCRIPTS, "check_$(solver)_runner_parity.jl")
    body = replace(read(path, String), "Main.capture_" => "Main.$name.capture_")
    root_line = only(filter(l -> startswith(l, "const FIXTURE_ROOT = "), split(body, '\n')))
    body = replace(body, root_line => "const FIXTURE_ROOT = " * repr(target))
    metadata = prefix == "c_kernel_order/native_fsz" ? PROVENANCE * NativeFSZEnergyReference.PROVENANCE[] : PROVENANCE
    body = replace(body, "function verify(name, actual)\n" => "function verify(name, actual)\n    Main.record(joinpath(FIXTURE_ROOT, name))\n    if name in (\"fixed-input.txt\", \"reference.txt\")\n        actual = $(repr(metadata)) * actual\n    end\n"; count=1)
    steps = solver == "sr_direct" && case == "hubbard" ? "1,2,3,4,5,6,7,8,9,10" : "1,2,3"
    empty!(ARGS); append!(ARGS, ["--case=$case", "--steps=$steps", "--write"])
    solver == "sr_direct" && push!(ARGS, "--store=$store")
    prefix == "c_kernel_order" && push!(ARGS, "--c-kernel-order")
    println("Independent ARM job ", job_name((solver, case, store, prefix)), "; core=$CORE")
    Base.include_string(scope, body, path)
end
mkpath(OUTPUT)
write(joinpath(OUTPUT, "PROVENANCE.txt"), PROVENANCE * "# generator sha256=$(bytes2hex(sha256(read(@__FILE__))))\n# command job=$SELECTED; OPENBLAS_CORETYPE=$(get(ENV,"OPENBLAS_CORETYPE","auto"))\n")
open(joinpath(OUTPUT, "SHA256-$(replace(SELECTED, '/' => '_')).tsv"), "w") do io
    for path in sort!(collect(WRITTEN))
        println(io, bytes2hex(sha256(read(path))), '\t', relpath(path, OUTPUT))
    end
end
if SELECTED in ("all", "remaining")
    open(joinpath(OUTPUT, "SHA256-all.tsv"), "w") do io
        for (directory, _, files) in walkdir(OUTPUT), file in sort(files)
            file in ("PROVENANCE.txt",) && continue
            endswith(file, ".tsv") && continue
            path = joinpath(directory, file)
            println(io, bytes2hex(sha256(read(path))), '\t', relpath(path, OUTPUT))
        end
    end
end
