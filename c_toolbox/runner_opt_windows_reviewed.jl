# Separate reviewed-fork lineage. Does not modify the historical wrapper.
# REQUIRED: --reviewed-commit=40HEX --reviewed-manifest=FILE
# Manifest: SHA256<space>relative-path, rooted at extern/Julia-mVMC.
# Optional C-compatible fork acquisition. Never called by Cargo.
# Usage: Julia 1.13.1 --project=extern/Julia-mVMC this.jl STAGE direct|cg [original script options]
using SHA, Printf, LinearAlgebra
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) >= 2 || error("usage: external-stage direct|cg [original script options]")
const REQUESTED_STAGE = abspath(popfirst!(ARGS))
ispath(REQUESTED_STAGE) && error("stage already exists; no overwrites")
const WINDOW_STAGE = joinpath(realpath(dirname(REQUESTED_STAGE)), basename(REQUESTED_STAGE))
const WINDOW_METHOD = popfirst!(ARGS)
WINDOW_METHOD in ("direct", "cg") || error("unknown solver")
const CASE_OPTIONS = filter(a -> startswith(a, "--case="), ARGS)
length(CASE_OPTIONS) == 1 || error("exactly one explicit --case per stage")
const WINDOW_CASE = split(only(CASE_OPTIONS), "="; limit=2)[2]
isempty(WINDOW_CASE) && error("empty case")
const STEP_OPTIONS = filter(a -> startswith(a, "--steps="), ARGS)
length(STEP_OPTIONS) == 1 || error("one explicit --steps list required")
const WINDOW_PREFIXES = parse.(Int, split(split(only(STEP_OPTIONS), "="; limit=2)[2], ','))
!isempty(WINDOW_PREFIXES) && all(>(0), WINDOW_PREFIXES) || error("positive prefixes required")
length(unique(WINDOW_PREFIXES)) == length(WINDOW_PREFIXES) || error("duplicate prefix destination")
Threads.nthreads(:default) == 1 || error("reference default worker pool must have one thread")
const WINDOW_REPO = normpath(joinpath(@__DIR__, ".."))
const REVIEWED_OPTIONS = Dict{String,String}()
for key in ("reviewed-commit", "reviewed-manifest")
    matches = filter(a -> startswith(a, "--" * key * "="), ARGS)
    length(matches) == 1 || error("one --$key option required; no unreviewed acquisition")
    REVIEWED_OPTIONS[key] = split(only(matches), "="; limit=2)[2]
end
filter!(a -> !startswith(a, "--reviewed-"), ARGS)
const STATE_OPTIONS = Dict{String,String}()
for key in ("sfmt-state-library", "sfmt-state-library-sha256", "sfmt-state-observer-sha256")
    matches = filter(a -> startswith(a, "--" * key * "="), ARGS)
    length(matches) == 1 || error("one --$key required for actual state/count acquisition")
    STATE_OPTIONS[key] = split(only(matches), "="; limit=2)[2]
end
filter!(a -> !startswith(a, "--sfmt-state-"), ARGS)
const STATE_LIBRARY = realpath(STATE_OPTIONS["sfmt-state-library"])
const STATE_OBSERVER = joinpath(@__DIR__, "reviewed_sfmt_state.jl")
for (key, path) in (("sfmt-state-library-sha256", STATE_LIBRARY),
                    ("sfmt-state-observer-sha256", STATE_OBSERVER))
    digest = STATE_OPTIONS[key]
    length(digest) == 64 && all(isxdigit, digest) || error("invalid reviewed diagnostic hash")
    bytes2hex(sha256(read(path))) == digest || error("reviewed SFMT diagnostic mismatch")
end
const REVIEWED_COMMIT = REVIEWED_OPTIONS["reviewed-commit"]
length(REVIEWED_COMMIT) == 40 && all(isxdigit, REVIEWED_COMMIT) || error("full published commit required")
const REVIEWED_ROOT = joinpath(WINDOW_REPO, "extern/Julia-mVMC")
strip(read(`git -C $REVIEWED_ROOT rev-parse HEAD`, String)) == REVIEWED_COMMIT ||
    error("reference checkout does not match reviewed commit")
success(`git -C $REVIEWED_ROOT diff --quiet HEAD -- MVMCOptimizers.jl/src MVMCExpertModeParsers.jl/src`) ||
    error("reference production source is dirty")
include("reviewed_source_manifest.jl")
const REVIEWED_HASHES = read_reviewed_manifest(REVIEWED_OPTIONS["reviewed-manifest"], REVIEWED_ROOT)
for directory in ("MVMCOptimizers.jl/src", "MVMCExpertModeParsers.jl/src", "SFMT.jl/src", "PfaPack.jl/src")
    for (root, _, files) in walkdir(joinpath(REVIEWED_ROOT, directory)), name in files
        endswith(name, ".jl") || continue
        relative = relpath(joinpath(root, name), REVIEWED_ROOT)
        haskey(REVIEWED_HASHES, relative) || error("unreviewed production source: $relative")
    end
end
for relative in ("Project.toml", "Manifest-v1.13.toml", "MVMCOptimizers.jl/Project.toml",
                 "MVMCExpertModeParsers.jl/Project.toml", "SFMT.jl/Project.toml", "PfaPack.jl/Project.toml")
    haskey(REVIEWED_HASHES, relative) || error("unreviewed environment: $relative")
end
const NEWLINEAGE_PROVENANCE = "# Reviewed fork " * REVIEWED_COMMIT *
    "; C-faithful CG recurrence; actual full pack, no historical slot shadow\n"

const NATIVE_BRIDGE = let args = filter(a -> startswith(a, "--native-fsz-bridge="), ARGS)
    isempty(args) ? nothing : split(only(args), "="; limit=2)[2]
end
filter!(a -> !startswith(a, "--native-fsz-bridge="), ARGS)
if NATIVE_BRIDGE !== nothing
    Base.include(Main, joinpath(WINDOW_REPO, "scripts/reference_native_fsz_energy.jl"))
    Base.invokelatest(Core.getglobal(Core.getglobal(Main, :NativeFSZEnergyReference), :install!),
        String(NATIVE_BRIDGE))
end
startswith(WINDOW_STAGE * "/", WINDOW_REPO * "/") && error("stage outside repository")
"--write" in ARGS && error("wrapper controls writes exclusively in external stage")
push!(ARGS, "--write")
const WINDOW_ROWS = Vector{Vector{ComplexF64}}()
const PARAMETER_AUDIT_SEQUENCE = Ref(0)
function declared_layout(data)
    counts = MVMCOptimizers._parameter_count_breakdown(data)
    widths = [counts.layout.n_gutzwiller, counts.layout.n_jastrow,
        counts.layout.n_dh2, counts.layout.n_dh4,
        MVMCOptimizers._parameter_section_width.(MVMCOptimizers._rbm_parameter_sections(data))...,
        counts.n_orbital_idx, counts.n_opt_trans]
    length(widths) == 15 || error("declared section count")
    widths[1] + widths[2] + 6*widths[3] + 10*widths[4] + sum(widths[5:end]) ==
        counts.n_para || error("C declared section widths do not sum to NPara")
    return counts, widths
end
function complete_declared_parameters(data)
    counts, _ = declared_layout(data)
    values = MVMCOptimizers.pack_parameters(data)
    length(values) == counts.n_para || error("actual full pack width")
    all(isfinite, values) || error("nonfinite actual full pack")
    return values
end
function assert_reference_threads!()
    Threads.nthreads(:default) == 1 || error("reference worker count changed")
    LinearAlgebra.BLAS.get_num_threads() == 1 || error("reference BLAS threads must be one")
end
function install_reviewed_sfmt_state!()
    Base.include(Main, STATE_OBSERVER)
    Base.invokelatest(Core.getglobal(Core.getglobal(Main, :ReviewedSFMTState), :install!), STATE_LIBRARY)
end
function capture_parameter_audit!(data, phase; boundary=phase)
    assert_reference_threads!()
    counts, widths = declared_layout(data)
    values = complete_declared_parameters(data)
    mapped = falses(counts.n_para)
    MVMCOptimizers._foreach_parameter_location(data, counts) do index, _
        1 <= index <= counts.n_para || error("mapped index outside declared pack")
        mapped[index] = true
    end
    length(data.optimization_flags) == 2*counts.n_para || error("component flag width")
    PARAMETER_AUDIT_SEQUENCE[] += 1
    mkpath(WINDOW_STAGE)
    state_path = joinpath(WINDOW_STAGE,
        "group-$(cld(PARAMETER_AUDIT_SEQUENCE[], 3))-$phase-state.txt")
    ispath(state_path) && error("duplicate actual state checkpoint")
    Base.invokelatest(Core.getglobal(Core.getglobal(Main, :ReviewedSFMTState), :capture!), state_path)
    open(joinpath(WINDOW_STAGE, "parameter-audit-events.tsv"), "a") do io
        println(io, join((PARAMETER_AUDIT_SEQUENCE[], phase, boundary, WINDOW_CASE,
            counts.n_para, join(widths, ','), Threads.nthreads(:default),
            Threads.nthreads(:interactive), LinearAlgebra.BLAS.get_num_threads()), '\t'))
    end
    open(joinpath(WINDOW_STAGE, "parameter-audit.tsv"), "a") do io
        for index in eachindex(values)
            value = values[index]
            println(io, join((PARAMETER_AUDIT_SEQUENCE[], phase, index-1,
                Int(mapped[index]), @sprintf("%.18e", real(value)),
                @sprintf("%.18e", imag(value)), Int(data.optimization_flags[2*index-1]),
                Int(data.optimization_flags[2*index])), '\t'))
        end
    end
end
function audited_sync!(sync_function, data, boundary)
    result = sync_function(data)
    capture_parameter_audit!(data, "synchronized"; boundary)
    return result
end

function capture_declared_output!(data, state, step; output_dir)
    MVMCOptimizers.output_data!(data, state, step; output_dir)
    open(joinpath(output_dir, "c_declared_var.dat"), step == 0 ? "w" : "a") do io
        for value in vcat([state.energy.etot, state.energy.etot2], complete_declared_parameters(data))
            @printf(io, "% .18e % .18e 0.0 ", real(value), imag(value))
        end
        println(io)
    end
end

function save_independent_window(data, steps, failed, output_dir, namelist)
    destination = joinpath(WINDOW_STAGE, "step-$steps")
    ispath(destination) && error("prefix destination already exists; no overwrites")
    mkpath(destination)
    cp(joinpath(output_dir, "c_declared_var.dat"), joinpath(destination, "zvo_var.dat"))
    counts = MVMCOptimizers._parameter_count_breakdown(data)
    widths = [counts.layout.n_gutzwiller, counts.layout.n_jastrow,
        counts.layout.n_dh2, counts.layout.n_dh4,
        MVMCOptimizers._parameter_section_width.(MVMCOptimizers._rbm_parameter_sections(data))...,
        counts.n_orbital_idx, counts.n_opt_trans]
    length(widths) == 15 || error("C section count")
    widths[1] + widths[2] + 6 * widths[3] + 10 * widths[4] + sum(widths[5:end]) == counts.n_para ||
        error("C declared section widths do not sum to NPara")
    window = data.modpara.nsr_opt_itr_smp
    window > 0 || error("nonpositive effective window")
    !failed && length(WINDOW_ROWS) < window && error("incomplete effective window; no invented clamp")
    rows = failed ? WINDOW_ROWS : WINDOW_ROWS[end-window+1:end]
    history_path = joinpath(destination, failed ? "successful-history-not-final-output.txt" : "c-window-input.txt")
    open(history_path, "w") do io
        println(io, length(rows), " ", counts.n_para, " ", join(widths, " "))
        for row in rows
            length(row) == counts.n_para + 2 || error("incomplete declared history")
            println(io, join((@sprintf("%.18e %.18e", real(v), imag(v)) for v in row), " "))
        end
    end
    open(joinpath(destination, "provenance.txt"), "w") do io
        println(io, NEWLINEAGE_PROVENANCE)
        println(io, "actual_sfmt_state_observer_sha256=", STATE_OPTIONS["sfmt-state-observer-sha256"])
        println(io, "actual_sfmt_state_library_sha256=", STATE_OPTIONS["sfmt-state-library-sha256"])
        println(io, "Reviewed-fork Julia observer; C declared slots and pre-SR Etot/Etot2, post-SR synchronized Para; not full C execution")
        assert_reference_threads!()
        println(io, "Julia=", VERSION, " BLAS=", LinearAlgebra.BLAS.get_config(),
            " BLAS_threads=", LinearAlgebra.BLAS.get_num_threads(),
            " default_workers=", Threads.nthreads(:default),
            " interactive_threads=", Threads.nthreads(:interactive))
        println(io, "options=", join(ARGS, " "), " prefix=", steps, " SR_failed=", failed)
        println(io, "successful_steps=", length(WINDOW_ROWS), " effective_window=", window,
            " selected_rows=", length(rows), " aggregatable_final_output=", !failed)
        println(io, "modpara=", repr(data.modpara))
        println(io, "declared_parameter_pack_complex_count=", counts.n_para,
            " schema=C-declared-dense-pack (window input and complete history)")
        println(io, "mapped_snapshot_complex_count=", length(Main.SNAPSHOTS[][1]),
            " schema=expanded-mapped-term-values (step-N-parameters.txt); NOT declared NPara")
        if NATIVE_BRIDGE !== nothing
            println(io, Main.NativeFSZEnergyReference.PROVENANCE[])
            for path in (joinpath(NATIVE_BRIDGE, "libmvmc_fsz_reference.so"),
                         joinpath(WINDOW_REPO, "scripts/reference_native_fsz_energy.jl"))
                println(io, path, " sha256=", bytes2hex(sha256(read(path))))
            end
        end
        for path in (WINDOW_SOURCE, @__FILE__, joinpath(WINDOW_REPO, "extern/Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl"), history_path)
            println(io, path, " sha256=", bytes2hex(sha256(read(path))))
        end
        println(io, "namelist=", abspath(namelist))
        for (directory, _, files) in walkdir(dirname(namelist))
            for name in sort(files)
                path = joinpath(directory, name)
                println(io, path, " sha256=", bytes2hex(sha256(read(path))))
            end
        end
        for (relative, digest) in sort!(collect(REVIEWED_HASHES); by=first)
            println(io, "reviewed_source ", relative, " sha256=", digest)
        end
        println(io, "JuliaProject sha256=", bytes2hex(sha256(read(joinpath(WINDOW_REPO, "extern/Julia-mVMC/Project.toml")))))
        println(io, "JuliaManifest sha256=", bytes2hex(sha256(read(joinpath(WINDOW_REPO, "extern/Julia-mVMC/Manifest-v1.13.toml")))))
        println(io, "C kernel shim sha256=", bytes2hex(sha256(read(joinpath(WINDOW_REPO, "scripts/reference_c_kernel_order.jl")))))
    end
end

const WINDOW_SOURCE = joinpath(WINDOW_REPO, "scripts/check_sr_$(WINDOW_METHOD)_runner_parity.jl")
source = read(WINDOW_SOURCE, String)
for helper in ("reference_numerical_comparison.jl", "reference_c_kernel_order.jl")
    global source = replace(source, "include(\"$helper\")" =>
        "include($(repr(joinpath(dirname(WINDOW_SOURCE), helper))))")
end
function rewrite_once(source, needle, replacement)
    length(findall(needle, source)) == 1 || error("observer boundary changed: $needle")
    replace(source, needle => replacement; count=1)
end
# Read-only observations around the original initializer/overlay/sync calls.
for (needle, phase) in (("MVMCExpertModeParsers.init_parameter!(data; rng)", "initialized"),
                         ("MVMCExpertModeParsers.read_input_parameters!(data,namelist)", "overlaid"))
    isempty(findall(needle, source)) && error("parameter audit boundary changed: $needle")
    global source = replace(source, needle => needle *
        "\n        Main.capture_parameter_audit!(data, " * repr(phase) * ")" *
        (phase == "initialized" ?
            "\n        !LOADED_CASE && Main.capture_parameter_audit!(data, \"overlaid\"; boundary=\"no-overlay-files\")" : ""))
end
for module_name in ("MVMCExpertModeParsers", "MVMCOptimizers")
    call = module_name * ".sync_modified_parameter!(data)"
    isempty(findall(call, source)) && error("actual sync boundary changed: $call")
    global source = replace(source, call =>
        "Main.audited_sync!(" * module_name * ".sync_modified_parameter!, data, " * repr(call) * ")")
end
source = rewrite_once(source, "BLAS.set_num_threads(1)",
    "BLAS.set_num_threads(1)\nMain.assert_reference_threads!()\nMain.install_reviewed_sfmt_state!()")
source = replace(source, "Julia-mVMC 8bb1b9e; numerical sources c2ea432;" =>
    "reviewed Julia fork $REVIEWED_COMMIT; C-faithful CG;")
source = rewrite_once(source, "    path = joinpath(FIXTURE_ROOT, name)",
    "    name == \"reference.txt\" && (actual = NEWLINEAGE_PROVENANCE * actual)\n    path = joinpath(FIXTURE_ROOT, name)")
source = rewrite_once(source, "const FIXTURE_ROOT = joinpath", "const FIXTURE_ROOT = WINDOW_STAGE # original: joinpath")
source = rewrite_once(source, "    failed && (FAILURE_STEP[] = step)",
    "    failed && (FAILURE_STEP[] = step)\n    failed || push!(WINDOW_ROWS, vcat([state.energy.etot, state.energy.etot2], complete_declared_parameters(data)))")
source = rewrite_once(source, "Base.include_string(MVMCOptimizers, body)",
    "body = replace(body, \"output_data!(data, state, step; output_dir=output_dir)\" => \"Main.capture_declared_output!(data, state, step; output_dir=output_dir)\"; count=1)\nBase.include_string(MVMCOptimizers, body)")
source = rewrite_once(source, "            FAILURE_STEP[] = -1", "            FAILURE_STEP[] = -1\n            empty!(WINDOW_ROWS)")
source = rewrite_once(source, "            params, energy, configs = SNAPSHOTS[]",
    "            save_independent_window(data, steps, FAILURE_STEP[] != -1, dir, namelist)\n            params, energy, configs = SNAPSHOTS[]")
using MVMCOptimizers, MVMCExpertModeParsers
for module_ref in (MVMCOptimizers, MVMCExpertModeParsers)
    startswith(realpath(pathof(module_ref)), realpath(REVIEWED_ROOT) * "/") ||
        error("loaded package is not from the reviewed fork")
end
Base.include_string(Main, source, WINDOW_SOURCE)
