# Explicit optional oracle for historical runner workloads. Never called by Cargo.
# Usage: Julia 1.13.1 --project=extern/Julia-mVMC this.jl STAGE direct|cg [original script options]
using SHA, Printf
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) >= 2 || error("usage: external-stage direct|cg [original script options]")
const WINDOW_STAGE = abspath(popfirst!(ARGS))
const WINDOW_METHOD = popfirst!(ARGS)
WINDOW_METHOD in ("direct", "cg") || error("unknown solver")
const WINDOW_REPO = normpath(joinpath(@__DIR__, ".."))
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
const DECLARED_SLOTS = IdDict{Any, Dict{Int, ComplexF64}}()

function capture_initial_declared_rbm!(data, values)
    offset = MVMCExpertModeParsers.projection_layout(data).n_proj
    DECLARED_SLOTS[data] = Dict(offset + i => value for (i, value) in enumerate(values))
end

function capture_declared_delta!(data, index, delta)
    slots = get!(DECLARED_SLOTS, data, Dict{Int, ComplexF64}())
    slots[index] = get(slots, index, 0.0 + 0.0im) + delta
end

function complete_declared_parameters(data)
    values = MVMCOptimizers.pack_parameters(data)
    counts = MVMCOptimizers._parameter_count_breakdown(data)
    mapped = falses(counts.n_para)
    MVMCOptimizers._foreach_parameter_location(data, counts) do index, _
        mapped[index] = true
    end
    slots = get(DECLARED_SLOTS, data, Dict{Int, ComplexF64}())
    for index in eachindex(values)
        if !mapped[index]
            haskey(slots, index) || error("uncaptured declared slot $index")
            values[index] = slots[index]
        end
    end
    values
end

function capture_declared_overlays!(data, namelist)
    names = ("InChargeRBM_PhysLayer", "InSpinRBM_PhysLayer", "InGeneralRBM_PhysLayer",
        "InChargeRBM_HiddenLayer", "InSpinRBM_HiddenLayer", "InGeneralRBM_HiddenLayer",
        "InChargeRBM_PhysHidden", "InSpinRBM_PhysHidden", "InGeneralRBM_PhysHidden")
    widths = MVMCOptimizers._parameter_section_width.(MVMCOptimizers._rbm_parameter_sections(data))
    slots = get!(DECLARED_SLOTS, data, Dict{Int, ComplexF64}())
    for line in readlines(namelist)
        fields = split(first(split(line, '#'; limit=2)))
        length(fields) == 2 || continue
        section = findfirst(==(fields[1]), names)
        section === nothing && continue
        path = normpath(joinpath(dirname(namelist), fields[2]))
        offset = MVMCExpertModeParsers.projection_layout(data).n_proj + sum(widths[1:section-1]; init=0)
        for record in readlines(path)[6:end]
            fields = split(record)
            isempty(fields) && continue
            length(fields) == 3 || error("RBM overlay fields")
            index = parse(Int, fields[1])
            0 <= index < widths[section] || error("RBM overlay index")
            slots[offset + index + 1] = complex(parse(Float64, fields[2]), parse(Float64, fields[3]))
        end
    end
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
    mkpath(destination)
    cp(joinpath(output_dir, "c_declared_var.dat"), joinpath(destination, "zvo_var.dat"); force=true)
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
        println(io, "Independent Julia observer; C declared slots and pre-SR Etot/Etot2, post-SR synchronized Para; not full C execution")
        println(io, "Julia=", VERSION, " BLAS=", LinearAlgebra.BLAS.get_config(), " threads=1")
        println(io, "options=", join(ARGS, " "), " prefix=", steps, " SR_failed=", failed)
        println(io, "successful_steps=", length(WINDOW_ROWS), " effective_window=", window,
            " selected_rows=", length(rows), " aggregatable_final_output=", !failed)
        println(io, "modpara=", repr(data.modpara))
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
# Observe copies of complete source functions, without changing arithmetic or RNG.
function install_declared_slot_observers!()
    path = joinpath(WINDOW_REPO, "extern/Julia-mVMC/MVMCExpertModeParsers.jl/src/utils/parameter_init.jl")
    raw = read(path, String)
    start = first(findfirst("function init_parameter!", raw))
    stop = last(findnext("\nend\n", raw, start))
    body = raw[start:stop]
    body = rewrite_once(body, "        # Scatter RBM values by", "        Main.capture_initial_declared_rbm!(data, rbm_values)\n        # Scatter RBM values by")
    Base.include_string(MVMCExpertModeParsers, body, path)
    path = joinpath(WINDOW_REPO, "extern/Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl")
    raw = read(path, String)
    start = first(findfirst("function _add_parameter_delta_direct!", raw))
    stop = last(findnext("\nend\n", raw, start))
    body = raw[start:stop]
    body = rewrite_once(body, "    counts = _parameter_count_breakdown(data)",
        "    Main.capture_declared_delta!(data, para_idx, delta)\n    counts = _parameter_count_breakdown(data)")
    Base.include_string(MVMCOptimizers, body, path)
end
function rewrite_once(source, needle, replacement)
    length(findall(needle, source)) == 1 || error("observer boundary changed: $needle")
    replace(source, needle => replacement; count=1)
end
source = rewrite_once(source, "BLAS.set_num_threads(1)", "BLAS.set_num_threads(1)\nMain.install_declared_slot_observers!()")
qp_boundary = "        MVMCExpertModeParsers.init_qp_weight!(data)"
length(findall(qp_boundary, source)) == (WINDOW_METHOD == "cg" ? 2 : 1) ||
    error("QP observation boundaries changed")
source = replace(source, qp_boundary =>
    "        Main.capture_declared_overlays!(data, namelist)\n        MVMCExpertModeParsers.init_qp_weight!(data)")
source = rewrite_once(source, "const FIXTURE_ROOT = joinpath", "const FIXTURE_ROOT = WINDOW_STAGE # original: joinpath")
source = rewrite_once(source, "    failed && (FAILURE_STEP[] = step)",
    "    failed && (FAILURE_STEP[] = step)\n    failed || push!(WINDOW_ROWS, vcat([state.energy.etot, state.energy.etot2], complete_declared_parameters(data)))")
source = rewrite_once(source, "Base.include_string(MVMCOptimizers, body)",
    "body = replace(body, \"output_data!(data, state, step; output_dir=output_dir)\" => \"Main.capture_declared_output!(data, state, step; output_dir=output_dir)\"; count=1)\nBase.include_string(MVMCOptimizers, body)")
source = rewrite_once(source, "            FAILURE_STEP[] = -1", "            FAILURE_STEP[] = -1\n            empty!(WINDOW_ROWS)")
source = rewrite_once(source, "            params, energy, configs = SNAPSHOTS[]",
    "            save_independent_window(data, steps, FAILURE_STEP[] != -1, dir, namelist)\n            params, energy, configs = SNAPSHOTS[]")
Base.include_string(Main, source, WINDOW_SOURCE)
