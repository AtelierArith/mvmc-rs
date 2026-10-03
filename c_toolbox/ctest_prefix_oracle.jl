# Optional independent #180 observer; no Rust results or toolbox Cargo dependency.
# Writes only to the explicitly supplied external staging directory.
using MVMCOptimizers, MVMCExpertModeParsers, SFMT, Random, LinearAlgebra, SHA, Printf
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) in (2,3) || error("usage: output-dir model[,model...] [steps; default1,2,3,20]")
const REPO = normpath(joinpath(@__DIR__, ".."))
const DESTINATION = abspath(ARGS[1])
startswith(DESTINATION * "/", REPO * "/") && error("stage outside the repository")
const MODELS = split(ARGS[2], ',')
const PREFIXES = length(ARGS)==3 ? parse.(Int, split(ARGS[3], ',')) : [1,2,3,20]
all(>(0), PREFIXES) || error("prefixes must be positive")
const REFERENCE_COMMIT = strip(read(`git -C $(joinpath(REPO,"extern/Julia-mVMC")) rev-parse HEAD`,String))
if 20 in PREFIXES
    REFERENCE_COMMIT=="62b0f97f076fb55c71c3ab0caa041a9adff94e04" ||
        error("fresh20 requires the reviewed62b reference snapshot, not historical/current unrelated Julia source")
end
BLAS.set_num_threads(1)
const PROJECT = joinpath(REPO, "extern/Julia-mVMC")
const IDENTITY = joinpath(REPO, "reviewed-source.sha256")
bytes2hex(sha256(read(IDENTITY))) == "4a22c8b21d2901bd3d88e08b65dcefbb349c5ec259583fd8d3c0482b73cec0b4" || error("reviewed source manifest identity")
identity_rows = split.(filter(line -> !startswith(line, "#") && !isempty(strip(line)), readlines(IDENTITY)))
length(identity_rows) == 63 || error("reviewed snapshot must contain all63 source identities")
seen_sources = Set{String}()
for row in identity_rows
    length(row) == 2 || error("invalid source identity row")
    hash, file = row
    file in seen_sources && error("duplicate source identity")
    push!(seen_sources, file)
    bytes2hex(sha256(read(joinpath(PROJECT, file)))) == hash || error("source identity mismatch: $file")
end
"Manifest-v1.13.toml" in seen_sources || error("Manifest identity missing")
for package in (MVMCOptimizers, MVMCExpertModeParsers, SFMT)
    path = realpath(pathof(package))
    startswith(path, realpath(PROJECT) * "/") || error("loaded package outside snapshot: $path")
    relpath(path, PROJECT) in seen_sources || error("loaded package not source-audited: $path")
end
const SFMT_LIBRARY = joinpath(REPO, "c_toolbox/libsfmt-observed-final.so")
bytes2hex(sha256(read(SFMT_LIBRARY))) == "7e77954acae2073591edf17b5c9825f0c020a2525026ae450ac31b780d7a7c86" || error("SFMT library identity")
const SFMT_OBSERVER = joinpath(REPO, "c_toolbox/reviewed_sfmt_state.jl")
bytes2hex(sha256(read(SFMT_OBSERVER))) == "91102d67e9674c3bbc3fb1ceca8b4dbfa32c2440219a8b4e48e2a70313ec2591" || error("SFMT observer identity")
include(SFMT_OBSERVER)
ReviewedSFMTState.install!(SFMT_LIBRARY)

# Reviewed62b contains the C-faithful kernels already. Never install the
# historical retained-coefficient/initialization replacements on this source.
function source_function(file, signature)
    source = read(file, String)
    a = first(findfirst(signature, source))
    b = last(findnext("\nend\n", source, a))
    source[a:b]
end
const CTEST_BRIDGE = get(ENV, "MVMC_CTEST_NATIVE_FSZ_BRIDGE", "")
if !isempty(CTEST_BRIDGE)
    include(joinpath(REPO, "scripts/reference_native_fsz_energy.jl"))
    NativeFSZEnergyReference.install!(CTEST_BRIDGE)
end
const SNAPSHOT = Ref{Any}()
const PRE_SR = Ref{Any}()
const ORACLE_RNG = Ref{Any}()
const FAILURES = String[]
const WINDOW_HISTORY = Vector{Vector{ComplexF64}}()
const RBM_SOLVER = get(ENV, "MVMC_CTEST_RBM_SOLVER", "canonical")
RBM_SOLVER in ("canonical", "direct", "cg") || error("invalid RBM solver")
function ctest_input_order!(data)
    if any(!isempty, MVMCOptimizers._rbm_parameter_sections(data))
        # Actual C arrays are spin-major. These are complete canonical tables,
        # not the sparse archived internal RBM workloads.
        sort!(data.general_rbm_phys_layer_terms; by=t -> (t.spin, t.site))
        sort!(data.general_rbm_phys_hidden_terms; by=t -> (t.spin, t.site1, t.site2))
        if RBM_SOLVER != "canonical"
            data.modpara.nsrcg = RBM_SOLVER == "cg" ? 1 : 0
            data.modpara.nstore_o = RBM_SOLVER == "cg" ? 0 : 1
        end
    end
    data
end
ctest_capture(data, state, rng) = begin
    SNAPSHOT[] = (deepcopy(data), deepcopy(state))
    ORACLE_RNG[] = rng
    push!(WINDOW_HISTORY, vcat([state.energy.etot, state.energy.etot2], MVMCOptimizers.pack_parameters(data)))
    nothing
end
ctest_capture_sr(state) = (PRE_SR[] = deepcopy(state.sr_opt); nothing)

# C vmcmain.c outputData(), lines 653-657, emits Para[0:NPara], not
# Julia's mapped orbital_terms (which may duplicate or reorder slots).
# Observe the same pre-SR data; do not alter parameters, state, or RNG.
function ctest_output(data, state, step; output_dir)
    MVMCOptimizers.output_data!(data, state, step; output_dir=output_dir)
    head = isempty(data.modpara.c_data_file_head) ? "zvo" : data.modpara.c_data_file_head
    open(joinpath(output_dir, head * "_c_slots_var.dat"), step == 0 ? "w" : "a") do io
        @printf(io, "% .18e % .18e 0.0 % .18e % .18e 0.0 ",
            real(state.energy.etot), imag(state.energy.etot),
            real(state.energy.etot2), imag(state.energy.etot2))
        for value in MVMCOptimizers.pack_parameters(data)
            @printf(io, "% .18e % .18e 0.0 ", real(value), imag(value))
        end
        println(io)
    end
end

# Extract complete Julia function bodies through their top-level end, retaining
# every original operation and draw. Insert only read-only observation hooks.
const SOURCE = joinpath(REPO, "extern/Julia-mVMC/MVMCOptimizers.jl/src")
function observe_copy(file, signature, new_signature, substitutions)
    body = source_function(joinpath(SOURCE, file), signature)
    body = replace(body, signature => new_signature; count=1)
    for (needle, replacement) in substitutions
        length(findall(needle, body)) == 1 || error("observation boundary changed: $needle")
        body = replace(body, needle => replacement; count=1)
    end
    Base.include_string(MVMCOptimizers, body)
end
observe_copy("vmc_para_opt.jl", "function vmc_para_opt!(", "function ctest_observed_opt!(", [
    "output_data!(data, state, step; output_dir=output_dir)" =>
        "Main.ctest_output(data, state, step; output_dir=output_dir)",
    "        # 8. Stochastic optimization" =>
        "        Main.ctest_capture_sr(state)\n        # 8. Stochastic optimization",
    "        # Callback" =>
        "        Main.ctest_capture(data, state, rng)\n        # Callback",
])
observe_copy("run_para_opt_from_namelist.jl", "function run_para_opt_from_namelist(",
    "function ctest_observed_runner(", [
    "data = MVMCExpertModeParsers.parse_expert_mode_files(namelist_str)" =>
        "data = Main.ctest_input_order!(MVMCExpertModeParsers.parse_expert_mode_files(namelist_str))",
    "    status = vmc_para_opt!(" => "    status = ctest_observed_opt!(",
])

function numeric(path, values)
    open(path, "w") do io
        println(io, join(repr.(vec(collect(reinterpret(Float64, ComplexF64.(vec(values)))))), " "))
    end
end
ispath(DESTINATION) && error("refuse to overwrite existing oracle stage")
mkdir(DESTINATION)
open(joinpath(DESTINATION, "provenance.txt"), "w") do io
    println(io, "Julia=$VERSION platform=$(Sys.MACHINE) threads=$(BLAS.get_num_threads()) BLAS=$(BLAS.get_config())")
    println(io, "Manifest-v1.13.toml sha256=", bytes2hex(sha256(read(joinpath(REPO, "extern/Julia-mVMC/Manifest-v1.13.toml")))))
    println(io, "observer sha256=", bytes2hex(sha256(read(@__FILE__))))
    println(io, "reference_commit=$REFERENCE_COMMIT")
    println(io, "reviewed-source.sha256 sha256=", bytes2hex(sha256(read(IDENTITY))))
    println(io, read(IDENTITY, String))
    for package in (MVMCOptimizers, MVMCExpertModeParsers, SFMT)
        println(io, "loaded_package=", nameof(package), " path=", realpath(pathof(package)))
    end
    println(io, "SFMT_library sha256=", bytes2hex(sha256(read(SFMT_LIBRARY))))
    println(io, "SFMT_observer sha256=", bytes2hex(sha256(read(joinpath(REPO,"c_toolbox/reviewed_sfmt_state.jl")))))
    println(io, "reviewed62b original kernels; no historical retention/initialization/CG replacements")
    println(io, "GeneralRBM complete canonical tables: C spin-major physical/coupling order; solver_override=$RBM_SOLVER")
    println(io, "C output layout: vmcmain.c outputData lines 653-657; contiguous declared Para slots; original Julia zvo_var retained")
    println(io, "vmcmain.c sha256=", bytes2hex(sha256(read(joinpath(REPO, "extern/mVMC-1.3.0/src/mVMC/vmcmain.c")))))
    if !isempty(CTEST_BRIDGE)
        println(io, NativeFSZEnergyReference.PROVENANCE[])
        println(io, "native bridge sha256=", bytes2hex(sha256(read(joinpath(CTEST_BRIDGE, "libmvmc_fsz_reference.so")))))
    end
    # Direct and CG implementations share stochastic_opt.jl in this checkout.
    for file in ("vmc_para_opt.jl", "run_para_opt_from_namelist.jl", "stochastic_opt.jl")
        path = joinpath(SOURCE, file)
        isfile(path) && println(io, file, " sha256=", bytes2hex(sha256(read(path))))
    end
end
for model in MODELS
    input = joinpath(REPO, "extern/Julia-mVMC/test/integration/reference", model, "inputs/namelist.def")
    original = parse_expert_mode_files(input)
    ctest_input_order!(original)
    mode = original.i_flg_orbital_general != 0 ? :fsz : endswith(model, "cmp") ? :cmp : :real
    for steps in PREFIXES
        case = joinpath(DESTINATION, model, "step-$steps")
        model_directory = dirname(case)
        isdir(model_directory) || mkdir(model_directory)
        ispath(case) && error("refuse to overwrite existing oracle case")
        mkdir(case)
        open(joinpath(case, "model-settings.txt"), "w") do io
            println(io, "model=$model mode=$mode RndSeed=$(original.modpara.rnd_seed)")
            println(io, "Julia_default_threads=$(Threads.nthreads(:default)) Julia_interactive_threads=$(Threads.nthreads(:interactive)) BLAS_threads=$(BLAS.get_num_threads()) MPI=serial workers=1")
            has_rbm = any(!isempty, MVMCOptimizers._rbm_parameter_sections(original))
            println(io, "energy_backend=", original.i_flg_orbital_general != 0 &&
                !isempty(CTEST_BRIDGE) && !has_rbm ? "native_C_FSZ_bridge" : "reviewed_Julia_kernel")
            println(io, "steps=$steps window=$steps NSRCG=$(original.modpara.nsrcg) NStore=$(original.modpara.nstore_o)")
            println(io, "canonical_NSROptItrStep=$(original.modpara.nsr_opt_itr_step) canonical_NSROptItrSmp=$(original.modpara.nsr_opt_itr_smp)")
            println(io, "effective_NSROptItrStep=$steps effective_NSROptItrSmp=$steps override=both_no_clamp")
            initial = joinpath(dirname(input), "initial.def")
            println(io, "initial_overlay=$(isfile(initial)) initial_order=after_parameter_initialization_before_In_files_and_sync")
            isfile(initial) && println(io, "initial.def sha256=", bytes2hex(sha256(read(initial))))
            println(io, "In_overlays=canonical_namelist_entries; full_input_hashes=inputs.sha256")
        end
        SNAPSHOT[] = nothing
        PRE_SR[] = nothing
        empty!(WINDOW_HISTORY)
        try
            result = Base.invokelatest(MVMCOptimizers.ctest_observed_runner, input;
                nsteps=steps, nsmp=steps, mode=mode, output_dir=case)
            result.status == 0 || error("SR status $(result.status)")
            data, state = SNAPSHOT[]
            # Standalone native avevar.c probe consumes this independent Julia
            # history. This is not Julia's final-parameter zqp extension.
            counts = MVMCOptimizers._parameter_count_breakdown(data)
            widths = [counts.layout.n_gutzwiller, counts.layout.n_jastrow,
                counts.layout.n_dh2, counts.layout.n_dh4,
                MVMCOptimizers._parameter_section_width.(MVMCOptimizers._rbm_parameter_sections(data))...,
                counts.n_orbital_idx, counts.n_opt_trans]
            length(WINDOW_HISTORY) == steps || error("missing window snapshots")
            length(widths) == 15 && sum(widths) == counts.n_para || error("declared family schema")
            all(row -> length(row) == 2 + counts.n_para && all(isfinite, row), WINDOW_HISTORY) || error("invalid dense window rows")
            open(joinpath(case, "c-window-input.txt"), "w") do io
                println(io, steps, " ", counts.n_para, " ", join(widths, " "))
                for row in WINDOW_HISTORY
                    println(io, join(repr.(collect(reinterpret(Float64, row))), " "))
                end
            end
            open(joinpath(case, "configs.txt"), "w") do io
                cfg = state.electron_config
                rows = Any[cfg.ele_idx, cfg.ele_cfg, cfg.ele_num, cfg.ele_proj_cnt]
                data.i_flg_orbital_general != 0 && push!(rows, cfg.ele_spn)
                append!(rows, [cfg.burn_ele_idx, vcat(cfg.counter[1:9], cfg.counter[11])])
                for values in rows
                    println(io, join(values, " "))
                end
            end
            ReviewedSFMTState.capture!(joinpath(case, "rng-state.txt"))
            before = ReviewedSFMTState.snapshot()
            next_words = Vector{UInt32}(undef, 624)
            SFMT.C_API.sfmt_dump_rand32(next_words, 624)
            ReviewedSFMTState.snapshot() == before || error("next624 peek changed SFMT state")
            write(joinpath(case, "rng.txt"), join(next_words, " ") * "\n")
            numeric(joinpath(case, "parameters.txt"), MVMCOptimizers.pack_parameters(data))
            numeric(joinpath(case, "energy.txt"), [state.energy.etot])
            sr = PRE_SR[]
            numeric(joinpath(case, "sr_oo.txt"), MVMCOptimizers.get_all_complex_flag(data) ? sr.sr_opt_oo : sr.sr_opt_oo_real)
            numeric(joinpath(case, "sr_ho.txt"), MVMCOptimizers.get_all_complex_flag(data) ? sr.sr_opt_ho : sr.sr_opt_ho_real)
            write(joinpath(case, "status.txt"), "0\n")
            open(joinpath(case, "inputs.sha256"), "w") do io
                for file in sort(readdir(dirname(input)))
                    path = joinpath(dirname(input), file)
                    isfile(path) && println(io, bytes2hex(sha256(read(path))), "  ", file)
                end
            end
            println("ORACLE EXECUTED $model prefix=$steps seed=$(original.modpara.rnd_seed) output=$case")
        catch error
            push!(FAILURES, "$model prefix=$steps")
            write(joinpath(case, "UNVERIFIED.txt"), sprint(showerror, error) * "\n")
            println(stderr, "ORACLE UNVERIFIED $model prefix=$steps: ", sprint(showerror, error))
        end
    end
end
isempty(FAILURES) || error("Unverified oracle prefixes: $(join(FAILURES, ", "))")
