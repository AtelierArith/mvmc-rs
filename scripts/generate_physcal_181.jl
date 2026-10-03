# Optional developer command; normal Cargo tests never invoke this generator.
# julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_181.jl
using MVMCOptimizers, MVMCExpertModeParsers, SFMT, Random, LinearAlgebra, SHA, Dates, Libdl, Printf

VERSION == v"1.13.1" || error("Julia 1.13.1 is required")
const repo = normpath(joinpath(@__DIR__, ".."))
const generator_hash = bytes2hex(sha256(read(@__FILE__)))
const reference = joinpath(repo, "extern", "Julia-mVMC")
const two_samples = get(ENV, "PHYSCAL181_TWO_SAMPLES", "0") == "1"
const base_destination = isempty(ARGS) ? joinpath(repo, "tests", "fixtures", "physcal_181") : abspath(ARGS[1])
const destination = two_samples ? joinpath(base_destination, "two-samples") : base_destination
const models = [("heisenberg_chain_real", :real), ("heisenberg_chain_cmp", :cmp),
    ("heisenberg_chain_fsz", :fsz), ("hubbard_chain_real", :real),
    ("hubbard_chain_dh_real", :real), ("kondo_chain_real", :real),
    ("hubbard_chain_dh_overlays", :real),
    ("hubbard_chain_dh_opttrans", :real),
    ("hubbard_chain_dh_rbm_opttrans", :cmp),
    ("hubbard_all_terms_lanczos1", :real), ("hubbard_all_terms_lanczos2", :real)]
const selected = split(get(ENV, "PHYSCAL181_MODELS", join(first.(models), ",")), ",")
const metadata_only = get(ENV, "PHYSCAL181_METADATA_ONLY", "0") == "1"
all(name -> name in first.(models), selected) || error("unknown PHYSCAL181_MODELS selection")
const draws = Ref(0)
const model_directory = Ref("")
const fixed_parameters = Ref(ComplexF64[])
const phase_order_case = Ref(false)
BLAS.set_num_threads(1)
const blas_library = first(BLAS.get_config().loaded_libs)
const blas_version = unsafe_string(ccall(Libdl.dlsym(blas_library.handle,
    Symbol("openblas_get_config", blas_library.suffix)), Cstring, ()))
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

function generate_average()
    dir = joinpath(destination, "weighted-average")
    mkpath(dir)
    state = MVMCOptimizers.VMCOptimizationState(2, 1, 0, 0, 1, 1, true, false)
    state.phys_quantities = MVMCOptimizers.PhysicalQuantities(3, 2, 2)
    for (field, value) in zip((:wc, :etot, :etot2, :sztot, :sztot2),
        ComplexF64[2.5+0.125im, -3.25+0.5im, 7.75-0.25im, 0.5+0.25im, 1.25-0.125im])
        setproperty!(state.energy, field, value)
    end
    state.phys_quantities.phys_cis_ajs .= ComplexF64[0.5+0.25im, -0.75-0.125im, 1.5+0.5im]
    state.phys_quantities.phys_cis_ajs_ckt_alt .= ComplexF64[0.25+0.125im, -0.5+0.375im]
    state.phys_quantities.phys_cis_ajs_ckt_alt_dc .= ComplexF64[1.25-0.25im, -1.5+0.5im]
    function save_average(stage)
        open(joinpath(dir, "$stage.txt"), "w") do io
            energy = [getproperty(state.energy, field) for field in (:wc, :etot, :etot2, :sztot, :sztot2)]
            phys = state.phys_quantities
            for value in vcat(energy, phys.phys_cis_ajs, phys.phys_cis_ajs_ckt_alt, phys.phys_cis_ajs_ckt_alt_dc)
                println(io, repr(real(value)), " ", repr(imag(value)))
            end
        end
    end
    save_average("accumulated")
    MVMCOptimizers.weight_average_we!(state)
    MVMCOptimizers.weight_average_green_func!(state)
    save_average("averaged")
    open(joinpath(dir, "provenance.txt"), "w") do io
        println(io, "generator_sha256=$generator_hash\njulia=$VERSION\nreference_revision=$revision\nblas_version=$blas_version")
        println(io, "scope=standalone weighted-average kernels, exact binary-rational synthetic inputs; not a full sampling/measurement trace")
        println(io, "layout=wc,etot,etot2,sztot,sztot2; three one-body; two factored; two direct complex values")
        println(io, "C_authority=src/mVMC/average.c WeightAverageWE/WeightAverageGreenFunc: reciprocal Wc then ordered multiplication")
    end
end

function integers(path, values)
    open(path, "w") do io
        println(io, join(values, " "))
    end
end

function c_definition_flags(inputs)
    definitions = Dict(split(line)[1] => split(line)[2] for line in
        split(read(joinpath(inputs, "namelist.def"), String), '\n') if length(split(line)) == 2)
    flags = Int[]
    written = Int[]
    n_proj = 0
    n_slater = 0
    order = ["Gutzwiller", "Jastrow", "DH2", "DH4",
        "ChargeRBM_PhysLayer", "SpinRBM_PhysLayer", "GeneralRBM_PhysLayer",
        "ChargeRBM_HiddenLayer", "SpinRBM_HiddenLayer", "GeneralRBM_HiddenLayer",
        "ChargeRBM_PhysHidden", "SpinRBM_PhysHidden", "GeneralRBM_PhysHidden",
        "Orbital", "OrbitalGeneral", "OrbitalParallel"]
    for key in order
        haskey(definitions, key) || continue
        lines = filter(!isempty, strip.(split(read(joinpath(inputs, definitions[key]), String), '\n')))
        width = parse(Int, split(lines[2])[2]) * (key == "DH2" ? 6 : key == "DH4" ? 10 : 1)
        complex = parse(Int, split(lines[3])[2]) > 0
        records = lines[end-width+1:end]
        for record in records
            fields = split(record)
            length(fields) == 2 || error("incomplete C flags in $key")
            raw = parse(Int, fields[2])
            append!(flags, key == "OrbitalParallel" ? [raw, Int(complex), raw, Int(complex)] :
                [raw, complex ? raw : 0])
            append!(written, key == "OrbitalParallel" ? [1, 1, 1, 1] : [1, Int(complex)])
        end
        key in order[1:4] && (n_proj += width)
        startswith(key, "Orbital") && (n_slater += key == "OrbitalParallel" ? 2width : width)
    end
    if haskey(definitions, "OptTrans")
        lines = split(read(joinpath(inputs, definitions["OptTrans"]), String), '\n')
        count = parse(Int, split(lines[2])[2])
        append!(flags, zeros(Int, 2count))
        append!(written, zeros(Int, 2count))
        # Native readdef.c:1064–1070,2385–2402 writes contiguous entries at
        # NProj+NOrbitalIdx, NOT interleaved entries at the parameter tail.
        flags[n_proj+n_slater+1:n_proj+n_slater+count] .= 1
        written[n_proj+n_slater+1:n_proj+n_slater+count] .= 1
    end
    flags, written
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
    fixed_parameters[] = copy(values)
    open(joinpath(model_directory[], "fixed-parameters.txt"), "w") do io
        for value in values
            println(io, repr(real(value)), " ", repr(imag(value)))
        end
    end
end

function measurement_checkpoint(stage, state, data, sample)
    dir = joinpath(model_directory[], stage)
    mkpath(dir)
    energy = [getproperty(state.energy, field) for field in (:wc, :etot, :etot2, :sztot, :sztot2)]
    phys = state.phys_quantities
    for (name, values) in [("energy", energy), ("one", phys.phys_cis_ajs),
        ("factored", phys.phys_cis_ajs_ckt_alt), ("direct", phys.phys_cis_ajs_ckt_alt_dc)]
        open(joinpath(dir, "$name.txt"), "w") do io
            for value in values; println(io, repr(real(value)), " ", repr(imag(value))); end
        end
    end
    checkpoint(stage) # Measurement and averaging must not consume RNG.
    startswith(stage, "averaged-") || return
    expected = joinpath(model_directory[], "expected")
    mkpath(expected)
    index = sample + data.modpara.n_data_idx_start
    label = lpad(index, 3, '0')
    e = state.energy
    variance = real((e.etot2 - e.etot*e.etot)/(e.etot*e.etot))
    open(joinpath(expected, "zvo_out_$label.dat"), "w") do io
        @printf(io, "% .18e % .18e % .18e % .18e %.18e %.18e\n", real(e.etot), imag(e.etot), real(e.etot2), variance, real(e.sztot), real(e.sztot2))
    end
    open(joinpath(expected, "zvo_var_$label.dat"), "w") do io
        @printf(io, "% .18e % .18e 0.0 % .18e % .18e 0.0 ", real(e.etot), imag(e.etot), real(e.etot2), imag(e.etot2))
        for value in MVMCOptimizers.pack_parameters(data)
            @printf(io, "% .18e % .18e 0.0 ", real(value), imag(value))
        end
        println(io)
    end
end

function restored_parameters(data)
    if phase_order_case[]
        # Explicit C phase-order harness, NOT a pass of Julia's PhysCal runner:
        # its save/restore omits RBM/OptTrans. Keep numerical kernels untouched.
        MVMCOptimizers.read_opt_para_file!(data, joinpath(model_directory[], "zqp_opt.dat"))
        MVMCOptimizers.read_input_parameters!(data, joinpath(model_directory[], "inputs", "namelist.def"))
        MVMCOptimizers.sync_modified_parameter!(MVMCOptimizers.serial_context(), data;
            shift_correlations = false)
        loaded_parameters(data)
    end
    actual = MVMCOptimizers.pack_parameters(data)
    actual == fixed_parameters[] || error("Julia PhysCal InitParameter save/restore changed fixed parameters; not C-compatible")
end

const physcal_path = joinpath(pkgdir(MVMCOptimizers), "src", "vmc_phys_cal.jl")
source = read(physcal_path, String)
for (needle, replacement) in [
    "    # Save current parameter values" => "    Main.loaded_parameters(data)\n    # Save current parameter values",
    "    # Restore the original parameter values" => "    Main.checkpoint(\"initialized\")\n    # Restore the original parameter values",
    "    # Get parameters" => "    Main.restored_parameters(data)\n    # Get parameters",
    "        # Main calculation (energy + Green's functions)" => "        Main.checkpoint(\"sample-\$(ismp)\", state)\n        # Main calculation (energy + Green's functions)",
    "        # Weighted averages" => "        Main.measurement_checkpoint(\"accumulated-\$(ismp)\", state, data, ismp)\n        # Weighted averages",
    "        # Reduce counters" => "        Main.measurement_checkpoint(\"averaged-\$(ismp)\", state, data, ismp)\n        # Reduce counters",
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
    name in selected || continue
    if metadata_only
        # No initialization or sampling: independently capture loader counts
        # and binary component flags from the checked-in complete inputs.
        dir = joinpath(destination, name)
        data = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(dir, "inputs", "namelist.def"))
        consumed = MVMCOptimizers.read_opt_para_file!(data, joinpath(dir, "zqp_opt.dat"))
        MVMCOptimizers.read_input_parameters!(data, joinpath(dir, "inputs", "namelist.def"))
        integers(joinpath(dir, "consumed-count.txt"), [consumed])
        flags, written = c_definition_flags(joinpath(dir, "inputs"))
        integers(joinpath(dir, "optimization-flags.txt"), flags)
        integers(joinpath(dir, "optimization-flags-written.txt"), written)
        open(joinpath(dir, "metadata-provenance.txt"), "w") do io
            println(io, "generator_sha256=$generator_hash\njulia=$VERSION\nreference_revision=$revision")
            println(io, "stage=after fixed record and named In overlays; no initialization or sampling")
            println(io, "flags=independent C GetInfoOpt/OptTrans enumeration from definition records, not Julia Boolean flags or Rust state")
            println(io, "flags_written=exact C reader writes; unwritten cells are not independent expectations for native allocation contents")
            println(io, "C_flag_source_sha256=$(bytes2hex(sha256(read(joinpath(repo, "extern", "mVMC-1.3.0", "src", "mVMC", "readdef.c")))))")
        end
        continue
    end
    overlay_case = name == "hubbard_chain_dh_overlays"
    phase_order_case[] = name in ("hubbard_chain_dh_opttrans", "hubbard_chain_dh_rbm_opttrans")
    combination_case = phase_order_case[]
    all_terms_case = startswith(name, "hubbard_all_terms_lanczos")
    source_name = all_terms_case ? "hubbard_chain_real" :
        overlay_case || combination_case ? "hubbard_chain_dh_real" : name
    fixture = joinpath(reference, "test", "integration", "reference", source_name, "physcal_ref")
    model_directory[] = joinpath(destination, name)
    mkpath(model_directory[])
    inputs = joinpath(model_directory[], "inputs")
    # Copy only on an explicit developer regeneration. No Cargo dependency on
    # vendor/runtime. The namelist and all sampling counts are unchanged.
    cp(joinpath(fixture, "inputs"), inputs; force = true)
    cp(joinpath(fixture, "zqp_opt.dat"), joinpath(model_directory[], "zqp_opt.dat"); force = true)
    if all_terms_case
        for (key, file, coefficient) in [("CoulombInter", "coulombinter", 0.375),
            ("Hund", "hund", -0.125), ("Exchange", "exchange", 0.25),
            ("PairHop", "pairhop", -0.0625)]
            write(joinpath(inputs, "$file.def"),
                "======================\nN$key 1\n======================\n$key\n======================\n0 1 $coefficient\n")
            open(joinpath(inputs, "namelist.def"), "a") do io
                println(io, "$key $file.def")
            end
        end
        path = joinpath(inputs, "modpara.def")
        write(path, replace(read(path, String), r"(?m)^\s*NLanczosMode\s+\d+\s*$" =>
            "NLanczosMode $(endswith(name, "1") ? 1 : 2)"))
    end
    if two_samples
        path = joinpath(inputs, "modpara.def")
        text = read(path, String)
        count(r"(?m)^\s*NDataQtySmp\s+1\s*$", text) == 1 || error("sampling quantity boundary changed")
        text = replace(text, r"(?m)^\s*NDataQtySmp\s+1\s*$" => "NDataQtySmp 2")
        text = replace(text, r"(?m)^\s*NDataIdxStart\s+1\s*$" => "NDataIdxStart 7")
        write(path, text)
    end
    if overlay_case
        overlay_source = joinpath(repo, "tests", "fixtures", "dh4", "production_dh24_real")
        for file in ["indh2.def", "indh4.def"]
            cp(joinpath(overlay_source, file), joinpath(inputs, file); force = true)
        end
        open(joinpath(inputs, "namelist.def"), "a") do io
            println(io, "\nInDH2 indh2.def\nInDH4 indh4.def")
        end
    end
    if combination_case
        rbm_case = name == "hubbard_chain_dh_rbm_opttrans"
        cp(joinpath(repo, "tests", "fixtures", "opttrans", "run_opt_dh24_rbm_cmp", "opttrans.def"),
            joinpath(inputs, "opttrans.def"); force = true)
        open(joinpath(inputs, "inopttrans.def"), "w") do io
            println(io, "===\nNParameter 3\nComplexType 0\n===\n===\n0 0.25 0\n1 -0.5 0\n2 0.75 0")
        end
        open(joinpath(inputs, "namelist.def"), "a") do io
            println(io, "\nOptTrans opttrans.def\nInOptTrans inopttrans.def")
            overlay_source = joinpath(repo, "tests", "fixtures", "dh4", "production_dh24_real")
            for file in ["indh2.def", "indh4.def"]
                cp(joinpath(overlay_source, file), joinpath(inputs, file); force = true)
            end
            println(io, "InDH2 indh2.def\nInDH4 indh4.def")
            if rbm_case
                rbm_source = joinpath(repo, "tests", "fixtures", "c_orbital_inputs", "historical_binary_rbm")
                for family in ["Charge", "Spin", "General"], layer in ["PhysLayer", "HiddenLayer", "PhysHidden"]
                    section = "$(family)RBM_$layer"
                    cp(joinpath(rbm_source, "$section.def"), joinpath(inputs, "$section.def"); force = true)
                    if layer == "HiddenLayer"
                        # Julia infers width from mappings whereas C reserves
                        # the declaration. Remove the unused fourth hidden
                        # slot so the explicit input widths agree (not a
                        # Julia-only declared-storage repair).
                        path = joinpath(inputs, "$section.def")
                        text = replace(read(path, String), "NParameter 4" => "NParameter 3")
                        write(path, replace(text, "3 0\n" => ""))
                    end
                    println(io, "$section $section.def\nIn$section ", layer == "HiddenLayer" ? "inrbmhidden.def" : "inrbm.def")
                end
                cp(joinpath(rbm_source, "rbm_input_opt_dh24_rbm_cmp.def"), joinpath(inputs, "inrbm.def"); force = true)
                write(joinpath(inputs, "inrbmhidden.def"), replace(read(joinpath(inputs, "inrbm.def"), String),
                    "NParameter 4" => "NParameter 3", "3 0.0 0.0\n" => ""))
            end
        end
        if rbm_case
            open(joinpath(inputs, "modpara.def"), "a") do io
                println(io, "\nNneuronCharge 2\nNneuronSpin 2\nNneuronGeneral 2")
            end
            orbital = joinpath(inputs, "orbitalidx.def")
            write(orbital, replace(read(orbital, String), r"ComplexType\s+0" => "ComplexType 1"))
        end
        # Derive complete C records from independent source zqp triples, not
        # from Rust state. DH source has 23 projections followed by 12 Slater.
        tokens = split(read(joinpath(fixture, "zqp_opt.dat"), String))
        width = 6 + 3 * 35
        length(tokens) % width == 0 || error("DH source record layout changed")
        open(joinpath(model_directory[], "zqp_opt.dat"), "w") do io
            for start in 1:width:length(tokens)
                record = tokens[start:start+width-1]
                rbm = rbm_case ? repeat(["0.03125", "-0.015625", "0"], 33) : String[]
                println(io, join(vcat(record[1:6+3*23], rbm, record[7+3*23:end],
                    ["0.2", "0", "0", "-0.3", "0", "0", "0.5", "0", "0"]), " "))
            end
        end
    end
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
            expected = joinpath(model_directory[], "expected")
            for file in readdir(work)
                (occursin(r"^zvo_cisajs.*_[0-9]+\.dat$", file) ||
                    startswith(file, "zvo_ls_")) || continue
                cp(joinpath(work, file), joinpath(expected, file); force = true)
            end
            for (file, compare) in (overlay_case || combination_case || two_samples || all_terms_case ? [] : [("zvo_cisajs_001.dat", compare_green_one),
                ("zvo_cisajscktalt_001.dat", compare_green_two_dc),
                ("zvo_cisajscktaltex_001.dat", compare_green_factored)])
                r = compare(joinpath(work, file), joinpath(fixture, "expected", file))
                push!(comparisons, "$file: ok=$(r.ok), max_abs=$(r.max_abs_err), $(r.detail)")
                actual_rows = _green_lines(read(joinpath(work, file), String))
                expected_rows = _green_lines(read(joinpath(fixture, "expected", file), String))
                first_difference = nothing
                first_numeric = file == "zvo_cisajs_001.dat" ? 5 : file == "zvo_cisajscktalt_001.dat" ? 9 : 1
                for (row, (actual, expected)) in enumerate(zip(actual_rows, expected_rows))
                    for column in first_numeric:length(expected)
                        a, e = parse(Float64, actual[column]), parse(Float64, expected[column])
                        if a - e != 0 && first_difference === nothing
                            first_difference = "row=$row column=$column actual=$(repr(a)) expected=$(repr(e)) delta=$(repr(a-e))"
                        end
                    end
                end
                difference_label = something(first_difference, "none")
                push!(comparisons, "$file first_numerical_difference=$difference_label")
            end
        end
    end
    open(joinpath(model_directory[], "provenance.txt"), "w") do io
        println(io, "generator=scripts/generate_physcal_181.jl")
        println(io, "generated_utc=$(now(UTC))\ngenerator_sha256=$generator_hash")
        println(io, "julia=$VERSION\nreference_revision=$revision\narchitecture=$(Sys.MACHINE)")
        println(io, "blas=$(BLAS.get_config())\nblas_threads=$(BLAS.get_num_threads())\njulia_threads=$(Threads.nthreads())")
        println(io, "blas_version=$blas_version\njulia_commit=$(Base.GIT_VERSION_INFO.commit)\ncpu=$(Sys.CPU_NAME)")
        println(io, "seed=1\nmode=$mode\nmanifest_sha256=$(bytes2hex(sha256(read(manifest))))")
        println(io, "physcal_source_sha256=$(bytes2hex(sha256(read(physcal_path))))")
        println(io, "stages=seeded; initialized (single InitParameter, before fixed restoration); sample-i for i=0:$(two_samples ? 1 : 0) (after that frame's complete saved samples, before measurement)")
        println(io, "rng=next 624 UInt32 outputs, non-consuming C SFMT peek; draw-count=primitive UInt32 words since seed")
        println(io, "input_contract=explicit zqp; no automatic initial.def; unchanged sampling counts; serial; correlation shifts disabled")
        println(io, "trajectory_authority=Julia independently generated; C compatibility of trajectory requires a separate C trace")
        println(io, "measurement_stages=accumulated-i before weighted averages; averaged-i after averages, all energy and ordered Green arrays; same RNG position at both stages")
        println(io, "output_contract=C indexed out/var lifecycle and parameter layout; Julia numeric reference records, NOT a byte-format oracle; no Julia shared-file append extension")
        println(io, "sample_quantity=$(two_samples ? 2 : 1); sample_index_start=$(two_samples ? 7 : 1)")
        if overlay_case
            println(io, "overlay_origin=tests/fixtures/dh4/production_dh24_real/{indh2,indh4}.def; complete indexed C InDH2/InDH4 records applied after explicit zqp")
            println(io, "C_output_check=not available for the overlaid parameter set; no comparison to the unoverlaid C output")
        end
        if combination_case
            println(io, "harness=C phase order: initialize once then reload fixed zqp, In overlays, synchronization; unmodified Julia sampling/measurement kernels; NOT original Julia PhysCal runner parity")
            println(io, "combination=DH2+DH4+OptTrans+In overlays", name == "hubbard_chain_dh_rbm_opttrans" ? "+all nine RBM sections" : "")
            println(io, "C_output_check=unavailable for combination; full C trajectory/input acceptance not established by Julia generation")
        end
        for line in comparisons; println(io, "C_output_check=$line"); end
        for (dir, _, files) in walkdir(inputs), file in sort(files)
            path = joinpath(dir, file)
            println(io, "input_sha256=$(bytes2hex(sha256(read(path)))) $(relpath(path, model_directory[]))")
        end
        println(io, "input_sha256=$(bytes2hex(sha256(read(joinpath(model_directory[], "zqp_opt.dat"))))) zqp_opt.dat")
    end
    println(name, ": saved trajectory; total UInt32 draws=", draws[], "; ", join(comparisons, "; "))
end

metadata_only && generate_average()
