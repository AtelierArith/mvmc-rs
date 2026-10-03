# Separate offline DH histories + actual C aggregation; no Rust oracle.
using SHA
prefix_source = read(joinpath(@__DIR__, "ctest_prefix_oracle.jl"), String)
boundary = findfirst("\nfor model in MODELS\n", prefix_source)
boundary === nothing && error("prefix observer boundary changed")
# Reuse identical read-only source hooks, C shim and native FSZ bridge setup.
Base.include_string(Main, prefix_source[1:first(boundary)-1],
    joinpath(@__DIR__, "ctest_prefix_oracle.jl"))
const DH_PROBE = get(ENV, "MVMC_CTEST_WINDOW_PROBE", "")
isfile(DH_PROBE) || error("explicit native C aggregation probe required")
PREFIXES == [3] || error("DH runtime contract requires steps=window=3")
for name in MODELS
    family, mode = split(name, '_')
    input = mode == "cmp" ? joinpath(REPO, "tests/fixtures/c_orbital_inputs/namelist_$name.def") :
        family == "dh2" ? joinpath(REPO, "tests/fixtures/dh2/production_$mode/namelist.def") :
        joinpath(REPO, "tests/fixtures/dh4/production_$name/namelist.def")
    original = parse_expert_mode_files(input)
    original.i_flg_orbital_general == 0 || !isempty(CTEST_BRIDGE) || error("native FSZ bridge required")
    case = joinpath(DESTINATION, name)
    mkpath(case)
    empty!(WINDOW_HISTORY)
    result = Base.invokelatest(MVMCOptimizers.ctest_observed_runner, input;
        nsteps=3, nsmp=3, mode=Symbol(mode), output_dir=case)
    result.status == 0 || error("DH solver failed: $name")
    data, state = SNAPSHOT[]
    counts = MVMCOptimizers._parameter_count_breakdown(data)
    anti = data.n_orbital_anti_parallel
    parallel = data.i_flg_orbital_parallel == 0 ? 0 : div(counts.n_orbital_idx - anti, 2)
    data.i_flg_orbital_parallel == 0 || anti + 2 * parallel == counts.n_orbital_idx || error("C orbital split")
    widths = [counts.layout.n_gutzwiller, counts.layout.n_jastrow,
        counts.layout.n_dh2, counts.layout.n_dh4,
        MVMCOptimizers._parameter_section_width.(MVMCOptimizers._rbm_parameter_sections(data))...,
        counts.n_orbital_idx, counts.n_opt_trans]
    length(WINDOW_HISTORY) == 3 || error("incomplete independent history")
    history = joinpath(case, "c-window-input.txt")
    open(history, "w") do io
        println(io, "3 ", counts.n_para, " ", join(widths, " "))
        for row in WINDOW_HISTORY
            length(row) == counts.n_para + 2 || error("declared slot count")
            println(io, join(repr.(collect(reinterpret(Float64, row))), " "))
        end
    end
    run(`$DH_PROBE $history $(joinpath(case, "c_zqp")) $(data.i_flg_orbital_general) $anti $parallel`)
    open(joinpath(case, "provenance.txt"), "w") do io
        println(io, "Actual C avevar aggregation of independently captured Julia post-SR/sync histories; no Rust values")
        println(io, "case=$name mode=$mode seed=$(original.modpara.rnd_seed) steps=3 window=3 NSRCG=$(original.modpara.nsrcg) NStore=$(original.modpara.nstore_o)")
        println(io, "canonical_steps=$(original.modpara.nsr_opt_itr_step) canonical_window=$(original.modpara.nsr_opt_itr_smp) override=both_no_clamp")
        println(io, "iFlgOrbitalGeneral=$(data.i_flg_orbital_general) iNOrbitalAntiParallel=$anti iNOrbitalParallel=$parallel")
        println(io, "initial_overlay=canonical_auto_initial_def In_overlays=canonical_namelist_entries")
        println(io, "Julia=$VERSION platform=$(Sys.MACHINE) BLAS=$(BLAS.get_config()) threads=$(BLAS.get_num_threads())")
        println(io, "Compiler=GCC_13.3.0 flags=-O0,-ffp-contract=off,-lm C_backend=libm/libc_no_BLAS")
        for path in (input, history, @__FILE__, joinpath(@__DIR__, "ctest_prefix_oracle.jl"),
                DH_PROBE, joinpath(@__DIR__, "ctest_dh_opt_window.c"), joinpath(@__DIR__, "ctest_opt_window.c"),
                joinpath(@__DIR__, "ctest_opt_window_upstream.inc"),
                joinpath(REPO, "extern/mVMC-1.3.0/src/mVMC/avevar.c"),
                joinpath(REPO, "extern/Julia-mVMC/Manifest-v1.13.toml"))
            println(io, bytes2hex(sha256(read(path))), "  ", relpath(path, REPO))
        end
        println(io, read(joinpath(DESTINATION, "provenance.txt"), String))
    end
    println("DH C WINDOW EXECUTED $name")
end
