# Optional standalone regeneration. No Cargo/Rust oracle calls.
using MVMCOptimizers, MVMCExpertModeParsers, SHA
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: independent-stage native-probe")
stage, probe = abspath.(ARGS)
repo = normpath(joinpath(@__DIR__, ".."))
startswith(stage * "/", repo * "/") && error("generate outside repository")
for model in sort(readdir(stage))
    isdir(joinpath(stage, model)) || continue
    input = joinpath(repo, "extern/Julia-mVMC/test/integration/reference", model, "inputs/namelist.def")
    data = parse_expert_mode_files(input)
    counts = MVMCOptimizers._parameter_count_breakdown(data)
    widths = [counts.layout.n_gutzwiller, counts.layout.n_jastrow,
        counts.layout.n_dh2, counts.layout.n_dh4,
        MVMCOptimizers._parameter_section_width.(MVMCOptimizers._rbm_parameter_sections(data))...,
        counts.n_orbital_idx, counts.n_opt_trans]
    for prefix in sort(readdir(joinpath(stage, model)))
        case = joinpath(stage, model, prefix)
        isdir(case) || continue
        raw = joinpath(case, "c-window-input.txt")
        lines = readlines(raw)
        steps = parse(Int, split(lines[1])[1])
        length(lines) == steps + 1 || error("incomplete independent history")
        # Rebuild metadata from declared slots, never mapped term row counts.
        # Numerical history lines are preserved verbatim, never regenerated.
        open(joinpath(case, "c-window-declared-input.txt"), "w") do io
            println(io, steps, " ", counts.n_para, " ", join(widths, " "))
            for line in lines[2:end]
                length(split(line)) == 2 * (counts.n_para + 2) || error("history width")
                println(io, line)
            end
        end
        run(`$probe $(joinpath(case, "c-window-declared-input.txt")) $(joinpath(case, "zqp_c_window"))`)
        open(joinpath(case, "c-window-provenance.txt"), "w") do io
            println(io, "Mixed oracle: independent Julia C-contract history aggregated/formatted by actual C avevar.c bodies; no Rust results")
            for path in (raw, joinpath(case, "c-window-declared-input.txt"), probe,
                @__FILE__, joinpath(repo, "c_toolbox/ctest_opt_window.c"),
                joinpath(repo, "c_toolbox/ctest_opt_window_upstream.inc"),
                joinpath(repo, "extern/mVMC-1.3.0/src/mVMC/avevar.c"))
                println(io, basename(path), " sha256=", bytes2hex(sha256(read(path))))
            end
            println(io, "Compiler: GCC Ubuntu 13.3.0-6ubuntu2~24.04.1; -O0 -ffp-contract=off -lm; native Linux x86_64; no BLAS in aggregator")
            println(io, "StoreOptData after SR/sync: Etot,Etot2,declared Para; chronological window; CalcAveVar sequential complex mean/sample deviation")
        end
        println("C WINDOW EXECUTED $model $prefix")
    end
end
