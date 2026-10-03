# Explicit offline aggregation of existing independent histories, no Rust input.
using MVMCExpertModeParsers, MVMCOptimizers, SHA
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: external-stage compiled-dh-probe")
stage, probe = abspath.(ARGS)
repo = normpath(joinpath(@__DIR__, ".."))
startswith(stage * "/", repo * "/") && error("stage outside repository")
for name in ("dh2_real", "dh2_cmp", "dh2_fsz", "dh4_real", "dh4_cmp", "dh4_fsz", "dh24_real", "dh24_cmp", "dh24_fsz")
    family, mode = split(name, '_')
    input = mode == "cmp" ? joinpath(repo, "tests/fixtures/c_orbital_inputs/namelist_$name.def") :
        family == "dh2" ? joinpath(repo, "tests/fixtures/dh2/production_$mode/namelist.def") :
        joinpath(repo, "tests/fixtures/dh4/production_$name/namelist.def")
    data = parse_expert_mode_files(input)
    counts = MVMCOptimizers._parameter_count_breakdown(data)
    anti = data.n_orbital_anti_parallel
    parallel = data.i_flg_orbital_parallel == 0 ? 0 : div(counts.n_orbital_idx - anti, 2)
    data.i_flg_orbital_parallel == 0 || anti + 2 * parallel == counts.n_orbital_idx || error("C split")
    case = joinpath(stage, name)
    history = joinpath(case, "c-window-input.txt")
    run(`$probe $history $(joinpath(case, "native_zqp")) $(data.i_flg_orbital_general) $anti $parallel`)
    open(joinpath(case, "orbital-aggregation-provenance.txt"), "w") do io
        println(io, "Actual C avevar.c auxiliary orbital branches; independent history unchanged; no Rust results")
        println(io, "iFlgOrbitalGeneral=$(data.i_flg_orbital_general) iNOrbitalAntiParallel=$anti iNOrbitalParallel=$parallel")
        println(io, "compiler=GCC_13.3.0 flags=-O0,-ffp-contract=off,-lm backend=libm/libc_no_BLAS Linux_x86_64")
        for path in (history, probe, @__FILE__, joinpath(@__DIR__, "ctest_dh_opt_window.c"),
                joinpath(@__DIR__, "ctest_opt_window.c"), joinpath(@__DIR__, "ctest_opt_window_upstream.inc"),
                joinpath(repo, "extern/mVMC-1.3.0/src/mVMC/avevar.c"))
            println(io, bytes2hex(sha256(read(path))), "  ", relpath(path, repo))
        end
    end
    println("C DH BRANCH EXECUTED $name general=$(data.i_flg_orbital_general) anti=$anti parallel=$parallel")
end
