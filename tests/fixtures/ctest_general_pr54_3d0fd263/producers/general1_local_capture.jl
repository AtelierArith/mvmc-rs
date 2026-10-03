# External read-only diagnostic; does not edit reviewed sources or fixtures.
using MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra, SHA
length(ARGS) == 3 || error("stage model prefixes required")
const LOCAL_STAGE = abspath(ARGS[1])
const LOCAL_REPO = normpath(joinpath(@__DIR__, ".."))
function local_capture(sample, ip, e, w, derivatives, counters, occupations, data, state)
    destination = joinpath(LOCAL_STAGE, "local", "sample-$sample")
    isdir(destination) && error("duplicate local sample boundary")
    mkpath(destination)
    for (name, values) in (("ip", [ip]), ("energy", [e]),
            ("derivatives", derivatives), ("pfaffians", state.slater_matrix.pf_m),
            ("inverse", state.slater_matrix.inv_m), ("rbm-counters", counters))
        open(joinpath(destination, "$name.txt"), "w") do io
            println(io, join((repr(part) for z in values for part in (real(z), imag(z))), " "))
        end
    end
    open(joinpath(destination, "settings.txt"), "w") do io
        println(io, "sample=$sample weight=$w seed=$(data.modpara.rnd_seed)")
    end
    if sample == 0
        open(joinpath(destination, "counter-operands.txt"), "w") do io
            println(io, "nsite=", data.modpara.nsite)
            println(io, "occupations ", join(occupations, " "))
            for term in data.general_rbm_hidden_layer_terms
                println(io, "hidden ", term.site, " ", term.idx, " ", repr(real(term.value)), " ", repr(imag(term.value)))
            end
            for term in data.general_rbm_phys_hidden_terms
                println(io, "coupling ", term.site1, " ", term.spin, " ", term.site2, " ", term.idx, " ", repr(real(term.value)), " ", repr(imag(term.value)))
            end
        end
    end
    nothing
end
source = joinpath(LOCAL_REPO, "extern/Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl")
original = read(source, String)
start = first(findfirst("function vmc_main_cal!(", original))
stop = last(findnext("\nend\n", original, start))
body = original[start:stop]
needle = "                # Accumulate OO and HO ([43] calculate OO and HO)"
length(findall(needle, body)) == 1 || error("local observation boundary changed")
observed = replace(body, needle =>
    "                Main.local_capture(sample, ip, e, w, sr_opt_o, rbm_cnt, ele_num, data, worker_state)\n" * needle; count=1)
Base.include_string(MVMCOptimizers, observed, "external_general1_local_observed")
open(LOCAL_STAGE * "-observer-provenance.txt", "w") do io
    println(io, "Read-only actual pre-SR per-sample observation; not full C executable validation")
    println(io, "Julia=$VERSION BLAS=$(BLAS.get_config())")
    println(io, "original_source_sha256=", bytes2hex(sha256(original)))
    println(io, "observed_body_sha256=", bytes2hex(sha256(observed)))
    println(io, "observer_sha256=", bytes2hex(sha256(read(@__FILE__))))
end
include(joinpath(@__DIR__, "ctest_prefix_oracle.jl"))
