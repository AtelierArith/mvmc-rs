using SHA
root = normpath(joinpath(@__DIR__, ".."))
source = joinpath(root, "extern/Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl")
original = read(source, String)
matched = match(r"(?ms)^function stochastic_opt!\(.*?^end\b", original)
matched === nothing && error("extraction boundary")
body = matched.match
for (needle, replacement) in (
    "        info = _solve_direct_sr!(S, g)" => "        Main.metrics_before(S, g, smat_to_para_idx, data)\n        info = _solve_direct_sr!(S, g)",
    "    ctimer_stop!(c_timer, 57)" => "    Main.metrics_after(g, info)\n    ctimer_stop!(c_timer, 57)")
    length(findall(needle, body)) == 1 || error("capture boundary")
    global body = replace(body, needle => replacement; count=1)
end
destination = joinpath(root, "direct-v2-injection-identity.txt")
ispath(destination) && error("refuse overwrite")
open(destination, "w") do io
    println(io, "original_stochastic_opt_file_sha256=", bytes2hex(sha256(original)))
    println(io, "original_extracted_function_sha256=", bytes2hex(sha256(matched.match)))
    println(io, "runtime_injected_function_sha256=", bytes2hex(sha256(body)))
    for file in ("ctest_direct_sr_metrics.jl", "ctest_direct_sr_capture.jl", "ctest_prefix_oracle.jl", "capture_injection_identity.jl")
        println(io, file, " sha256=", bytes2hex(sha256(read(joinpath(@__DIR__, file)))))
    end
    println(io, "Read-only observer method replacement is outside the 63-file on-disk reference manifest; original _solve_direct_sr! is called unchanged.")
    println(io, "94143 preflight source separately reconstructed by reversal of the recorded one-line patch; stdout transcribed from actual tool output, not a replay.")
end
