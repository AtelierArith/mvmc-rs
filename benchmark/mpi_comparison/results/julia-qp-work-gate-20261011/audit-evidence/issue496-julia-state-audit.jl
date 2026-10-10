# Optional final-state observer; run separately from primary benchmarks.
using MPI, MVMCOptimizers, SHA

function record_final_state!(data, state)
    destination = ENV["MVMC_STATE_AUDIT_DIRECTORY"]
    mkpath(destination)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    config = state.electron_config
    open(joinpath(destination, "state-rank-$rank.txt"), "w") do io
        println(io, "sites=", data.modpara.nsite)
        for name in (:ele_idx, :ele_cfg, :ele_num, :ele_proj_cnt, :ele_spn,
                     :tmp_ele_idx, :tmp_ele_cfg, :tmp_ele_num, :tmp_ele_proj_cnt,
                     :burn_ele_idx, :burn_ele_cfg, :burn_ele_num, :burn_ele_proj_cnt,
                     :counter)
            println(io, name, "=", join(getproperty(config, name), ','))
        end
    end
    inverse = state.slater_matrix.inv_m
    real_inverse = state.slater_matrix.inv_m_real
    if length(inverse) == length(real_inverse) && !isempty(real_inverse)
        error = maximum(abs(inverse[i] - ComplexF64(real_inverse[i], 0.0))
                        for i in eachindex(real_inverse))
        budget = 8eps(Float64) * (1 + maximum(abs, real_inverse))
        @assert isfinite(error) && error <= budget
        open(joinpath(destination, "cache-rank-$rank.txt"), "w") do io
            println(io, "inverse_cache_max_abs=", error)
            println(io, "inverse_cache_budget=", budget)
        end
    end
    return nothing
end

source = read(joinpath(dirname(pathof(MVMCOptimizers)), "vmc_para_opt.jl"), String)
start = first(findfirst("function vmc_para_opt!(", source))
stop = last(findnext("\nend", source, start))
method = source[start:stop]
original = "\n    return info\nend"
@assert count(original, method) == 1
replacement = "\n    Main.record_final_state!(data, state)\n    return info\nend"
Core.eval(MVMCOptimizers, Meta.parse(replace(method, original => replacement)))
mkpath(ENV["MVMC_STATE_AUDIT_DIRECTORY"])
write(joinpath(ENV["MVMC_STATE_AUDIT_DIRECTORY"], "observed-source-sha256-$(getpid()).txt"),
      bytes2hex(sha256(source)) * "\n")
include("/home/vscode/.cache/mvmc/issue496-julia-rng-capture.jl")
