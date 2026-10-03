# Optional instrumented reference worker. Observation-only source overlays;
# fail if the upstream observation boundaries change. Never write vendored code.
using MVMCOptimizers, MVMCExpertModeParsers, MPI, SFMT, Random, LinearAlgebra
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: <namelist> <state-dir>")
ENV["JULIA_MVMC_MPI"] = "1"
MPI.Init()
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const out = ARGS[2]
mkpath(out)
const evidence = open(joinpath(out, "rank-$rank.txt"), "w")
function record(key, values)
    isempty(values) && return
    print(evidence, key)
    for v in values
        if v isa Complex
            print(evidence, " ", repr(real(v)), " ", repr(imag(v)))
        else
            print(evidence, " ", repr(v))
        end
    end
    println(evidence)
    flush(evidence)
end
function local_state(state)
    e = state.energy
    record("n:local-energy", [e.wc,e.etot,e.etot2,e.sztot,e.sztot2])
    for (key, name) in (("oo", :sr_opt_oo), ("ho", :sr_opt_ho), ("oo-real", :sr_opt_oo_real), ("ho-real", :sr_opt_ho_real))
        record("n:local-$key", getproperty(state.sr_opt,name))
    end
end
function final_state(data, state, rng, info)
    record("d:status", [info == 0 ? 0 : 1])
    ec = state.electron_config
    for name in (:ele_idx,:ele_cfg,:ele_num,:ele_proj_cnt,:ele_spn)
        record("d:$name", getproperty(ec,name))
    end
    record("d:counter", vcat(ec.counter[1:9], ec.counter[11]))
    record("n:parameters", MVMCOptimizers.pack_parameters(data))
    e = state.energy
    record("n:energy", [e.wc,e.etot,e.etot2,e.sztot,e.sztot2])
    words = zeros(UInt32,624)
    SFMT.C_API.sfmt_dump_rand32(words,624) # saves/restores global SFMT, consumes no draws
    record("d:rng", words)
end
function overlay(file, replacements)
    source = read(joinpath(pkgdir(MVMCOptimizers),"src",file),String)
    for (needle, replacement) in replacements
        count(needle,source) == 1 || error("observation boundary changed: $file: $needle")
        source = replace(source,needle => replacement)
    end
    Base.include_string(MVMCOptimizers,source,file)
end
overlay("vmc_para_opt.jl", [
    "        # [21] WeightAverage:" => "        Main.local_state(state)\n        # [21] WeightAverage:",
    "    # Final output (rank0" => "    Main.final_state(data,state,rng,info)\n    # Final output (rank0",
])
overlay("weight_average.jl", [
    "    allreduce_sum!(ctx, buf; which = :comm0)" => "    allreduce_sum!(ctx, buf; which = :comm0)\n    Main.record(\"n:reduced-energy\",buf)",
    "    allreduce_sum!(ctx, state.sr_opt.sr_opt_oo; which = :comm0)" => "    allreduce_sum!(ctx, state.sr_opt.sr_opt_oo; which = :comm0)\n    Main.record(\"n:reduced-oo\",state.sr_opt.sr_opt_oo)",
    "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho; which = :comm0)" => "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho; which = :comm0)\n    Main.record(\"n:reduced-ho\",state.sr_opt.sr_opt_ho)",
    "    allreduce_sum!(ctx, @view(state.sr_opt.sr_opt_oo_real[1:oo_len]); which = :comm0)" => "    allreduce_sum!(ctx, @view(state.sr_opt.sr_opt_oo_real[1:oo_len]); which = :comm0)\n    Main.record(\"n:reduced-oo-real\",state.sr_opt.sr_opt_oo_real)",
    "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho_real; which = :comm0)" => "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho_real; which = :comm0)\n    Main.record(\"n:reduced-ho-real\",state.sr_opt.sr_opt_ho_real)",
])
try
    data = parse_expert_mode_files(ARGS[1])
    isempty(data.inter_all_terms) || error("InterAll excluded")
    ctx = MVMCOptimizers.build_parallel_context(data.modpara.nsplit_size)
    seed = MVMCOptimizers.resolve_rnd_seed(ctx,data.modpara.rnd_seed,nothing)
    record("d:seed",[seed])
    rng = SFMT.SFMT19937RNG()
    Random.seed!(rng,seed)
    MVMCExpertModeParsers.init_parameter!(data;rng=rng)
    initial = joinpath(dirname(ARGS[1]),"initial.def")
    if isfile(initial)
        MVMCOptimizers.read_initial_def!(data,initial) || error("initial.def failed")
    end
    MVMCExpertModeParsers.read_input_parameters!(data,ARGS[1])
    MVMCOptimizers.sync_modified_parameter!(ctx,data)
    MVMCExpertModeParsers.init_qp_weight!(data)
    words = zeros(UInt32,624)
    SFMT.C_API.sfmt_dump_rand32(words,624)
    record("d:initial-rng",words)
    record("n:initial-parameters",MVMCOptimizers.pack_parameters(data))
    data.modpara.nsr_opt_itr_step=1
    data.modpara.nsr_opt_itr_smp=1
    status = MVMCOptimizers.vmc_para_opt!(data;rng=rng,ctx=ctx,output_dir=out)
    status == 0 || error("SR status=$status")
catch err
    showerror(stderr,err,catch_backtrace())
    MPI.Abort(MPI.COMM_WORLD,1)
    rethrow()
finally
    close(evidence)
end
