# Optional instrumented reference worker. Observation-only source overlays;
# fail if the upstream observation boundaries change. Never write vendored code.
using MVMCOptimizers, MVMCExpertModeParsers, MPI, SFMT, PfaPack, Random, LinearAlgebra
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) == 2 || error("usage: <namelist> <state-dir>")
ENV["JULIA_MVMC_MPI"] = "1"
MPI.Init()
include(joinpath(@__DIR__,"mpi_issue179_reference_provenance.jl"))
issue179_reference_provenance(ARGS[2],ARGS[1],[MVMCOptimizers,MVMCExpertModeParsers,SFMT,PfaPack])
println("Julia=", VERSION, " BLAS=", BLAS.get_config(), " threads=", BLAS.get_num_threads())
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const out = ARGS[2]
mkpath(out)
const evidence = open(joinpath(out, "rank-$rank.txt"), "w")
const sampling_events = Vector{Vector{Int}}()
const trace_words_enabled = Ref(false)
const total_word_count = Ref(0)
const sr_systems = Any[]
const sr_pending = Ref{Any}(nothing)
const cg_events = Any[]
const cg_product_call = Ref(0)
function cg_event(kind, fields...)
    # Copy actual buffers at the reference boundary; no inferred solver values.
    push!(cg_events,[("kind",[kind]); [(name,copy(values)) for (name,values) in fields]])
end
const sample_step = Ref(-1)
function trace_word(word)
    total_word_count[] += 1
    if trace_words_enabled[]
        push!(sampling_events,[9,Int(word)])
    end
end
function trace_checkpoint(state,rng)
    sample_step[] += 1
    ec = state.electron_config
    words = zeros(UInt32,624)
    SFMT.C_API.sfmt_dump_rand32(words,624)
    fields = Int[8]
    for values in (ec.ele_idx,ec.ele_cfg,ec.ele_num,ec.ele_proj_cnt,ec.ele_spn,vcat(ec.counter[1:9],ec.counter[11]),words)
        push!(fields,length(values))
        append!(fields,Int.(values))
    end
    push!(sampling_events,fields)
end
function trace_candidate(kind, candidate, layout)
    push!(sampling_events, vcat(kind, Int[candidate[i] for i in layout]))
    return candidate
end
function trace_update(update)
    push!(sampling_events, [0, Int(update)])
    return update
end
function trace_decision(weight, draw)
    # Exact integer image of the actual SFMT real2 draw, not a second draw.
    push!(sampling_events, [7, Int(weight > draw), Int(draw * 4294967296.0)])
end
function record(key, values)
    isempty(values) && return
    if sample_step[]>0 && (occursin(":local-",key) || occursin(":reduced-",key))
        parts=split(key,":";limit=2)
        key=parts[1]*":step-"*string(sample_step[])*"-"*parts[2]
    end
    print(evidence, key)
    for v in values
        if v isa Complex
            print(evidence, " ", repr(real(v)), " ", repr(imag(v)))
        elseif v isa Integer
            print(evidence, " ", string(v))
        else
            print(evidence, " ", repr(v))
        end
    end
    println(evidence)
    flush(evidence)
end
function local_state(data,state)
    cg_product_call[] = 0
    e = state.energy
    record("n:local-energy", [e.wc,e.etot,e.etot2,e.sztot,e.sztot2])
    active = MVMCOptimizers.get_all_complex_flag(data) ?
        (("oo", :sr_opt_oo), ("ho", :sr_opt_ho)) :
        (("oo-real", :sr_opt_oo_real), ("ho-real", :sr_opt_ho_real))
    for (key, name) in active
        record("n:local-$key", getproperty(state.sr_opt,name))
    end
end
function final_state(data, state, rng, info)
    open(joinpath(out,"cg-rank-$rank.txt"),"w") do io
        println(io,"d:events ",length(cg_events))
        for (index,event) in enumerate(cg_events), (name,values) in event
            isempty(values) && continue
            discrete = name in ("kind","phase","mapping","iteration","iterations")
            print(io,discrete ? "d:" : "n:","event-",lpad(string(index-1),6,'0'),"-",name)
            for value in values
                print(io," ",discrete ? string(Int(value)) : repr(value))
            end
            println(io)
        end
    end
    record("d:sr-kind",[data.modpara.nsrcg])
    record("d:sr-systems",[length(sr_systems)])
    for (index, item) in enumerate(sr_systems)
        before, after = item
        key = "sr-system-" * lpad(string(index-1),6,'0')
        skipped = before.not_solved !== nothing
        record("d:$key-dimension",[skipped ? 0 : length(before.rhs)])
        record("d:$key-not-solved",[skipped ? 2 : 0])
        record("d:$key-status",[after.status])
        record("d:$key-active",before.active_indices)
        record("d:$key-flags",Int.(before.flags))
        settings = before.settings
        record("d:$key-settings",[settings.seed,settings.steps,settings.window,settings.nsrcg,settings.nstore])
        record("n:$key-regularization",[settings.diagonal_shift,settings.redundant_cut,settings.step_dt])
        if !skipped
            record("n:$key-matrix",before.matrix)
            record("n:$key-rhs",before.rhs)
            record("n:$key-increment",after.increment)
            # Post-solve diagnostics consume owned observations only. They never
            # replace the actual solve or supply a comparison tolerance.
            open(joinpath(out,"$key-diagnostics-rank-$rank.txt"),"w") do io
                println(io,"condition2_estimate is not a certified bound; triangle=U")
                for (name,value) in pairs(CTestDirectSRCapture.diagnose(before,after))
                    println(io,name,"=",value)
                end
            end
        end
    end
    record("d:sampling-draw-count",[count(event -> event[1]==9,sampling_events)])
    record("d:total-draw-count",[total_word_count[]])
    record("d:trace-events",[length(sampling_events)])
    record("d:acceptance-events",[count(event -> event[1]==7 && event[2]==1,sampling_events),count(event -> event[1]==7 && event[2]==0,sampling_events)])
    for (index,event) in enumerate(sampling_events)
        record("d:trace-" * lpad(string(index-1),6,'0'),event)
    end
    record("d:status", [info == 0 ? 0 : 1])
    record("d:chain-samples",[data.modpara.nvmc_sample])
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
# Observation-only hooks around the original function returns and actual
# acceptance draws. Preserve the reference's proposals, arithmetic and RNG.
function overlay_sampling_trace()
    source = read(joinpath(pkgdir(MVMCOptimizers),"src","vmc_sampling.jl"),String)
    for (name,kind,layout) in (
        ("make_candidate_hopping",1,[1,2,3,4,5]),
        ("make_candidate_exchange",2,[1,2,4,5,3,6,7]),
        ("make_candidate_hopping_fsz",3,[1,2,3,4,5,6]),
        ("make_candidate_local_spin_flip_conduction",4,[1,2,3,4,5,6]),
        ("make_candidate_local_spin_flip_localspin",5,[1,2,3,4,5,6]),
        ("make_candidate_exchange_fsz",6,[1,2,3,4,5]),
        ("get_update_type",0,Int[]),
    )
        pattern = Regex("^function " * name * "\\([\\s\\S]*?^end\$","m")
        matches = collect(eachmatch(pattern,source))
        length(matches)==1 || error("trace function boundary changed: $name")
        original = matches[1].match
        returns = 0
        observed = replace(original, r"^([ \t]*)return ([^\n]+)$"m => line -> begin
            m = match(r"^([ \t]*)return ([^#]+)(.*)$",line)
            m === nothing && error("unrecognized return: $line")
            returns += 1
            expr = strip(m.captures[2])
            wrapper = kind==0 ? "Main.trace_update(($expr))" : "Main.trace_candidate($kind,($expr),$(repr(layout)))"
            m.captures[1] * "return " * wrapper * m.captures[3]
        end)
        returns>0 || error("no candidate return observed: $name")
        source = replace(source,original=>observed)
    end
    for (pattern,expected,replacement) in (
        (r"^([ \t]*)r_metro = rng_real2\(rng\)$"m,2,
         m -> m.match * "\n" * m.captures[1] * "Main.trace_decision(w,r_metro)"),
        (r"^([ \t]*)accept = w > r$"m,3,
         m -> m.captures[1] * "Main.trace_decision(w,r)\n" * m.match),
        (r"^([ \t]*)if w > rng_real2\(rng\)$"m,5,
         m -> m.captures[1] * "r_metro_179 = rng_real2(rng)\n" * m.captures[1] * "Main.trace_decision(w,r_metro_179)\n" * m.captures[1] * "if w > r_metro_179"),
    )
        count(pattern,source)==expected || error("acceptance trace boundary changed: $pattern")
        source = replace(source,pattern=>line -> replacement(match(pattern,line)))
    end
    Base.include_string(MVMCOptimizers,source,"vmc_sampling.jl")
end
overlay_sampling_trace()
# Instrument the real SFMT API return, not a replay or inferred stream offset.
# These are the two primitive APIs used by the pinned optimizer/parser.
let api_source = read(joinpath(pkgdir(SFMT),"src","C_API.jl"),String)
    for (name,conversion) in (("gen_rand32","Int(value)"), ("genrand_real2","Int(value * 4294967296.0)"))
        pattern = Regex("^function " * name * "\\(\\)[\\s\\S]*?^end\$","m")
        matches = collect(eachmatch(pattern,api_source))
        length(matches)==1 || error("SFMT trace boundary changed: $name")
        original = matches[1].match
        count(r"^    ccall"m,original)==1 || error("SFMT primitive body changed: $name")
        observed = replace(original,r"^    (ccall[^\n]+)$"m => line -> begin
            call = match(r"^    (ccall[^\n]+)$",line).captures[1]
            "    value = $call\n    Main.trace_word($conversion)\n    return value"
        end)
        Base.include_string(SFMT.C_API,observed,"C_API.jl")
    end
end
overlay("vmc_para_opt.jl", [
    "        sync_modified_parameter!(ctx, data)" => "        sync_modified_parameter!(ctx, data)\n        Main.record(\"n:sr-step-\" * string(Main.sample_step[]),pack_parameters(data))",
    "        # [4] VMCMainCal" => "        Main.trace_checkpoint(state,rng)\n        # [4] VMCMainCal",
    "        # [21] WeightAverage:" => "        Main.local_state(data,state)\n        # [21] WeightAverage:",
    "        reduce_counter!(ctx, state)" => "        Main.record(\"d:local-counter\",vcat(state.electron_config.counter[1:9],state.electron_config.counter[11]))\n        reduce_counter!(ctx, state)\n        Main.record(\"d:reduced-counter\",vcat(state.electron_config.counter[1:9],state.electron_config.counter[11]))",
    "    # Final output (rank0" => "    Main.final_state(data,state,rng,info)\n    # Final output (rank0",
])
overlay("weight_average.jl", [
    "    allreduce_sum!(ctx, buf; which = :comm0)" => "    allreduce_sum!(ctx, buf; which = :comm0)\n    Main.record(\"n:reduced-energy\",buf)",
    "    allreduce_sum!(ctx, state.sr_opt.sr_opt_oo; which = :comm0)" => "    allreduce_sum!(ctx, state.sr_opt.sr_opt_oo; which = :comm0)\n    Main.record(\"n:reduced-oo\",state.sr_opt.sr_opt_oo)",
    "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho; which = :comm0)" => "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho; which = :comm0)\n    Main.record(\"n:reduced-ho\",state.sr_opt.sr_opt_ho)",
    "    allreduce_sum!(ctx, @view(state.sr_opt.sr_opt_oo_real[1:oo_len]); which = :comm0)" => "    allreduce_sum!(ctx, @view(state.sr_opt.sr_opt_oo_real[1:oo_len]); which = :comm0)\n    Main.record(\"n:reduced-oo-real\",state.sr_opt.sr_opt_oo_real)",
    "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho_real; which = :comm0)" => "    allreduce_sum!(ctx, state.sr_opt.sr_opt_ho_real; which = :comm0)\n    Main.record(\"n:reduced-ho-real\",state.sr_opt.sr_opt_ho_real)",
])
include(joinpath(@__DIR__,"..","c_toolbox","ctest_direct_sr_capture.jl"))
CTestDirectSRCapture.install!(before -> begin
    sr_pending[] === nothing || error("nested actual SR observation")
    sr_pending[] = before
end, after -> begin
    sr_pending[] === nothing && error("missing pre-factorization SR observation")
    push!(sr_systems,(sr_pending[],after))
    sr_pending[] = nothing
end)
function overlay_cg_function(name,replacements)
    source=read(joinpath(pkgdir(MVMCOptimizers),"src","stochastic_opt.jl"),String)
    matches=collect(eachmatch(Regex("(?ms)^function "*name*"\\(.*?^end\\b"),source))
    length(matches)==1 || error("CG observation function boundary changed: $name")
    body=only(matches).match
    for (needle,replacement) in replacements
        count(needle,body)==1 || error("CG observation boundary changed: $name: $needle")
        body=replace(body,needle=>replacement;count=1)
    end
    Base.include_string(MVMCOptimizers,body,"mpi179_actual_CG_observation_"*name)
end
overlay_cg_function("operate_by_s!",[
    "    bcast!(ctx, x; root = 0, which = :comm0)" =>
        "    bcast!(ctx, x; root = 0, which = :comm0)\n    Main.cg_event(1,(\"phase\",[0]),(\"search\",x))",
    "    allreduce_sum!(ctx, z; which = :comm0)" =>
        "    Main.cg_event(1,(\"phase\",[1]),(\"search\",x),(\"product\",z))\n    Main.record(\"n:local-collective-cg-product-\"*lpad(string(Main.cg_product_call[]),6,'0'),z)\n    allreduce_sum!(ctx, z; which = :comm0)\n    Main.record(\"n:reduced-collective-cg-product-\"*lpad(string(Main.cg_product_call[]),6,'0'),z)\n    Main.cg_product_call[] += 1\n    Main.cg_event(1,(\"phase\",[2]),(\"search\",x),(\"product\",z))",
    "        z[si] = inv_w * z[si] - coef * ws.stcO[si] + dsr_opt_sta_del * ws.sdiag[si] * x[si]\n    end" =>
        "        z[si] = inv_w * z[si] - coef * ws.stcO[si] + dsr_opt_sta_del * ws.sdiag[si] * x[si]\n    end\n    Main.cg_event(1,(\"phase\",[3]),(\"search\",x),(\"product\",z))",
])
overlay_cg_function("stochastic_opt_cg_main!",[
    "    iter = 0" => "    Main.cg_event(2,(\"iteration\",[0]),(\"solution\",ws.x),(\"residual\",ws.r),(\"direction\",ws.d),(\"delta\",[delta]))\n    iter = 0",
    "            ws.d[si] = ws.r[si] + beta * ws.d[si]\n        end" =>
        "            ws.d[si] = ws.r[si] + beta * ws.d[si]\n        end\n        Main.cg_event(2,(\"iteration\",[iter_idx]),(\"solution\",ws.x),(\"residual\",ws.r),(\"direction\",ws.d),(\"delta\",[delta]),(\"alpha\",[alpha]))",
    "    return iter" => "    Main.cg_event(3,(\"iterations\",[iter]),(\"solution\",ws.x),(\"residual\",ws.r),(\"direction\",ws.d))\n    return iter",
])
overlay_cg_function("stochastic_opt_cg!",[
    "    # Phase 3: Run CG iteration" =>
        "    Main.cg_event(0,(\"mapping\",smat_to_para_idx),(\"mean\",ws.stcO),(\"diagonal\",ws.sdiag),(\"gradient\",ws.g),(\"real-samples\",ws.stcOs_real),(\"imag-samples\",ws.stcOs_imag))\n    # Phase 3: Run CG iteration",
])
try
    data = parse_expert_mode_files(ARGS[1])
    isempty(data.inter_all_terms) || error("InterAll excluded")
    ctx = MVMCOptimizers.build_parallel_context(data.modpara.nsplit_size)
    record("d:group",[ctx.group1,ctx.rank1,ctx.size1])
    record("d:configured-samples",[data.modpara.nvmc_sample])
    record("d:requested-width",[data.modpara.nsplit_size])
    seed = MVMCOptimizers.resolve_rnd_seed(ctx,data.modpara.rnd_seed,nothing)
    record("d:seed",[seed])
    rng = SFMT.SFMT19937RNG()
    Random.seed!(rng,seed)
    MVMCExpertModeParsers.init_parameter!(data;rng=rng)
    initial = joinpath(dirname(ARGS[1]),"initial.def")
    if isfile(initial) && !any(line -> startswith(strip(line),"OptTrans "), readlines(ARGS[1]))
        MVMCOptimizers.read_initial_def!(data,initial) || error("initial.def failed")
    end
    MVMCExpertModeParsers.read_input_parameters!(data,ARGS[1])
    MVMCOptimizers.sync_modified_parameter!(ctx,data)
    MVMCExpertModeParsers.init_qp_weight!(data)
    open(joinpath(out,"draw-provenance-rank-$rank.txt"),"w") do io
        println(io,"actual_initial_words=",total_word_count[])
    end
    trace_words_enabled[] = true
    record("d:initial-draw-count",[total_word_count[]])
    words = zeros(UInt32,624)
    SFMT.C_API.sfmt_dump_rand32(words,624)
    record("d:initial-rng",words)
    record("n:initial-parameters",MVMCOptimizers.pack_parameters(data))
    steps=parse(Int,get(ENV,"MPI179_STEPS","1"))
    1<=steps<=3 || error("bounded prefixes 1/2/3 required")
    record("d:steps",[steps])
    data.modpara.nsr_opt_itr_step=steps
    data.modpara.nsr_opt_itr_smp=steps
    status = MVMCOptimizers.vmc_para_opt!(data;rng=rng,ctx=ctx,output_dir=out)
    status == 0 || error("SR status=$status")
catch err
    showerror(stderr,err,catch_backtrace())
    MPI.Abort(MPI.COMM_WORLD,1)
    rethrow()
finally
    close(evidence)
end
