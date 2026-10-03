# Observation-only OPT FSZ CG audit; numerical files go to an explicit temp dir.
using Test, Random, SFMT, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
VERSION == v"1.13.1" || error("Use Julia 1.13.1")
BLAS.set_num_threads(1)
option(key, default) = let matches = filter(a -> startswith(a, key * "="), ARGS)
    isempty(matches) ? default : split(only(matches), "="; limit=2)[2]
end
directory = option("--output-dir", "")
isempty(directory) && error("Pass --output-dir=TEMP_DIRECTORY")
mode = option("--mode", "historical")
mode in ("historical", "native") || error("Unknown mode")
steps = option("--steps", "50")
trace_step = parse(Int, option("--trace-step", "0"))
if mode == "native"
    include("reference_native_fsz_energy.jl")
    bridge = option("--bridge-dir", "")
    isempty(bridge) && error("Native mode requires --bridge-dir=DIR")
    NativeFSZEnergyReference.install!(bridge)
end
namespace = Module(:OPTFSZCGAudit)
Core.eval(Main, :(const OPTFSZCGAudit = $namespace))
Core.eval(namespace, :(include(path) = Base.include($namespace, path)))
Core.eval(namespace, :(const AUDIT_DIR = $directory))
Core.eval(namespace, :(const TRACE_STEP = $trace_step))
observer = """
using SHA
const SAMPLE_STEP = Ref(0)
const DRAWS = UInt32[]
function begin_sample!()
    SAMPLE_STEP[] += 1
    empty!(DRAWS)
end
function record_draw!(value)
    push!(DRAWS,value)
    return value
end
function trace!(tag,args...)
    SAMPLE_STEP[] == TRACE_STEP || return
    open(joinpath(AUDIT_DIR,"sampling-trace.txt"),"a") do io
        println(io,tag," ",join(args," "))
    end
end
function capture_audit!(step, params, energy, configs)
    mkpath(AUDIT_DIR)
    write(joinpath(AUDIT_DIR,"step-\$step-parameters.txt"),hex(collect(reinterpret(Float64,params)))*"\\n")
    write(joinpath(AUDIT_DIR,"step-\$step-energy.txt"),hex([real(energy),imag(energy)])*"\\n")
    io=IOBuffer()
    for vals in (configs.ele_idx,configs.ele_cfg,configs.ele_num,configs.ele_proj_cnt,
                 configs.ele_spn,configs.burn_ele_idx,vcat(configs.counter[1:9],configs.counter[11]))
        println(io,join(vals," "))
    end
    write(joinpath(AUDIT_DIR,"step-\$step-configs.txt"),String(take!(io)))
    if TRACE_STEP > 0
        write(joinpath(AUDIT_DIR,"step-\$step-rng-draws.txt"),string(length(DRAWS))*" "*bytes2hex(sha256(collect(reinterpret(UInt8,DRAWS))))*"\\n")
        if SAMPLE_STEP[] == TRACE_STEP
            write(joinpath(AUDIT_DIR,"sampling-draws.txt"),join(DRAWS," ")*"\\n")
        end
    end
end
"""
Base.include_string(namespace, observer)
if trace_step > 0
    mkpath(directory)
    write(joinpath(directory,"sampling-trace.txt"),"")
    sampling_path = joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCOptimizers.jl","src","vmc_sampling.jl")
    sampling = read(sampling_path,String)
    start = first(findfirst("function vmc_make_sample_fsz!(",sampling))
    stop = last(findnext("\nend\n",sampling,start))
    body = sampling[start:stop]
    body = replace(body,"    n_site = data.modpara.nsite" => "    Main.OPTFSZCGAudit.begin_sample!()\n    n_site = data.modpara.nsite";count=1)
    body = replace(body,"                if reject_flag\n" => "                Main.OPTFSZCGAudit.trace!(\"candidate\",out_step,in_step,Int(update_type),mi,ri,rj,s,reject_flag)\n                if reject_flag\n")
    body = replace(body,"                accept = w > r\n" => "                accept = w > r\n                Main.OPTFSZCGAudit.trace!(\"accept\",out_step,in_step,Int(update_type),mi,ri,rj,s,t,accept,Main.OPTFSZCGAudit.hex([w,r,x,real(log_ip_new),imag(log_ip_new),real(log_ip_old),imag(log_ip_old)]))\n")
    Base.include_string(MVMCOptimizers,body,sampling_path)
    Base.include_string(MVMCOptimizers,"@inline rng_rand32(rng::AbstractRNG) = Main.OPTFSZCGAudit.record_draw!(rand(rng,UInt32))",sampling_path)
end
path = joinpath(@__DIR__, "check_sr_cg_runner_parity.jl")
source = replace(read(path, String), "Main.capture_" => "Main.OPTFSZCGAudit.capture_")
root_line = only(filter(l -> startswith(l, "const FIXTURE_ROOT = "), split(source, '\n')))
source = replace(source, root_line => "const FIXTURE_ROOT = " * repr(joinpath(directory, "prefix")); count=1)
source = replace(source, "SNAPSHOTS[] = (copy(params), state.energy.etot, deepcopy(state.electron_config))" =>
    "SNAPSHOTS[] = (copy(params), state.energy.etot, deepcopy(state.electron_config))\n    capture_audit!(step, SNAPSHOTS[]...)"; count=1)
empty!(ARGS); append!(ARGS, ["--case=opt_fsz", "--steps=$steps", "--write"])
Base.include_string(namespace, source, path)
