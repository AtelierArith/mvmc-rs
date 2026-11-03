# Independent resident PhysCal oracle. Use Juliaup +1.13 and the pinned GPU lock.
# julia +1.13 --project=/path/to/Julia-mVMC/test/gpu gen_physcal_julia.jl INPUT PARAMS OUTPUT
using CUDA, MVMCOptimizers, LinearAlgebra, SHA
CUDA.device!(0); CUDA.allowscalar(false); BLAS.set_num_threads(1)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
data,_=GPU.prepared_gpu_data(abspath(ARGS[1]);seed=79,opt_para=abspath(ARGS[2]),physcal=true)
# Product-side correlations must be formed per walker, before ensemble means.
push!(data.green_two_ex_terms,
    GPU.Parsers.GreenTwoExTerm(0,0,1,0,0,0,2,0),
    GPU.Parsers.GreenTwoExTerm(0,1,1,1,0,1,1,1),
    GPU.Parsers.GreenTwoExTerm(0,0,0,0,0,1,0,1))
walkers=2
state,pf=GPU.initial_gpu_state(data,walkers;seed=79,return_pfaffians=true,needs_derivatives=false)
cache=GPU.GPUSamplingCache(state)
inverse=GPU.BatchedSamplingInverse(state)
GPU.refresh_sampling_state!(state,cache,inverse;initial_pfaffians=pf)
observables=GPU.CUDAObservables(state,data)
steps=20
tape=reshape([UInt32((UInt64(k)*0x9e3779b9+0xbf58476d)&0xffffffff) for k in 1:5walkers*steps],5,walkers,steps)
println("Julia=",VERSION," CUDA=",pkgversion(CUDA)," parameter_sha256=",bytes2hex(sha256(read(ARGS[2]))))
CUDA.versioninfo()
open(ARGS[3],"w") do io
    function dump(name,x)
        values=vec(Array(x)); println(io,name," ",length(values))
        foreach(x->println(io,x),values)
    end
    dump("initial_cfg",state.configurations)
    dump("parameters",state.parameters)
    dump("initial_pf",cache.pfaffians)
    MVMCOptimizers.gpu_observables!(state,observables;sampling_cache=cache)
    dump("initial_values",observables.values)
    dump("tape",tape)
    MVMCOptimizers.gpu_sample!(state;steps,tape=CuArray(tape),cache,
        empty_destinations=true,incremental_correlation=true,triangle_update=true)
    dump("final_cfg",state.configurations)
    dump("accepted",state.accepted)
    MVMCOptimizers.gpu_observables!(state,observables;sampling_cache=cache)
    dump("final_values",observables.values)
end
CUDA.synchronize()
println("wrote ",ARGS[3])
