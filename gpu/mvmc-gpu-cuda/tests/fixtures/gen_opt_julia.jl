# Independent resident Opt stage oracle, Julia-mVMC 47452f9061ec3de049d891a7aa7fd9cfeac228a4.
# julia +1.13.1 --project=/path/to/Julia-mVMC/test/gpu gen_opt_julia.jl namelist.def output.txt [walkers]
using CUDA, MVMCOptimizers, LinearAlgebra
CUDA.device!(0); CUDA.allowscalar(false); BLAS.set_num_threads(1)
const ext=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
data,rng=ext.prepared_gpu_data(abspath(ARGS[1]);seed=79)
operators=get(ENV,"MVMC_OPT_FIXTURE_OPERATORS","0")=="1"
if operators
    P=ext.Parsers
    parameters=MVMCOptimizers.pack_parameters(data)
    for p in 1:P.projection_layout(data).n_proj
        parameters[p]=(mod(p,7)-3)/100
    end
    MVMCOptimizers.unpack_parameters!(data,parameters)
    push!(data.coulomb_inter_terms,P.CoulombInterTerm(0,1,0.125))
    push!(data.hund_terms,P.HundTerm(0,1,0.0625))
    push!(data.pair_hop_terms,P.PairHopTerm(0,1,-0.03125),P.PairHopTerm(1,0,-0.03125))
    push!(data.exchange_terms,P.ExchangeTerm(0,1,0.25))
    for i in 0:2,j in 0:2,k in 0:2,l in 0:2,s in 0:1,t in 0:1
        push!(data.inter_all_terms,P.InterAllTerm(i,s,j,s,k,t,l,t,0.015625+0im,false))
    end
end
walkers=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 8
s,pf=ext.initial_gpu_state(data,walkers;seed=79,return_pfaffians=true)
cache=ext.GPUSamplingCache(s)
inverse=ext.BatchedSamplingInverse(s)
ext.refresh_sampling_state!(s,cache,inverse;initial_pfaffians=pf)
o=ext.CUDAObservables(s,data;measure_greens=false)
sr=ext.CUDASR(s,data)
MVMCOptimizers.gpu_observables!(s,o;sampling_cache=o.batched_inverse === nothing ? nothing : cache)
ext.derivatives!(s,sr,o)
CUDA.synchronize()
println("Julia=",VERSION," CUDA=",pkgversion(CUDA)," curand=",CUDA.cuRAND.version()," cublas=",CUDA.cuBLAS.version()," cusolver=",CUDA.CUSOLVER.version())
open(ARGS[2],"w") do io
    function dump(name,x)
        v=vec(Array(x));println(io,name," ",length(v));foreach(a->println(io,a),v)
    end
    dump("parameters",s.parameters)
    full=get(ENV,"MVMC_OPT_FIXTURE_FULL","0")=="1"
    full && dump("slater",s.slater)
    dump("pf",o.pfaffians)
    full && dump("inverse",permutedims(Array(ext.cache_view(o.inverse)),(2,1,3,4)))
    dump("derivatives",sr.derivatives)
    dump("energy",Array(o.values)[1,:])
    operators && dump("terms",o.termvalues)
    dump("overlaps",s.overlaps)
    dump("cfg",s.configurations)
    moments=ext.sr_moments(sr,copy(vec(view(o.values,1,:))))
    solver=ext.CUDASRSolver(length(s.parameters))
    ext.resident_sr_update!(s,sr,moments,data.modpara,solver)
    ext.sync_gpu_parameters!(s,ext.CUDAParameterGauge(data))
    ext.check_sr_solver(solver)
    dump("parameters_after",s.parameters)
    dump("energies",moments.meanE)
end
