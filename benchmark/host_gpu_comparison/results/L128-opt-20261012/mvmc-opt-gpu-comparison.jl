using CUDA, MVMCOptimizers, LinearAlgebra, Serialization
CUDA.device!(0)
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
const INPUT = abspath(ARGS[1])
const OUTPUT = abspath(ARGS[2])
mkpath(OUTPUT)
println("Julia=",VERSION," GPU=",CUDA.name(CUDA.device())," BLAS=",BLAS.get_config(),
        " Julia_threads=",Threads.nthreads()," BLAS_threads=",BLAS.get_num_threads())
println("optimizer_source=",pathof(MVMCOptimizers))
println("READY"); flush(stdout)
for line in eachline(stdin)
    line == "quit" && break
    rep = parse(Int,line)
    GC.gc(); CUDA.synchronize()
    start = time_ns()
    result = run_gpu_para_opt_from_namelist(INPUT;nsteps=20,walkers=320,device=0,
        seed=79,warmup=128,interval=128)
    CUDA.synchronize()
    elapsed = (time_ns()-start)/1e9
    @assert result.status == 0
    @assert all(isfinite,result.energies) && all(isfinite,result.parameters)
    serialize(joinpath(OUTPUT,"rep-$(rep).bin"), (;seconds=elapsed,
        energies=result.energies,parameters=result.parameters,
        cfg=Array(result.state.configurations),accepted=Array(result.state.accepted)))
    println("rep=",rep," seconds=",elapsed," energy_per_site=",last(result.energies)/128)
    GC.gc(); CUDA.synchronize()
    println("DONE ",rep); flush(stdout)
end
