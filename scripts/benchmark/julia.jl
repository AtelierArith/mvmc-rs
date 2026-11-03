# Warm public API runner; setup/JIT (rep 0), explicit GC and serialization excluded.
using MVMCOptimizers, LinearAlgebra, Serialization
BLAS.set_num_threads(1)
mode, workload, input, output = ARGS[1:4]
steps, samples, reps = parse.(Int, ARGS[5:7])
mkpath(output)
println("Julia=", VERSION, " threads=", Threads.nthreads(), " BLAS=", BLAS.get_config())
if mode == "gpu"
    @eval using CUDA
    CUDA.device!(0)
    CUDA.allowscalar(false)
    println("GPU=", CUDA.name(CUDA.device()))
    CUDA.versioninfo()
end
flush(stdout)
for rep in 0:reps
    dest = joinpath(output, "rep$(rep)")
    mkpath(dest)
    GC.gc()
    mode == "gpu" && CUDA.synchronize()
    start = time_ns()
    result = if mode == "cpu" && workload == "opt"
        run_para_opt_from_namelist(joinpath(input,"namelist.def"); nsteps=steps,
            nsmp=1, mode=:real, seed=79, output_dir=dest)
    elseif mode == "cpu"
        run_phys_cal_from_namelist(joinpath(input,"namelist.def");
            opt_para=joinpath(input,"zqp_opt.dat"), mode=:real, seed=79, output_dir=dest)
    elseif workload == "opt"
        run_gpu_para_opt_from_namelist(joinpath(input,"namelist.def"); nsteps=steps,
            walkers=samples, device=0, seed=79, warmup=128, interval=128)
    else
        run_gpu_phys_cal_from_namelist(joinpath(input,"namelist.def");
            opt_para=joinpath(input,"zqp_opt.dat"), nsamples=1, walkers=samples,
            device=0, seed=79, warmup=128, interval=128)
    end
    mode == "gpu" && CUDA.synchronize()
    seconds = (time_ns()-start)/1e9
    @assert result.status == 0
    # Save returned GPU values only after the synchronized API timing.
    if mode == "gpu"
        values = workload == "opt" ? (energies=result.energies, parameters=result.parameters) :
            (energy=result.energy, one=result.green_one, two=result.green_two, two_ex=result.green_two_ex)
        serialize(joinpath(dest,"result.bin"), values)
        for key in keys(values)
            xs = getproperty(values,key)
            xs = xs isa Number ? [xs] : vec(Array(xs))
            @assert all(isfinite,xs)
            open(joinpath(dest,"$(key).txt"),"w") do io
                foreach(x -> println(io,x),xs)
            end
        end
    else
        @assert any(startswith(name,"zvo_out") for name in readdir(dest))
    end
    println("BENCH rep=",rep," seconds=",seconds)
    flush(stdout)
end
