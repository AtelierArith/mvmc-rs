using MVMCOptimizers, LinearAlgebra, Statistics, SHA
const CPU=MVMCOptimizers
BLAS.set_num_threads(1)
mode,input,output=ARGS
mkpath(output)
namelist=joinpath(input,"namelist.def"); checkpoint=joinpath(input,"zqp_opt.dat")
println("mode=",mode," Julia=",VERSION," threads=",Threads.nthreads()," BLAS=",BLAS.get_config()," parameter_sha256=",bytes2hex(sha256(read(checkpoint))))
if mode == "gpu"
    @eval using CUDA
    CUDA.device!(0); CUDA.allowscalar(false)
    println("GPU=",CUDA.name(CUDA.device()))
end
for rep in 0:3
    GC.gc()
    mode == "gpu" && CUDA.synchronize()
    start=time_ns()
    if mode == "cpu"
        dest=joinpath(output,"julia-$(rep)"); @assert !ispath(dest); mkpath(dest)
        result=CPU.run_phys_cal_from_namelist(namelist;opt_para=checkpoint,mode=:real,seed=79,output_dir=dest)
        seconds=(time_ns()-start)/1e9
        @assert result.status == 0
        energy=parse(Float64,first(split(read(joinpath(dest,"zvo_out_001.dat"),String))))
    else
        result=CPU.run_gpu_phys_cal_from_namelist(namelist;opt_para=checkpoint,nsamples=1,walkers=320,device=0,seed=79,warmup=128,interval=128)
        CUDA.synchronize()
        seconds=(time_ns()-start)/1e9
        @assert result.status == 0 && all(isfinite,result.green_one) && all(isfinite,result.green_two) && all(isfinite,result.green_two_ex)
        energy=result.energy
    end
    @assert isfinite(energy)
    if mode == "gpu"
        # Serialization is outside the synchronized API timing above.
        for (name,values) in [("one",result.green_one),("two",result.green_two),("two_ex",result.green_two_ex)]
            open(joinpath(output,"rep$(rep)-$(name).txt"),"w") do io
                foreach(value -> println(io,value), values)
            end
        end
    end
    println("BENCH rep=",rep," seconds=",seconds," energy=",energy); flush(stdout)
end
