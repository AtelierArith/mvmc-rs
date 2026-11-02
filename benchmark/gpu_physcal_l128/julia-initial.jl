using CUDA, MVMCOptimizers, LinearAlgebra
BLAS.set_num_threads(1); CUDA.device!(0); CUDA.allowscalar(false)
const ext=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
namelist,parameters,cpu_config,output=ARGS
mkpath(output)
data,rng=ext.prepared_gpu_data(namelist;seed=79,opt_para=parameters,physcal=true)
ne=data.modpara.nelec; ns=data.modpara.nsite
CUDA.seed!(79)
tape=Array(CUDA.rand(UInt32,2ne+2ns+16,320))
open(joinpath(output,"first-initial-words.txt"),"w") do io
    foreach(x->println(io,x),tape[1:8,1])
end
s,pf=ext.initial_gpu_state(data,320;seed=79,return_pfaffians=true,needs_derivatives=false)
configurations=Array(s.configurations)
open(joinpath(output,"first-walker.txt"),"w") do io
    foreach(x->println(io,x),configurations[:,1])
end
cfg=reshape(parse.(Int32,readlines(cpu_config)),:,1)
r=MVMCOptimizers.gpu_phys_cal!(data;configurations=cfg,nsamples=1,warmup=0,interval=0,seed=79)
CUDA.synchronize(); @assert r.status==0
open(joinpath(output,"common-configuration-energy.txt"),"w") do io
    println(io,r.energy)
end
for (name,values) in [("one",r.green_one),("two",r.green_two),("two_ex",r.green_two_ex)]
    @assert all(isfinite,values)
    open(joinpath(output,"common-configuration-$(name).txt"),"w") do io
        foreach(x->println(io,x),values)
    end
end
println("Julia=",VERSION," first_XORWOW=",tape[1,1]," first_site=",configurations[1,1]," common_configuration_energy=",r.energy)
