# Independent UInt32 cuRAND fixture for #560, generated with Julia 1.13.1.
# Reference Julia-mVMC: 47452f9061ec3de049d891a7aa7fd9cfeac228a4.
# CUDA_VISIBLE_DEVICES=1 julia +1.13.1 --project=/path/to/Julia-mVMC/test/gpu \
#   gen_curand_julia.jl curand_julia.txt
# Consecutive calls intentionally cross the XORWOW library's 4096-state boundary;
# odd/zero lengths and reseeding must preserve CUDA.rand's actual call contract.
using CUDA
CUDA.device!(0)
CUDA.allowscalar(false)
CUDA.versioninfo()
open(ARGS[1], "w") do io
    println(io, "# Julia ", VERSION, " CUDA.jl ", pkgversion(CUDA))
    println(io, "# Reference 47452f9061ec3de049d891a7aa7fd9cfeac228a4; cuRAND ", CUDA.cuRAND.version(), "; ", CUDA.name(CUDA.device()))
    for seed in [UInt64(0), UInt64(79), UInt64(2)^40+123, UInt64(79)]
        println(io, "seed ", seed)
        CUDA.seed!(seed)
        for n in [0, 1, 5, 31, 32, 33, 4095, 4097, 75]
            words = Array(CUDA.rand(UInt32, n))
            println(io, "words ", n)
            println(io, join(words, ' '))
        end
    end
end
