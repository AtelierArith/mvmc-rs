# Generate a shared-tape parity fixture for the Rust interval sampler (issue #547).
#
# Runs the Julia GPU sampler (`gpu_sample!` -> `cached_sampling_kernel!`) on a small
# synthetic, projection-free state with a fixed deterministic tape, and dumps the
# state, the tape and the resulting trajectory. The Rust ignored test
# `interval_parity` replays the same tape through `IntervalSampler` and compares.
#
# Run in the reference environment:
#   julia +1.13.1 --project=extern/Julia-mVMC/test/gpu \
#       gpu/mvmc-gpu-cuda/tests/fixtures/gen_interval_parity.jl
#
# Julia 1.13.1, CUDA.jl 6.4.2, A100 (sm_80), no projections (nproj=0, ngutz=0),
# exchange_fraction=0 (NExUpdatePath=0), Julia `empty_destinations=true` proposal.

using MVMCOptimizers, CUDA, Printf

const GPU = Base.get_extension(MVMCOptimizers, :MVMCCUDAExt)
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)

# Deterministic 32-bit word shared with the Rust generator (no host RNG, so the
# same tape is reproducible without reproducing Julia's MersenneTwister).
word(k, s) = UInt32((UInt64(k) * 0x9E3779B97F4A7C15 + UInt64(s) * 0xBF58476D1CE4E5B9) & 0xffffffff)

function main()
    ns = 8
    ne = 4
    nq = 2
    batch = 1
    steps = 20
    n = 2ne

    rng = nothing
    n = 2ne       # electron slots / inverse side
    m = 2ns       # Slater table side
    slater = zeros(Float64, m, m, nq)
    for q in 1:nq, i in 1:m, j in 1:m
        slater[i, j, q] = sin(0.37 * (i + 3j + 7q)) + 0.11 * cos(0.19 * (2i - j + q))
    end
    for q in 1:nq
        slater[:, :, q] .-= transpose(slater[:, :, q])
    end
    weights = [1.0, 0.4]

    cfg = zeros(Int32, n, batch)
    cfg[1:ne, 1] .= Int32.(0:(ne - 1))          # up electrons at 0..ne-1
    cfg[(ne + 1):n, 1] .= Int32.(ne:(n - 1))    # down electrons at ne..n-1

    tape = Array{UInt32}(undef, 5, batch, steps)
    for k in 1:5, s in 1:steps
        tape[k, 1, s] = word(k, s - 1)
    end

    s = GPU.CUDAState(ns, ne, 0, nq, 32, CUDA.zeros(Float64, 0), nothing,
        CuArray(slater), CuArray(weights), CuArray(cfg),
        CUDA.zeros(Int32, 2ns, batch), CUDA.zeros(Int32, 0, batch),
        CuArray(fill(Int32(-1), ns)), CuArray(fill(Int32(-1), ns, ns)), 0, 0,
        CUDA.zeros(Float64, batch, n, n), CUDA.zeros(Int32, batch, n),
        CUDA.zeros(Float64, batch), CUDA.zeros(Float64, batch), CUDA.zeros(Float64, batch),
        CuArray(zeros(Int32, ns)), CUDA.zeros(Int32, batch),
        CUDA.zeros(Float64, n, n, batch), nothing)

    CPU = MVMCOptimizers
    CPU.gpu_wavefunctions!(s)
    cache = GPU.GPUSamplingCache(s)
    CPU.gpu_sample!(s; steps=steps, exchange_fraction=0.0, tape=CuArray(tape), cache=cache,
        empty_destinations=true)

    out = joinpath(@__DIR__, "interval_parity_j8.txt")
    open(out, "w") do io
        @printf(io, "# julia 1.13.1 CUDA.jl 6.4.2 A100 sm_80; generate with gen_interval_parity.jl\n")
        @printf(io, "nsite %d\n", ns)
        @printf(io, "ne %d\n", ne)
        @printf(io, "nqp %d\n", nq)
        @printf(io, "npair %d\n", n)
        @printf(io, "steps %d\n", steps)
        @printf(io, "slater %d\n", m * m * nq)
        for q in 1:nq, j in 1:m, i in 1:m   # column-major (i fastest) to match CUDA
            @printf(io, "%.17g\n", slater[i, j, q])
        end
        @printf(io, "weights %d\n", nq)
        for q in 1:nq
            @printf(io, "%.17g\n", weights[q])
        end
        @printf(io, "ele %d\n", n)
        for j in 1:n
            @printf(io, "%d\n", cfg[j, 1])
        end
        @printf(io, "tape %d\n", 5 * steps)
        for s in 1:steps, k in 1:5
            @printf(io, "%d\n", tape[k, 1, s])
        end
        final = Array(s.configurations)
        @printf(io, "final_ele %d\n", n)
        for j in 1:n
            @printf(io, "%d\n", final[j, 1])
        end
        @printf(io, "final_occupancy %d\n", 2ns)
        occ = Array(s.occupancy)
        for r in 1:2ns
            @printf(io, "%d\n", occ[r, 1])
        end
        @printf(io, "accepted %d\n", Array(s.accepted)[1])
    end
    println("wrote ", out)
    println("final cfg = ", vec(Array(s.configurations)))
    println("accepted  = ", Array(s.accepted)[1])
    return nothing
end

main()
