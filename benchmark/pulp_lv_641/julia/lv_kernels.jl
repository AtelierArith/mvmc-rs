# Issue #641: LoopVectorization (@turbo) versus the Julia-mVMC loop forms on the same
# L128 hot kernels as ../rust (K1 UpdateMAll one-hop, K2 dsktf2 rank-2 sweep,
# K3 two-hop bilinear, K4 skew-tridiagonal solve). Flat 0-based-offset vectors with the
# C/Rust memory layout.
#
#   julia +1.13.1 --project=extern/Julia-mVMC benchmark/pulp_lv_641/julia/lv_kernels.jl [n] [rounds]
#
# LoopVectorization is not a direct dependency of the pinned workspace project; it is reached
# through PfaPack (which loads it), so the pinned Manifest-v1.13.toml is used unchanged.
using PfaPack
using LinearAlgebra
const LV = PfaPack.LTLDecomposition.LoopVectorization
const VB = LV.VectorizationBase
using .LV: @turbo

# ---------------------------------------------------------------- K1 UpdateMAll one-hop
# Julia-mVMC form (vmc_sampling.jl update_m_all_real!): @inbounds + @simd ivdep inner loops.
function k1_simd!(inv, slt, vec1, vec2, n, msa, pf)
    fill!(vec1, 0.0)
    @inbounds for msj = 0:(n-1)
        s = slt[msj+1]
        @simd ivdep for msi = 0:(n-1)
            vec1[msi+1] += -inv[msj*n+msi+1] * s
        end
    end
    tmp = vec1[msa+1]
    pf *= -tmp
    ia = -1.0 / tmp
    @inbounds for msi = 0:(n-1)
        vec2[msi+1] = inv[msa*n+msi+1] * ia
    end
    @inbounds for msi = 0:(n-1)
        v1i = vec1[msi+1]
        v2i = vec2[msi+1]
        @simd ivdep for msj = 0:(n-1)
            inv[msi*n+msj+1] += v1i * vec2[msj+1] - vec1[msj+1] * v2i
        end
        inv[msi*n+msa+1] -= v2i
    end
    @inbounds for msj = 0:(n-1)
        inv[msa*n+msj+1] += vec2[msj+1]
    end
    return pf
end

# Same algorithm with @turbo on both O(n^2) passes (LV reorders and contracts to FMA).
function k1_turbo!(inv, slt, vec1, vec2, n, msa, pf)
    fill!(vec1, 0.0)
    @turbo for msj = 0:(n-1)
        for msi = 0:(n-1)
            vec1[msi+1] += -inv[msj*n+msi+1] * slt[msj+1]
        end
    end
    tmp = vec1[msa+1]
    pf *= -tmp
    ia = -1.0 / tmp
    @inbounds for msi = 0:(n-1)
        vec2[msi+1] = inv[msa*n+msi+1] * ia
    end
    @turbo for msi = 0:(n-1)
        for msj = 0:(n-1)
            inv[msi*n+msj+1] += vec1[msi+1] * vec2[msj+1] - vec1[msj+1] * vec2[msi+1]
        end
    end
    @inbounds for msi = 0:(n-1)
        inv[msi*n+msa+1] -= vec2[msi+1]
    end
    @inbounds for msj = 0:(n-1)
        inv[msa*n+msj+1] += vec2[msj+1]
    end
    return pf
end

# ---------------------------------------------------------------- K2 dsktf2 rank-2 sweep
# Julia-mVMC form (PfaPack.jl julia_dsktf2!): column j, @simd over i < j.
function k2_simd!(A::Matrix{Float64}, n)
    alpha = 1.0 / n
    k = n
    while k >= 3
        kk = k - 1
        @inbounds for j in 1:k-2
            t1 = alpha * A[j, kk]
            t2 = alpha * A[j, k]
            @simd for i in 1:j-1
                A[i, j] = (A[i, j] + A[i, k] * t1) - A[i, kk] * t2
            end
            A[j, j] = 0.0
        end
        k -= 2
    end
    return A
end

function k2_turbo!(A::Matrix{Float64}, n)
    alpha = 1.0 / n
    k = n
    while k >= 3
        kk = k - 1
        @inbounds for j in 1:k-2
            t1 = alpha * A[j, kk]
            t2 = alpha * A[j, k]
            @turbo for i in 1:j-1
                A[i, j] = (A[i, j] + A[i, k] * t1) - A[i, kk] * t2
            end
            A[j, j] = 0.0
        end
        k -= 2
    end
    return A
end

# ---------------------------------------------------------------- K3 two-hop bilinear
# C order: per row a serial left-to-right sum, rows accumulated in order (no FMA).
function k3_plain(inv, n, a, b)
    bma = 0.0
    @inbounds for i = 0:(n-1)
        tmp = 0.0
        for j = 0:(n-1)
            tmp += inv[i*n+j+1] * a[j+1]
        end
        bma += b[i+1] * tmp
    end
    return bma
end

function k3_turbo(inv, n, a, b)
    bma = 0.0
    @turbo for i = 0:(n-1)
        tmp = 0.0
        for j = 0:(n-1)
            tmp += inv[i*n+j+1] * a[j+1]
        end
        bma += b[i+1] * tmp
    end
    return bma
end

# ---------------------------------------------------------------- K4 sktd solve
# Columns are independent; the row recurrence is serial. Julia/C form: one column at a time.
function k4_plain!(vt, B, C, n)
    @inbounds for w in 1:n
        C[2, w] = B[1, w] / -vt[1]
        for i in 3:2:n
            C[i+1, w] = (B[i, w] - C[i-1, w] * vt[i-1]) / -vt[i]
        end
        C[n-1, w] = B[n, w] / vt[n-1]
        for i in (n-2):-2:2
            C[i-1, w] = (B[i, w] + C[i+1, w] * vt[i]) / vt[i-1]
        end
    end
    return C
end

# @turbo across columns inside each recurrence step (lanes = columns, strided).
function k4_turbo!(vt, B, C, n)
    d0 = -vt[1]
    @turbo for w in 1:n
        C[2, w] = B[1, w] / d0
    end
    @inbounds for i in 3:2:n
        v = vt[i-1]
        d = -vt[i]
        @turbo for w in 1:n
            C[i+1, w] = (B[i, w] - C[i-1, w] * v) / d
        end
    end
    dn = vt[n-1]
    @turbo for w in 1:n
        C[n-1, w] = B[n, w] / dn
    end
    @inbounds for i in (n-2):-2:2
        v = vt[i]
        d = vt[i-1]
        @turbo for w in 1:n
            C[i-1, w] = (B[i, w] + C[i+1, w] * v) / d
        end
    end
    return C
end

# ---------------------------------------------------------------- harness
bits_differ(a, b) = count(i -> reinterpret(UInt64, a[i]) != reinterpret(UInt64, b[i]), eachindex(a))
max_rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function run_bench(title, rounds, iters, variants)
    for (_, f) in variants
        for _ in 1:iters
            f()
        end
    end
    samples = [Float64[] for _ in variants]
    for r in 0:(rounds-1)
        for k in 0:(length(variants)-1)
            idx = (k + r) % length(variants) + 1
            f = variants[idx][2]
            t = time_ns()
            for _ in 1:iters
                f()
            end
            push!(samples[idx], (time_ns() - t) / iters)
        end
    end
    println("## ", title)
    for (k, (name, _)) in enumerate(variants)
        s = sort(samples[k])
        println(title, "\t", name, "\tmedian_ns=", round(s[div(length(s), 2)+1]; digits=1),
                "\tmin_ns=", round(s[1]; digits=1), "\trounds=", length(s))
    end
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
    rounds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 15
    println("n=$n rounds=$rounds julia=$(VERSION) cpu=$(Sys.CPU_NAME) threads=$(Threads.nthreads())")
    println("LoopVectorization=$(pkgversion(LV)) VectorizationBase=$(pkgversion(VB)) ",
            "register_size=$(VB.register_size()) bytes, pick_vector_width(Float64)=$(VB.pick_vector_width(Float64)), ",
            "fma=$(VB.fma_fast()) avx512f=$(VB.has_feature(Val(:x86_64_avx512f)))")
    println("BLAS: ", BLAS.get_config())
    rng_state = Ref(UInt64(0x9E3779B97F4A7C15))
    function rnd()
        x = rng_state[]
        x ⊻= x << 13; x ⊻= x >> 7; x ⊻= x << 17
        rng_state[] = x
        return Float64(x >> 11) / Float64(UInt64(1) << 53) * 2 - 1
    end

    # K1: genuine inverse, chain of 64 hops, residual check
    M = zeros(n, n)
    for i in 1:n, j in (i+1):n
        v = rnd(); M[i, j] = v; M[j, i] = -v
    end
    X0 = inv(M)
    rowmajor(A) = vec(permutedims(A))
    inv0 = rowmajor(X0)
    resid(Mr, x) = maximum(abs.(reshape(Mr, n, n)' * reshape(x, n, n)' - I))
    results = Dict{String,Vector{Float64}}()
    for (name, f) in (("simd", k1_simd!), ("turbo", k1_turbo!))
        st = Ref(UInt64(12345))
        nxt() = (y = st[]; y ⊻= y << 13; y ⊻= y >> 7; y ⊻= y << 17; st[] = y; y)
        unif() = Float64(nxt() >> 11) / Float64(UInt64(1) << 53) * 2 - 1
        mr = rowmajor(M); xw = copy(inv0); v1 = zeros(n); v2 = zeros(n); pf = 1.0
        for _ in 1:64
            msa = Int(nxt() % UInt64(n))
            slt = [unif() for _ in 1:n]; slt[msa+1] = 0.0
            pf = f(xw, slt, v1, v2, n, msa, pf)
            for j in 0:(n-1)
                mr[msa*n+j+1] = slt[j+1]; mr[j*n+msa+1] = -slt[j+1]
            end
        end
        results[name] = xw
        println("check\tK1 $name\tsteps=64\tresidual=", resid(mr, xw),
                name == "turbo" ? "\tbits_differ_vs_simd=$(bits_differ(xw, results["simd"]))/$(n*n)\tmax_rel_vs_simd=$(max_rel(xw, results["simd"]))" : "")
    end
    let msa = 37, slt = [rnd() for _ in 1:n]
        slt[msa+1] = 0.0
        w1 = copy(inv0); w2 = copy(inv0); w0 = copy(inv0)
        v = [zeros(n) for _ in 1:4]
        run_bench("K1 update_one_real (copy+kernel)", rounds, 2000, [
            "copy-only" => () -> copyto!(w0, inv0),
            "julia-simd(mVMC form)" => () -> (copyto!(w1, inv0); k1_simd!(w1, slt, v[1], v[2], n, msa, 1.0)),
            "julia-turbo" => () -> (copyto!(w2, inv0); k1_turbo!(w2, slt, v[3], v[4], n, msa, 1.0)),
        ])
    end

    # K2
    A0 = zeros(n, n)
    for i in 1:n, j in (i+1):n
        v = rnd(); A0[i, j] = v; A0[j, i] = -v
    end
    rs = k2_simd!(copy(A0), n); rt = k2_turbo!(copy(A0), n)
    println("check\tK2 turbo\tbits_differ_vs_simd=$(bits_differ(rt, rs))/$(n*n)\tmax_rel_vs_simd=$(max_rel(rt, rs))")
    let w0 = copy(A0), w1 = copy(A0), w2 = copy(A0)
        run_bench("K2 dsktf2 rank-2 sweep (copy+kernel)", rounds, 1000, [
            "copy-only" => () -> copyto!(w0, A0),
            "julia-simd(PfaPack form)" => () -> (copyto!(w1, A0); k2_simd!(w1, n)),
            "julia-turbo" => () -> (copyto!(w2, A0); k2_turbo!(w2, n)),
        ])
    end

    # K3
    a = [rnd() for _ in 1:n]; b = [rnd() for _ in 1:n]
    r0 = k3_plain(inv0, n, a, b); r1 = k3_turbo(inv0, n, a, b)
    println("check\tK3 turbo\tbitwise_equal=$(r0 === r1)\trel=$(abs((r1 - r0) / r0))")
    sink = Ref(0.0)
    run_bench("K3 two_hop_bilinear", rounds, 5000, [
        "julia-plain(C order)" => () -> (sink[] += k3_plain(inv0, n, a, b)),
        "julia-turbo" => () -> (sink[] += k3_turbo(inv0, n, a, b)),
    ])

    # K4
    vt = [0.5 + abs(rnd()) for _ in 1:(n-1)]
    B = reshape([rnd() for _ in 1:n*n], n, n)
    C0 = zeros(n, n); C1 = zeros(n, n)
    k4_plain!(vt, B, C0, n); k4_turbo!(vt, B, C1, n)
    println("check\tK4 turbo\tbits_differ_vs_plain=$(bits_differ(C1, C0))/$(n*n)")
    run_bench("K4 solve_sktd", rounds, 2000, [
        "julia-plain(per column)" => () -> k4_plain!(vt, B, C0, n),
        "julia-turbo(across columns)" => () -> k4_turbo!(vt, B, C1, n),
    ])
end

main()
