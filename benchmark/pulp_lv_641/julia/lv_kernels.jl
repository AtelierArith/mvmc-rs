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

# ---------------------------------------------------------------- numerical acceptance (#697)
# Finite values within explicit, derived bounds; computed results are never compared through
# their bit patterns. The bounds mirror ../rust/src/main.rs (see the README).
const U = eps(Float64) / 2
gamma(k) = (k * U < 0.5 || error("gamma($k) is not meaningful"); k * U / (1 - k * U))
inf_norm_rowmajor(a, n) = maximum(sum(abs, @view a[i*n+1:(i+1)*n]) for i in 0:(n-1))

struct Cmp
    max_abs::Float64
    max_ratio::Float64
    differing::Int
    nonfinite::Int
    zero_bound_violations::Int
end
ok(c::Cmp) = c.nonfinite == 0 && c.zero_bound_violations == 0 && c.max_ratio <= 1

function compare(a, b, bound)
    length(a) == length(b) || error("length mismatch")
    max_abs = 0.0; max_ratio = 0.0; differing = 0; nonfinite = 0; zb = 0
    for k in eachindex(a, b)
        x, y, bk = a[k], b[k], bound(k)
        if !(isfinite(x) && isfinite(y) && isfinite(bk))
            nonfinite += 1
            continue
        end
        d = abs(x - y)
        max_abs = max(max_abs, d)
        differing += x != y
        if bk > 0
            max_ratio = max(max_ratio, d / bk)
        elseif d > 0
            zb += 1
        end
    end
    return Cmp(max_abs, max_ratio, differing, nonfinite, zb)
end

fmt(x) = string(round(x; sigdigits=4))

mutable struct Acceptance
    checks::Int
    failed::Vector{String}
end
Acceptance() = Acceptance(0, String[])

function record!(acc::Acceptance, label, passed::Bool, detail)
    acc.checks += 1
    println("check\t", label, "\t", detail, "\tstatus=", passed ? "PASS" : "FAIL")
    passed || push!(acc.failed, label)
    return passed
end

function compare!(acc::Acceptance, label, a, b, bound, why)
    c = compare(a, b, bound)
    record!(acc, label, ok(c),
            "max_abs=$(fmt(c.max_abs))\tmax_ratio_to_bound=$(fmt(c.max_ratio))\t" *
            "differing_values=$(c.differing)/$(length(a))\tnonfinite=$(c.nonfinite)\tbound=$why")
end

function finish!(acc::Acceptance)
    if isempty(acc.failed)
        println("correctness\tPASS\tchecks=$(acc.checks)\tfailed=0")
    else
        println("correctness\tFAIL\tchecks=$(acc.checks)\tfailed=$(length(acc.failed))\t",
                join(acc.failed, ","))
        exit(1)
    end
end

function acceptance_self_test()
    tight = _ -> 1e-15
    @assert ok(compare([1.0], [1.0], tight))
    @assert !ok(compare([NaN], [0.0], tight)) "NaN accepted"
    @assert !ok(compare([Inf], [Inf], tight)) "infinity accepted"
    @assert !ok(compare([1.0 + 1e-12], [1.0], tight)) "out-of-bound value accepted"
    @assert !ok(compare([1e-300], [0.0], _ -> 0.0)) "zero bound accepted a difference"
    @assert ok(compare([0.0], [-0.0], _ -> 0.0)) "signed zeros are equal"
    println("check\tacceptance self-test (NaN, infinity, out-of-bound, zero bound)\tstatus=PASS")
end

# K2 sweep on magnitudes (subtractions become additions): bounds every partial result.
function k2_magnitude!(A::Matrix{Float64}, n)
    alpha = 1.0 / n
    A .= abs.(A)
    k = n
    while k >= 3
        kk = k - 1
        for j in 1:k-2
            t1 = alpha * A[j, kk]
            t2 = alpha * A[j, k]
            for i in 1:j-1
                A[i, j] = (A[i, j] + A[i, k] * t1) + A[i, kk] * t2
            end
            A[j, j] = 0.0
        end
        k -= 2
    end
    return A
end

# K4 recurrences on magnitudes.
function k4_magnitude!(vt, B, S, n)
    for w in 1:n
        S[2, w] = abs(B[1, w]) / abs(vt[1])
        for i in 3:2:n
            S[i+1, w] = (abs(B[i, w]) + S[i-1, w] * abs(vt[i-1])) / abs(vt[i])
        end
        S[n-1, w] = abs(B[n, w]) / abs(vt[n-1])
        for i in (n-2):-2:2
            S[i-1, w] = (abs(B[i, w]) + S[i+1, w] * abs(vt[i])) / abs(vt[i-1])
        end
    end
    return S
end

# Relative residual of the skew-tridiagonal relations that k4_plain! solves (1-based):
# forward -vt[i] C[i+1] + vt[i-1] C[i-1] = B[i] (i odd; -vt[1] C[2] = B[1]),
# backward vt[i-1] C[i-1] - vt[i] C[i+1] = B[i] (i even; vt[n-1] C[n-1] = B[n]).
function k4_residual(vt, B, C, n)
    worst = 0.0
    rel(lhs, scale, rhs) = (s = scale + abs(rhs); s > 0 ? abs(lhs - rhs) / s : 0.0)
    for w in 1:n
        worst = max(worst, rel(-vt[1] * C[2, w], abs(vt[1] * C[2, w]), B[1, w]))
        for i in 3:2:n
            p = -vt[i] * C[i+1, w]; q = vt[i-1] * C[i-1, w]
            worst = max(worst, rel(p + q, abs(p) + abs(q), B[i, w]))
        end
        worst = max(worst, rel(vt[n-1] * C[n-1, w], abs(vt[n-1] * C[n-1, w]), B[n, w]))
        for i in (n-2):-2:2
            p = vt[i-1] * C[i-1, w]; q = -vt[i] * C[i+1, w]
            worst = max(worst, rel(p + q, abs(p) + abs(q), B[i, w]))
        end
    end
    return worst
end

# ---------------------------------------------------------------- harness

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
    acceptance_self_test()
    acc = Acceptance()
    rng_state = Ref(UInt64(0x9E3779B97F4A7C15))
    function rnd()
        x = rng_state[]
        x ⊻= x << 13; x ⊻= x >> 7; x ⊻= x << 17
        rng_state[] = x
        return Float64(x >> 11) / Float64(UInt64(1) << 53) * 2 - 1
    end

    # K1: genuine inverse, chain of 64 hops. Bounds as ../rust (README "Correctness"):
    # residual r0 + (steps+1) gamma(n+4) |M| |X|, orders 2 steps gamma(n+4) kappa max|X|.
    M = zeros(n, n)
    for i in 1:n, j in (i+1):n
        v = rnd(); M[i, j] = v; M[j, i] = -v
    end
    X0 = inv(M)
    rowmajor(A) = vec(permutedims(A))
    inv0 = rowmajor(X0)
    resid(Mr, x) = maximum(abs.(reshape(Mr, n, n)' * reshape(x, n, n)' - I))
    steps = 64
    b = gamma(n + 4)
    r0 = resid(rowmajor(M), inv0)
    println("info\tK1 initial inverse residual max|M X - I| = ", fmt(r0))
    results = Dict{String,Tuple{Vector{Float64},Float64,Vector{Float64}}}()
    reference = Vector{Vector{Float64}}()
    for (name, f) in (("simd", k1_simd!), ("turbo", k1_turbo!))
        st = Ref(UInt64(12345))
        nxt() = (y = st[]; y ⊻= y << 13; y ⊻= y >> 7; y ⊻= y << 17; st[] = y; y)
        unif() = Float64(nxt() >> 11) / Float64(UInt64(1) << 53) * 2 - 1
        mr = rowmajor(M); xw = copy(inv0); v1 = zeros(n); v2 = zeros(n); pf = 1.0
        first_divergence = "none"
        for hop in 1:steps
            msa = Int(nxt() % UInt64(n))
            slt = [unif() for _ in 1:n]; slt[msa+1] = 0.0
            pf = f(xw, slt, v1, v2, n, msa, pf)
            for j in 0:(n-1)
                mr[msa*n+j+1] = slt[j+1]; mr[j*n+msa+1] = -slt[j+1]
            end
            if name == "simd"
                push!(reference, copy(xw))
            elseif first_divergence == "none" && xw != reference[hop]
                first_divergence = string(hop - 1)
            end
        end
        results[name] = (xw, pf, mr)
        res = resid(mr, xw)
        res_bound = r0 + (steps + 1) * b * inf_norm_rowmajor(mr, n) * inf_norm_rowmajor(xw, n)
        record!(acc, "K1 $name residual", isfinite(res) && res <= res_bound,
                "steps=$steps\tmax|M X - I|=$(fmt(res))\tbound=$(fmt(res_bound))")
        if name == "turbo"
            x0, pf0, _ = results["simd"]
            kappa = inf_norm_rowmajor(mr, n) * inf_norm_rowmajor(x0, n)
            entry = 2 * steps * b * kappa * maximum(abs, x0)
            compare!(acc, "K1 turbo vs simd", xw, x0, _ -> entry,
                     "2*steps*gamma(n+4)*kappa*max|X|=$(fmt(entry)) (kappa=$(fmt(kappa)))\t" *
                     "first_divergent_hop=$first_divergence")
            pf_rel = abs((pf - pf0) / pf0)
            pf_bound = 2 * steps * b * kappa
            record!(acc, "K1 turbo Pfaffian ratio vs simd", isfinite(pf) && pf_rel <= pf_bound,
                    "relative=$(fmt(pf_rel))\tbound=$(fmt(pf_bound))")
        end
    end
    let mr = results["simd"][3]
        fresh = rowmajor(inv(Matrix(reshape(mr, n, n)')))
        kappa = inf_norm_rowmajor(mr, n) * inf_norm_rowmajor(fresh, n)
        entry = 2 * (steps + 1) * b * kappa * maximum(abs, fresh)
        for name in ("simd", "turbo")
            compare!(acc, "K1 $name vs fresh inverse", results[name][1], fresh, _ -> entry,
                     "2*(steps+1)*gamma(n+4)*kappa*max|X|=$(fmt(entry))")
        end
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
    # Two evaluations differ by at most 2 gamma(5n/2+1) S_ij, S the magnitude sweep.
    let S = k2_magnitude!(copy(A0), n), factor = 2 * gamma(5 * div(n, 2) + 1)
        compare!(acc, "K2 turbo vs simd", rt, rs, k -> factor * S[k],
                 "2*gamma(5n/2+1)*S_ij (S: magnitude sweep)")
    end
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
    # Any order of either summation level: within gamma(2n+1) S of the exact value,
    # S = sum_i |b_i| sum_j |M_ij a_j|. a^T X a = 0 exactly only for an exactly skew X; the
    # LAPACK inverse is not, so the cancellation case uses (X - X^T)/2, exactly skew because
    # fl(x - y) = -fl(y - x).
    k3_scale(X, bv, av) = sum(abs(bv[i+1]) * sum(abs(X[i*n+j+1] * av[j+1]) for j in 0:(n-1)) for i in 0:(n-1))
    skew0 = rowmajor((X0 .- X0') ./ 2)
    let s3 = k3_scale(inv0, b, a), s0 = k3_scale(skew0, a, a), g = gamma(2 * n + 1)
        compare!(acc, "K3 turbo vs plain", [r1], [r0], _ -> 2 * g * s3,
                 "2*gamma(2n+1)*S=$(fmt(2 * g * s3))")
        for (name, f) in (("plain", k3_plain), ("turbo", k3_turbo))
            compare!(acc, "K3 $name cancellation a^T M a vs exact 0", [f(skew0, n, a, a)], [0.0],
                     _ -> g * s0, "gamma(2n+1)*S=$(fmt(g * s0))")
        end
    end
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
    # Two evaluations differ by at most 2 gamma(2n+1) S_ij (S: magnitude recurrence); each
    # solution satisfies the relations with relative residual <= 4 eps.
    let S = k4_magnitude!(vt, B, zeros(n, n), n), factor = 2 * gamma(2 * n + 1)
        compare!(acc, "K4 turbo vs plain", C1, C0, k -> factor * S[k],
                 "2*gamma(2n+1)*S_ij (S: magnitude recurrence)")
        for (name, C) in (("plain", C0), ("turbo", C1))
            res = k4_residual(vt, B, C, n)
            record!(acc, "K4 $name residual", all(isfinite, C) && res <= 4 * eps(Float64),
                    "relative residual=$(fmt(res))\tbound=4eps=$(fmt(4 * eps(Float64)))")
        end
    end
    run_bench("K4 solve_sktd", rounds, 2000, [
        "julia-plain(per column)" => () -> k4_plain!(vt, B, C0, n),
        "julia-turbo(across columns)" => () -> k4_turbo!(vt, B, C1, n),
    ])
    finish!(acc)
end

main()
