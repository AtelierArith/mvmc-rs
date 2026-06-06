using PfaPack

function next_f64!(state::Base.RefValue{UInt64})
    state[] = state[] * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    bits = UInt32(state[] >> 32)
    return (Float64(bits) / Float64(typemax(UInt32))) * 2.0 - 1.0
end

function skew_real(n::Int, seed::UInt64)
    s = Ref(seed + UInt64(0xdeadbeef))
    A = zeros(Float64, n, n)
    for j in 1:n, i in 1:j-1
        v = next_f64!(s)
        A[i, j] = v
        A[j, i] = -v
    end
    return A
end

function skew_complex(n::Int, seed::UInt64)
    s = Ref(seed + UInt64(0xdeadbeef))
    A = zeros(ComplexF64, n, n)
    for j in 1:n, i in 1:j-1
        v = ComplexF64(next_f64!(s), next_f64!(s))
        A[i, j] = v
        A[j, i] = -v
    end
    return A
end

function median_ms(samples)
    sort!(samples)
    return samples[cld(length(samples), 2)] / 1e6
end

function bench(f, iters::Int)
    for _ in 1:3
        f()
    end
    samples = Vector{Float64}(undef, iters)
    for i in 1:iters
        t0 = time_ns()
        f()
        samples[i] = Float64(time_ns() - t0)
    end
    return median_ms(samples)
end

iters_for(n) = n <= 64 ? 200 : n <= 128 ? 80 : n <= 256 ? 25 : 8

println("impl,kind,n,op,median_ms")
for n in (32, 64, 128, 256)
    iters = iters_for(n)

    orig = skew_real(n, UInt64(42))
    t = bench(iters) do
        A = copy(orig)
        Base.inferencebarrier(pfaffian_ltl!(A))
    end
    println("julia,real,$n,pfaffian_ltl,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(orig)
        piv = zeros(Int, n)
        Base.inferencebarrier(julia_dsktf2!(A, piv))
    end
    println("julia,real,$n,ltl,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(orig)
        piv = zeros(Int, n)
        julia_dsktf2!(A, piv)
        Base.inferencebarrier(utu2pfa(n, A, n, piv))
    end
    println("julia,real,$n,ltl_utu2pfa,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(orig)
        piv = zeros(Int, n)
        julia_dsktf2!(A, piv)
        vt = zeros(Float64, n - 1)
        m = zeros(Float64, n, n)
        utu2inv!(n, A, n, piv, vt, m, n)
        Base.inferencebarrier(A[1, 1])
    end
    println("julia,real,$n,ltl_utu2inv,$(round(t; digits=6))")

    origc = skew_complex(n, UInt64(42))
    t = bench(iters) do
        A = copy(origc)
        Base.inferencebarrier(pfaffian_ltl!(A))
    end
    println("julia,complex,$n,pfaffian_ltl,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(origc)
        piv = zeros(Int, n)
        Base.inferencebarrier(julia_zsktf2!(A, piv))
    end
    println("julia,complex,$n,ltl,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(origc)
        piv = zeros(Int, n)
        Base.inferencebarrier(julia_zsktf2_turbo!(A, piv))
    end
    println("julia_lv,complex,$n,ltl,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(origc)
        piv = zeros(Int, n)
        julia_zsktf2!(A, piv)
        Base.inferencebarrier(utu2pfa(n, A, n, piv))
    end
    println("julia,complex,$n,ltl_utu2pfa,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(origc)
        piv = zeros(Int, n)
        julia_zsktf2_turbo!(A, piv)
        Base.inferencebarrier(utu2pfa(n, A, n, piv))
    end
    println("julia_lv,complex,$n,ltl_utu2pfa,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(origc)
        piv = zeros(Int, n)
        julia_zsktf2!(A, piv)
        vt = zeros(ComplexF64, n - 1)
        m = zeros(ComplexF64, n, n)
        utu2inv!(n, A, n, piv, vt, m, n)
        Base.inferencebarrier(A[1, 1])
    end
    println("julia,complex,$n,ltl_utu2inv,$(round(t; digits=6))")

    t = bench(iters) do
        A = copy(origc)
        piv = zeros(Int, n)
        julia_zsktf2_turbo!(A, piv)
        vt = zeros(ComplexF64, n - 1)
        m = zeros(ComplexF64, n, n)
        utu2inv!(n, A, n, piv, vt, m, n)
        Base.inferencebarrier(A[1, 1])
    end
    println("julia_lv,complex,$n,ltl_utu2inv,$(round(t; digits=6))")
end
