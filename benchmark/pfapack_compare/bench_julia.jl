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

function print_result(impl::String, kind::String, n::Int, op::String, t)
    println("$impl,$kind,$n,$op,$(round(t; digits=6))")
end

function run_benchmark()
    for n in (32, 64, 128, 256)
        iters = iters_for(n)

        orig = skew_real(n, UInt64(42))
        A = similar(orig)
        t = bench(iters) do
            copyto!(A, orig)
            Base.inferencebarrier(pfaffian_ltl!(A))
        end
        print_result("julia", "real", n, "pfaffian_ltl", t)

        A = similar(orig)
        piv = Vector{Int}(undef, n)
        t = bench(iters) do
            copyto!(A, orig)
            fill!(piv, 0)
            Base.inferencebarrier(julia_dsktf2!(A, piv))
        end
        print_result("julia", "real", n, "ltl", t)

        A = similar(orig)
        piv = Vector{Int}(undef, n)
        t = bench(iters) do
            copyto!(A, orig)
            fill!(piv, 0)
            julia_dsktf2!(A, piv)
            Base.inferencebarrier(utu2pfa(n, A, n, piv))
        end
        print_result("julia", "real", n, "ltl_utu2pfa", t)

        A = similar(orig)
        piv = Vector{Int}(undef, n)
        vt = Vector{Float64}(undef, n - 1)
        m = Matrix{Float64}(undef, n, n)
        t = bench(iters) do
            copyto!(A, orig)
            fill!(piv, 0)
            julia_dsktf2!(A, piv)
            utu2inv!(n, A, n, piv, vt, m, n)
            Base.inferencebarrier(A[1, 1])
        end
        print_result("julia", "real", n, "ltl_utu2inv", t)

        origc = skew_complex(n, UInt64(42))
        A = similar(origc)
        t = bench(iters) do
            copyto!(A, origc)
            Base.inferencebarrier(pfaffian_ltl!(A))
        end
        print_result("julia", "complex", n, "pfaffian_ltl", t)

        A = similar(origc)
        piv = Vector{Int}(undef, n)
        t = bench(iters) do
            copyto!(A, origc)
            fill!(piv, 0)
            Base.inferencebarrier(julia_zsktf2!(A, piv))
        end
        print_result("julia", "complex", n, "ltl", t)

        A = similar(origc)
        piv = Vector{Int}(undef, n)
        t = bench(iters) do
            copyto!(A, origc)
            fill!(piv, 0)
            Base.inferencebarrier(julia_zsktf2_turbo!(A, piv))
        end
        print_result("julia_lv", "complex", n, "ltl", t)

        A = similar(origc)
        piv = Vector{Int}(undef, n)
        t = bench(iters) do
            copyto!(A, origc)
            fill!(piv, 0)
            julia_zsktf2!(A, piv)
            Base.inferencebarrier(utu2pfa(n, A, n, piv))
        end
        print_result("julia", "complex", n, "ltl_utu2pfa", t)

        A = similar(origc)
        piv = Vector{Int}(undef, n)
        t = bench(iters) do
            copyto!(A, origc)
            fill!(piv, 0)
            julia_zsktf2_turbo!(A, piv)
            Base.inferencebarrier(utu2pfa(n, A, n, piv))
        end
        print_result("julia_lv", "complex", n, "ltl_utu2pfa", t)

        A = similar(origc)
        piv = Vector{Int}(undef, n)
        vt = Vector{ComplexF64}(undef, n - 1)
        m = Matrix{ComplexF64}(undef, n, n)
        t = bench(iters) do
            copyto!(A, origc)
            fill!(piv, 0)
            julia_zsktf2!(A, piv)
            utu2inv!(n, A, n, piv, vt, m, n)
            Base.inferencebarrier(A[1, 1])
        end
        print_result("julia", "complex", n, "ltl_utu2inv", t)

        A = similar(origc)
        piv = Vector{Int}(undef, n)
        vt = Vector{ComplexF64}(undef, n - 1)
        m = Matrix{ComplexF64}(undef, n, n)
        t = bench(iters) do
            copyto!(A, origc)
            fill!(piv, 0)
            julia_zsktf2_turbo!(A, piv)
            utu2inv!(n, A, n, piv, vt, m, n)
            Base.inferencebarrier(A[1, 1])
        end
        print_result("julia_lv", "complex", n, "ltl_utu2inv", t)
    end
end

println("impl,kind,n,op,median_ms")
run_benchmark()
