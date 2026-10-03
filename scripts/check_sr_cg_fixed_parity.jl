include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Run: julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_fixed_parity.jl
# Pass --write to regenerate. Inputs use exact binary fractions and no RNG.
# The upstream CG implementation (including its BLAS calls) is the oracle.
using MVMCOptimizers, LinearAlgebra, Test

VERSION == v"1.13.1" || error("SR-CG reference fixtures require Julia 1.13.1")
const FIXTURE_ROOT = joinpath(@__DIR__, "..", "tests", "fixtures", "sr_cg")
BLAS.set_num_threads(1)

bits(v) = join((string(reinterpret(UInt64, x); base=16, pad=16) for x in v), " ")
numerical(s) = join(filter(line -> !startswith(line, "#"), split(s, '\n')), '\n')

function fixture_workspace(all_complex)
    if "--sampled" in ARGS
        lines = filter(l -> !startswith(l,"#"), readlines(joinpath(FIXTURE_ROOT,"sampled_complex.txt")))
        n, samples, _ = parse.(Int, split(lines[1]))
        ws = MVMCOptimizers.CGWorkspace(n,samples,true)
        parse_bits(l) = reinterpret.(Float64,parse.(UInt64,split(l);base=16))
        ws.stcO .= parse_bits(lines[2]);ws.sdiag .= parse_bits(lines[3])
        ws.stcOs_real .= reshape(parse_bits(lines[4]),n,samples)
        ws.stcOs_imag .= reshape(parse_bits(lines[5]),n,samples)
        ws.g .= parse_bits(lines[6])
        return ws,n,samples
    end
    n, samples = 32, 48
    ws = MVMCOptimizers.CGWorkspace(n, samples, all_complex)
    # Fourteen powers of two produce an ill-conditioned Gram matrix, keeping
    # the iteration alive through both the 20th and 40th residual refresh.
    for j in 1:samples, i in 1:n
        scale = 2.0^(-mod(i, 14))
        ws.stcOs_real[i, j] = Float64(mod(17i + 13j + 7i*j, 101) - 50) / 64 * scale
        if all_complex
            ws.stcOs_imag[i, j] = Float64(mod(11i + 19j + 3i*j, 97) - 48) / 128 * scale
        end
    end
    for i in 1:n
        ws.stcO[i] = Float64(mod(7i, 19) - 9) / 1024 * 2.0^(-mod(i, 14))
        imag_gram = all_complex ? sum(abs2, ws.stcOs_imag[i, :]) / samples : 0.0
        ws.sdiag[i] = sum(abs2, ws.stcOs_real[i, :]) / samples + imag_gram - ws.stcO[i]^2
        ws.g[i] = Float64(mod(11i, 23) - 11) / 64
    end
    return ws, n, samples
end

@testset "Fixed-input SR-CG numerical values (Julia 1.13.1)" begin
    for all_complex in ("--sampled" in ARGS ? (true,) : (false,true))
        ws, n, samples = fixture_workspace(all_complex)
        io = IOBuffer()
        println(io, "# Julia ", VERSION, "; ", BLAS.get_config(), "; threads=1")
        println(io, "# Julia-mVMC v0.5.0 c2ea432785bc14364a3cd5e9eef44db464289cc9")
        println(io, n, " ", samples, " ", Int(all_complex))
        for v in (ws.stcO, ws.sdiag, vec(ws.stcOs_real), vec(ws.stcOs_imag), ws.g)
            println(io, bits(v))
        end
        z = zeros(n)
        MVMCOptimizers.operate_by_s!(z, ws.g, ws, n, samples, 1/samples, 1e-5, all_complex)
        println(io, bits(z))
        # Restart from the same input with each iteration limit. This records
        # every prefix using the unmodified upstream solver, rather than a
        # second implementation of the CG loop in the fixture generator.
        for limit in 1:41
            trial = deepcopy(ws)
            fill!(trial.x, 0.0)
            iterations = MVMCOptimizers.stochastic_opt_cg_main!(
                trial, n, samples, 1/samples, 1e-5, 0.0, limit, all_complex)
            @test iterations <= limit
            println(io, limit, " ", iterations, " ", bits(trial.x))
            println(io, bits(trial.r))
            println(io, bits(trial.d))
        end
        actual = String(take!(io))
        path = joinpath(FIXTURE_ROOT, "--sampled" in ARGS ? "sampled_complex.txt" : all_complex ? "complex.txt" : "real.txt")
        if "--write" in ARGS
            mkpath(FIXTURE_ROOT)
            write(path, actual)
        else
            # Library names are informational; numerical bounds and explicit residuals are checked.
            @test compare_cg_fixed(actual,read(path,String))
        end
    end
end
