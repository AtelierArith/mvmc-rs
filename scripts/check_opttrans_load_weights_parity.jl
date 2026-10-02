# Exact OptTrans full-record loading and data-level QP-weight contracts.
using Test, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("OptTrans fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const P = MVMCExpertModeParsers
const O = MVMCOptimizers
const root = normpath(joinpath(@__DIR__, "..", "tests", "fixtures", "opttrans"))
hex(values) = join([string(reinterpret(UInt64, x); base=16, pad=16) for v in values for x in (real(v), imag(v))], " ")
function verify(file, actual)
    path = joinpath(root, file)
    "--write" in ARGS ? write(path, actual) : @test(actual == read(path, String))
end
function model(name)
    base = name in ("short_opt", "long_opt", "empty_opt") ? "layout" : name
    d = parse_expert_mode_files(joinpath(root, "namelist_$base.def"))
    name == "empty_opt" && empty!(d.opt_trans)
    name == "short_opt" && resize!(d.opt_trans, 1)
    name == "long_opt" && push!(d.opt_trans, 0.5 - 0.25im)
    return d
end
function values(d)
    vcat(P.projection_parameters(d), [t.value for terms in O._rbm_parameter_sections(d) for t in terms], [t.value for t in d.orbital_terms], d.opt_trans)
end
function failure(f, path)
    try
        return (f(), "")
    catch e
        return (-1, replace(sprint(showerror, e), path => basename(path)))
    end
end
@testset "Canonical OptTrans loaders and weights" begin
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test_unit", "test_unit_read_opt_para.jl"))
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCExpertModeParsers.jl", "test", "test_qp_weight.jl"))
    io = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; source 8bb1b9e / c2ea432; exact loader contract")
    for name in ("valid", "layout", "qp_only", "empty_opt", "short_opt", "long_opt")
        n = P.count_variational_parameters(model(name))
        tokens = vcat([string(i) for i in 1:6], [string(v) for i in 0:n-1 for v in (i + 0.125, -i - 0.375, 123.0)])
        for case in ("valid", "short", "extra", "extra_triple", "garbled", "nan_gradient", "nan_parameter", "hex", "underflow", "overflow", "absent")
            altered = copy(tokens)
            case == "short" && resize!(altered, length(altered)-3)
            case == "extra" && push!(altered, "0.125")
            case == "extra_triple" && append!(altered, ["0.5", "0.25", "123"])
            case == "garbled" && (altered[end-1] = "bogus")
            case == "nan_gradient" && (altered[end] = "NaN")
            case == "nan_parameter" && (altered[end-1] = "NaN")
            case == "hex" && (altered[end-2] = "0x1.8p-1")
            case == "underflow" && (altered[end-1] = "1e-999")
            case == "overflow" && (altered[end-1] = "1e999")
            file = "load_$(name)_$case.def"
            case != "absent" && verify(file, join(altered, " ") * "\n")
            path = joinpath(root, file)
            for loader in ("initial", "optimized")
                d = model(name)
                result, err = failure(() -> loader == "initial" ? O.read_initial_def!(d, path) : O.read_opt_para_file!(d, path), path)
                println(io, "$name $case $loader $(Int(result))")
                println(io, err)
                println(io, hex(values(d)))
                println(io, hex(d.para_qp_opt_trans))
            end
        end
    end
    verify("loaders.txt", String(take!(io)))
    io = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; exact initialized and updated QP weights")
    for leg in (1, 2, 4, 8), stot in (0, 1, 2), mp in (-2, 2), mode in ("empty", "one", "complex", "zero")
        d = ExpertModeData()
        d.modpara = P.ModParaParameters(nsp_gauss_leg=leg, nsp_stot=stot, nmp_trans=mp)
        d.para_qp_trans = [1.0 + 0.25im, -0.5 - 0.125im]
        d.opt_trans = mode == "empty" ? ComplexF64[] : mode == "one" ? [0.75 - 0.0im] : mode == "zero" ? [0.0 + 0.0im, -0.0 - 0.0im] : [0.25 - 0.125im, -0.75 + 0.5im]
        P.init_qp_weight!(d)
        for phase in ("initial", "replace", "grow", "shrink", "clear", "repeat")
            phase == "replace" && (d.opt_trans = [0.5 + 0.25im, -0.125 - 0.75im])
            phase == "grow" && push!(d.opt_trans, 1.5 - 0.5im)
            phase == "shrink" && resize!(d.opt_trans, 1)
            phase == "clear" && empty!(d.opt_trans)
            phase != "initial" && O.update_qp_weight!(d)
            println(io, "$leg $stot $mp $mode $phase")
            for field in (:qp_full_weight, :qp_fix_weight, :spgl_cos, :spgl_sin, :spgl_cos_sin, :spgl_cos_cos, :spgl_sin_sin)
                println(io, hex(getfield(d.qp_weights, field)))
            end
        end
    end
    verify("weights.txt", String(take!(io)))
end
