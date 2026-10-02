# OptTrans input contracts from unmodified Julia 1.13.1 reference sources.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers
VERSION == v"1.13.1" || error("OptTrans fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const P = MVMCExpertModeParsers
const root = normpath(joinpath(@__DIR__, "..", "tests", "fixtures", "opttrans"))
hex(values) = join([string(reinterpret(UInt64, x); base=16, pad=16) for v in values for x in (real(v), imag(v))], " ")
function verify(file, actual)
    path = joinpath(root, file)
    "--write" in ARGS ? write(path, actual) : @test(actual == read(path, String))
end
function failure(f, path)
    try
        f()
        return ""
    catch e
        return replace(sprint(showerror, e), path => basename(path))
    end
end
function emit_state(io, d)
    println(io, d.n_qp_opt_trans)
    println(io, hex(d.para_qp_opt_trans))
    println(io, hex(d.opt_trans))
    println(io, join(vcat(d.qp_opt_trans...), " "))
    println(io, join(vcat(d.qp_opt_trans_sgn...), " "))
end
@testset "Canonical OptTrans input contract" begin
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCExpertModeParsers.jl", "test", "test_read_input_parameters.jl"))
    io = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; Julia-mVMC 8bb1b9e; parser c2ea432")
    cases = sort(filter(n -> endswith(n, ".def") && !(startswith(n, "namelist_") || startswith(n, "overlay") || startswith(n, "load_") || n in ("layout.def", "modpara.def", "qptrans.def")), readdir(root)))
    for name in cases, nsite in (0, 2), nmp in (-1, 0, 1)
        d = ExpertModeData()
        d.modpara = ModParaParameters(nsite=2, nmp_trans=-1)
        P.parse_opttrans_def!(d, joinpath(root, "valid.def"))
        d.opt_trans .= ComplexF64(7, -9)
        d.modpara.nsite = nsite
        d.modpara.nmp_trans = nmp
        path = joinpath(root, name)
        err = failure(() -> P.parse_opttrans_def!(d, path), path)
        println(io, "$name $nsite $nmp $(Int(isempty(err)))")
        println(io, err)
        emit_state(io, d)
    end
    verify("parser.txt", String(take!(io)))
    io = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; seed=11272; exact values and next 624 UInt32 words")
    for name in ("valid", "layout", "opt_first", "failed_replacement", "failed_first", "before_modpara", "missing", "qp_first", "qp_last", "qp_failed", "qp_only"), mode in ("parsed", "complex", "inactive", "empty_para")
        d = parse_expert_mode_files(joinpath(root, "namelist_$name.def"))
        mode == "complex" && (d.modpara.complex_flag = 1)
        mode == "inactive" && fill!(d.optimization_flags, false)
        mode == "empty_para" && empty!(d.para_qp_opt_trans)
        d.opt_trans .= ComplexF64(7, -9)
        println(io, "$name $mode")
        println(io, join(Int.(d.optimization_flags), " "))
        println(io, P.count_opt_trans_parameters(d))
        rng = SFMT19937RNG(); Random.seed!(rng, 11272)
        P.init_parameter!(d; rng)
        emit_state(io, d)
        vals = vcat(P.projection_parameters(d), [t.value for field in (:charge_rbm_phys_layer_terms, :spin_rbm_phys_layer_terms, :general_rbm_phys_layer_terms, :charge_rbm_hidden_layer_terms, :spin_rbm_hidden_layer_terms, :general_rbm_hidden_layer_terms, :charge_rbm_phys_hidden_terms, :spin_rbm_phys_hidden_terms, :general_rbm_phys_hidden_terms) for t in getfield(d, field)], [t.value for t in d.orbital_terms])
        println(io, hex(vals))
        println(io, join([rand(rng, UInt32) for _ in 1:624], " "))
    end
    verify("initial.txt", String(take!(io)))
    io = IOBuffer()
    println(io, "# Julia $VERSION; strict atomic complex InOptTrans overlays")
    for name in ("valid", "hex", "underflow", "overflow", "duplicate", "missing", "extra", "nonfinite", "count", "absent", "inactive")
        d = parse_expert_mode_files(joinpath(root, "namelist_valid.def"))
        name == "inactive" && empty!(d.opt_trans)
        path = joinpath(root, name in ("valid", "inactive") ? "overlay.def" : "overlay_$name.def")
        err = failure(() -> P.read_input_parameters!(d, joinpath(root, "namelist_overlay_$name.def")), path)
        println(io, "$name $(Int(isempty(err)))")
        println(io, err)
        println(io, hex(d.opt_trans))
        println(io, hex(d.para_qp_opt_trans))
    end
    verify("overlays.txt", String(take!(io)))
end
