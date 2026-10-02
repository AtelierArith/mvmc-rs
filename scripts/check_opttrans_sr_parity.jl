# Canonical indexed SR updates and optimizer-only OptTrans normalization.
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
    d = P.parse_expert_mode_files(joinpath(root, "namelist_$base.def"))
    name == "empty_opt" && empty!(d.opt_trans)
    name == "short_opt" && resize!(d.opt_trans, 1)
    name == "long_opt" && push!(d.opt_trans, 0.5 - 0.25im)
    for i in 1:P.count_variational_parameters(d)
        O.update_parameter_value(d, i, i/64, -i/128)
    end
    d.modpara.nmp_trans = 2
    d.para_qp_trans = [1.0+0.25im, -0.5-0.125im]
    P.init_qp_weight!(d)
    return d
end
values(d) = vcat(P.projection_parameters(d), [t.value for terms in O._rbm_parameter_sections(d) for t in terms], [t.value for t in d.orbital_terms], d.opt_trans)
@testset "Canonical OptTrans SR and synchronization" begin
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test_unit", "test_unit_stochastic_opt.jl"))
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test_unit", "test_unit_parameter_sync.jl"))
    io = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; source 8bb1b9e / c2ea432; exact indexed OptTrans SR updates")
    for name in ("valid", "layout", "qp_only", "empty_opt", "short_opt", "long_opt"), complex in (false, true), solver in ("direct", "cg"), flags in ("real", "imag", "both", "fixed")
        d = model(name)
        d.modpara.complex_flag = Int(complex)
        d.complex_flags = [Int(complex)]
        for t in d.orbital_terms; t.is_complex = complex; end
        for t in d.gutzwiller_terms; t.is_complex = complex; end
        for t in d.jastrow_terms; t.is_complex = complex; end
        d.doublon_holon_2site_complex = complex
        d.doublon_holon_4site_complex = complex
        d.modpara.dsr_opt_sta_del = 0.0
        d.modpara.dsr_opt_step_dt = 0.125
        d.modpara.dsr_opt_red_cut = 1e-8
        d.modpara.nvmc_sample = 8
        n = P.count_variational_parameters(d)
        no = length(d.opt_trans)
        d.optimization_flags = falses(2*n)
        for i in n-no:n-1
            d.optimization_flags[2*i+1] = flags in ("real", "both")
            d.optimization_flags[2*i+2] = flags in ("imag", "both")
        end
        state = O.VMCOptimizationState(d.modpara.nsite, max(1,d.modpara.nelec), P.projection_layout(d).n_proj, n, 1, 8, complex, false)
        state.energy.wc = 8.0 + 0.0im
        off = complex ? 2 : 1
        size = off * (n+1)
        oo = complex ? state.sr_opt.sr_opt_oo : state.sr_opt.sr_opt_oo_real
        ho = complex ? state.sr_opt.sr_opt_ho : state.sr_opt.sr_opt_ho_real
        store = complex ? state.sr_opt.sr_opt_o_store : state.sr_opt.sr_opt_o_store_real
        for p in 0:off*no-1
            idx = off*(n-no+1)+p
            oo[solver == "cg" ? size+idx+1 : idx*size+idx+1] = 1.0
            ho[idx+1] = (p+1)/16
            for s in 0:7
                store[s*size+idx+1] = isodd(count_ones(s & (p+1))) ? -1.0 : 1.0
            end
        end
        info = solver == "cg" ? O.stochastic_opt_cg!(d,state) : O.stochastic_opt!(d,state)
        println(io, "$name $(Int(complex)) $solver $flags $n $info")
        println(io, hex(values(d)))
        println(io, hex(d.qp_weights.qp_full_weight))
        println(io, hex(d.para_qp_opt_trans))
    end
    verify("sr.txt", String(take!(io)))
    io = IOBuffer()
    println(io, "# Julia $VERSION; optimizer sync versus parser sync; exact bits and weights")
    vectors = (ComplexF64[], [0.0+0.0im,-0.0-0.0im], [3.0+4.0im,-6.0+8.0im], [0.125-0.75im,1.5-0.5im], [1e-300+2e-300im,-3e-300+4e-300im], [1e300+2e300im,-3e300+4e300im])
    for name in ("valid", "layout", "qp_only"), (case, vector) in enumerate(vectors), shift in (false,true), path in ("parser","optimizer")
        d = model(name)
        d.opt_trans = copy(vector)
        P.update_qp_weight!(d.qp_weights,d.opt_trans)
        path == "parser" ? P.sync_modified_parameter!(d) : O.sync_modified_parameter!(d; shift_correlations=shift)
        println(io, "$name $case $(Int(shift)) $path")
        println(io, hex(values(d)))
        # Local normalization does not refresh weights; the next loop refreshes.
        println(io, hex(d.qp_weights.qp_full_weight))
        println(io, hex(d.para_qp_opt_trans))
        O.update_qp_weight!(d)
        println(io, hex(d.qp_weights.qp_full_weight))
    end
    verify("sync.txt", String(take!(io)))
end
