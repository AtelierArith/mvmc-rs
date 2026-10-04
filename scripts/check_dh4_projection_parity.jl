include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Original-source exhaustive DH4 counters/updates and gauge synchronization.
using Test, LinearAlgebra, MVMCOptimizers, MVMCExpertModeParsers
using MVMCExpertModeParsers: ExpertModeData, ModParaParameters, GutzwillerTerm,
    JastrowTerm, DoublonHolon4SiteIndex, DoublonHolon2SiteIndex, projection_layout
VERSION == v"1.13.1" || error("DH4 fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
for name in ("test_unit_vmc_sampling_proj.jl", "test_unit_parameter_sync.jl",
             "test_unit_stochastic_opt.jl", "test_unit_read_opt_para.jl")
    include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCOptimizers.jl", "test_unit", name))
end
include(joinpath(@__DIR__, "..", "extern", "Julia-mVMC", "MVMCExpertModeParsers.jl", "test", "test_read_input_parameters.jl"))
const COMBINED = "--combined" in ARGS
const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures", "dh4", COMBINED ? "combined" : "kernels")
mkpath(ROOT)
hex(v) = join(string.(reinterpret(UInt64, collect(reinterpret(Float64, v))); base=16, pad=16), " ")
function verify(name, actual)
    path = joinpath(ROOT, name)
    if "--write" in ARGS
        write(path, actual)
    else
        @test name=="counts.txt" ? compare_record_blocks(actual,read(path,String),3,(2,3)) :
              name=="moves.txt" ? compare_hex_text(actual,read(path,String),(r,c,t)->c==7 ? (1e-13,1e-13) : nothing) :
              compare_record_blocks(actual,read(path,String),2,(2,))
    end
end
@testset "DH4 original structured validation" begin
    indices=[DoublonHolon4SiteIndex(zeros(Int,2,4)),
             DoublonHolon4SiteIndex([-1 3 0 0; 2 typemax(Int) 0 0; 0 0 0 0]),
             DoublonHolon4SiteIndex(zeros(Int,3,4))]
    result=MVMCExpertModeParsers.validate_doublon_holon_4site_indices(indices,3)
    @test !result.is_valid
    @test isempty(result.warnings)
    @test result.errors==[
        "DH4 index 0: neighbors must be 3 x 4",
        "DH4 index 1 site 0 neighbor 0=-1 out of range [0, 2]",
        "DH4 index 1 site 0 neighbor 1=3 out of range [0, 2]",
        "DH4 index 1 site 1 neighbor 1=$(typemax(Int)) out of range [0, 2]",
    ]
end
function model()
    d = ExpertModeData()
    d.modpara = ModParaParameters(nsite=4)
    d.n_gutzwiller_idx = 3
    d.gutzwiller_idx = [0,1,0,1]
    d.gutzwiller_terms = [GutzwillerTerm(i, ComplexF64((i+1)/8, -(i+1)/16), true) for i=0:1]
    d.n_jastrow_idx = 4
    d.jastrow_idx = [-1 0 1 2; 0 -1 2 1; 1 2 -1 0; 2 1 0 -1]
    d.jastrow_terms = [JastrowTerm(0, i+1, ComplexF64((i+1)/16, -(i+1)/32), true) for i=0:2]
    d.doublon_holon_4site_indices = [
        DoublonHolon4SiteIndex([1 2 3 0; 0 2 3 1; 0 1 3 2; 0 1 2 3]),
        DoublonHolon4SiteIndex([1 1 1 1; 1 1 3 3; 3 3 0 0; 2 2 2 2]),
    ]
    d.doublon_holon_4site_params = [ComplexF64(i/10, -i/13) for i=1:20]
    if COMBINED
        d.doublon_holon_2site_indices = [DoublonHolon2SiteIndex([1 2; 0 2; 0 1; 0 1]), DoublonHolon2SiteIndex([0 0; 0 0; 3 3; 2 2])]
        d.doublon_holon_2site_params = [ComplexF64(i/8,-i/16) for i=1:12]
    end
    d.optimization_flags = fill(true, 2projection_layout(d).n_proj)
    d
end
occupancy(mask) = [(mask >> i) & 1 for i=0:7]
@testset "DH4 original-source exhaustive counts and moves" begin
    d = model(); n = projection_layout(d).n_proj
    counts = [zeros(Int,n) for _=0:255]
    io = IOBuffer(); moves = IOBuffer()
    println(io, "# Julia $VERSION; $(BLAS.get_config()); threads=1; sources c2ea432")
    println(moves, "# mask, source site/spin, destination site/spin, resulting mask, log-ratio bits")
    for mask=0:255
        num = occupancy(mask); c = counts[mask+1]
        MVMCOptimizers.make_proj_cnt!(c, num, d)
        println(io, mask, " ", join(c," "))
        println(io, hex([MVMCOptimizers.log_proj_val(c,d)]))
        diff = zeros(ComplexF64, 2(n+1)); MVMCOptimizers.set_projection_diff!(diff,c,n)
        println(io, hex(diff))
    end
    for mask=0:255, s=0:1, t=0:1, ri=0:3, rj=0:3
        from = ri+4s; to = rj+4t
        num = occupancy(mask)
        num[from+1] == 1 && num[to+1] == 0 || continue
        next = (mask & ~(1 << from)) | (1 << to)
        num[from+1] = 0; num[to+1] = 1
        old = counts[mask+1]; expected = counts[next+1]
        inc = copy(old)
        MVMCOptimizers.update_proj_cnt_fsz!(ri,rj,s,t,inc,old,num,d)
        @test inc == expected
        alias = copy(old)
        MVMCOptimizers.update_proj_cnt_fsz!(ri,rj,s,t,alias,alias,num,d)
        @test alias == expected
        if s == t
            normal = copy(old)
            MVMCOptimizers.update_proj_cnt!(ri,rj,s,normal,old,num,d)
            @test normal == expected
        end
        println(moves, join((mask,ri,s,rj,t,next)," "), " ", hex([MVMCOptimizers.log_proj_ratio(expected,old,d)]))
    end
    verify("counts.txt",String(take!(io)))
    verify("moves.txt",String(take!(moves)))
end
@testset "DH4 original-source direct SR write-back" begin
    io=IOBuffer()
    println(io,"# Original stochastic_opt!; diagonal 2, HO 3, dt .25; exact Cholesky result")
    for complex in (false,true), local_index=0:19, imaginary=0:Int(complex)
        d=model(); layout=projection_layout(d); target=layout.dh4_offset+local_index
        d.complex_flags=[Int(complex)]
        d.modpara.dsr_opt_red_cut=0.0; d.modpara.dsr_opt_sta_del=0.0; d.modpara.dsr_opt_step_dt=.25
        d.optimization_flags=falses(2layout.n_proj); d.optimization_flags[2target+imaginary+1]=true
        state=MVMCOptimizers.VMCOptimizationState(4,1,layout.n_proj,layout.n_proj,1,1,complex,false)
        if complex
            component=2target+imaginary+2; dim=2state.sr_opt.sr_opt_size
            state.sr_opt.sr_opt_oo[component*dim+component+1]=2.0
            state.sr_opt.sr_opt_ho[component+1]=3.0
        else
            component=target+1; dim=state.sr_opt.sr_opt_size
            state.sr_opt.sr_opt_oo_real[component*dim+component+1]=2.0
            state.sr_opt.sr_opt_ho_real[component+1]=3.0
        end
        @test MVMCOptimizers.stochastic_opt!(d,state)==0
        println(io,Int(complex)," ",local_index," ",imaginary)
        println(io,hex(MVMCExpertModeParsers.projection_parameters(d)))
    end
    verify("sr-writeback.txt",String(take!(io)))
end
@testset "DH4 original-source gauge flag and compensation boundaries" begin
    io = IOBuffer()
    println(io, "# Original flag_shift_dh4 and sync_modified_parameter!; real shifts preserve imaginary parts")
    for name in ("all", "disabled", "fixed_gutz", "fixed_dh", "empty_flags", "short_flags", "partial_params", "no_gutz", "no_jast", "no_dh", "fixed_dh2", "cancellation")
        d=model()
        name == "fixed_gutz" && (d.optimization_flags[5]=false)
        name == "fixed_dh" && (d.optimization_flags[2(projection_layout(d).dh4_offset+19)+1]=false)
        name == "empty_flags" && empty!(d.optimization_flags)
        name == "short_flags" && resize!(d.optimization_flags,15)
        name == "partial_params" && resize!(d.doublon_holon_4site_params,18)
        if name == "no_gutz"; d.n_gutzwiller_idx=0; empty!(d.gutzwiller_terms); end
        if name == "no_jast"; d.n_jastrow_idx=0; empty!(d.jastrow_terms); end
        if name == "no_dh"; empty!(d.doublon_holon_4site_indices); empty!(d.doublon_holon_4site_params); end
        if name == "fixed_dh2" && COMBINED; d.optimization_flags[2projection_layout(d).dh2_offset+1]=false; end
        if name == "cancellation"
            for (idx,value) in zip((1,5,9,13,17),(1e16,1.0,-1e16,1.0,1.0)); d.doublon_holon_4site_params[idx]=ComplexF64(value,-value/13); end
        end
        enabled=MVMCOptimizers.flag_shift_dh4(d)
        d.orbital_terms=[MVMCExpertModeParsers.OrbitalTerm(0,0,0,2.0+0im,true,1),MVMCExpertModeParsers.OrbitalTerm(0,1,1,0.0+8im,true,1)]
        MVMCOptimizers.sync_modified_parameter!(d;shift_correlations=name!="disabled")
        println(io, name, " ", Int(enabled))
        println(io, hex(vcat([t.value for t in d.gutzwiller_terms], [t.value for t in d.jastrow_terms],d.doublon_holon_2site_params,d.doublon_holon_4site_params,[t.value for t in d.orbital_terms])))
    end
    verify("gauge.txt",String(take!(io)))
end
