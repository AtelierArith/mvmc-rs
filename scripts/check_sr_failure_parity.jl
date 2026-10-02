using Test, MVMCExpertModeParsers, MVMCOptimizers, LinearAlgebra
using MVMCExpertModeParsers: ExpertModeData, OrbitalTerm
VERSION == v"1.13.1" || error("SR failure checks require Julia 1.13.1")
BLAS.set_num_threads(1)
function problem(complex)
    data = ExpertModeData()
    data.modpara.n_orbital_idx = 2
    data.modpara.dsr_opt_sta_del = 0.0
    data.optimization_flags = [true,false,true,false]
    data.complex_flags = [Int(complex)]
    data.orbital_terms = [OrbitalTerm(0,0,i,ComplexF64(2+i),complex,1) for i in 0:1]
    state = MVMCOptimizers.VMCOptimizationState(1,1,0,2,1,1,complex,false)
    if complex
        state.sr_opt.sr_opt_oo[15] = 1.0
        state.sr_opt.sr_opt_oo[29] = 1.0
        state.sr_opt.sr_opt_ho[3] = 1.0
        state.sr_opt.sr_opt_ho[5] = 1.0
    else
        state.sr_opt.sr_opt_oo_real[5] = 1.0
        state.sr_opt.sr_opt_oo_real[9] = 1.0
        state.sr_opt.sr_opt_ho_real[2] = 1.0
        state.sr_opt.sr_opt_ho_real[3] = 1.0
    end
    data,state
end

@testset "Direct SR canonical positive potrf status handling" begin
    io = IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; positive potrf INFO is ignored by canonical stochastic_opt!")
    for complex in (false,true), kind in ("indefinite","singular")
        data,state = problem(complex)
        off = complex ? 2 : 1
        size = off*3
        oo = complex ? state.sr_opt.sr_opt_oo : state.sr_opt.sr_opt_oo_real
        covariance = kind == "indefinite" ? 2.0 : 1.0
        oo[off*size+2off+1] = covariance
        oo[2off*size+off+1] = covariance
        info = MVMCOptimizers.stochastic_opt!(data,state)
        @test info == (kind == "indefinite" ? 0 : 1)
        println(io,"$(Int(complex)) $kind $info")
        println(io,join([string(reinterpret(UInt64,x);base=16,pad=16) for t in data.orbital_terms for x in (real(t.value),imag(t.value))]," "))
    end
    path = joinpath(@__DIR__,"..","tests","fixtures","sr_failure","potrf_status.txt")
    actual = String(take!(io))
    if "--write" in ARGS
        mkpath(dirname(path)); write(path,actual)
    else
        @test actual == read(path,String)
    end
end
@testset "Direct SR finite check and missing component flags" begin
    for complex in (false,true), kind in (:variance,:gradient,:flags)
        data,state = problem(complex)
        if kind == :variance
            if complex; state.sr_opt.sr_opt_oo[15] = NaN
            else; state.sr_opt.sr_opt_oo_real[5] = NaN; end
        else
            if complex; state.sr_opt.sr_opt_ho[5] = NaN
            else; state.sr_opt.sr_opt_ho_real[3] = NaN; end
            kind == :flags && (data.optimization_flags = [true,false])
        end
        before = [t.value for t in data.orbital_terms]
        info = MVMCOptimizers.stochastic_opt!(data,state)
        if kind == :flags
            @test info == 0
            @test data.orbital_terms[2].value == before[2]
            @test data.orbital_terms[1].value == 1.98+0im
        else
            @test info == 1
            @test [t.value for t in data.orbital_terms] == before
        end
    end
end
