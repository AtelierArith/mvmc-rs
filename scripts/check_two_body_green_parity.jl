# General fixed-Sz four-fermion kernels, using the original Julia source and
# independent occupation-basis operator application with analytic 4x4 Pfaffians.
using Test, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("Two-body fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"dh2_green_model.jl"))
include(joinpath(@__DIR__,"dh4_green_model.jl"))
const DH2 = "--dh2" in ARGS
const DH4 = "--dh4" in ARGS
const COMBINED = "--dh24" in ARGS
const DH = DH2 || DH4 || COMBINED
hex(v) = join(string.(reinterpret(UInt64, collect(reinterpret(Float64, v))); base=16, pad=16), " ")

function occupation(idx)
    num = zeros(Int,8); cfg = fill(-1,8)
    for m in 0:3
        rs = idx[m+1] + (m÷2)*4
        num[rs+1] = 1; cfg[rs+1] = m%2
    end
    num, cfg
end

function analytic_overlap(num, slater)
    rs = findall(==(1), num) .- 1
    @assert length(rs) == 4
    total = 0.0 + 0.0im
    for (qp,w) in ((0,1.0),(1,-0.375))
        a(i,j) = slater[qp*64+rs[i]*8+rs[j]+1]
        total += w * (a(1,2)*a(3,4) - a(1,3)*a(2,4) + a(1,4)*a(2,3))
    end
    total
end

function explicit_operator(ri,rj,rk,rl,s,t,num,cnt,data,state)
    moved = copy(num); sign = 1
    # Apply rightmost operator first in the canonical sorted Fock basis.
    for (rs,create) in ((rl+4t,false),(rk+4t,true),(rj+4s,false),(ri+4s,true))
        moved[rs+1] == Int(create) && return 0.0 + 0.0im
        isodd(sum(moved[1:rs])) && (sign = -sign)
        moved[rs+1] = Int(create)
    end
    new_cnt = zeros(Int,length(cnt))
    MVMCOptimizers.make_proj_cnt!(new_cnt,moved,data)
    ratio = if DH
        z=0.0
        for (i,value) in enumerate(MVMCExpertModeParsers.projection_parameters(data))
            z += real(value)*(new_cnt[i]-cnt[i])
        end
        exp(z)
    else
        exp(0.125*(new_cnt[1]-cnt[1]) - 0.2*(new_cnt[2]-cnt[2]))
    end
    conj(sign*ratio*analytic_overlap(moved,state.slater_matrix.slater_elm) /
         analytic_overlap(num,state.slater_matrix.slater_elm))
end

io = IOBuffer()
println(io,"# Julia ",VERSION,"; ",BLAS.get_config(),"; threads=1")
println(io,"# Original green_func2; exhaustive 4-site indices and spins; analytic Fock check atol=2e-12 rtol=0")
@testset "General fixed-Sz two-body Green ratios" begin
    for complex in (false,true), idx in ([0,2,1,3],[0,3,0,3])
        data = MVMCExpertModeParsers.ExpertModeData()
        data.modpara.nsite = 4; data.modpara.nelec = 2; data.modpara.nmp_trans = 2
        data.complex_flags = [Int(complex)]
        data.n_qp_trans = 2; data.para_qp_trans = ComplexF64[1,-0.375]
        data.n_gutzwiller_idx = 1; data.gutzwiller_idx = zeros(Int,4)
        data.gutzwiller_terms = [MVMCExpertModeParsers.GutzwillerTerm(0,0.125+0im,false)]
        data.n_jastrow_idx = 1; data.jastrow_idx = [i==j ? -1 : 0 for i in 1:4,j in 1:4]
        data.jastrow_terms = [MVMCExpertModeParsers.JastrowTerm(0,1,-0.2+0im,false)]
        (DH2 || COMBINED) && add_dh2_green_model!(data)
        (DH4 || COMBINED) && add_dh4_green_model!(data)
        MVMCExpertModeParsers.init_qp_weight!(data)
        state = MVMCOptimizers.VMCOptimizationState(4,2,MVMCExpertModeParsers.projection_layout(data).n_proj,0,2,1,complex,false)
        mat = state.slater_matrix
        for qp in 0:1, i in 0:7, j in (i+1):7
            # Normal orbitals couple up/down blocks only.
            z = i÷4 == j÷4 ? 0.0+0.0im :
                ComplexF64(((17i+13j+7qp)%31-15)/7+0.125, complex ? ((11i+3j+qp)%19-9)/13 : 0.0)
            mat.slater_elm[qp*64+i*8+j+1] = z
            mat.slater_elm[qp*64+j*8+i+1] = -z
        end
        num,cfg = occupation(idx); cnt = zeros(Int,MVMCExpertModeParsers.projection_layout(data).n_proj)
        MVMCOptimizers.make_proj_cnt!(cnt,num,data)
        if complex
            @test MVMCOptimizers.calculate_m_all_fcmp!(idx,1,3,data,state) == 0
            ip = MVMCOptimizers.calculate_ip_fcmp(mat.pf_m,1,3,data; reduce=:none)
            pf = mat.pf_m; inv = mat.inv_m[1:32]
        else
            mat.slater_elm_real .= real.(mat.slater_elm)
            @test MVMCOptimizers.calculate_m_all_real!(idx,1,3,data,state) == 0
            ip = ComplexF64(MVMCOptimizers.calculate_ip_real(mat.pf_m_real,1,3,data; reduce=:none))
            pf = ComplexF64.(mat.pf_m_real); inv = ComplexF64.(mat.inv_m_real[1:32])
        end
        @test isapprox(ip,analytic_overlap(num,mat.slater_elm); atol=2e-14,rtol=0)
        println(io,Int(complex)); println(io,join(idx," "))
        println(io,join(cfg," ")); println(io,join(num," ")); println(io,join(cnt," "))
        println(io,hex(mat.slater_elm)); println(io,hex(pf)); println(io,hex(inv)); println(io,hex(ComplexF64[ip]))
        data.coulomb_intra_terms = [MVMCExpertModeParsers.CoulombIntraTerm(0,0.7)]
        data.coulomb_inter_terms = [MVMCExpertModeParsers.CoulombInterTerm(0,3,-0.3)]
        data.hund_terms = [MVMCExpertModeParsers.HundTerm(0,3,0.125)]
        data.exchange_terms = [MVMCExpertModeParsers.ExchangeTerm(0,1,0.2),
                               MVMCExpertModeParsers.ExchangeTerm(0,0,0.15)]
        energy = MVMCOptimizers.calculate_hamiltonian(ip,idx,cfg,num,cnt,data,state; all_complex=complex)
        g(i,j,k,l,s,t) = MVMCOptimizers.green_func2(i,j,k,l,s,t,ip,idx,cfg,num,cnt,data,state; all_complex=complex)
        equivalent = 0.7*g(0,0,0,0,0,1)
        for s in 0:1, t in 0:1; equivalent -= 0.3*g(0,0,3,3,s,t); end
        equivalent -= 0.125*(g(0,0,3,3,0,0)+g(0,0,3,3,1,1))
        equivalent += 0.2*(g(0,1,1,0,0,1)+g(0,1,1,0,1,0))
        equivalent += 0.15*(g(0,0,0,0,0,1)+g(0,0,0,0,1,0))
        @test isapprox(energy,equivalent; atol=2e-14,rtol=0)
        println(io,hex(ComplexF64[energy]))
        for s in 0:1, t in 0:1, ri in 0:3, rj in 0:3, rk in 0:3, rl in 0:3
            g = MVMCOptimizers.green_func2(ri,rj,rk,rl,s,t,ip,idx,cfg,num,cnt,data,state; all_complex=complex)
            expected = explicit_operator(ri,rj,rk,rl,s,t,num,cnt,data,state)
            @test isapprox(g,expected; atol=2e-12,rtol=0)
            println(io,ri," ",rj," ",rk," ",rl," ",s," ",t," ",hex(ComplexF64[g]))
        end
    end
    actual = String(take!(io))
    path = joinpath(@__DIR__,"..","tests","fixtures",COMBINED ? "dh4/combined" : DH4 ? "dh4/kernels" : DH2 ? "dh2" : "interall","green_normal.txt")
    if "--write" in ARGS; write(path,actual); else; @test actual == read(path,String); end
end
