# Original FSZ Green kernels versus independent Fock-basis operators.
using Test, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("FSZ Green fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"dh2_green_model.jl"))
const DH2 = "--dh2" in ARGS
hex(v) = join(string.(reinterpret(UInt64, collect(reinterpret(Float64, v))); base=16, pad=16), " ")

function overlap(num, slater)
    rs = findall(==(1),num) .- 1
    @assert length(rs) == 4
    total = 0.0 + 0.0im
    for (qp,w) in ((0,1.0),(1,-0.375))
        a(i,j) = slater[qp*64+rs[i]*8+rs[j]+1]
        total += w*(a(1,2)*a(3,4)-a(1,3)*a(2,4)+a(1,4)*a(2,3))
    end
    total
end

function explicit_operator(operators,num,cnt,data,state)
    moved = copy(num); sign = 1
    for (rs,create) in operators
        moved[rs+1] == Int(create) && return 0.0 + 0.0im
        isodd(sum(moved[1:rs])) && (sign = -sign)
        moved[rs+1] = Int(create)
    end
    new_cnt = zeros(Int,length(cnt))
    MVMCOptimizers.make_proj_cnt!(new_cnt,moved,data)
    ratio = if DH2
        z=0.0
        for (i,value) in enumerate(MVMCExpertModeParsers.projection_parameters(data))
            z += real(value)*(new_cnt[i]-cnt[i])
        end
        exp(z)
    else
        exp(0.125*(new_cnt[1]-cnt[1])-0.2*(new_cnt[2]-cnt[2]))
    end
    conj(sign*ratio*overlap(moved,state.slater_matrix.slater_elm)/overlap(num,state.slater_matrix.slater_elm))
end

io = IOBuffer()
println(io,"# Julia ",VERSION,"; ",BLAS.get_config(),"; threads=1")
println(io,"# Original green_func1_fsz/2 and green_func2_fsz/2; independent Fock check atol=2e-12 rtol=0")
@testset "General FSZ Green operators" begin
    for complex in (false,true), (idx,spins) in (([0,2,1,3],[0,0,1,1]),
                                               ([0,1,2,3],[0,0,0,1]),
                                               ([3,0,2,0],[1,0,0,1]))
        data = MVMCExpertModeParsers.ExpertModeData()
        data.modpara.nsite = 4; data.modpara.nelec = 2; data.modpara.nmp_trans = 2
        data.i_flg_orbital_general = 1; data.complex_flags = [Int(complex)]
        data.n_qp_trans = 2; data.para_qp_trans = ComplexF64[1,-0.375]
        data.n_gutzwiller_idx = 1; data.gutzwiller_idx = zeros(Int,4)
        data.gutzwiller_terms = [MVMCExpertModeParsers.GutzwillerTerm(0,0.125+0im,false)]
        data.n_jastrow_idx = 1; data.jastrow_idx = [i==j ? -1 : 0 for i in 1:4,j in 1:4]
        data.jastrow_terms = [MVMCExpertModeParsers.JastrowTerm(0,1,-0.2+0im,false)]
        DH2 && add_dh2_green_model!(data)
        MVMCExpertModeParsers.init_qp_weight!(data)
        state = MVMCOptimizers.VMCOptimizationState(4,2,MVMCExpertModeParsers.projection_layout(data).n_proj,0,2,1,complex,true)
        mat = state.slater_matrix
        for qp in 0:1, i in 0:7, j in (i+1):7
            z = ComplexF64(((17i+13j+7qp)%31-15)/7+0.125,
                          complex ? ((11i+3j+qp)%19-9)/13 : 0.0)
            mat.slater_elm[qp*64+i*8+j+1] = z
            mat.slater_elm[qp*64+j*8+i+1] = -z
        end
        num = zeros(Int,8); cfg = fill(-1,8)
        for m in 0:3
            rs = idx[m+1]+spins[m+1]*4
            num[rs+1] = 1; cfg[rs+1] = m
        end
        cnt = zeros(Int,MVMCExpertModeParsers.projection_layout(data).n_proj); MVMCOptimizers.make_proj_cnt!(cnt,num,data)
        @test MVMCOptimizers.calculate_m_all_fsz!(idx,spins,1,3,data,state) == 0
        # Both authoritative FSZ Green families read complex inverse/Pfaffian
        # buffers, even for a real Slater table. Serialize those original inputs.
        ip = MVMCOptimizers.calculate_ip_fcmp(mat.pf_m,1,3,data; reduce=:none)
        println(io,Int(complex)); println(io,join(idx," ")); println(io,join(spins," "))
        println(io,join(cfg," ")); println(io,join(num," ")); println(io,join(cnt," "))
        println(io,hex(mat.slater_elm)); println(io,hex(mat.pf_m)); println(io,hex(mat.inv_m[1:32])); println(io,hex(ComplexF64[ip]))
        one = ComplexF64[]; two = ComplexF64[]
        for s in 0:1,t in 0:1,ri in 0:3,rj in 0:3
            g = MVMCOptimizers.green_func1_fsz2(ri,rj,s,t,ip,idx,cfg,num,cnt,spins,data,state; all_complex=complex)
            expected = explicit_operator(((rj+4t,false),(ri+4s,true)),num,cnt,data,state)
            @test isapprox(g,expected; atol=2e-12,rtol=0)
            push!(one,g)
        end
        for s in 0:1,t in 0:1,u in 0:1,v in 0:1,ri in 0:3,rj in 0:3,rk in 0:3,rl in 0:3
            g = if s==t && u==v
                MVMCOptimizers.green_func2_fsz(ri,rj,rk,rl,s,u,ip,idx,cfg,num,cnt,spins,data,state; all_complex=complex)
            else
                MVMCOptimizers.green_func2_fsz2(ri,rj,rk,rl,s,t,u,v,ip,idx,cfg,num,cnt,spins,data,state; all_complex=complex)
            end
            expected = explicit_operator(((rl+4v,false),(rk+4u,true),(rj+4t,false),(ri+4s,true)),num,cnt,data,state)
            @test isapprox(g,expected; atol=2e-12,rtol=0)
            push!(two,g)
        end
        println(io,hex(one)); println(io,hex(two))
        data.coulomb_intra_terms = [MVMCExpertModeParsers.CoulombIntraTerm(0,0.7)]
        data.coulomb_inter_terms = [MVMCExpertModeParsers.CoulombInterTerm(0,3,-0.3)]
        data.hund_terms = [MVMCExpertModeParsers.HundTerm(0,3,0.125)]
        data.exchange_terms = [MVMCExpertModeParsers.ExchangeTerm(0,1,0.2),
                               MVMCExpertModeParsers.ExchangeTerm(0,0,0.15)]
        base = MVMCOptimizers.calculate_hamiltonian_fsz(ip,idx,cfg,num,cnt,spins,data,state; all_complex=complex)
        g(i,j,k,l,s,t,u,v) = two[((((s*2+t)*2+u)*2+v)*256)+((i*4+j)*4+k)*4+l+1]
        equivalent = 0.7*g(0,0,0,0,0,0,1,1)
        for s in 0:1,t in 0:1; equivalent -= 0.3*g(0,0,3,3,s,s,t,t); end
        equivalent -= 0.125*(g(0,0,3,3,0,0,0,0)+g(0,0,3,3,1,1,1,1))
        equivalent += 0.2*(g(0,1,1,0,0,0,1,1)+g(0,1,1,0,1,1,0,0))
        equivalent += 0.15*(g(0,0,0,0,0,0,1,1)+g(0,0,0,0,1,1,0,0))
        @test isapprox(base,equivalent; atol=2e-14,rtol=0)
        data.inter_all_terms = MVMCExpertModeParsers.parse_interall_content("""
        0 0 0 0 3 1 3 1 -0.5 0.125
        0 0 0 1 3 1 3 0 -0.375 0.1875
        0 0 0 1 2 0 2 1 0.125 -0.25
        1 1 2 0 2 0 0 1 -0.25 -0.375
        1 1 0 0 2 0 3 1 0.5 0.125
        0 0 0 1 3 1 3 0 -0.375 0.1875
        -1 0 0 1 3 1 3 0 0.25 0.125
        """).data
        full = MVMCOptimizers.calculate_hamiltonian_fsz(ip,idx,cfg,num,cnt,spins,data,state; all_complex=complex)
        expected_full = equivalent
        for term in data.inter_all_terms
            min(term.site0,term.site1,term.site2,term.site3) < 0 && continue
            expected_full += term.value*g(term.site0,term.site1,term.site2,term.site3,term.spin0,term.spin1,term.spin2,term.spin3)
        end
        @test isapprox(full,expected_full; atol=2e-14,rtol=0)
        println(io,hex(ComplexF64[base,full]))
    end
    actual = String(take!(io)); path = joinpath(@__DIR__,"..","tests","fixtures",DH2 ? "dh2" : "interall","green_fsz.txt")
    if "--write" in ARGS; write(path,actual); else; @test actual == read(path,String); end
end
