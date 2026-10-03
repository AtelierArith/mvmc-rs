include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Reuse the original-source Green input states, including supplied inverse bits.
# Check PairHop accumulation and its explicit InterAll operator representation.
using Test, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("PairHop fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
root = joinpath(@__DIR__,"..","tests","fixtures")
hex(v) = join(string.(reinterpret(UInt64,collect(reinterpret(Float64,v))); base=16,pad=16)," ")
integers(line) = parse.(Int,split(line))
complex_bits(line) = collect(reinterpret(ComplexF64,reinterpret(Float64,parse.(UInt64,split(line);base=16))))
function green_data(complex,fsz)
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = 4; data.modpara.nelec = 2; data.modpara.nmp_trans = 2
    data.i_flg_orbital_general = Int(fsz); data.complex_flags = [Int(complex)]
    data.n_qp_trans = 2; data.para_qp_trans = ComplexF64[1,-0.375]
    data.n_gutzwiller_idx = 1; data.gutzwiller_idx = zeros(Int,4)
    data.gutzwiller_terms = [MVMCExpertModeParsers.GutzwillerTerm(0,0.125+0im,false)]
    data.n_jastrow_idx = 1; data.jastrow_idx = [i==j ? -1 : 0 for i in 1:4,j in 1:4]
    data.jastrow_terms = [MVMCExpertModeParsers.JastrowTerm(0,1,-0.2+0im,false)]
    MVMCExpertModeParsers.init_qp_weight!(data)
    data
end
function analytic_overlap(num,slater)
    rs = findall(==(1),num) .- 1
    @assert length(rs) == 4
    total = 0.0+0.0im
    for (qp,w) in ((0,1.0),(1,-0.375))
        a(i,j) = slater[qp*64+rs[i]*8+rs[j]+1]
        total += w*(a(1,2)*a(3,4)-a(1,3)*a(2,4)+a(1,4)*a(2,3))
    end
    total
end
function explicit_pair(ri,rj,num,cnt,data,state)
    moved = copy(num); sign = 1
    for (rs,create) in ((rj+4,false),(ri+4,true),(rj,false),(ri,true))
        moved[rs+1] == Int(create) && return 0.0+0.0im
        isodd(sum(moved[1:rs])) && (sign = -sign)
        moved[rs+1] = Int(create)
    end
    new_cnt = zeros(Int,2); MVMCOptimizers.make_proj_cnt!(new_cnt,moved,data)
    ratio = exp(0.125*(new_cnt[1]-cnt[1])-0.2*(new_cnt[2]-cnt[2]))
    conj(sign*ratio*analytic_overlap(moved,state.slater_matrix.slater_elm)/
         analytic_overlap(num,state.slater_matrix.slater_elm))
end
@testset "PairHop normal/FSZ original energy and InterAll equivalence" begin
    for fsz in (false,true)
        input = filter(l -> !startswith(l,"#"),readlines(joinpath(root,"interall",fsz ? "green_fsz.txt" : "green_normal.txt")))
        cursor = 1
        next() = (line=input[cursor]; cursor+=1; line)
        io = IOBuffer()
        println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1")
        println(io,"# Original Hamiltonian; pure PairHop, combined, equivalent InterAll; Fock atol=2e-12 rtol=0")
        for case in 1:(fsz ? 6 : 4)
            complex = next() == "1"
            idx = integers(next()); spins = fsz ? integers(next()) : [0,0,1,1]
            cfg = integers(next()); num = integers(next()); cnt = integers(next())
            data = green_data(complex,fsz)
            state = MVMCOptimizers.VMCOptimizationState(4,2,2,0,2,1,complex,fsz)
            mat = state.slater_matrix
            mat.slater_elm .= complex_bits(next()); mat.pf_m .= complex_bits(next())
            mat.inv_m[1:32] .= complex_bits(next()); ip = only(complex_bits(next()))
            if !complex && !fsz
                mat.slater_elm_real .= real.(mat.slater_elm)
                mat.pf_m_real .= real.(mat.pf_m); mat.inv_m_real[1:32] .= real.(mat.inv_m[1:32])
            end
            if fsz
                next(); next(); next() # one-body, two-body, Hamiltonian outputs
            else
                next() # Hamiltonian
                for _ in 1:1024; next(); end
            end
            original = MVMCExpertModeParsers.parse_pairhop_def(joinpath(root,"pairhop","hamiltonian.def"))
            @test original.success
            data.pair_hop_terms = copy(original.data)
            push!(data.pair_hop_terms,MVMCExpertModeParsers.PairHopTerm(-1,0,0.125))
            energy() = fsz ? MVMCOptimizers.calculate_hamiltonian_fsz(ip,idx,cfg,num,cnt,spins,data,state;all_complex=complex) :
                            MVMCOptimizers.calculate_hamiltonian(ip,idx,cfg,num,cnt,data,state;all_complex=complex)
            pure = energy()
            explicit = 0.0+0.0im
            interall = MVMCExpertModeParsers.InterAllTerm[]
            for t in data.pair_hop_terms
                if 0 <= t.site1 < 4 && 0 <= t.site2 < 4
                    explicit += t.value*explicit_pair(t.site1,t.site2,num,cnt,data,state)
                    push!(interall,MVMCExpertModeParsers.InterAllTerm(t.site1,0,t.site2,0,t.site1,1,t.site2,1,ComplexF64(t.value),false))
                end
            end
            @test isapprox(pure,explicit;atol=2e-12,rtol=0)
            data.coulomb_intra_terms = [MVMCExpertModeParsers.CoulombIntraTerm(0,0.7)]
            data.coulomb_inter_terms = [MVMCExpertModeParsers.CoulombInterTerm(0,3,-0.3)]
            data.hund_terms = [MVMCExpertModeParsers.HundTerm(0,3,0.125)]
            data.exchange_terms = [MVMCExpertModeParsers.ExchangeTerm(0,1,0.2),MVMCExpertModeParsers.ExchangeTerm(0,0,0.15)]
            combined = energy()
            empty!(data.pair_hop_terms)
            if fsz
                data.inter_all_terms = interall
                equivalent = energy()
            else
                # The canonical normal accumulator's term.sites defect is kept
                # visible; evaluate this parsed operator representation via G2.
                equivalent = energy()
                for t in interall
                    equivalent += t.value*MVMCOptimizers.green_func2(t.site0,t.site1,t.site2,t.site3,t.spin0,t.spin2,ip,idx,cfg,num,cnt,data,state;all_complex=complex)
                end
            end
            @test isapprox(combined,equivalent;atol=2e-14,rtol=0)
            println(io,hex(ComplexF64[pure,combined,equivalent]))
        end
        @test cursor == length(input)+1
        path = joinpath(root,"pairhop",fsz ? "energy_fsz.txt" : "energy_normal.txt")
        actual = String(take!(io))
        if "--write" in ARGS; write(path,actual); else; @test compare_hex_text(actual,read(path,String),(r,c,t)->(1e-12,1e-12)); end
    end
end
