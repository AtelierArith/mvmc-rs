include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Canonical declared orbital layout, boundary caches, initialization and loading.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("Orbital contract fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const P=MVMCExpertModeParsers
const O=MVMCOptimizers
const root=joinpath(@__DIR__,"..","tests","fixtures","orbital_general")
definition(name,count,complex,rows)="===\n$name $count\nComplexType $complex\n===\n===\n$rows"
hex(values)=join([string(reinterpret(UInt64,x);base=16,pad=16) for z in values for x in (real(z),imag(z))]," ")
function matrix(io,m)
    m === nothing && return println(io,"0")
    println(io,size(m,1)," ",join(vec(permutedims(m))," "))
end
@testset "Canonical orbital layout and boundary contracts" begin
    include(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCExpertModeParsers.jl","test","test_orbital_qptrans_utils.jl"))
    io=IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; source 8bb1b9e / c2ea432; seed=1")
    for case in ("ap_only","ap_sparse","ap_empty","p_empty","both_empty","ap_zero","p_only","general_sparse","general_empty","order_error"), complex in (0,1), boundary in (-2,2)
        mktempdir() do dir
            write(joinpath(dir,"modpara.def"),"Nsite 3\nNElec 1\nNMPTrans $boundary\nNSPGaussLeg 1\n")
            ap_empty=case in ("ap_empty","both_empty","ap_zero")
            write(joinpath(dir,"ap.def"),definition("NOrbitalIdx",case=="ap_zero" ? 0 : 7,complex,ap_empty ? "" : "0 1 0 -1\n1 0 1 1\n0 0\n1 1\n"))
            write(joinpath(dir,"p.def"),definition("NOrbitalParallel",3,complex,case in ("p_empty","both_empty") ? "" : "0 1 0 -1\n0 1\n1 0\n2 1\n"))
            write(joinpath(dir,"general.def"),definition("NOrbitalGeneral",17,complex,case=="general_empty" ? "" : "0 4 1 -1\n2 3 4 1\n1 0\n4 1\n"))
            qp="===\nNQPTrans 2\n===\n===\n===\n0 1.0\n1 -0.375\n"
            for mp in 0:1, site in 0:2
                qp*="$mp $site $(mod(site+mp,3)) $(isodd(site+mp) ? -1 : 1)\n"
            end
            write(joinpath(dir,"qp.def"),qp)
            orbitals=case=="order_error" ? "OrbitalParallel p.def\nOrbitalAntiParallel ap.def\n" : startswith(case,"general") ? "OrbitalGeneral general.def\n" : case=="p_only" ? "OrbitalParallel p.def\n" : "OrbitalAntiParallel ap.def\n"*(case=="ap_only" ? "" : "OrbitalParallel p.def\n")
            file=joinpath(dir,"namelist.def")
            write(file,"ModPara modpara.def\n"*orbitals*"QPTrans qp.def\n")
            println(io,"$case $complex $boundary")
            d=try P.parse_expert_mode_files(file) catch e
                @test case=="order_error"
                println(io,"error ",sprint(showerror,e))
                return
            end
            @test case != "order_error"
            println(io,"ok $(d.modpara.n_orbital_idx) $(d.n_orbital_anti_parallel) $(d.i_flg_orbital_anti_parallel) $(d.i_flg_orbital_parallel) $(d.i_flg_orbital_general)")
            println(io,join([x for t in d.orbital_terms for x in (t.site1,t.site2,t.idx,t.sign,Int(t.is_complex))]," "))
            println(io,join(Int.(d.optimization_flags)," "))
            matrix(io,d.orbital_idx_matrix);matrix(io,d.orbital_sgn)
            println(io,join([s for row in d.qp_trans_sgn for s in row]," "))
            rng=SFMT19937RNG();Random.seed!(rng,1)
            P.init_parameter!(d;rng)
            println(io,hex([t.value for t in d.orbital_terms]))
            println(io,join([rand(rng,UInt32) for _ in 1:624]," "))
            n=P.count_variational_parameters(d)
            record="1 2 3 4 5 6 "*join(["$((i+1)/32) $(-i/64) 99" for i in 0:n-1]," ")
            write(joinpath(dir,"initial.def"),record)
            @test O.read_opt_para_file!(d,joinpath(dir,"initial.def"))==n
            println(io,hex([t.value for t in d.orbital_terms]))
        end
    end
    actual=String(take!(io));path=joinpath(root,"contracts.txt")
    "--write" in ARGS ? write(path,actual) : @test(compare_initialization_text(actual,read(path,String)))
end
