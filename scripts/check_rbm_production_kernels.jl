include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
# Unmodified canonical RBM loaders, counters, ratios and SR derivatives.
using Test, Random, SFMT, LinearAlgebra, MVMCExpertModeParsers, MVMCOptimizers
VERSION == v"1.13.1" || error("RBM fixtures require Julia 1.13.1")
BLAS.set_num_threads(1)
const P=MVMCExpertModeParsers
const O=MVMCOptimizers
root=joinpath(@__DIR__,"..","tests","fixtures","rbm","production")
fields=[Symbol(lowercase(f),"_rbm_",s,"_terms") for s in ("phys_layer","hidden_layer","phys_hidden") for f in ("Charge","Spin","General")]
hex(v)=join([string(reinterpret(UInt64,x);base=16,pad=16) for z in v for x in (real(z),imag(z))]," ")
values(d)=vcat(P.projection_parameters(d),[t.value for f in fields for t in getfield(d,f)],[t.value for t in d.orbital_terms])
function verify(name,actual)
    path=joinpath(root,name)
    "--write" in ARGS ? write(path,actual) : @test(compare_hex_text(actual,read(path,String),(r,c,t)->begin
        ishex=occursin(r"^[0-9a-fA-F]{16}$",t[c])
        computed=name=="kernels.txt" || (name=="loading.txt" && mod1(r,6) in (2,6)) || (name=="updates.txt" && mod1(r,6)>=5)
        ishex && computed ? (1e-13,1e-13) : nothing
    end))
end
@testset "Original RBM production kernels" begin
    include(joinpath(@__DIR__,"..","extern","Julia-mVMC","MVMCOptimizers.jl","test_unit","test_unit_vmc_sampling_rbm.jl"))
    io=IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; canonical Julia-mVMC 8bb1b9e; seed=11272")
    for case in ("all","tied","mixed","dh24")
        file=joinpath(root,"namelist_$case.def")
        data=P.parse_expert_mode_files(file)
        rng=SFMT19937RNG(); Random.seed!(rng,11272)
        P.init_parameter!(data;rng)
        println(io,case)
        println(io,hex(values(data)))
        if case=="all"
            @test O.read_initial_def!(data,joinpath(root,"initial.def"))
            @test O.read_opt_para_file!(data,joinpath(root,"initial.def"))==36
        end
        println(io,hex(values(data)))
        P.read_input_parameters!(data,file)
        println(io,hex(values(data)))
        println(io,join([rand(rng,UInt32) for _ in 1:624]," "))
        P.sync_modified_parameter!(data)
        println(io,hex(values(data)))
    end
    verify("loading.txt",String(take!(io)))
    io=IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; original indexed SR update boundaries")
    for case in ("all","tied","mixed","dh24")
        file=joinpath(root,"namelist_$case.def")
        d=P.parse_expert_mode_files(file)
        P.read_input_parameters!(d,file)
        nproj=P.projection_layout(d).n_proj
        nrbm=P.count_rbm_parameters(d)
        npara=P.count_variational_parameters(d)
        println(io,case)
        println(io,"$nproj $nrbm $npara")
        println(io,hex(values(d)))
        println(io,hex([O.get_parameter_value(d,nproj+i) for i in 1:nrbm]))
        for i in 1:npara
            O.update_parameter_value(d,i,i/128,-i/256)
        end
        println(io,hex(values(d)))
        println(io,hex([O.get_parameter_value(d,nproj+i) for i in 1:nrbm]))
    end
    verify("updates.txt",String(take!(io)))
    io=IOBuffer()
    println(io,"# Julia $VERSION; $(BLAS.get_config()); threads=1; original make/update/log-ratio/RBMDiff kernels")
    for case in ("all","tied","mixed","dh24","empty")
        d = case=="empty" ? P.parse_expert_mode_files(joinpath(root,"..","namelist_empty.def")) : P.parse_expert_mode_files(joinpath(root,"namelist_$case.def"))
        case!="empty" && P.read_input_parameters!(d,joinpath(root,"namelist_$case.def"))
        for mask in 0:63
            n=[(mask>>i)&1 for i in 0:5]
            cnt=O.make_rbm_cnt(n,d)
            derivative=zeros(ComplexF64,2P.count_rbm_parameters(d))
            O.set_rbm_diff!(derivative,cnt,n,d)
            println(io,case," ",mask)
            println(io,hex(cnt))
            println(io,hex(derivative))
            println(io,hex([O.log_rbm_val(n,d)]))
            for spin in 0:1, ri in 0:2, rj in 0:2
                new=fill(ComplexF64(99,-99),length(cnt))
                O.update_rbm_cnt_hopping!(new,cnt,ri,rj,spin,d)
                println(io,hex(new))
                println(io,hex([O.log_rbm_ratio(new,cnt,d)]))
            end
        end
    end
    verify("kernels.txt",String(take!(io)))
end
