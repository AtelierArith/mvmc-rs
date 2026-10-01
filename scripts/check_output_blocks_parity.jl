# Julia oracle for output_blocks.rs: compare full bytes, not only parsed floats.
using Test, MVMCOptimizers, MVMCExpertModeParsers
import MVMCExpertModeParsers: ExpertModeData, GutzwillerTerm, JastrowTerm, OrbitalTerm
@testset "optimized parameter output bytes" begin
    mktempdir() do dir
        d = ExpertModeData()
        d.modpara.c_para_file_head = "custom"
        d.gutzwiller_terms = [GutzwillerTerm(9,1.5-2.0im,true)]
        d.jastrow_terms = [JastrowTerm(3,7,-2.0+1.5im,true)]
        d.orbital_terms = [OrbitalTerm(0,1,3,1.5-2.0im,true,1) for _ in 1:2]
        MVMCOptimizers.output_opt_data!(d;output_dir=dir)
        row = " 1.500000000000000000e+00 -2.000000000000000000e+00 \n"
        reverse = "-2.000000000000000000e+00  1.500000000000000000e+00 \n"
        header(name,count) = "===============================\n$name $count\n===============================\n===============================\n"
        @test read(joinpath(dir,"custom_opt.dat"),String) == row*reverse*row*row
        @test read(joinpath(dir,"custom_gutzwiller_opt.dat"),String) == header("NGutzwillerIdx",1)*"0 "*row
        @test read(joinpath(dir,"custom_jastrow_opt.dat"),String) == header("NJastrowIdx",1)*"0 "*reverse
        @test read(joinpath(dir,"custom_orbital_opt.dat"),String) == header("NOrbitalIdx",2)*"0 "*row*"1 "*row
    end
    mktempdir() do dir
        d = ExpertModeData()
        d.modpara.c_para_file_head = joinpath(dir,"prefix")
        MVMCOptimizers.output_opt_data!(d)
        @test read(joinpath(dir,"prefix_opt.dat"),String) == ""
        @test !isfile(joinpath(dir,"prefix_gutzwiller_opt.dat"))
        @test !isfile(joinpath(dir,"prefix_jastrow_opt.dat"))
        @test !isfile(joinpath(dir,"prefix_orbital_opt.dat"))
    end
end
