using Test,SHA
include("reference_native_fsz_fixture_inheritance.jl")
const Inheritance=NativeFSZFixtureInheritance
const Comparison=Inheritance.ReferenceNumericalComparison
hex(v)=join(string.(reinterpret.(UInt64,v);base=16,pad=16)," ")
@testset "Optional numerical references preserve exact controls" begin
    reference="# source sha256=abc\n7 "*hex([.5,1.0])*"\n"
    selector=(r,c,t)->c==3 ? (1e-13,1e-13) : nothing
    @test Comparison.compare_hex_text(replace(reference,hex([1.0])=>hex([nextfloat(1.0)])),reference,selector)
    for changed in (replace(reference,"sha256=abc"=>"sha256=abd"),
                    replace(reference,"7 "=>"8 "),
                    replace(reference,hex([.5])=>hex([nextfloat(.5)])),
                    replace(reference,hex([1.0])=>hex([1.001])),reference*"0\n")
        @test_throws ErrorException Comparison.compare_hex_text(changed,reference,selector)
    end
    @test Comparison.within(0.0,-0.0,1e-13,1e-13)
    @test !Comparison.within(Inf,-Inf,1e-13,1e-13)
    @test Comparison.within(NaN,NaN,1e-13,1e-13)
    @test !Comparison.within(floatmax(Float64),-floatmax(Float64),floatmax(Float64),1e-10)
    gram="2 3\n"*hex([1.,2.,3.,4.,5.,6.])*"\n"*hex([5.,7.,9.,11.])*"\n"
    @test Comparison.compare_unused_snapshot("gram.txt",replace(gram,hex([11.])=>hex([nextfloat(11.)])),gram)
    @test_throws ErrorException Comparison.compare_unused_snapshot("gram.txt",replace(gram,"2 3"=>"2 4"),gram)
    @test_throws ErrorException Comparison.compare_unused_snapshot("gram.txt",replace(gram,hex([11.])=>hex([11.1])),gram)
    mktempdir() do dir
        path=joinpath(dir,"fixed-input.txt")
        write(path,gram)
        run(pipeline(`gzip -c $path`,stdout=path*".gz"))
        digest=bytes2hex(sha256(gram))
        @test Inheritance.read_archived(path,digest)==gram
        write(path,gram*"corrupted\n")
        run(pipeline(`gzip -c $path`,stdout=path*".gz"))
        @test_throws ErrorException Inheritance.read_archived(path,digest)
    end
    for name in ("step-3-rng.txt","step-3-configs.txt","step-3-status.txt")
        @test_throws ErrorException Comparison.compare_runner(name,"8\n","7\n")
        @test !Inheritance.unused_allowed("sr_direct/fsz_runner/"*name,name)
    end
    @test !Inheritance.unused_allowed("sr_direct/opt_fsz_runner/fixed-input.txt","fixed-input.txt")
    @test !Inheritance.unused_allowed("sr_direct/rbm_fsz_runner/fixed-input.txt","fixed-input.txt")
    @test !Inheritance.unused_allowed("sr_direct/unknown_runner/fixed-input.txt","fixed-input.txt")
    @test_throws ErrorException Comparison.compare_unused_snapshot("step-3-rng.txt","7\n","7\n")
    words=join(fill("7",624)," ")*"\n"
    mixed="# exact schema\n1234567890123456\n"*hex([.5])*"\n"*words*hex([.25])*"\n"
    @test Comparison.compare_initialization_text(replace(mixed,hex([.5])=>hex([nextfloat(.5)])),mixed)
    @test_throws ErrorException Comparison.compare_initialization_text(replace(mixed,"1234567890123456"=>"1234567890123457"),mixed)
    @test_throws ErrorException Comparison.compare_initialization_text(replace(mixed,hex([.25])=>hex([nextfloat(.25)])),mixed)
    cutoff=hex([0.0,1e-14])*"\n"
    @test_throws ErrorException Comparison.compare_computed_text(hex([1e-15,1e-14])*"\n",cutoff;zero_pattern=true)
    tiny="case 7\n"*hex([1e-300])*"\n"
    @test Comparison.compare_record_blocks(replace(tiny,hex([1e-300])=>hex([nextfloat(1e-300)])),tiny,2,(2,);preserve_tiny=true)
    @test_throws ErrorException Comparison.compare_record_blocks(replace(tiny,hex([1e-300])=>hex([0.0])),tiny,2,(2,);preserve_tiny=true)
    @test Comparison.compare_runner("step-3-SRinfo.txt","# exact header\n3 10 4 4 1.00001e+00 0.00000e+00 .1 13, 40\n","# exact header\n3 10 4 4 1.00000e+00 0.00000e+00 .1 13, 40\n")
    @test_throws ErrorException Comparison.compare_runner("step-3-SRinfo.txt","# exact header\n3 10 5 4 1.00000e+00 0.00000e+00 .1 13, 40\n","# exact header\n3 10 4 4 1.00000e+00 0.00000e+00 .1 13, 40\n")
end

@testset "Runner schema uses exact discrete columns" begin
    line="3.00000e+00 10.00000e+00 4.00000e+00 4.00000e+00 1.00000e+00 0.00000e+00 0.10000e+00 13.00000e+00 40.00000e+00\n"
    for column in (1,2,3,4,8,9)
        tokens=split(chomp(line)); tokens[column]=string(parse(Float64,tokens[column])+1e-8)
        @test_throws ErrorException Comparison.compare_runner("step-3-SRinfo.txt",join(tokens," ")*"\n",line)
    end
    @test_throws ErrorException Comparison.compare_runner("step-3-zqp_orbital_opt.dat","0.00001 1.0 2.0\n","0.00000 1.0 2.0\n")
    @test_throws ErrorException Comparison.compare_runner("step-3-zvo_var.dat","1.0 2.0 1e-15\n","1.0 2.0 0.0\n")
end
