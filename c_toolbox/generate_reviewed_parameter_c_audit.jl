# Optional C-only expectation acquisition; no reference Julia package imported.
using SHA, Dates
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) in (4,5) || error("usage: NAMELIST EXTERNAL_NEW_STAGE SEED INITIAL_RECORD_OR_MINUS [GROUPS]")
repo = normpath(joinpath(@__DIR__, ".."))
namelist, output, seed_text, initial = ARGS[1:4]
groups=length(ARGS)==5 ? parse(Int,ARGS[5]) : 1
1 <= groups <= 100 || error("invalid acquisition group count")
stage_prefix=groups==1 ? "" : "group-1-"
namelist = realpath(namelist)
output = abspath(output)
startswith(output * "/", repo * "/") && error("stage must be outside repository")
ispath(output) && error("stage already exists; no overwrite")
seed = parse(UInt32, seed_text)
initial == "-" || (initial = realpath(initial))
vendor = joinpath(repo, "extern/mVMC-1.3.0/src")
function extract(source, start, stop)
    a = findfirst(start, source); a === nothing && error("missing extraction start")
    b = findnext(stop, source, last(a)+1); b === nothing && error("missing extraction stop")
    source[first(a):prevind(source, first(b))]
end
definitions = Dict{String,String}()
for line in eachline(namelist)
    fields = split(line)
    isempty(fields) && continue
    length(fields) == 2 || error("unsupported namelist record")
    haskey(definitions, fields[1]) && error("duplicate namelist key")
    definitions[fields[1]] = realpath(joinpath(dirname(namelist), fields[2]))
end
for key in ("OrbitalGeneral", "OrbitalParallel")
    haskey(definitions, key) && error("not supported by this scoped anti-parallel probe")
end
modpara = Dict(f[1] => f[2] for f in split.(readlines(definitions["ModPara"])) if length(f) == 2)
nsite = parse(Int, modpara["Nsite"])
neurons = [parse(Int, get(modpara, key, "0")) for key in
    ("NneuronCharge", "NneuronSpin", "NneuronGeneral")]
neuron_total = sum(neurons) + parse(Int, get(modpara, "Nneuron", "0"))
keys15 = ["Gutzwiller", "Jastrow", "DH2", "DH4",
    "ChargeRBM_PhysLayer", "SpinRBM_PhysLayer", "GeneralRBM_PhysLayer",
    "ChargeRBM_HiddenLayer", "SpinRBM_HiddenLayer", "GeneralRBM_HiddenLayer",
    "ChargeRBM_PhysHidden", "SpinRBM_PhysHidden", "GeneralRBM_PhysHidden", "Orbital", "OptTrans"]
widths = zeros(Int, 15); complex = zeros(Int, 15)
flag_records = [String[] for _ in 1:15]
mapped = Int[]; optweights = Float64[]
for (section, key) in enumerate(keys15)
    haskey(definitions, key) || continue
    lines = readlines(definitions[key]); length(lines) >= 5 || error("short header")
    widths[section] = parse(Int, split(lines[2])[2])
    widths[section] > 0 || error("nonpositive declared width")
    complex[section] = parse(Int, split(lines[3])[2])
    complex[section] in (0,1) || error("unsupported complex header")
    records = filter(!isempty, strip.(lines[6:end]))
    width = widths[section] * (section == 3 ? 6 : section == 4 ? 10 : 1)
    if section == 15
        length(records) == width + width*nsite || error("incomplete OptTrans records")
        weights = zeros(width)
        for (i, line) in enumerate(records[1:width])
            f = split(line); parse(Int,f[1]) == i-1 || error("OptTrans ordering")
            weights[i] = parse(Float64,f[2])
        end
        append!(optweights, weights); append!(mapped, ones(Int,width)); continue
    end
    # Counts from the actual C reader's geometry loops, not a Julia pack.
    rows = section == 1 ? nsite : section == 2 ? nsite*(nsite-1) :
        section in (3,4) ? nsite*widths[section] : section == 14 ? nsite*nsite :
        section in 5:7 ? (section == 7 ? 2nsite : nsite) :
        section in 8:10 ? neurons[section-7] :
        (section == 13 ? 2nsite : nsite)*neurons[section-10]
    length(records) == rows + width || error("C-invalid count for $key: geometry=$rows flags=$width, actual=$(length(records)); no flag expansion")
    geometry = split.(records[1:rows])
    active = falses(width)
    for f in geometry
        index = parse(Int, f[end])
        if section in (3,4)
            0 <= index < widths[section] || error("DH mapping outside declared width")
            for block in 0:(section == 3 ? 5 : 9)
                active[index+block*widths[section]+1] = true
            end
        else
            0 <= index < width || error("mapping outside declared width")
            active[index+1] = true
        end
    end
    for (i, line) in enumerate(records[rows+1:end])
        f=split(line); length(f)==2 || error("bad flag record")
        parse(Int,f[1]) == i-1 || error("unsupported flag label order")
        parse(Int,f[2]); push!(flag_records[section], f[2])
    end
    append!(mapped, Int.(active))
end
npara = widths[1]+widths[2]+6widths[3]+10widths[4]+sum(widths[5:end])
length(mapped) == npara || error("declared pack width")
allcomplex = sum(complex[1:4])+complex[14] # C readdef.c:641–642, not RBM headers.
parameter_path = joinpath(vendor,"mVMC/parameter.c")
reader_path = joinpath(vendor,"mVMC/readdef.c")
parameter = read(parameter_path,String); reader=read(reader_path,String)
excerpt = extract(parameter,"void InitParameter()","void SetFlagShift()") *
    extract(parameter,"void SetFlagShift()","#endif") *
    extract(reader,"int ReadInputParameters(char", "/**********************************************************************/") *
    extract(reader,"int GetInfoOpt(FILE", "int GetInfoGutzwiller(FILE") *
    extract(reader[first(findlast("int GetInfoOptTrans(",reader)):end],
        "int GetInfoOptTrans(", "int GetInfoInterAll(FILE")
header=read(joinpath(vendor,"mVMC/include/readdef.h"),String)
keywords=[m.captures[1] for m in eachmatch(r"\"([^\"]+)\"",extract(header,"static char cKWListOfFileNameList","/**"))]
mkpath(output)
write(joinpath(output,"reviewed_parameter_c_upstream.inc"),
    parameter[1:prevind(parameter,first(findfirst("#include",parameter)))] *
    "/* Verbatim parameter.c/readdef.c extraction; see reviewed_parameter_c_audit.md. */\n"*excerpt)
spec=joinpath(output,"spec.txt")
open(spec,"w") do io
    println(io,join(vcat(widths,[allcomplex,neuron_total,nsite,parse(Int,modpara["NVMCCalMode"])])," "))
    for section in 1:14
        println(io,complex[section]," ",join(flag_records[section]," "))
    end
    println(io,join(mapped," ")); println(io,join(optweights," "))
    println(io,get(definitions,"OptTrans","-"))
    for (key,path) in sort!(collect(definitions); by=first)
        startswith(key,"In") || continue
        i=findfirst(==(key),keywords); i===nothing && error("unsupported C overlay keyword")
        println(io,i-1," ",path)
    end
end
compiler=get(ENV,"CC","gcc")
driver=joinpath(@__DIR__,"reviewed_parameter_c_audit.c")
binary=joinpath(output,"parameter-audit")
options=["-std=gnu11","-O0","-ffp-contract=off","-DMEXP=19937",
    "-I$(joinpath(vendor,"sfmt"))","-I$(joinpath(vendor,"mVMC/include"))","-I$output"]
command=`$compiler $options $driver -lm -o $binary`
run(command)
execution=`$binary $spec $initial $output $seed $groups`
run(execution)
open(joinpath(output,"provenance.txt"),"w") do io
    println(io,"authority=C\nupstream_sha256=",bytes2hex(sha256(read(parameter_path))))
    println(io,"adapter_sha256=",bytes2hex(sha256(read(driver))))
    println(io,"input_sha256=",bytes2hex(sha256(read(spec))))
    println(io,"expected_sha256=",bytes2hex(sha256(read(joinpath(output,"parameter-audit.tsv")))))
    println(io,"rng_sha256=",bytes2hex(sha256(read(joinpath(output,stage_prefix*"initialized-next624.txt")))))
    println(io,"command=",execution,"; build=",command)
    println(io,"compiler=",replace(strip(read(`$compiler --version`,String)), '\n'=>';'))
    println(io,"extraction=parameter.c InitParameter through SetFlagShift; readdef.c ReadInputParameters/GetInfoOpt verbatim; scalar serial adapter, not full geometry reader/executable/MPI")
    println(io,"numerical_budget_justification=UNREVIEWED; no parameter acceptance budget assigned by generator")
    println(io,"seed=$seed\nNPara=$npara\nwidths15=",join(widths," "))
    println(io,"declared_npara=$npara\naudit_groups=$groups")
    println(io,"readdef_sha256=",bytes2hex(sha256(read(reader_path))))
    println(io,"excerpt_sha256=",bytes2hex(sha256(read(joinpath(output,"reviewed_parameter_c_upstream.inc")))))
    println(io,"binary_sha256=",bytes2hex(sha256(read(binary))))
    for group in 1:groups, phase in ("initialized","overlaid","synchronized")
        prefix=groups==1 ? "" : "group-$group-"
        println(io,"$(prefix)$(phase)_state_sha256=",bytes2hex(sha256(read(joinpath(output,"$prefix$phase-state.txt")))))
        println(io,"$(prefix)$(phase)_next624_sha256=",bytes2hex(sha256(read(joinpath(output,"$prefix$phase-next624.txt")))))
    end
    println(io,"architecture=$(Sys.MACHINE)\nJulia=$VERSION orchestration only; no reference package; BLAS=none\ngenerated_utc=$(now(UTC))")
    for (index,path) in enumerate(sort!(unique(vcat([namelist,parameter_path,reader_path,driver,@__FILE__,joinpath(vendor,"sfmt/SFMT.c"),joinpath(repo,"extern/mVMC-1.3.0/COPYING")],collect(values(definitions)),initial=="-" ? String[] : [initial]))))
        println(io,"source_or_input_sha256.$index=",bytes2hex(sha256(read(path)))," ",path)
    end
end
println("GENERATED C acquisition NPara=$npara three phases; no parity acceptance; $output")
