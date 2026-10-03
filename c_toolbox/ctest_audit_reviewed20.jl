# Optional pre-import audit; pure offline reference files, no Rust results.
using SHA, MVMCExpertModeParsers
length(ARGS) == 2 || error("usage: frozen-repo reference-stage")
repo, stage = abspath.(ARGS)
models = ["heisenberg_chain_real", "heisenberg_chain_cmp", "hubbard_chain_real",
    "hubbard_chain_cmp", "heisenberg_chain_fsz", "hubbard_chain_fsz",
    "kondo_chain_real", "kondo_chain_cmp", "kondo_chain_stot1_cmp",
    "hubbard_tetragonal_real", "hubbard_tetragonal_momentum_projection_real",
    "kondo_chain_fsz", "general_rbm_cmp"]
required = ["c-window-input.txt", "configs.txt", "energy.txt", "inputs.sha256",
    "model-settings.txt", "parameters.txt", "rng-state.txt", "rng.txt",
    "sr_ho.txt", "sr_oo.txt", "status.txt", "zvo_c_slots_var.dat", "zvo_out.dat"]
records = 0
for model in models, steps in (1, 2, 3, 20)
    case = joinpath(stage, model, "step-$steps")
    all(file -> isfile(joinpath(case, file)), required) || error("missing artifact $case")
    strip(read(joinpath(case, "status.txt"), String)) == "0" || error("nonzero status")
    inputdir = joinpath(repo, "extern/Julia-mVMC/test/integration/reference", model, "inputs")
    names = Set{String}()
    for row in readlines(joinpath(case, "inputs.sha256"))
        fields = split(row)
        length(fields) == 2 || error("hash row")
        hash, name = fields
        basename(name) == name || error("nonflat hash input")
        name in names && error("duplicate input hash")
        push!(names, name)
        bytes2hex(sha256(read(joinpath(inputdir, name)))) == hash || error("changed input $name")
    end
    all(name -> name in names, ("namelist.def", "modpara.def")) || error("missing required input")
    for (_, name) in MVMCExpertModeParsers.parse_namelist_content(read(joinpath(inputdir, "namelist.def"), String))
        name in names || error("omitted referenced input $name")
    end
    isfile(joinpath(inputdir, "initial.def")) && !("initial.def" in names) && error("omitted auto initial")
    lines = readlines(joinpath(case, "c-window-input.txt"))
    header = parse.(Int, split(lines[1]))
    length(header) == 17 && header[1] == steps || error("window header")
    npara = header[2]
    widths = header[3:end]
    all(>=(0), widths) && sum(widths) == npara || error("15 declared slot widths")
    length(lines) == steps + 1 || error("window rows")
    for row in lines[2:end]
        values = parse.(Float64, split(row))
        length(values) == 2 * (npara + 2) && all(isfinite, values) || error("dense finite window")
    end
    for (file, width) in (("parameters.txt", 2*npara), ("energy.txt", 2))
        values = parse.(Float64, split(read(joinpath(case, file), String)))
        length(values) == width && all(isfinite, values) || error("$file schema")
    end
    state = readlines(joinpath(case, "rng-state.txt"))
    length(state) == 3 && length(parse.(UInt32, split(state[1]))) == 624 || error("SFMT state shape")
    0 <= parse(Int, state[2]) <= 624 || error("SFMT index")
    parse(UInt64, state[3])
    length(parse.(UInt32, split(read(joinpath(case, "rng.txt"), String)))) == 624 || error("next624 shape")
    out = readlines(joinpath(case, "zvo_out.dat"))
    var = readlines(joinpath(case, "zvo_c_slots_var.dat"))
    length(out) == length(var) == steps || error("preSR output history rows")
    all(row -> length(split(row)) == 6 + 3*npara, var) || error("full declared preSR var schema")
    global records += 1
    println("AUDITED $model steps=$steps NPara=$npara inputs=$(length(names))")
end
records == 52 || error("expected52 cases")
println("AUDIT PASS cases=$records inputclosure/window/parameter/energy/SFMT/preSR-var schemas; not Rust numerical acceptance")
