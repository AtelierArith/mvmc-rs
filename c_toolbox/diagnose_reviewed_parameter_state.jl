# Optional read-only acquisition diagnostic, never invoked by Cargo.
# ACTUAL_STAGE C_STAGE NPARA GROUPS. Initial acquisition only; never CG outputs.
using SHA
length(ARGS) == 4 || error("actual-stage C-stage declared-npara groups")
actual_root, c_root = ARGS[1:2]
width, groups = parse.(Int, ARGS[3:4])
width > 0 && groups > 0 || error("positive declared width/groups required")
for line in eachline(joinpath(c_root, "archive.sha256"))
    fields = split(strip(line); limit=2)
    length(fields) == 2 || error("malformed C archive manifest")
    relative = strip(fields[2])
    isabspath(relative) && error("absolute C archive entry")
    ".." in splitpath(relative) && error("C archive path escape")
    bytes2hex(sha256(read(joinpath(c_root, relative)))) == fields[1] ||
        error("C archive hash mismatch: $relative")
end
phases = ("initialized", "overlaid", "synchronized")
const INITIAL_ABSOLUTE = 8.7e-19 # Explicit user approval; relative tolerance zero.
function audit(root)
    records = Dict{Tuple{Int,String,Int},Vector{String}}()
    for (row, line) in enumerate(eachline(joinpath(root, "parameter-audit.tsv")))
        f = String.(split(line, '\t'))
        length(f) == 8 || error("eight audit fields required")
        key = (parse(Int, f[1]), f[2], parse(Int, f[3]))
        sequence = div(row-1, width) + 1
        key == (sequence, phases[mod1(sequence, 3)], mod(row-1, width)) ||
            error("physical audit sequence/phase/slot order differs at row $row")
        haskey(records, key) && error("duplicate audit record")
        records[key] = f
    end
    expected = Set((seq, phases[mod1(seq, 3)], index)
        for seq in 1:3*groups for index in 0:width-1)
    Set(keys(records)) == expected || error("missing phase/slot or unexpected record")
    records
end
a, c = audit(actual_root), audit(c_root)
mask = parse.(Int, split(read(joinpath(c_root, "flags-written.txt"), String)))
length(mask) == 2*width && all(v -> v in (0, 1), mask) || error("invalid C written mask")
max_difference = 0.0
first_difference = nothing
flag_checks = 0
for key in sort!(collect(keys(c)))
    av, cv = a[key], c[key]
    parse(Int, av[4]) == parse(Int, cv[4]) || error("mapped marker differs: $key")
    for component in 1:2
        position = 2*key[3] + component
        if mask[position] == 1
            parse(Int, av[6+component]) == parse(Int, cv[6+component]) ||
                error("C-written flag differs: $key component $component")
            global flag_checks += 1
        end
        x, y = parse(Float64, av[4+component]), parse(Float64, cv[4+component])
        isfinite(x) && isfinite(y) || error("nonfinite parameter")
        difference = abs(x-y)
        difference <= INITIAL_ABSOLUTE ||
            error("initial acquisition numerical bound exceeded: $key component $component error=$difference")
        global max_difference = max(max_difference, difference)
        if difference != 0 && first_difference === nothing
            global first_difference = (key, component, x, y, difference)
        end
    end
end
function state(path)
    lines = readlines(path)
    length(lines) == 3 || error("three state lines required: $path")
    words = parse.(UInt32, split(lines[1]))
    length(words) == 624 || error("exactly 624 raw state words required")
    index = parse(Int, lines[2])
    0 <= index <= 624 || error("SFMT index out of range")
    count = parse(UInt64, lines[3])
    (words, index, count)
end
for group in 1:groups, phase in phases
    filename = "group-$group-$phase-state.txt"
    actual = state(joinpath(actual_root, filename))
    expected = state(joinpath(c_root, filename))
    actual == expected || error("actual raw state/index/count differs: $filename")
    actual[3] == parse(UInt64, strip(read(joinpath(c_root,
        "group-$group-$phase-draw-count.txt"), String))) || error("C separate draw count differs")
    println("EXACT $filename words=624 index=$(actual[2]) primitive_draws=$(actual[3])")
end
println("EXACT records=$(length(a)) mappings=$(length(a)) C-written-flag-checks=$flag_checks",
    " undefined-C-components-per-phase=$(count(==(0), mask))")
println("INITIAL ACQUISITION ONLY abs=$INITIAL_ABSOLUTE rel=0 max_abs_difference=$max_difference first=$first_difference")
println("Undefined C flags are NOT native zero expectations; no CG/MPI numerical allowance or automatic adoption.")
