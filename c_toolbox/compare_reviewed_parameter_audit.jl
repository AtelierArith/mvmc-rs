# Optional read-only comparator; never called by Cargo or fixture generation.
# ACTUAL.tsv C_EXPECTED.tsv C_PROVENANCE.txt ACTUAL_RNG.txt C_RNG.txt ABS REL
using SHA
length(ARGS) == 7 || error("actual expected C-provenance actual-rng C-rng abs rel")
actual_path, expected_path, provenance_path, actual_rng, expected_rng = ARGS[1:5]
absolute, relative = parse.(Float64, ARGS[6:7])
all(x -> isfinite(x) && x >= 0, (absolute, relative)) || error("invalid numerical budget")
provenance = Dict{String,String}()
for line in eachline(provenance_path)
    (isempty(line) || startswith(line, "#")) && continue
    fields = split(line, '='; limit=2)
    length(fields) == 2 || error("malformed C provenance")
    haskey(provenance, fields[1]) && error("duplicate C provenance key")
    provenance[fields[1]] = fields[2]
end
get(provenance, "authority", "") == "C" || error("independent C acquisition required")
for key in ("upstream_sha256", "adapter_sha256", "input_sha256", "expected_sha256", "rng_sha256")
    digest = get(provenance, key, "")
    length(digest) == 64 && all(isxdigit, digest) || error("missing C identity: $key")
end
for key in ("command", "compiler", "extraction", "numerical_budget_justification")
    isempty(get(provenance, key, "")) && error("missing C provenance: $key")
end
bytes2hex(sha256(read(expected_path))) == provenance["expected_sha256"] || error("C expected hash")
bytes2hex(sha256(read(expected_rng))) == provenance["rng_sha256"] || error("C RNG hash")
declared_width = parse(Int, get(provenance, "declared_npara", "0"))
groups = parse(Int, get(provenance, "audit_groups", "0"))
declared_width > 0 && groups > 0 || error("positive C declared_npara and audit_groups required")
for (key, value) in (("absolute_tolerance", absolute), ("relative_tolerance", relative))
    parse(Float64, get(provenance, key, "NaN")) == value ||
        error("caller budget differs from reviewed C provenance: $key")
end
function rows(path)
    result = Dict{Tuple{Int,String,Int},Tuple{Int,Float64,Float64,Int,Int}}()
    for (line_number, line) in enumerate(eachline(path))
        words = split(line, '\t')
        length(words) == 8 || error("$path:$line_number: eight fields required")
        sequence, phase, index = parse(Int, words[1]), words[2], parse(Int, words[3])
        sequence > 0 && index >= 0 || error("invalid sequence/index")
        key = (sequence, phase, index)
        haskey(result, key) && error("duplicate slot record $key")
        mapped = parse(Int, words[4])
        mapped in (0, 1) || error("invalid mapped marker")
        re, im = parse.(Float64, words[5:6])
        isfinite(re) && isfinite(im) || error("nonfinite declared slot")
        flags = parse.(Int, words[7:8])
        result[key] = (mapped, re, im, flags[1], flags[2])
    end
    isempty(result) && error("empty audit cannot pass")
    stages = Set((key[1], key[2]) for key in keys(result))
    ordered_phases = ("initialized", "overlaid", "synchronized")
    expected_stages = Set((sequence, ordered_phases[mod1(sequence, 3)]) for sequence in 1:3*groups)
    stages == expected_stages || error("missing/reordered audit phase or noncontiguous sequence")
    for stage in expected_stages
        indices = sort([key[3] for key in keys(result) if (key[1], key[2]) == stage])
        indices == collect(0:declared_width-1) || error("C declared width/slot set differs: $stage")
    end
    result
end
actual, expected = rows(actual_path), rows(expected_path)
Set(keys(actual)) == Set(keys(expected)) || error("audit stage/declared-slot sets differ")
for key in sort!(collect(keys(expected)))
    a, e = actual[key], expected[key]
    (a[1], a[4], a[5]) == (e[1], e[4], e[5]) || error("mapping/flags differ: $key")
    for component in (2, 3)
        abs(a[component] - e[component]) <= absolute + relative*max(abs(a[component]), abs(e[component])) ||
            error("declared parameter differs: $key component $component actual=$(a[component]) C=$(e[component])")
    end
end
function words624(path)
    values = parse.(UInt32, split(read(path, String)))
    length(values) == 624 || error("next624 requires exactly 624 words: $path")
    values
end
words624(actual_rng) == words624(expected_rng) || error("initialization RNG next624 differs")
println("PASS ", length(actual), " complete mapped/unmapped slot records; flags/mapping and next624 exact; parameters numerical abs=", absolute, " rel=", relative)
println("Scope: next624 only, NOT full RNG state or draw count. Input/source hashes are recorded C provenance, not independently verified input/source bytes by this comparator.")
