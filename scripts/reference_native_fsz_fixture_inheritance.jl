# Explicit, SHA-verified reuse of unchanged historical fixture bytes.
module NativeFSZFixtureInheritance
using SHA
const FIXTURES = normpath(joinpath(@__DIR__, "..", "tests", "fixtures"))
const NATIVE = joinpath(FIXTURES, "c_kernel_order", "native_fsz")

# Full matrix/Gram snapshots which ordinary Cargo gates never read may be
# represented by an explicit independently generated hash. This is not a
# fallback for missing numerical or trajectory expectations.
function verify_unused(directory, name, actual)
    manifest = joinpath(NATIVE, "unused-output-sha256.tsv")
    isfile(manifest) || return false
    path = joinpath(directory, name)
    relative = replace(relpath(path, NATIVE), '\\' => '/')
    matches = String[]
    for line in eachline(manifest)
        (isempty(line) || startswith(line, "#")) && continue
        fields = split(line, '\t')
        length(fields) == 2 || error("Malformed unused-output SHA-256 entry")
        fields[1] == relative || continue
        push!(matches, fields[2])
    end
    isempty(matches) && return false
    length(matches) == 1 || error("Duplicate unused-output SHA-256 entry: $relative")
    parts = split(relative, '/')
    solver = parts[end-2]
    case = parts[end-1]
    allowed = solver in ("sr_direct", "sr_cg") &&
              !startswith(case, "rbm_") && name in ("fixed-input.txt", "gram.txt") &&
              !(solver == "sr_direct" && startswith(case, "opt_"))
    allowed || error("Consumed Cargo output cannot use a SHA-only reference: $relative")
    digest = only(matches)
    length(digest) == 64 && all(isxdigit, digest) || error("Invalid unused-output SHA-256: $relative")
    bytes2hex(sha256(actual)) == digest || error("Changed independent unused output: $relative")
    isfile(path) && bytes2hex(sha256(read(path))) != digest && error("Stored unused output contradicts SHA-256: $relative")
    return true
end

function resolve(directory, name)
    path = joinpath(directory, name)
    manifest = joinpath(NATIVE, "inheritance.tsv")
    isfile(manifest) || return path
    relative = replace(relpath(path, NATIVE), '\\' => '/')
    for line in eachline(manifest)
        (isempty(line) || startswith(line, "#")) && continue
        fields = split(line)
        length(fields) == 3 || error("Malformed native FSZ inheritance entry")
        fields[1] == relative || continue
        historical = joinpath(FIXTURES, fields[2])
        bytes2hex(sha256(read(historical))) == fields[3] || error("Changed historical fixture: $(fields[2])")
        if isfile(path)
            bytes2hex(sha256(read(path))) == fields[3] || error("Native fixture contradicts inheritance: $relative")
            return path
        end
        return historical
    end
    return path # A missing changed expectation fails at the caller's read.
end
end
