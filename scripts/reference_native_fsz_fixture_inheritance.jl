# Explicit, SHA-verified reuse of unchanged historical fixture bytes.
module NativeFSZFixtureInheritance
using SHA
include("reference_numerical_comparison.jl")
using .ReferenceNumericalComparison
const FIXTURES = normpath(joinpath(@__DIR__, "..", "tests", "fixtures"))
const NATIVE = joinpath(FIXTURES, "c_kernel_order", "native_fsz")

function read_archived(path,digest)
    compressed=path*".gz"
    isfile(compressed) || error("Missing compressed independent snapshot: $path")
    stored=read(`gzip -dc $compressed`)
    bytes2hex(sha256(stored))==digest || error("Stored unused output contradicts SHA-256: $path")
    return String(stored)
end
function unused_allowed(relative,name)
    parts=split(relative,'/')
    length(parts)>=3 || return false
    solver,case=parts[end-2:end-1]
    return solver in ("sr_direct","sr_cg") &&
        case in (family*suffix for family in ("fsz","interall","pairhop_fsz","dh2_fsz","dh4_fsz","dh24_fsz","opt_fsz") for suffix in ("_runner","_store_runner")) &&
        name in ("fixed-input.txt","gram.txt") &&
        !(solver=="sr_direct" && startswith(case,"opt_"))
end

# Full matrix/Gram snapshots which ordinary Cargo gates never read may be
# stored compressed with an explicit independently generated integrity hash. This is not a
# fallback for missing numerical or trajectory expectations.
function verify_unused(directory, name, actual; sample_count=2000)
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
    unused_allowed(relative,name) || error("Consumed Cargo output cannot use an archived unused reference: $relative")
    digest = only(matches)
    length(digest) == 64 && all(isxdigit, digest) || error("Invalid unused-output SHA-256: $relative")
    # The digest protects independent stored bytes, never a new computation.
    stored=read_archived(path,digest)
    ReferenceNumericalComparison.compare_unused_snapshot(name, actual, stored; sample_count)
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
