# Read-only metadata validation shared by the optional generator and regressions.
using SHA
function read_reviewed_manifest(manifest, source_root)
    root = realpath(source_root)
    result = Dict{String,String}()
    for (number, line) in enumerate(eachline(manifest))
        stripped = strip(line)
        (isempty(stripped) || startswith(stripped, "#")) && continue
        matched = match(r"^([0-9a-fA-F]{64})[ \t]+(.+)$", stripped)
        matched === nothing && error("malformed SHA256 manifest record $number")
        digest, relative = lowercase(matched[1]), strip(matched[2])
        isempty(relative) && error("empty manifest path")
        isabspath(relative) && error("manifest paths must be relative")
        path = normpath(joinpath(root, relative))
        startswith(path, root * "/") || error("manifest path escapes fork")
        isfile(path) || error("missing reviewed source: $relative")
        startswith(realpath(path), root * "/") || error("manifest symlink escapes fork")
        canonical = relpath(path, root)
        haskey(result, canonical) && error("duplicate manifest path: $canonical")
        bytes2hex(sha256(read(path))) == digest || error("reviewed source mismatch: $canonical")
        result[canonical] = digest
    end
    isempty(result) && error("empty reviewed manifest")
    return result
end
