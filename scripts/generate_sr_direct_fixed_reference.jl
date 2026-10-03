# Optional independent LAPACK replay of archived direct-SR systems.
# Preserve all input rows, matrix and gradient; replace only factor/solution.
using LinearAlgebra, Libdl, SHA
VERSION == v"1.13.1" || error("Julia 1.13.1 is required")
Sys.isapple() && Sys.ARCH == :aarch64 || error("macOS ARM is required")
include("reference_lp64_blas.jl")
bridge = only(filter(a -> startswith(a, "--blas-bridge="), ARGS))
provenance = ReferenceLP64BLAS.install!(split(bridge, "="; limit=2)[2])
core = unsafe_string(ccall(Libdl.dlsym(ReferenceLP64BLAS.HANDLE[], :mvmc_reference_blas_core), Cstring, ()))
fixtures = normpath(joinpath(@__DIR__, "..", "tests", "fixtures"))
output = joinpath(fixtures, "macos_arm_julia", core)
parse_bits(line) = reinterpret.(Float64, parse.(UInt64, split(line); base=16))
hex(values) = join(string.(reinterpret.(UInt64, values); base=16, pad=16), " ")
hashes = String[]
for directory in sort(readdir(joinpath(fixtures, "sr_direct")))
    archived = joinpath(fixtures, "sr_direct", directory, "fixed-input.txt")
    isfile(archived) || continue
    payload = read(archived, String)
    lines = filter(l -> !isempty(l) && !startswith(l, "#"), split(payload, '\n'))
    length(lines) == 8 || error("Unexpected archived system: $archived")
    n = parse(Int, split(lines[1])[2])
    matrix = reshape(parse_bits(lines[5]), n, n)
    gradient = parse_bits(lines[6])
    LinearAlgebra.LAPACK.potrf!('U', matrix)
    LinearAlgebra.LAPACK.potrs!('U', matrix, gradient)
    relative = joinpath("sr_direct_fixed", directory, "factor-solution.txt")
    path = joinpath(output, relative)
    mkpath(dirname(path))
    text = "# Independent Julia LAPACK replay; archived inputs/matrix/gradient unchanged.\n" * provenance *
        "# input $(relpath(archived, fixtures)); SHA-256 $(bytes2hex(sha256(read(archived))))\n" *
        hex(vec(matrix)) * "\n" * hex(gradient) * "\n"
    write(path, text)
    push!(hashes, bytes2hex(sha256(read(path))) * "\t" * relative * "\n")
end
write(joinpath(output, "SHA256-fixed-direct.tsv"), join(hashes))
println("Independently replayed $(length(hashes)) archived direct-SR systems; core=$core")
