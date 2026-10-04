# Optional metadata-only regressions. No reference generation or numerical calls.
using Test, SHA
include("reviewed_source_manifest.jl")
@testset "Reviewed source manifest metadata" begin
    mktempdir() do root
        source = joinpath(root, "source with internal spaces.jl")
        write(source, "metadata fixture\n")
        digest = bytes2hex(sha256(read(source)))
        manifest = joinpath(root, "manifest.sha256")
        # Actual standard sha256sum emits TWO separator spaces. Preserve the
        # filename's internal spaces; do not reserialize to bypass parsing.
        standard = read(`sha256sum $source`, String)
        @test startswith(standard, digest * "  ")
        record = replace(standard, root * "/" => ""; count=1)
        write(manifest, record)
        @test read_reviewed_manifest(manifest, root) == Dict(basename(source) => digest)
        write(manifest, "# reviewed metadata\n\n" * digest * "\t" * basename(source) * "\n")
        @test read_reviewed_manifest(manifest, root) == Dict(basename(source) => digest)
        for invalid in (record * record,
                        digest * "  ../outside.jl\n",
                        digest * "  " * source * "\n",
                        "not-a-sha  " * basename(source) * "\n",
                        digest * basename(source) * "\n",
                        digest * "  missing.jl\n",
                        repeat("0", 64) * "  " * basename(source) * "\n",
                        "# empty manifest\n")
            write(manifest, invalid)
            @test_throws ErrorException read_reviewed_manifest(manifest, root)
        end
        write(manifest, record * digest * "  ./" * basename(source) * "\n")
        @test_throws ErrorException read_reviewed_manifest(manifest, root)
        mktempdir() do external
            outside = joinpath(external, "outside.jl")
            write(outside, "outside\n")
            symlink(outside, joinpath(root, "escaped.jl"))
            write(manifest, bytes2hex(sha256(read(outside))) * "  escaped.jl\n")
            @test_throws ErrorException read_reviewed_manifest(manifest, root)
        end
    end
end
