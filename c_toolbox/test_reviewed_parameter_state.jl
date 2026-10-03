# Optional comparator regressions on actual acquisition artifacts; no adoption.
using Test
length(ARGS) == 2 || error("actual-stage C-stage")
actual, expected = abspath.(ARGS)
comparator = joinpath(@__DIR__, "diagnose_reviewed_parameter_state.jl")
function succeeds(stage)
    command = `$(Base.julia_cmd()) $comparator $stage $expected 102 3`
    success(pipeline(command; stdout=devnull, stderr=devnull))
end
@testset "actual phase/state/count and C-written mask fail closed" begin
    @test succeeds(actual)
    mktempdir() do root
        function case(name, mutation)
            stage = joinpath(root, name)
            cp(actual, stage)
            mutation(stage)
            succeeds(stage)
        end
        @test !case("swapped-rows", stage -> begin
            path = joinpath(stage, "parameter-audit.tsv")
            rows = readlines(path)
            rows[1], rows[2] = rows[2], rows[1]
            write(path, join(rows, '\n') * "\n")
        end)
        @test !case("wrong-phase", stage -> begin
            path = joinpath(stage, "parameter-audit.tsv")
            write(path, replace(read(path, String), "1\tinitialized" => "1\toverlaid"; count=1))
        end)
        for (name, line) in (("raw-word", 1), ("index", 2), ("draw-count", 3))
            @test !case(name, stage -> begin
                path = joinpath(stage, "group-1-initialized-state.txt")
                rows = readlines(path)
                if line == 1
                    words = split(rows[1])
                    words[1] = string(xor(parse(UInt32, words[1]), UInt32(1)))
                    rows[1] = join(words, ' ')
                else
                    rows[line] = string(parse(Int, rows[line]) + 1)
                end
                write(path, join(rows, '\n') * "\n")
            end)
        end
        @test !case("short-state", stage -> begin
            path = joinpath(stage, "group-1-initialized-state.txt")
            rows = readlines(path)
            rows[1] = join(split(rows[1])[1:623], ' ')
            write(path, join(rows, '\n') * "\n")
        end)
        @test !case("written-flag", stage -> begin
            path = joinpath(stage, "parameter-audit.tsv")
            rows = readlines(path)
            fields = split(rows[1], '\t')
            fields[7] = "0"
            rows[1] = join(fields, '\t')
            write(path, join(rows, '\n') * "\n")
        end)
        @test case("undefined-flag-not-native-zero", stage -> begin
            path = joinpath(stage, "parameter-audit.tsv")
            rows = readlines(path)
            fields = split(rows[1], '\t')
            fields[8] = "17"
            rows[1] = join(fields, '\t')
            write(path, join(rows, '\n') * "\n")
        end)
        for (name, value, accepted) in (("initial-approved-bound", "8.7e-19", true),
                                        ("initial-outside-bound", "8.8e-19", false))
            @test case(name, stage -> begin
                path = joinpath(stage, "parameter-audit.tsv")
                rows = readlines(path)
                fields = split(rows[1], '\t')
                fields[5] = value # Independent C first real value is exactly zero.
                rows[1] = join(fields, '\t')
                write(path, join(rows, '\n') * "\n")
            end) == accepted
        end
    end
end
