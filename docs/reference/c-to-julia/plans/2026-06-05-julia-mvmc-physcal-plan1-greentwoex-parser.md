# Julia-mVMC PhysCal Plan 1 — `greentwoex.def` / `TwoBodyGEx` Parser

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict parser for `greentwoex.def` (`TwoBodyGEx`) to `MVMCExpertModeParsers.jl`, producing a `GreenTwoExTerm` list on `ExpertModeData`, with a fatal dispatch branch — the parser foundation the factored two-body Green computation (Plan 2) builds on.

**Architecture:** Mirror the existing `parse_green_two_def` structure but make it **strict** (Review Findings 3, 4 of the spec): read and validate the `greentwoex.def` header count, require exactly eight integer fields per data row, fail to the caller on any malformed row, and make the `TwoBodyGEx` dispatch branch fatal so a bad factored definition can never silently degrade into an empty list. The 8 columns `x0..x7` are stored as two one-body Green specs with C's reorder absorbed (second Green = `(x6,x7)`→`(x4,x5)`).

**Tech Stack:** Julia 1.11+, the `MVMCExpertModeParsers.jl` subpackage, its `Test`-based suite under `test/`.

**Spec:** `docs/superpowers/specs/2026-06-05-julia-mvmc-physcal-factored-green-and-fixtures-design.md` (§2 Data structures, §6 Findings 3–4, Input format §1).

**Scope note:** This plan is parser-only. It does **not** compute Green functions, build the canonical one-body list, resolve index pairs, or touch `MVMCOptimizers.jl` — those are Plan 2. It only makes `data.green_two_ex_terms` available and correct.

---

## Plan Review Findings and Required Updates

These review findings must be applied while executing the tasks below. Without
them, Plan 1 would not fully satisfy spec Findings 3 and 4.

### Finding 1: `parse_file_by_type!` fatal is not enough

Severity: High.

Review result: Task 3 makes `parse_file_by_type!(..., "TwoBodyGEx", ...)`
throw on malformed `greentwoex.def`, but the public namelist path
`parse_expert_mode_files` currently catches exceptions from
`parse_file_by_type!`, downgrades them to warnings, and continues. Missing
files are also warnings. Therefore a bad or missing `TwoBodyGEx` entry in
`namelist.def` can still silently degrade into an empty
`green_two_ex_terms` list.

Required update to Task 3:

- In addition to adding the `TwoBodyGEx` branch in `parse_file_by_type!`,
  update `parse_expert_mode_files` so `file_type == "TwoBodyGEx"` is fatal:
  - if the referenced file is missing/unreadable, throw an error instead of
    warning and continuing;
  - if `parse_file_by_type!` throws while parsing `TwoBodyGEx`, rethrow (or
    throw a new error preserving the original message) instead of pushing a
    warning.
- Add tests that exercise the public path, not only direct dispatch:
  - a temporary `namelist.def` with a valid `TwoBodyGEx greentwoex.def`
    populates `parse_expert_mode_files(...).green_two_ex_terms`;
  - a temporary `namelist.def` with a malformed `greentwoex.def` throws;
  - a temporary `namelist.def` with `TwoBodyGEx missing.def` throws.

Implementation sketch for the missing-file branch:

```julia
if !validate_file_exists(full_path)
    if file_type == "TwoBodyGEx"
        error("Required TwoBodyGEx file not found: $full_path")
    end
    warning_msg = "File not found: $full_path"
    push!(warnings, warning_msg)
    @warn warning_msg
    continue
end
```

Implementation sketch for the parse-exception branch:

```julia
try
    parse_file_by_type!(data, file_type, full_path)
catch e
    if file_type == "TwoBodyGEx"
        error("Error parsing required TwoBodyGEx file $full_path: $e")
    end
    error_msg = "Error parsing $file_type file $full_path: $e"
    push!(warnings, error_msg)
    @warn error_msg
end
```

### Finding 2: Header count parsing is too permissive

Severity: Medium.

Review result: Task 2's parser sketch reads the first token on line 2 that
can be parsed as an integer. That is not strict enough for the stated
contract (`<keyword> <count>` on line 2) and can accept malformed headers such
as `0 bogus`. C's `ReadBuffInt()` reads the integer in the second token
position (`sscanf(ctmp2, "%s %d", ...)`), so the Julia strict parser should
require the second token to be a non-negative integer.

Required update to Task 2:

- Replace the "first integer token on line 2" logic with:
  - `tokens = split_def_line(clean_line(lines[2]))`;
  - require `length(tokens) >= 2`;
  - require `tryparse(Int, tokens[2]) !== nothing`;
  - require the parsed count is `>= 0`.
- Add malformed-header tests:
  - `NCisAjsCktAlt x` fails;
  - `0 bogus` fails even though the first token is an integer;
  - negative count fails.

Replacement implementation sketch:

```julia
header_count = -1
if length(lines) >= 2
    header_tokens = split_def_line(clean_line(lines[2]))
    if length(header_tokens) >= 2
        parsed_count = tryparse(Int, header_tokens[2])
        if parsed_count !== nothing && parsed_count >= 0
            header_count = parsed_count
        end
    end
end
if header_count < 0
    return ParseResult{Vector{GreenTwoExTerm}}(
        false,
        nothing,
        "greentwoex.def: missing or invalid header count on line 2",
        2,
    )
end
```

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `MVMCExpertModeParsers.jl/src/types/expert_types.jl` | Type definitions | Add `GreenTwoExTerm` struct; add `green_two_ex_terms` field + initializer to `ExpertModeData` |
| `MVMCExpertModeParsers.jl/src/parsers/green_parser.jl` | Green def parsers | Add `parse_green_two_ex_def` + `parse_green_two_ex_content` |
| `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl` | File-type dispatch | Add fatal `TwoBodyGEx` branch in `parse_file_by_type!` |
| `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl` | Tests | New test file |
| `MVMCExpertModeParsers.jl/test/runtests.jl` | Test registration | Register the new test file |

**Reference facts (verified against the code):**
- `ParseResult{T}` = `(success::Bool, data::Union{T,Nothing}, error_message::String, line_number::Int)` (`expert_types.jl:13`).
- `ParsingContext(name::String)` has mutable `.line_number::Int`, `.errors::Vector{String}`, `.warnings::Vector{String}` (`expert_types.jl:49`).
- Helpers: `read_def_file`, `clean_line`, `split_def_line`, `safe_parse_int` (`utils/file_utils.jl`).
- `.def` header layout (from `greentwo.def`): line 1 `=====`, **line 2 `<keyword>  <count>`** (e.g. `NCisAjsCktAlt   8`), lines 3–5 more separators, data from line 6. `IGNORE_LINES_IN_DEF = 5`.
- `ExpertModeData` is a `mutable struct` (`expert_types.jl:598`) with a single positional `new(...)` constructor (`expert_types.jl:674`); `green_two_terms::Vector{GreenTwoTerm}` is field declaration line 626, its initializer `GreenTwoTerm[]` is line 692.
- Dispatch lives in `parse_file_by_type!(data, file_type, file_path)` (`MVMCExpertModeParsers.jl:294`); the `TwoBodyG` branch is lines 516–520.
- Existing green parsers are **not** exported; tests reference them as `MVMCExpertModeParsers.parse_green_two_ex_content` etc.
- The `TwoBodyGEx → greentwoex.def` keyword mapping already exists (`utils/constants.jl:39`); no change needed there.

**Column → field mapping** (input row `x0 x1 x2 x3 x4 x5 x6 x7`, tokens 1-based):

| `GreenTwoExTerm` field | token | meaning |
|---|---|---|
| `site_i1` | `tokens[1]` (x0) | first Green creation site |
| `spin_i1` | `tokens[2]` (x1) | first Green creation spin |
| `site_j1` | `tokens[3]` (x2) | first Green annihilation site |
| `spin_j1` | `tokens[4]` (x3) | first Green annihilation spin |
| `site_i2` | `tokens[7]` (x6) | second Green creation site |
| `spin_i2` | `tokens[8]` (x7) | second Green creation spin |
| `site_j2` | `tokens[5]` (x4) | second Green annihilation site |
| `spin_j2` | `tokens[6]` (x5) | second Green annihilation spin |

---

## Setup: Branch

- [ ] **Step 0: Create the feature branch off `develop`**

The git repo is the `Julia-mVMC/` subdirectory. Branch first (do not commit on `develop`). Remote pushes are out of scope for this plan (confirm separately per CLAUDE.md).

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git switch develop
git switch -c feature/v0.3-physcal-greentwoex-parser
```
Expected: `Switched to a new branch 'feature/v0.3-physcal-greentwoex-parser'`

---

## Task 1: `GreenTwoExTerm` type + `ExpertModeData` field

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/types/expert_types.jl` (add struct after `GreenTwoTerm` at line 380; add field after line 626; add initializer after line 692)
- Test: `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl`

- [ ] **Step 1: Write the failing test**

Create `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl`:

```julia
using Test
using MVMCExpertModeParsers
using MVMCExpertModeParsers: GreenTwoExTerm, ExpertModeData

@testset "GreenTwoExTerm type and ExpertModeData field" begin
    # The struct stores two one-body Green specs (C reorder already absorbed).
    t = GreenTwoExTerm(0, 0, 1, 0, 2, 1, 3, 1)
    @test t.site_i1 == 0
    @test t.spin_i1 == 0
    @test t.site_j1 == 1
    @test t.spin_j1 == 0
    @test t.site_i2 == 2
    @test t.spin_i2 == 1
    @test t.site_j2 == 3
    @test t.spin_j2 == 1

    # A fresh ExpertModeData has an empty factored-term list by default.
    data = ExpertModeData()
    @test isa(data.green_two_ex_terms, Vector{GreenTwoExTerm})
    @test isempty(data.green_two_ex_terms)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test/test_green_two_ex_parser.jl")'
```
Expected: FAIL — `UndefVarError: GreenTwoExTerm not defined` (the struct does not exist yet).

- [ ] **Step 3: Add the `GreenTwoExTerm` struct**

In `MVMCExpertModeParsers.jl/src/types/expert_types.jl`, immediately after the `GreenTwoTerm` struct (after its closing `end` at line 380), insert:

```julia
"""
    GreenTwoExTerm

Factored (product-side) two-body Green's function term, `TwoBodyGEx` /
`greentwoex.def`. Each term names two one-body Green's functions whose product
`⟨c†_{i1} c_{j1}⟩ · conj(⟨c†_{i2} c_{j2}⟩)` is accumulated by PhysCal.

The 8 input columns `x0 x1 x2 x3 x4 x5 x6 x7` map with C's reorder absorbed
(see `GetInfoTwoBodyGEx` in mVMC `readdef.c`):
- first one-body Green  `⟨c†_{x0,x1} c_{x2,x3}⟩`  → `(site_i1,spin_i1,site_j1,spin_j1)`
- second one-body Green `⟨c†_{x6,x7} c_{x4,x5}⟩`  → `(site_i2,spin_i2,site_j2,spin_j2)`

Spins are kept as integers (0 = up, 1 = down) to match the integer lookup used
when resolving these to one-body Green indices (Plan 2).
"""
struct GreenTwoExTerm
    site_i1::Int
    spin_i1::Int
    site_j1::Int
    spin_j1::Int
    site_i2::Int
    spin_i2::Int
    site_j2::Int
    spin_j2::Int
end
```

- [ ] **Step 4: Add the `green_two_ex_terms` field declaration**

In the `ExpertModeData` struct, the field is declared right after `green_two_terms`. Change (line 624–626):

```julia
    # Green's function measurements
    green_one_terms::Vector{GreenOneTerm}
    green_two_terms::Vector{GreenTwoTerm}
```
to:
```julia
    # Green's function measurements
    green_one_terms::Vector{GreenOneTerm}
    green_two_terms::Vector{GreenTwoTerm}
    green_two_ex_terms::Vector{GreenTwoExTerm}  # TwoBodyGEx / greentwoex.def (factored)
```

- [ ] **Step 5: Add the field initializer in the constructor**

`ExpertModeData()` calls `new(...)` positionally. The initializer for `green_two_terms` is `GreenTwoTerm[]` at line 692, in this block:

```julia
            GreenOneTerm[],
            GreenTwoTerm[],
            QPTransTerm[],
```
Change it to:
```julia
            GreenOneTerm[],
            GreenTwoTerm[],
            GreenTwoExTerm[],
            QPTransTerm[],
```
(The new initializer sits in the same relative position as the new field declaration — both immediately after the `green_two_terms` slot — so the positional `new(...)` stays aligned.)

- [ ] **Step 6: Run test to verify it passes**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test/test_green_two_ex_parser.jl")'
```
Expected: PASS — the `GreenTwoExTerm type and ExpertModeData field` testset passes.

- [ ] **Step 7: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCExpertModeParsers.jl/src/types/expert_types.jl MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl
git commit -m "feat(parsers): add GreenTwoExTerm type and ExpertModeData.green_two_ex_terms"
```

---

## Task 2: Strict `parse_green_two_ex_def` / `parse_green_two_ex_content`

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/parsers/green_parser.jl` (append the new functions at end of file)
- Test: `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl`:

```julia
using MVMCExpertModeParsers: parse_green_two_ex_content

# Standard 5-line .def header followed by data rows.
function _greentwoex(count::Int, rows::Vector{String})
    header = join([
        "=============================================",
        "NCisAjsCktAlt   $count",
        "=============================================",
        "======== Factored two-body Green ============",
        "=============================================",
    ], "\n")
    return header * "\n" * join(rows, "\n") * "\n"
end

# Build content with an arbitrary line-2 (count) string, for header tests.
function _greentwoex_raw(line2::String, rows::Vector{String})
    header = join([
        "=============================================",
        line2,
        "=============================================",
        "======== Factored two-body Green ============",
        "=============================================",
    ], "\n")
    return header * "\n" * join(rows, "\n") * "\n"
end

@testset "parse_green_two_ex_content" begin
    @testset "valid content maps columns with C reorder" begin
        # Row x0..x7 = 0 0 1 0  2 1 3 1
        #   first  Green ⟨c†_{0,0} c_{1,0}⟩ -> (0,0,1,0)
        #   second Green ⟨c†_{3,1} c_{2,1}⟩ -> (3,1,2,1)   (x6,x7,x4,x5)
        content = _greentwoex(1, ["    0     0     1     0     2     1     3     1"])
        res = parse_green_two_ex_content(content)
        @test res.success
        @test length(res.data) == 1
        t = res.data[1]
        @test (t.site_i1, t.spin_i1, t.site_j1, t.spin_j1) == (0, 0, 1, 0)
        @test (t.site_i2, t.spin_i2, t.site_j2, t.spin_j2) == (3, 1, 2, 1)
    end

    @testset "header count must match parsed rows" begin
        content = _greentwoex(3, [
            "0 0 0 0 0 0 0 0",
            "0 0 0 0 0 1 0 1",
        ])  # header says 3, only 2 rows
        res = parse_green_two_ex_content(content)
        @test !res.success
        @test occursin("count", lowercase(res.error_message))
    end

    @testset "a data row that is not exactly 8 integers is rejected" begin
        content = _greentwoex(1, ["0 0 0 0 0 0 0"])  # 7 fields
        res = parse_green_two_ex_content(content)
        @test !res.success
        @test occursin("8", res.error_message)
    end

    @testset "non-integer field is rejected" begin
        content = _greentwoex(1, ["0 0 0 0 0 0 0 x"])
        res = parse_green_two_ex_content(content)
        @test !res.success
    end

    @testset "spin outside {0,1} is rejected" begin
        content = _greentwoex(1, ["0 2 0 0 0 0 0 0"])
        res = parse_green_two_ex_content(content)
        @test !res.success
    end

    @testset "header count must be a non-negative integer in token 2" begin
        # Missing header entirely.
        @test !parse_green_two_ex_content("no header here\njust one line\n").success
        # Keyword present but the count token is not an integer.
        @test !parse_green_two_ex_content(
            _greentwoex_raw("NCisAjsCktAlt x", ["0 0 0 0 0 0 0 0"])).success
        # First token is an integer but token 2 (the count) is not — guards
        # against the permissive "first integer token" shortcut (Finding 2).
        @test !parse_green_two_ex_content(
            _greentwoex_raw("0 bogus", ["0 0 0 0 0 0 0 0"])).success
        # Negative count.
        @test !parse_green_two_ex_content(
            _greentwoex_raw("NCisAjsCktAlt -1", ["0 0 0 0 0 0 0 0"])).success
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test/test_green_two_ex_parser.jl")'
```
Expected: FAIL — `UndefVarError: parse_green_two_ex_content not defined`.

- [ ] **Step 3: Implement the strict parser**

Append to `MVMCExpertModeParsers.jl/src/parsers/green_parser.jl`:

```julia
"""
    parse_green_two_ex_def(filepath::String) -> ParseResult{Vector{GreenTwoExTerm}}

Parse a `greentwoex.def` (`TwoBodyGEx`) file from a path.
"""
function parse_green_two_ex_def(filepath::String)::ParseResult{Vector{GreenTwoExTerm}}
    try
        content = read_def_file(filepath)
        return parse_green_two_ex_content(content)
    catch e
        return ParseResult{Vector{GreenTwoExTerm}}(false, nothing, "Error reading file: $e", 0)
    end
end

"""
    parse_green_two_ex_content(content::String) -> ParseResult{Vector{GreenTwoExTerm}}

Parse `greentwoex.def` content. Strict (spec Findings 3, 4):

- The header count is read from line 2, token 2 (`<keyword> <count>`) and must
  be a non-negative integer (token 2 specifically, per C's `ReadBuffInt`).
- Each non-empty data row (from line 6) must contain exactly eight integer
  fields; spins must be 0 or 1; sites must be non-negative. Any violation is an
  error surfaced to the caller (rows are not silently skipped).
- The number of parsed rows must equal the header count.

Columns `x0 x1 x2 x3 x4 x5 x6 x7` map to two one-body Green specs with C's
reorder: first `(x0,x1,x2,x3)`, second `(x6,x7,x4,x5)`.
"""
function parse_green_two_ex_content(content::String)::ParseResult{Vector{GreenTwoExTerm}}
    context = ParsingContext("greentwoex.def")
    terms = GreenTwoExTerm[]
    lines = split(content, '\n')

    # Header count from line 2: require `<keyword> <count>` with the count in
    # token position 2, matching C's ReadBuffInt (`sscanf(ctmp2, "%s %d", ...)`).
    # Do NOT accept "the first integer token anywhere on the line" (Finding 2).
    header_count = -1
    if length(lines) >= 2
        header_tokens = split_def_line(clean_line(lines[2]))
        if length(header_tokens) >= 2
            parsed_count = tryparse(Int, header_tokens[2])
            if parsed_count !== nothing && parsed_count >= 0
                header_count = parsed_count
            end
        end
    end
    if header_count < 0
        return ParseResult{Vector{GreenTwoExTerm}}(
            false, nothing,
            "greentwoex.def: missing or invalid header count on line 2 (expected `<keyword> <count>`)",
            2,
        )
    end

    IGNORE_LINES_IN_DEF = 5
    for line_num = (IGNORE_LINES_IN_DEF + 1):length(lines)
        context.line_number = line_num
        cleaned = clean_line(lines[line_num])
        isempty(cleaned) && continue

        tokens = split_def_line(cleaned)
        if length(tokens) != 8
            push!(context.errors,
                "Line $line_num: expected exactly 8 integer fields, got $(length(tokens))")
            continue
        end

        vals = Vector{Int}(undef, 8)
        ok = true
        for i = 1:8
            v = tryparse(Int, tokens[i])
            if v === nothing
                push!(context.errors, "Line $line_num: field $i is not an integer: '$(tokens[i])'")
                ok = false
                break
            end
            vals[i] = v
        end
        ok || continue

        # vals = (x0,x1,x2,x3,x4,x5,x6,x7)
        sites = (vals[1], vals[3], vals[7], vals[5])     # i1, j1, i2, j2
        spins = (vals[2], vals[4], vals[8], vals[6])     # spins of the above
        if any(<(0), sites)
            push!(context.errors, "Line $line_num: negative site index in $(Tuple(vals))")
            continue
        end
        if any(s -> s < 0 || s > 1, spins)
            push!(context.errors, "Line $line_num: spin must be 0 or 1 in $(Tuple(vals))")
            continue
        end

        push!(terms, GreenTwoExTerm(
            vals[1], vals[2], vals[3], vals[4],   # site_i1, spin_i1, site_j1, spin_j1 = x0,x1,x2,x3
            vals[7], vals[8], vals[5], vals[6],   # site_i2, spin_i2, site_j2, spin_j2 = x6,x7,x4,x5
        ))
    end

    if isempty(context.errors) && length(terms) != header_count
        push!(context.errors,
            "greentwoex.def: header count $header_count does not match parsed rows $(length(terms))")
    end

    success = isempty(context.errors)
    return ParseResult{Vector{GreenTwoExTerm}}(
        success,
        success ? terms : nothing,
        join(context.errors, "; "),
        context.line_number,
    )
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test/test_green_two_ex_parser.jl")'
```
Expected: PASS — all `parse_green_two_ex_content` testsets pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCExpertModeParsers.jl/src/parsers/green_parser.jl MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl
git commit -m "feat(parsers): add strict parse_green_two_ex_def for greentwoex.def"
```

---

## Task 3: Fatal `TwoBodyGEx` on dispatch **and** the public path

A `TwoBodyGEx` parse failure must be fatal not only in `parse_file_by_type!`
but also through the public `parse_expert_mode_files`. Today that function
catches per-file exceptions and downgrades them to warnings, and its **outer**
`try/catch` (around the whole loop) would also swallow an in-loop `error(...)`
and still `return data` with an empty list. So the public path must *record*
the fatal failure and rethrow it **after** the outer `try/catch` (spec Finding 4
+ Plan Review Finding 1).

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl`:
  - `parse_file_by_type!` — add the `TwoBodyGEx` branch after the `TwoBodyG` branch (lines 516–520);
  - `parse_expert_mode_files` — make `TwoBodyGEx` fatal for both missing file and parse exception, throwing after the outer `try/catch` (lines 82–110 and 148–154).
- Test: `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl`:

```julia
using MVMCExpertModeParsers: parse_file_by_type!

@testset "TwoBodyGEx dispatch" begin
    mktempdir() do dir
        good = joinpath(dir, "greentwoex.def")
        write(good, _greentwoex(1, ["0 0 1 0 2 1 3 1"]))
        data = ExpertModeData()
        parse_file_by_type!(data, "TwoBodyGEx", good)
        @test length(data.green_two_ex_terms) == 1
        @test data.green_two_ex_terms[1].site_i2 == 3  # reorder preserved through dispatch

        # A malformed factored file must be fatal, not silently empty.
        bad = joinpath(dir, "bad_greentwoex.def")
        write(bad, _greentwoex(2, ["0 0 1 0 2 1 3 1"]))  # header 2, one row
        data2 = ExpertModeData()
        @test_throws ErrorException parse_file_by_type!(data2, "TwoBodyGEx", bad)
    end
end

# parse_expert_mode_files is exported; `using MVMCExpertModeParsers` (top of file)
# brings it into scope.
@testset "TwoBodyGEx is fatal on the public parse_expert_mode_files path" begin
    # Valid: factored terms reach the public ExpertModeData.
    mktempdir() do dir
        write(joinpath(dir, "greentwoex.def"), _greentwoex(1, ["0 0 1 0 2 1 3 1"]))
        write(joinpath(dir, "namelist.def"), "      TwoBodyGEx  greentwoex.def\n")
        d = parse_expert_mode_files(joinpath(dir, "namelist.def"))
        @test length(d.green_two_ex_terms) == 1
    end
    # Malformed greentwoex.def is fatal (not a swallowed warning).
    mktempdir() do dir
        write(joinpath(dir, "greentwoex.def"), _greentwoex(2, ["0 0 1 0 2 1 3 1"]))
        write(joinpath(dir, "namelist.def"), "      TwoBodyGEx  greentwoex.def\n")
        @test_throws ErrorException parse_expert_mode_files(joinpath(dir, "namelist.def"))
    end
    # Missing greentwoex.def is fatal.
    mktempdir() do dir
        write(joinpath(dir, "namelist.def"), "      TwoBodyGEx  greentwoex.def\n")
        @test_throws ErrorException parse_expert_mode_files(joinpath(dir, "namelist.def"))
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test/test_green_two_ex_parser.jl")'
```
Expected: FAIL — `parse_file_by_type!` has no `"TwoBodyGEx"` branch, so `data.green_two_ex_terms` stays empty (the dispatch and public "valid" `@test` length checks fail); and `parse_expert_mode_files` still swallows the malformed/missing cases, so those `@test_throws` fail too.

- [ ] **Step 3: Add the dispatch branch**

In `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl`, the `TwoBodyG` branch is:

```julia
    elseif file_type == "TwoBodyG"
        result = parse_green_two_def(file_path)
        if result.success
            data.green_two_terms = result.data
        end
```
Insert the new branch immediately after it:
```julia
    elseif file_type == "TwoBodyGEx"
        result = parse_green_two_ex_def(file_path)
        if result.success
            data.green_two_ex_terms = result.data
        else
            # Fatal: a present-but-unparseable factored definition must never
            # degrade silently into an empty list (spec Finding 4).
            error("Failed to parse TwoBodyGEx file '$file_path': $(result.error_message)")
        end
```

This makes the **direct** `parse_file_by_type!` call throw. The public path is fixed in Step 4.

- [ ] **Step 4: Make `parse_expert_mode_files` fatal for `TwoBodyGEx`**

In `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl`, four edits to `parse_expert_mode_files`. The outer `try/catch` (around the whole loop) would swallow any in-loop `error(...)`, so record the fatal failure in a variable and throw it **after** the outer `try/catch`.

**(a) Declare the fatal flag before the outer `try`.** Change:
```julia
    data = ExpertModeData()
    errors = String[]
    warnings = String[]

    try
```
to:
```julia
    data = ExpertModeData()
    errors = String[]
    warnings = String[]
    # TwoBodyGEx (factored Green) is required-if-present: record a fatal failure
    # and rethrow AFTER the outer try/catch below, which would otherwise swallow
    # an in-loop error and return an empty green_two_ex_terms list.
    twobodygex_fatal = nothing

    try
```

**(b) Missing-file branch — fatal for `TwoBodyGEx`.** Change:
```julia
            if !validate_file_exists(full_path)
                warning_msg = "File not found: $full_path"
                push!(warnings, warning_msg)
                @warn warning_msg
                continue
            end
```
to:
```julia
            if !validate_file_exists(full_path)
                if file_type == "TwoBodyGEx"
                    twobodygex_fatal = "Required TwoBodyGEx file not found: $full_path"
                    break   # stop parsing; post-loop work is guarded below
                end
                warning_msg = "File not found: $full_path"
                push!(warnings, warning_msg)
                @warn warning_msg
                continue
            end
```

**(c) Parse-exception branch — fatal for `TwoBodyGEx`.** Change:
```julia
            try
                parse_file_by_type!(data, file_type, full_path)
            catch e
                error_msg = "Error parsing $file_type file $full_path: $e"
                push!(warnings, error_msg)  # Treat as warning, not error
                @warn error_msg
                # Continue processing other files
            end
```
to:
```julia
            try
                parse_file_by_type!(data, file_type, full_path)
            catch e
                if file_type == "TwoBodyGEx"
                    twobodygex_fatal = "Error parsing required TwoBodyGEx file $full_path: $e"
                    break   # stop parsing; post-loop work is guarded below
                end
                error_msg = "Error parsing $file_type file $full_path: $e"
                push!(warnings, error_msg)  # Treat as warning, not error
                @warn error_msg
                # Continue processing other files
            end
```

**(c2) Guard the post-loop processing so it is skipped on a fatal `TwoBodyGEx`.** After the `for` loop closes and before the outer `catch`, the function runs `apply_rbm_opt_flags_from_files!`, `judge_orbital_mode!`, the orbital-matrix build, the `NCond` logic, `read_input_parameters!`, and the summary `@info`. Wrap that whole block so it runs only when not fatal. Change:
```julia
        # RBM OptFlag is defined in RBM idx files and uses C-specific block offsets.
        # Apply it after all RBM files are parsed, independent of namelist order.
        apply_rbm_opt_flags_from_files!(data, file_list, base_dir)
```
to:
```julia
        # Skip all post-loop processing when a required TwoBodyGEx file failed:
        # the run is about to error, so avoid spurious warnings / side effects.
        if twobodygex_fatal === nothing
        # RBM OptFlag is defined in RBM idx files and uses C-specific block offsets.
        # Apply it after all RBM files are parsed, independent of namelist order.
        apply_rbm_opt_flags_from_files!(data, file_list, base_dir)
```
and close the new `if` by changing the summary block at the end of the `try` (just before `catch e`):
```julia
        # Print summary like C implementation
        if !isempty(warnings)
            @info "Parsing completed with $(length(warnings)) warnings"
        else
            @info "Parsing completed successfully"
        end

    catch e
```
to:
```julia
        # Print summary like C implementation
        if !isempty(warnings)
            @info "Parsing completed with $(length(warnings)) warnings"
        else
            @info "Parsing completed successfully"
        end
        end  # if twobodygex_fatal === nothing

    catch e
```
(The inner block keeps its existing indentation; only the wrapping `if ... end` is added. Re-indenting the body is optional and not required for correctness.)

**(d) Throw after the outer `try/catch`.** Change:
```julia
    catch e
        error_msg = "Critical error in parse_expert_mode_files: $e"
        push!(errors, error_msg)
        @error error_msg
    end

    return data
end
```
to:
```julia
    catch e
        error_msg = "Critical error in parse_expert_mode_files: $e"
        push!(errors, error_msg)
        @error error_msg
    end

    # TwoBodyGEx failures are fatal on the public path (spec Finding 4). Throw
    # here — outside the outer try/catch — so the error is not swallowed.
    if twobodygex_fatal !== nothing
        error(twobodygex_fatal)
    end

    return data
end
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test/test_green_two_ex_parser.jl")'
```
Expected: PASS — `TwoBodyGEx dispatch` and `TwoBodyGEx is fatal on the public parse_expert_mode_files path` testsets pass (valid input populates `green_two_ex_terms`; malformed and missing files throw `ErrorException` on both the direct and public paths).

- [ ] **Step 6: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl MVMCExpertModeParsers.jl/test/test_green_two_ex_parser.jl
git commit -m "feat(parsers): dispatch TwoBodyGEx and make it fatal on the public path"
```

---

## Task 4: Register the test file in the suite

**Files:**
- Modify: `MVMCExpertModeParsers.jl/test/runtests.jl` (after `include("test_parsers.jl")` at line 9)

- [ ] **Step 1: Register the new test file**

In `MVMCExpertModeParsers.jl/test/runtests.jl`, after:
```julia
include("test_parsers.jl")
```
add:
```julia
include("test_green_two_ex_parser.jl")
```

- [ ] **Step 2: Run the full subpackage test suite**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCExpertModeParsers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'using Pkg; Pkg.test()'
```
Expected: PASS — the whole `MVMCExpertModeParsers.jl` suite passes, including the new `test_green_two_ex_parser.jl` testsets, with no regressions.

- [ ] **Step 3: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCExpertModeParsers.jl/test/runtests.jl
git commit -m "test(parsers): register greentwoex parser tests in the suite"
```

---

## Done criteria for Plan 1

- `ExpertModeData().green_two_ex_terms` is an empty `Vector{GreenTwoExTerm}`.
- `parse_green_two_ex_def` reads the header count from line 2 token 2, parses 8-integer rows with the C reorder, and rejects malformed rows / count mismatch / bad header to the caller.
- A `TwoBodyGEx` parse failure is fatal on **both** the direct `parse_file_by_type!` call **and** the public `parse_expert_mode_files` path (missing file and parse error), not swallowed by the per-file or outer try/catch.
- The full `MVMCExpertModeParsers.jl` `Pkg.test()` passes.

**Not in this plan (Plan 2):** canonical one-body list, index-pair resolution, product accumulation, FSZ guard, output numbering / `output_dir`. **Plan 3:** PhysCal runner, fixtures, comparison helpers, e2e gate, docs.

---

## Self-Review

**Spec coverage (Plan 1 slice):**
- Spec Finding 3 (header count, not modpara) → Task 2 reads/validates the `greentwoex.def` header count; never touches `modpara.n_two_body_g_ex`. ✓
- Spec Finding 4 (strictness + fatal) → Task 2 rejects non-8-field/non-integer/bad-spin rows to the caller; Task 3 makes a `TwoBodyGEx` parse failure fatal on both the direct dispatch and the public path. ✓
- Plan Review Finding 1 (public-path fatal) → Task 3 Step 4 records the fatal failure and rethrows after the outer `try/catch` (which would otherwise swallow it); public-path tests cover valid/malformed/missing. ✓
- Plan Review Finding 2 (header token 2, strict) → Task 2 Step 3 requires `tokens[2]` to be a non-negative integer; tests cover `NCisAjsCktAlt x`, `0 bogus`, and negative count. ✓
- Spec §1 column reorder (`x6,x7→x4,x5`) → Task 1 mapping table + Task 2 `GreenTwoExTerm(vals[1..4], vals[7], vals[8], vals[5], vals[6])`, asserted in tests. ✓
- Spec §2 data structures (`GreenTwoExTerm`, `green_two_ex_terms`) → Task 1. ✓

**Placeholder scan:** No TBD/“handle errors”/“similar to”; every step has full code and exact commands. ✓

**Type consistency:** `GreenTwoExTerm` field names (`site_i1,spin_i1,site_j1,spin_j1,site_i2,spin_i2,site_j2,spin_j2`) are identical in Task 1 struct, Task 2 constructor call, and all tests. `ParseResult{Vector{GreenTwoExTerm}}(success, data, message, line)` matches the 4-field `ParseResult` definition. `parse_green_two_ex_def` / `parse_green_two_ex_content` / `parse_file_by_type!` names are consistent across tasks. ✓
