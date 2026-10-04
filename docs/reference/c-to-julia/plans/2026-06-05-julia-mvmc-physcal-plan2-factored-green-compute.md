# Julia-mVMC PhysCal Plan 2 — Factored Two-Body Green Computation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and output the factored / product-side two-body Green function in `MVMCOptimizers.jl`, faithful to C-mVMC: build the C-compatible canonical one-body list, resolve factored terms to 1-based index pairs, accumulate `⟨c†c⟩·conj(⟨c†c⟩)`, guard FSZ, and fix PhysCal output numbering / `output_dir`.

**Architecture:** All factored work hangs off the parsed `green_two_ex_terms` (Plan 1). At PhysCal init, build a canonical one-body Green list (explicit `greenone.def` terms, then de-duplicated factored constituents — only when `TwoBodyGEx` is present, matching C's `IndirectGFOn`) and resolve each factored term to a 1-based `(idx0,idx1)` pair into it. The per-sample one-body Greens (already computed) feed a product accumulator. A runtime guard rejects FSZ + factored. Output gains `output_dir` plumbing and C's `ismp + NDataIdxStart` file numbering.

**Tech Stack:** Julia 1.11+, `MVMCOptimizers.jl` (depends on `MVMCExpertModeParsers.jl` Plan 1), its `test_unit/` suite.

**Spec:** `docs/superpowers/specs/2026-06-05-julia-mvmc-physcal-factored-green-and-fixtures-design.md` (Findings 1, 2, 5, 6; §3, §4, §5).

**Depends on:** Plan 1 (`GreenTwoExTerm`, `ExpertModeData.green_two_ex_terms`). Base this branch on Plan 1's branch `feature/v0.3-physcal-greentwoex-parser` (or on `develop` once PR #8 is merged).

**Scope note:** No reference fixtures, comparison helpers, e2e gate, PhysCal runner, or docs — those are Plan 3. This plan makes the factored Green compute correct and unit/contract-tested.

---

## Plan Review Findings and Required Updates

This section records the implementation-plan review result and the required
remedies. Apply these updates before executing the task list below.

### Finding 1: PhysCal output numbering is not tested through the changed path

**Issue:** Task 4 changes `vmc_phys_cal!` to pass
`file_idx = ismp + data.modpara.n_data_idx_start` to `output_data_phys!`, but
the proposed test calls `output_green_func!(data, state, 1; output_dir = dir)`
directly. That direct writer test proves `_001` can be written, but it does not
prove that the `vmc_phys_cal!` sampling loop actually applies
`NDataIdxStart`. The file-indexing bug could remain and the test would still
pass.

**Required update:**
- Add a small helper in `vmc_phys_cal.jl`, for example:

  ```julia
  physcal_output_file_index(data::ExpertModeData, ismp::Int)::Int =
      ismp + data.modpara.n_data_idx_start
  ```

- Use that helper in the sample loop:

  ```julia
  file_idx = physcal_output_file_index(data, ismp)
  output_data_phys!(data, state, file_idx; output_dir = output_dir)
  ```

- Extend the Task 4 tests with a contract test that covers `NDataIdxStart`
  explicitly:

  ```julia
  @testset "PhysCal output file index uses NDataIdxStart" begin
      data = ExpertModeData()
      data.modpara.n_data_idx_start = 1
      @test MVMCOptimizers.physcal_output_file_index(data, 0) == 1
      @test MVMCOptimizers.physcal_output_file_index(data, 3) == 4

      data.modpara.n_data_idx_start = 7
      @test MVMCOptimizers.physcal_output_file_index(data, 0) == 7
      @test MVMCOptimizers.physcal_output_file_index(data, 2) == 9
  end
  ```

This keeps the unit test cheap while guarding the exact logic changed in
`vmc_phys_cal!`. The existing direct `output_green_func!` test should remain as
the writer-format test.

### Finding 2: FSZ guard should run before RNG / parameter side effects and needs public-path coverage

**Issue:** Task 5 currently inserts `validate_factored_green_supported(data)`
after `i_flg_orbital_general = data.i_flg_orbital_general`. In the current
`vmc_phys_cal!`, that point is after RNG setup and `init_parameter!`, so an
unsupported `FSZ + TwoBodyGEx` input can mutate state before failing. The tests
also call only the guard helper, not the public `vmc_phys_cal!` path.

**Required update:**
- Move the guard call immediately after `validate_supported_modpara(data.modpara)`
  near the top of `vmc_phys_cal!`:

  ```julia
  validate_supported_modpara(data.modpara)
  validate_factored_green_supported(data)
  ```

- Keep the direct helper test, and add a public-path test:

  ```julia
  @testset "vmc_phys_cal! rejects FSZ + factored before sampling" begin
      data = ExpertModeData()
      data.green_two_ex_terms = [GreenTwoExTerm(0, 0, 1, 0, 2, 1, 3, 1)]
      data.i_flg_orbital_general = 1

      err = try
          MVMCOptimizers.vmc_phys_cal!(data)
          nothing
      catch e
          e
      end

      @test err isa ErrorException
      msg = sprint(showerror, err)
      @test occursin("TwoBodyGEx", msg)
      @test occursin("FSZ", msg)
  end
  ```

Because the guard is now before sampling setup, this test should fail for the
intended reason without requiring a physically complete input fixture.

### Finding 3: The byte-identical no-TwoBodyGEx path needs an output regression test

**Issue:** The plan's Done criteria require that, with `TwoBodyGEx` absent,
PhysCal behavior remains byte-identical and `zvo_cisajs` is unchanged. After
Task 4, however, `output_green_func!` writes `zvo_cisajs` from
`phys.cis_ajs_idx` instead of `data.green_one_terms`. That is correct only if
`initialize_phys_quantities!` has populated `cis_ajs_idx` exactly like the old
`greenone.def` order, including duplicate rows when `TwoBodyGEx` is absent.

**Required update:**
- Extend Task 2 or Task 4 with a no-factored regression test that runs through
  `initialize_phys_quantities!` and `output_green_func!`:

  ```julia
  @testset "no TwoBodyGEx preserves greenone order and duplicates in output" begin
      data = ExpertModeData()
      data.modpara.nsite = 2
      data.modpara.c_data_file_head = "zvo"
      data.green_one_terms = [
          GreenOneTerm(0, 1, :up, :up),
          GreenOneTerm(0, 1, :up, :up),
          GreenOneTerm(1, 0, :down, :down),
      ]

      state = VMCOptimizationState(2, 2, 0, 0, 1, 2, true, false)
      MVMCOptimizers.initialize_phys_quantities!(state, data)
      pq = state.phys_quantities

      @test pq.cis_ajs_idx == NTuple{4,Int}[
          (0, 0, 1, 0),
          (0, 0, 1, 0),
          (1, 1, 0, 1),
      ]

      pq.phys_cis_ajs .= ComplexF64[1.0 + 0im, 2.0 + 0im, 3.0 + 0im]
      mktempdir() do dir
          MVMCOptimizers.output_green_func!(data, state, 1; output_dir = dir)
          lines = filter(!isempty, split(read(joinpath(dir, "zvo_cisajs_001.dat"), String), "\n"))
          @test length(lines) == 3
          @test split(lines[1])[1:4] == ["0", "0", "1", "0"]
          @test split(lines[2])[1:4] == ["0", "0", "1", "0"]
          @test split(lines[3])[1:4] == ["1", "1", "0", "1"]
      end
  end
  ```

This directly protects the no-`TwoBodyGEx` compatibility promise.

### Finding 4: `_spin_int` should reject invalid symbols instead of treating them as down spin

**Issue:** The proposed helper `_spin_int(s::Symbol) = s == :up ? 0 : 1`
silently maps any non-`:up` symbol, including `:both` or typos, to down spin.
That can hide malformed hand-constructed test data or future parser bugs.

**Required update:**
- Replace the helper with explicit validation:

  ```julia
  _spin_int(::Val{:up}) = 0
  _spin_int(::Val{:down}) = 1
  function _spin_int(s::Symbol)
      if s == :up
          return 0
      elseif s == :down
          return 1
      end
      error("one-body Green spin must be :up or :down, got $s")
  end
  ```

- Add a small test:

  ```julia
  @test_throws ErrorException MVMCOptimizers.build_canonical_cis_ajs_idx(
      [GreenOneTerm(0, 1, :both, :up)], GreenTwoExTerm[], 2)
  ```

This is not a C-compatibility issue for parsed files, but it makes the Julia
contract stricter and easier to debug.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `MVMCOptimizers.jl/src/types.jl` | `PhysicalQuantities` | Add `cis_ajs_idx` + `cis_ajs_ckt_alt_idx` fields + constructor init |
| `MVMCOptimizers.jl/src/green_func_calc.jl` | Green init + compute | Canonical list builder, resolver, rewritten `initialize_phys_quantities!`, factored accumulator, canonical one-body loop |
| `MVMCOptimizers.jl/src/vmc_phys_cal.jl` | PhysCal driver | FSZ guard, `output_dir` arg, `ismp + n_data_idx_start` file index, `output_data_phys!` plumbing |
| `MVMCOptimizers.jl/src/data_io.jl` | Output writer | `zvo_cisajs` from the canonical list |
| `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl` | Tests | New |
| `MVMCOptimizers.jl/test/runtests.jl` | Registration | Register the new test file |

**Reference facts (verified):**
- `PhysicalQuantities(n_cis_ajs, n_cis_ajs_ckt_alt, n_cis_ajs_ckt_alt_dc)` allocates the five `ComplexF64` arrays (`types.jl:265-293`).
- `initialize_phys_quantities!(state, data)` currently sets `n_cis_ajs_ckt_alt = 0` (`green_func_calc.jl:12-25`).
- `calculate_green_func!` computes the one-body Greens into `local_cis_ajs`/`phys_cis_ajs` over `data.green_one_terms` (`green_func_calc.jl:79-103`), the DC two-body (`:106-134`), and has an empty product stub (`:136-143`). `phys = state.phys_quantities`, `w::Float64`.
- The one-body loop uses `s = term.spin2` (sj) to match C's `CisAjsIdx[idx][3]`.
- `green_func1(ri, rj, s, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt, data, state; all_complex)` (`vmc_main_cal.jl:97`).
- `output_green_func!(data, state, ismp; output_dir=nothing)` writes `zvo_cisajs` from `data.green_one_terms` (`data_io.jl:233-267`), `zvo_cisajscktaltex` from `phys.phys_cis_ajs_ckt_alt` (`:269-278`), `zvo_cisajscktalt` (DC) from `data.green_two_terms` (`:281-309`). `_output_path(filename, output_dir)` (`data_io.jl:48`).
- `output_data!(data, state, step; output_dir=nothing)` already supports `output_dir` (`data_io.jl:66`).
- `output_data_phys!(data, state, ismp)` has **no** `output_dir` and calls `output_data!`/`output_green_func!` without it (`vmc_phys_cal.jl:284-292`).
- `vmc_phys_cal!(data; callback, rng)` has **no** `output_dir`; `i_flg_orbital_general = data.i_flg_orbital_general` (`vmc_phys_cal.jl:86`); the sample loop calls `output_data_phys!(data, state, ismp)` (`:248`); `data.modpara.n_data_idx_start` exists.
- `i_flg_orbital_general != 0` ⇔ FSZ / general-orbital mode.
- `reset_phys_quantities!` only zeros the value arrays (`green_func_calc.jl:32-43`); index fields persist across samples — no change needed.
- `weight_average_green_func!` already normalizes `phys_cis_ajs_ckt_alt` (`weight_average.jl`).
- Tests register in `test/runtests.jl` via `include("../test_unit/test_unit_<name>.jl")` (last entry line 71); tests use `using MVMCExpertModeParsers: ExpertModeData, ...`.

**Canonical representation:** the canonical one-body list is `Vector{NTuple{4,Int}}` of `(ri, si, rj, sj)` with integer spins (0=up,1=down). This is the same key C builds with `iOneBodyGIdx` and avoids importing `GreenOneTerm`. `green_func1` uses `(ri=t[1], rj=t[3], s=t[4]=sj)`; output prints `ri si rj sj` directly.

---

## Setup: Branch

- [ ] **Step 0: Create the Plan 2 branch on top of Plan 1**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git switch feature/v0.3-physcal-greentwoex-parser
git switch -c feature/v0.3-physcal-factored-green
```
Expected: `Switched to a new branch 'feature/v0.3-physcal-factored-green'`.
(If PR #8 has already merged to `develop`, instead branch off `origin/develop`: `git fetch origin && git switch -c feature/v0.3-physcal-factored-green origin/develop`.)

---

## Task 1: `PhysicalQuantities` index fields

**Files:**
- Modify: `MVMCOptimizers.jl/src/types.jl` (`PhysicalQuantities`, lines 265-293)
- Test: `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`

- [ ] **Step 1: Write the failing test**

Create `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`:

```julia
using Test
using MVMCOptimizers
using MVMCExpertModeParsers: ExpertModeData, GreenOneTerm, GreenTwoExTerm

@testset "PhysicalQuantities index fields" begin
    pq = MVMCOptimizers.PhysicalQuantities(2, 1, 3)
    @test pq.cis_ajs_idx == NTuple{4,Int}[]
    @test pq.cis_ajs_ckt_alt_idx == Tuple{Int,Int}[]
    @test length(pq.phys_cis_ajs) == 2
    @test length(pq.phys_cis_ajs_ckt_alt) == 1
    @test length(pq.phys_cis_ajs_ckt_alt_dc) == 3
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: FAIL — `type PhysicalQuantities has no field cis_ajs_idx`.

- [ ] **Step 3: Add the fields and initialize them**

In `MVMCOptimizers.jl/src/types.jl`, change the `PhysicalQuantities` struct body + constructor:

```julia
    # 2-body Green's function (direct): <c†_i c_j c†_k c_l>
    # Size: NCisAjsCktAltDC
    local_cis_ajs_ckt_alt_dc::Vector{ComplexF64}
    phys_cis_ajs_ckt_alt_dc::Vector{ComplexF64}

    function PhysicalQuantities(
        n_cis_ajs::Int,
        n_cis_ajs_ckt_alt::Int,
        n_cis_ajs_ckt_alt_dc::Int,
    )
        new(
            zeros(ComplexF64, n_cis_ajs),
            zeros(ComplexF64, n_cis_ajs),
            zeros(ComplexF64, n_cis_ajs_ckt_alt),
            zeros(ComplexF64, n_cis_ajs_ckt_alt_dc),
            zeros(ComplexF64, n_cis_ajs_ckt_alt_dc),
        )
    end
end
```
to:
```julia
    # 2-body Green's function (direct): <c†_i c_j c†_k c_l>
    # Size: NCisAjsCktAltDC
    local_cis_ajs_ckt_alt_dc::Vector{ComplexF64}
    phys_cis_ajs_ckt_alt_dc::Vector{ComplexF64}

    # Canonical one-body Green list (ri, si, rj, sj), C-compatible order.
    # When TwoBodyGEx is present this includes appended factored constituents
    # (C's iOneBodyGIdx); otherwise it equals greenone.def order.
    cis_ajs_idx::Vector{NTuple{4,Int}}
    # Factored two-body pairs: 1-based indices into cis_ajs_idx / local_cis_ajs.
    cis_ajs_ckt_alt_idx::Vector{Tuple{Int,Int}}

    function PhysicalQuantities(
        n_cis_ajs::Int,
        n_cis_ajs_ckt_alt::Int,
        n_cis_ajs_ckt_alt_dc::Int,
    )
        new(
            zeros(ComplexF64, n_cis_ajs),
            zeros(ComplexF64, n_cis_ajs),
            zeros(ComplexF64, n_cis_ajs_ckt_alt),
            zeros(ComplexF64, n_cis_ajs_ckt_alt_dc),
            zeros(ComplexF64, n_cis_ajs_ckt_alt_dc),
            NTuple{4,Int}[],
            Tuple{Int,Int}[],
        )
    end
end
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCOptimizers.jl/src/types.jl MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl
git commit -m "feat(physcal): add canonical/factored index fields to PhysicalQuantities"
```

---

## Task 2: Canonical one-body list + index resolution

**Files:**
- Modify: `MVMCOptimizers.jl/src/green_func_calc.jl` (add builders; rewrite `initialize_phys_quantities!`)
- Test: `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`:

```julia
@testset "canonical one-body list" begin
    # No TwoBodyGEx: canonical == greenone.def order, no dedup.
    g1 = [GreenOneTerm(0, 1, :up, :up), GreenOneTerm(1, 0, :down, :down)]
    canon = MVMCOptimizers.build_canonical_cis_ajs_idx(g1, GreenTwoExTerm[], 2)
    @test canon == NTuple{4,Int}[(0, 0, 1, 0), (1, 1, 0, 1)]

    # TwoBodyGEx present: explicit terms first, then appended factored
    # constituents in file order, de-duplicated. GreenTwoExTerm stores the two
    # one-body Greens directly (reorder already absorbed at parse time): here
    # second = <c†_{2,1} c_{3,1}> = key (2,1,3,1), NOT in greenone.def -> appended.
    g1b = [GreenOneTerm(0, 1, :up, :up)]                  # (0,0,1,0)
    ex = [GreenTwoExTerm(0, 0, 1, 0, 2, 1, 3, 1)]         # A=(0,0,1,0) present; B=(2,1,3,1) new
    canon2 = MVMCOptimizers.build_canonical_cis_ajs_idx(g1b, ex, 4)
    @test canon2 == NTuple{4,Int}[(0, 0, 1, 0), (2, 1, 3, 1)]

    # Out-of-range site is rejected.
    @test_throws ErrorException MVMCOptimizers.build_canonical_cis_ajs_idx(
        [GreenOneTerm(0, 5, :up, :up)], GreenTwoExTerm[], 2)

    # An invalid spin symbol is rejected, not silently treated as down (Finding 4).
    @test_throws ErrorException MVMCOptimizers.build_canonical_cis_ajs_idx(
        [GreenOneTerm(0, 1, :both, :up)], GreenTwoExTerm[], 2)
end

@testset "factored index resolution is 1-based" begin
    canon = NTuple{4,Int}[(0, 0, 1, 0), (2, 1, 3, 1)]
    ex = [GreenTwoExTerm(0, 0, 1, 0, 2, 1, 3, 1)]  # A=(0,0,1,0)->canon[1], B=(2,1,3,1)->canon[2]
    pairs = MVMCOptimizers.resolve_cis_ajs_ckt_alt_idx(canon, ex)
    @test pairs == [(1, 2)]

    # C index 0 (first one-body Green) must resolve to Julia index 1.
    canon2 = NTuple{4,Int}[(0, 0, 0, 0)]
    ex2 = [GreenTwoExTerm(0, 0, 0, 0, 0, 0, 0, 0)]  # both constituents = canon[1]
    @test MVMCOptimizers.resolve_cis_ajs_ckt_alt_idx(canon2, ex2) == [(1, 1)]
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: FAIL — `build_canonical_cis_ajs_idx` / `resolve_cis_ajs_ckt_alt_idx` not defined.

- [ ] **Step 3: Implement the builders and rewrite `initialize_phys_quantities!`**

In `MVMCOptimizers.jl/src/green_func_calc.jl`, replace the whole `initialize_phys_quantities!` function (lines 7-25) with:

```julia
# Symbol spin (:up/:down) -> integer (0/1), matching C's spin encoding.
# Reject anything else (e.g. :both or a typo) instead of silently mapping it to
# down spin, so malformed data surfaces (Plan Review Finding 4).
function _spin_int(s::Symbol)
    if s == :up
        return 0
    elseif s == :down
        return 1
    end
    error("one-body Green spin must be :up or :down, got :$s")
end

"""
    build_canonical_cis_ajs_idx(green_one_terms, green_two_ex_terms, n_site)
        -> Vector{NTuple{4,Int}}

Build the C-compatible canonical one-body Green list of `(ri, si, rj, sj)`.

- When `green_two_ex_terms` is empty, this is exactly `greenone.def` order with
  no de-duplication (C reads cisajs.def directly, `IndirectGFOn = false`).
- When `green_two_ex_terms` is non-empty, the list is de-duplicated by
  `(ri, si, rj, sj)`: explicit `greenone.def` terms first, then for each
  factored term its first constituent `(i1,j1)` and second constituent
  `(i2,j2)` appended if not already present (C's `CountOneBodyGForLanczos` /
  `iOneBodyGIdx`).

Sites are validated against `[0, n_site)`.
"""
function build_canonical_cis_ajs_idx(green_one_terms, green_two_ex_terms, n_site::Int)
    canonical = NTuple{4,Int}[]
    if isempty(green_two_ex_terms)
        for t in green_one_terms
            push!(canonical, (t.site1, _spin_int(t.spin1), t.site2, _spin_int(t.spin2)))
        end
    else
        seen = Dict{NTuple{4,Int},Int}()
        function add!(key::NTuple{4,Int})
            if !haskey(seen, key)
                push!(canonical, key)
                seen[key] = length(canonical)  # 1-based
            end
        end
        for t in green_one_terms
            add!((t.site1, _spin_int(t.spin1), t.site2, _spin_int(t.spin2)))
        end
        for t in green_two_ex_terms
            add!((t.site_i1, t.spin_i1, t.site_j1, t.spin_j1))
            add!((t.site_i2, t.spin_i2, t.site_j2, t.spin_j2))
        end
    end
    for (ri, si, rj, sj) in canonical
        if ri < 0 || ri >= n_site || rj < 0 || rj >= n_site
            error("one-body Green site out of range [0,$n_site): (ri=$ri, rj=$rj)")
        end
        if si < 0 || si > 1 || sj < 0 || sj > 1
            error("one-body Green spin must be 0 or 1: (si=$si, sj=$sj)")
        end
    end
    return canonical
end

"""
    resolve_cis_ajs_ckt_alt_idx(cis_ajs_idx, green_two_ex_terms)
        -> Vector{Tuple{Int,Int}}

Resolve each factored term to a 1-based `(idx0, idx1)` pair into `cis_ajs_idx`.
`idx0` is the first one-body constituent `(i1,j1)`, `idx1` the second `(i2,j2)`.
After `build_canonical_cis_ajs_idx` every constituent is present, so a miss is
an internal invariant error.
"""
function resolve_cis_ajs_ckt_alt_idx(cis_ajs_idx, green_two_ex_terms)
    lookup = Dict{NTuple{4,Int},Int}()
    for (i, key) in enumerate(cis_ajs_idx)
        # First occurrence wins (canonical is already de-duplicated).
        get!(lookup, key, i)
    end
    pairs = Tuple{Int,Int}[]
    for t in green_two_ex_terms
        k0 = (t.site_i1, t.spin_i1, t.site_j1, t.spin_j1)
        k1 = (t.site_i2, t.spin_i2, t.site_j2, t.spin_j2)
        idx0 = get(lookup, k0, 0)
        idx1 = get(lookup, k1, 0)
        if idx0 == 0 || idx1 == 0
            error("internal: factored constituent missing from canonical one-body list")
        end
        push!(pairs, (idx0, idx1))
    end
    return pairs
end

"""
    initialize_phys_quantities!(state::VMCOptimizationState, data::ExpertModeData)

Initialize PhysicalQuantities, including the canonical one-body list and the
resolved factored index pairs.
"""
function initialize_phys_quantities!(state::VMCOptimizationState, data::ExpertModeData)
    n_site = data.modpara.nsite
    cis_ajs_idx =
        build_canonical_cis_ajs_idx(data.green_one_terms, data.green_two_ex_terms, n_site)
    cis_ajs_ckt_alt_idx = resolve_cis_ajs_ckt_alt_idx(cis_ajs_idx, data.green_two_ex_terms)

    n_cis_ajs = length(cis_ajs_idx)
    n_cis_ajs_ckt_alt = length(cis_ajs_ckt_alt_idx)  # == length(green_two_ex_terms)
    n_cis_ajs_ckt_alt_dc = length(data.green_two_terms)

    phys = PhysicalQuantities(n_cis_ajs, n_cis_ajs_ckt_alt, n_cis_ajs_ckt_alt_dc)
    phys.cis_ajs_idx = cis_ajs_idx
    phys.cis_ajs_ckt_alt_idx = cis_ajs_ckt_alt_idx
    state.phys_quantities = phys
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: PASS — canonical-list and resolution testsets pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCOptimizers.jl/src/green_func_calc.jl MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl
git commit -m "feat(physcal): build C-compatible canonical one-body list and resolve factored pairs"
```

---

## Task 3: Factored accumulation + canonical one-body loop

**Files:**
- Modify: `MVMCOptimizers.jl/src/green_func_calc.jl` (`calculate_green_func!`: one-body loop + product loop; add `accumulate_factored_green!`)
- Test: `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl` (extend)

- [ ] **Step 1: Write the failing test**

Append to `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`:

```julia
@testset "factored accumulation: w * local[idx0] * conj(local[idx1])" begin
    pq = MVMCOptimizers.PhysicalQuantities(2, 1, 0)
    pq.cis_ajs_ckt_alt_idx = [(1, 2)]
    pq.local_cis_ajs[1] = 2.0 + 1.0im
    pq.local_cis_ajs[2] = 3.0 - 4.0im
    MVMCOptimizers.accumulate_factored_green!(pq, 0.5)
    # 0.5 * (2+1im) * conj(3-4im) = 0.5 * (2+1im) * (3+4im) = 0.5 * (2+11im)
    @test pq.phys_cis_ajs_ckt_alt[1] ≈ 0.5 * ((2.0 + 1.0im) * conj(3.0 - 4.0im))
    @test pq.phys_cis_ajs_ckt_alt[1] ≈ (1.0 + 5.5im)

    # Accumulates (adds), not overwrites.
    MVMCOptimizers.accumulate_factored_green!(pq, 0.5)
    @test pq.phys_cis_ajs_ckt_alt[1] ≈ (2.0 + 11.0im)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: FAIL — `accumulate_factored_green!` not defined.

- [ ] **Step 3: Add the accumulator and wire the loops**

In `MVMCOptimizers.jl/src/green_func_calc.jl`, add the accumulator function (place it just before `calculate_green_func!`):

```julia
"""
    accumulate_factored_green!(phys::PhysicalQuantities, w::Float64)

Accumulate the factored two-body Green:
`phys_cis_ajs_ckt_alt[idx] += w * local_cis_ajs[idx0] * conj(local_cis_ajs[idx1])`,
faithful to C's `calgrn.c` `PhysCisAjsCktAlt` loop. Indices are 1-based.
"""
function accumulate_factored_green!(phys::PhysicalQuantities, w::Float64)
    @inbounds for (idx, (idx0, idx1)) in enumerate(phys.cis_ajs_ckt_alt_idx)
        phys.phys_cis_ajs_ckt_alt[idx] +=
            w * phys.local_cis_ajs[idx0] * conj(phys.local_cis_ajs[idx1])
    end
    return nothing
end
```

In `calculate_green_func!`, replace the one-body loop over `data.green_one_terms`
(lines 79-103) with a loop over the canonical list:

```julia
    # 1-body Green's function: <c†_{ri,s} c_{rj,s}>, over the canonical list.
    # C uses s = CisAjsIdx[idx][3] = sj (annihilation spin); for sz-conserved
    # inputs si == sj.
    for (idx, (ri, si, rj, sj)) in enumerate(phys.cis_ajs_idx)
        local_val = green_func1(
            ri,
            rj,
            sj,
            ip,
            ele_idx,
            ele_cfg,
            ele_num,
            ele_proj_cnt,
            data,
            state;
            all_complex = all_complex,
        )

        phys.local_cis_ajs[idx] = local_val
        phys.phys_cis_ajs[idx] += w * local_val
    end
```

And replace the empty product stub (lines 136-143) with the accumulator call:

```julia
    # 2-body Green's function (product): <c†_i c_j> × conj(<c†_k c_l>)
    # using the resolved 1-based index pairs into the one-body Greens above.
    accumulate_factored_green!(phys, w)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: PASS — the factored accumulation testset passes.

- [ ] **Step 5: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCOptimizers.jl/src/green_func_calc.jl MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl
git commit -m "feat(physcal): accumulate factored two-body Green over the canonical one-body list"
```

---

## Task 4: Output — canonical `zvo_cisajs`, `output_dir`, `ismp + NDataIdxStart`

**Files:**
- Modify: `MVMCOptimizers.jl/src/data_io.jl` (`output_green_func!` `zvo_cisajs` block, lines 244-267)
- Modify: `MVMCOptimizers.jl/src/vmc_phys_cal.jl` (`output_data_phys!` `output_dir`; `vmc_phys_cal!` `output_dir` + file index)
- Test: `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl` (extend)

- [ ] **Step 1: Write the failing test**

Append to `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`:

```julia
using MVMCOptimizers: VMCOptimizationState
using Printf

@testset "output: canonical cisajs + factored ex, with output_dir and numbering" begin
    # Build a minimal state with a populated PhysicalQuantities directly.
    data = ExpertModeData()
    data.modpara.nsite = 4
    data.modpara.c_data_file_head = "zvo"
    state = VMCOptimizationState(4, 2, 0, 0, 1, 4, true, false)

    pq = MVMCOptimizers.PhysicalQuantities(1, 1, 0)
    pq.cis_ajs_idx = NTuple{4,Int}[(0, 0, 1, 0)]
    pq.cis_ajs_ckt_alt_idx = [(1, 1)]
    pq.phys_cis_ajs[1] = 1.25 + 0.0im
    pq.phys_cis_ajs_ckt_alt[1] = 2.5 - 1.0im
    state.phys_quantities = pq

    mktempdir() do dir
        # ismp = 1 => file index 001 (covers the C numbering when written via
        # output_green_func! with the C-visible index).
        MVMCOptimizers.output_green_func!(data, state, 1; output_dir = dir)

        cisajs = joinpath(dir, "zvo_cisajs_001.dat")
        ex = joinpath(dir, "zvo_cisajscktaltex_001.dat")
        @test isfile(cisajs)
        @test isfile(ex)

        # zvo_cisajs: ri si rj sj from the canonical list (not data.green_one_terms).
        line = first(filter(!isempty, split(read(cisajs, String), "\n")))
        cols = split(line)
        @test cols[1] == "0" && cols[2] == "0" && cols[3] == "1" && cols[4] == "0"

        # zvo_cisajscktaltex: value-only real/imag for the one factored term.
        exline = first(filter(!isempty, split(read(ex, String), "\n")))
        vals = parse.(Float64, split(exline))
        @test length(vals) == 2
        @test vals[1] ≈ 2.5
        @test vals[2] ≈ -1.0
    end
end

@testset "PhysCal output file index uses NDataIdxStart" begin
    data = ExpertModeData()
    data.modpara.n_data_idx_start = 1
    @test MVMCOptimizers.physcal_output_file_index(data, 0) == 1
    @test MVMCOptimizers.physcal_output_file_index(data, 3) == 4

    data.modpara.n_data_idx_start = 7
    @test MVMCOptimizers.physcal_output_file_index(data, 0) == 7
    @test MVMCOptimizers.physcal_output_file_index(data, 2) == 9
end

@testset "no TwoBodyGEx preserves greenone order and duplicates in output" begin
    data = ExpertModeData()
    data.modpara.nsite = 2
    data.modpara.c_data_file_head = "zvo"
    data.green_one_terms = [
        GreenOneTerm(0, 1, :up, :up),
        GreenOneTerm(0, 1, :up, :up),
        GreenOneTerm(1, 0, :down, :down),
    ]

    state = VMCOptimizationState(2, 2, 0, 0, 1, 2, true, false)
    MVMCOptimizers.initialize_phys_quantities!(state, data)
    pq = state.phys_quantities

    # No dedup when TwoBodyGEx absent: greenone order + duplicate preserved.
    @test pq.cis_ajs_idx == NTuple{4,Int}[
        (0, 0, 1, 0),
        (0, 0, 1, 0),
        (1, 1, 0, 1),
    ]

    pq.phys_cis_ajs .= ComplexF64[1.0 + 0im, 2.0 + 0im, 3.0 + 0im]
    mktempdir() do dir
        MVMCOptimizers.output_green_func!(data, state, 1; output_dir = dir)
        lines = filter(!isempty, split(read(joinpath(dir, "zvo_cisajs_001.dat"), String), "\n"))
        @test length(lines) == 3
        @test split(lines[1])[1:4] == ["0", "0", "1", "0"]
        @test split(lines[2])[1:4] == ["0", "0", "1", "0"]
        @test split(lines[3])[1:4] == ["1", "1", "0", "1"]
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: FAIL — `output_green_func!` writes `zvo_cisajs` from `data.green_one_terms` (empty here), so `zvo_cisajs_001.dat` either is absent or has no `0 0 1 0` row.

- [ ] **Step 3: Write `zvo_cisajs` from the canonical list**

In `MVMCOptimizers.jl/src/data_io.jl`, replace the `zvo_cisajs` block (lines 244-267):

```julia
    # zvo_cisajs_XXX.dat (1-body Green's function)
    if !isempty(data.green_one_terms)
        filename = _output_path(@sprintf("%s_cisajs_%03d.dat", data_file_head, ismp), output_dir)
        open(filename, "w") do f
            for (idx, term) in enumerate(data.green_one_terms)
                val = phys.phys_cis_ajs[idx]
                # C format: "%d %d %d %d % .18e  % .18e \n"
                # Format: ri, si, rj, sj, real, imag
                si = term.spin1 == :up ? 0 : 1
                sj = term.spin2 == :up ? 0 : 1
                @printf(
                    f,
                    "%d %d %d %d % .18e  % .18e \n",
                    term.site1,
                    si,
                    term.site2,
                    sj,
                    real(val),
                    imag(val)
                )
            end
            println(f)  # Empty line at end (C format)
        end
    end
```
with the canonical-list version:
```julia
    # zvo_cisajs_XXX.dat (1-body Green's function) — written from the canonical
    # one-body list (which, with TwoBodyGEx, includes appended factored
    # constituents in C order), not raw data.green_one_terms.
    if !isempty(phys.cis_ajs_idx)
        filename = _output_path(@sprintf("%s_cisajs_%03d.dat", data_file_head, ismp), output_dir)
        open(filename, "w") do f
            for (idx, (ri, si, rj, sj)) in enumerate(phys.cis_ajs_idx)
                val = phys.phys_cis_ajs[idx]
                # C format: "%d %d %d %d % .18e  % .18e \n" (ri, si, rj, sj, real, imag)
                @printf(
                    f,
                    "%d %d %d %d % .18e  % .18e \n",
                    ri,
                    si,
                    rj,
                    sj,
                    real(val),
                    imag(val)
                )
            end
            println(f)  # Empty line at end (C format)
        end
    end
```

- [ ] **Step 4: Thread `output_dir` through `output_data_phys!`**

In `MVMCOptimizers.jl/src/vmc_phys_cal.jl`, change `output_data_phys!`:

```julia
function output_data_phys!(data::ExpertModeData, state::VMCOptimizationState, ismp::Int)
    # Output energy and parameters (same as optimization mode)
    output_data!(data, state, ismp)

    # Output Green's functions
    if state.phys_quantities !== nothing
        output_green_func!(data, state, ismp)
    end
end
```
to:
```julia
function output_data_phys!(
    data::ExpertModeData,
    state::VMCOptimizationState,
    ismp::Int;
    output_dir::Union{String,Nothing} = nothing,
)
    # Output energy and parameters (same as optimization mode)
    output_data!(data, state, ismp; output_dir = output_dir)

    # Output Green's functions
    if state.phys_quantities !== nothing
        output_green_func!(data, state, ismp; output_dir = output_dir)
    end
end
```

- [ ] **Step 5: Add `output_dir` + C file numbering to `vmc_phys_cal!`**

In `MVMCOptimizers.jl/src/vmc_phys_cal.jl`, change the signature:
```julia
function vmc_phys_cal!(
    data::ExpertModeData;
    callback::Union{Nothing,Function} = nothing,
    rng::Union{AbstractRNG,Nothing} = nothing,
)::Int
```
to:
```julia
function vmc_phys_cal!(
    data::ExpertModeData;
    callback::Union{Nothing,Function} = nothing,
    rng::Union{AbstractRNG,Nothing} = nothing,
    output_dir::Union{String,Nothing} = nothing,
)::Int
```

Add a small, unit-testable helper near `output_data_phys!` in `vmc_phys_cal.jl`
(Plan Review Finding 1) so the numbering logic is covered through the changed
path, not only the direct writer:
```julia
"""
    physcal_output_file_index(data::ExpertModeData, ismp::Int) -> Int

C-visible PhysCal per-sampling file index: `ismp + NDataIdxStart`
(so the usual first files are `*_001.dat`).
"""
physcal_output_file_index(data::ExpertModeData, ismp::Int)::Int =
    ismp + data.modpara.n_data_idx_start
```

Then change the output call inside the sample loop:
```julia
        # Output data
        output_data_phys!(data, state, ismp)
```
to:
```julia
        # Output data. C names per-sampling files with idx = ismp + NDataIdxStart.
        file_idx = physcal_output_file_index(data, ismp)
        output_data_phys!(data, state, file_idx; output_dir = output_dir)
```

- [ ] **Step 6: Run test to verify it passes**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: PASS — `zvo_cisajs_001.dat` has the canonical `0 0 1 0` row and `zvo_cisajscktaltex_001.dat` has `2.5 -1.0`.

- [ ] **Step 7: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCOptimizers.jl/src/data_io.jl MVMCOptimizers.jl/src/vmc_phys_cal.jl MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl
git commit -m "feat(physcal): canonical cisajs output, output_dir plumbing, C file numbering"
```

---

## Task 5: FSZ runtime guard

**Files:**
- Modify: `MVMCOptimizers.jl/src/green_func_calc.jl` (add `validate_factored_green_supported`)
- Modify: `MVMCOptimizers.jl/src/vmc_phys_cal.jl` (call the guard early)
- Test: `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl` (extend)

- [ ] **Step 1: Write the failing test**

Append to `MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`:

```julia
@testset "FSZ + factored is rejected before sampling" begin
    data = ExpertModeData()
    data.green_two_ex_terms = [GreenTwoExTerm(0, 0, 1, 0, 2, 1, 3, 1)]

    # sz-conserved (i_flg_orbital_general == 0): allowed.
    data.i_flg_orbital_general = 0
    @test MVMCOptimizers.validate_factored_green_supported(data) === nothing

    # FSZ / general-orbital (== 1): rejected with a clear message.
    data.i_flg_orbital_general = 1
    threw = false
    msg = ""
    try
        MVMCOptimizers.validate_factored_green_supported(data)
    catch err
        threw = true
        msg = sprint(showerror, err)
    end
    @test threw
    @test occursin("TwoBodyGEx", msg)
    @test occursin("FSZ", msg)

    # No factored terms: FSZ is fine for the rest of PhysCal.
    data.green_two_ex_terms = GreenTwoExTerm[]
    @test MVMCOptimizers.validate_factored_green_supported(data) === nothing
end

@testset "vmc_phys_cal! rejects FSZ + factored before sampling" begin
    # The guard runs before RNG / init_parameter!, so a minimal (physically
    # incomplete) input still fails for the intended reason (Finding 2).
    data = ExpertModeData()
    data.green_two_ex_terms = [GreenTwoExTerm(0, 0, 1, 0, 2, 1, 3, 1)]
    data.i_flg_orbital_general = 1

    err = try
        MVMCOptimizers.vmc_phys_cal!(data)
        nothing
    catch e
        e
    end

    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("TwoBodyGEx", msg)
    @test occursin("FSZ", msg)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: FAIL — `validate_factored_green_supported` not defined.

- [ ] **Step 3: Add the guard function**

In `MVMCOptimizers.jl/src/green_func_calc.jl`, add (near `initialize_phys_quantities!`):

```julia
"""
    validate_factored_green_supported(data::ExpertModeData)

Reject the factored two-body Green (`TwoBodyGEx`) in FSZ / general-orbital mode.
The FSZ measurement path is not yet wired for Green functions (a separate spec),
so producing factored output there would be silently wrong. Non-mutating;
returns `nothing` when supported.
"""
function validate_factored_green_supported(data::ExpertModeData)
    if !isempty(data.green_two_ex_terms) && data.i_flg_orbital_general != 0
        error(
            "TwoBodyGEx (factored two-body Green) is not supported in FSZ / " *
            "general-orbital mode (i_flg_orbital_general = $(data.i_flg_orbital_general)). " *
            "The FSZ Green measurement path is not yet implemented; use sz-conserved " *
            "mode or remove the TwoBodyGEx input.",
        )
    end
    return nothing
end
```

- [ ] **Step 4: Call the guard at the very top of `vmc_phys_cal!`**

In `MVMCOptimizers.jl/src/vmc_phys_cal.jl`, place the guard immediately after the
existing `validate_supported_modpara(data.modpara)` call — **before** RNG setup
and `init_parameter!` — so an unsupported FSZ + `TwoBodyGEx` input fails before
any RNG / parameter side effects (Plan Review Finding 2). Change:
```julia
    # Reject unsupported ModPara inputs (e.g. NSplitSize > 1) before any work.
    validate_supported_modpara(data.modpara)
```
to:
```julia
    # Reject unsupported ModPara inputs (e.g. NSplitSize > 1) before any work.
    validate_supported_modpara(data.modpara)
    # Reject TwoBodyGEx in FSZ / general-orbital mode before any sampling or RNG
    # side effects (its Green measurement path is not yet wired).
    validate_factored_green_supported(data)
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'include("test_unit/test_unit_physcal_factored_green.jl")'
```
Expected: PASS — the FSZ-guard testset passes.

- [ ] **Step 6: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCOptimizers.jl/src/green_func_calc.jl MVMCOptimizers.jl/src/vmc_phys_cal.jl MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl
git commit -m "feat(physcal): reject TwoBodyGEx in FSZ mode before sampling"
```

---

## Task 6: Register tests + full subpackage suite

**Files:**
- Modify: `MVMCOptimizers.jl/test/runtests.jl` (after the `test_unit_unsupported_inputs.jl` include, line 71)
- Modify: `MVMCOptimizers.jl/test_unit/INDEX.md` (document the new tests)

- [ ] **Step 1: Register the new test file**

In `MVMCOptimizers.jl/test/runtests.jl`, after:
```julia
    include("../test_unit/test_unit_unsupported_inputs.jl")
```
add:
```julia
    include("../test_unit/test_unit_physcal_factored_green.jl")
```

- [ ] **Step 2: Add the INDEX.md entry**

In `MVMCOptimizers.jl/test_unit/INDEX.md`, add a section (mirroring the existing format):

```markdown
### `src/green_func_calc.jl` + `vmc_phys_cal.jl`（factored two-body Green）
- `test_unit/test_unit_physcal_factored_green.jl`
  - `PhysicalQuantities index fields` → `cis_ajs_idx` / `cis_ajs_ckt_alt_idx`
  - `canonical one-body list` → `build_canonical_cis_ajs_idx`（greenone 先頭 → greentwoex 構成 append・dedup・site 範囲）
  - `factored index resolution is 1-based` → `resolve_cis_ajs_ckt_alt_idx`（C index 0 → Julia 1）
  - `factored accumulation` → `accumulate_factored_green!`（`w·local[idx0]·conj(local[idx1])`）
  - `output: canonical cisajs + factored ex` → `output_green_func!`（canonical 出力 / output_dir / `_001` 番号）
  - `FSZ + factored is rejected` → `validate_factored_green_supported`
```

- [ ] **Step 3: Run the full subpackage test suite**

Run:
```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC/MVMCOptimizers.jl
OMP_NUM_THREADS=1 julia --project=@. -e 'using Pkg; Pkg.test()'
```
Expected: PASS — the whole `MVMCOptimizers.jl` unit suite passes (including the new factored-Green tests) with no regressions. (Integration tests against the C reference are gated by an environment variable and are not run here; they are Plan 3.)

- [ ] **Step 4: Commit**

```bash
cd /Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git add MVMCOptimizers.jl/test/runtests.jl MVMCOptimizers.jl/test_unit/INDEX.md
git commit -m "test(physcal): register factored two-body Green unit tests"
```

---

## Done criteria for Plan 2

- With `TwoBodyGEx` absent, PhysCal behavior is byte-identical to before (canonical list == `greenone.def` order, no dedup; factored arrays empty; `zvo_cisajs` unchanged).
- With `TwoBodyGEx` present: the canonical one-body list is C-compatible (explicit-then-appended, de-duplicated), factored pairs are 1-based, `accumulate_factored_green!` matches C's `w·local[idx0]·conj(local[idx1])`, `zvo_cisajscktaltex_*` is populated, and `zvo_cisajs_*` reflects the extended list.
- FSZ + `TwoBodyGEx` errors before sampling.
- PhysCal per-sampling files are numbered `ismp + NDataIdxStart` and honor `output_dir`.
- `MVMCOptimizers.jl` unit `Pkg.test()` passes.

**Not in this plan (Plan 3):** C reference fixtures, comparison helpers (6/10/value-only columns, rtol+atol), the PhysCal runner (`run_phys_cal_from_namelist` / `read_opt_para_file!`), the end-to-end gate, and docs.

---

## Self-Review

**Spec coverage (Plan 2 slice):**
- Finding 1 (canonical one-body list) → Task 2 `build_canonical_cis_ajs_idx` (explicit-then-appended, dedup only when `TwoBodyGEx` present — faithful to C's `IndirectGFOn`); Task 4 writes `zvo_cisajs` from it. ✓
- Finding 2 (1-based indices) → Task 2 `resolve_cis_ajs_ckt_alt_idx` (1-based; C-index-0 → Julia-1 test); Task 3 accumulator indexes `local_cis_ajs` 1-based. ✓
- Finding 5 (FSZ guard) → Task 5 `validate_factored_green_supported`, called early in `vmc_phys_cal!`. ✓
- Finding 6 (output numbering + output_dir) → Task 4 `ismp + n_data_idx_start` and `output_dir` through `vmc_phys_cal!` → `output_data_phys!` → `output_green_func!`/`output_data!`. ✓
- Finding 9 (C-only behavior) → Task 2 test includes a factored constituent absent from `greenone.def`, asserting the extended canonical order. ✓
- Spec §4 product formula (`conj` on the second factor) → Task 3 accumulator + hand-computed test. ✓

**Plan Review Findings (applied in the tasks):**
- PR Finding 1 (numbering tested through the changed path) → Task 4 adds `physcal_output_file_index` + a `NDataIdxStart` contract test, alongside the direct writer-format test. ✓
- PR Finding 2 (FSZ guard before side effects + public path) → Task 5 places the guard right after `validate_supported_modpara` (before RNG / `init_parameter!`) and adds a `vmc_phys_cal!` public-path test. ✓
- PR Finding 3 (no-`TwoBodyGEx` regression) → Task 4 adds a test that runs `initialize_phys_quantities!` + `output_green_func!` and asserts `greenone.def` order **and duplicates** are preserved. ✓
- PR Finding 4 (`_spin_int` strictness) → Task 2 makes `_spin_int` error on non-`:up`/`:down` symbols, with a test. ✓

**Placeholder scan:** No TBD/"handle errors"/"similar to"; every step has full code and exact commands. ✓

**Type consistency:** Canonical entries are `NTuple{4,Int}` `(ri,si,rj,sj)` everywhere (struct field, builder, resolver, one-body loop, output). `cis_ajs_ckt_alt_idx::Vector{Tuple{Int,Int}}` 1-based throughout. `accumulate_factored_green!(phys, w::Float64)`, `build_canonical_cis_ajs_idx(green_one_terms, green_two_ex_terms, n_site)`, `resolve_cis_ajs_ckt_alt_idx(cis_ajs_idx, green_two_ex_terms)`, `validate_factored_green_supported(data)` names are consistent across tasks and tests. `GreenTwoExTerm` field names match Plan 1. ✓
