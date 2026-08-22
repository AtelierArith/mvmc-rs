# Julia-mVMC PhysCal: Factored Two-Body Green + Reference Fixtures Design

## Scope

This design covers the next v0.3 PhysCal step: porting the **factored /
product-side two-body Green function** path from C-mVMC and adding
**committed C reference fixtures** plus an **end-to-end PhysCal gate** that
protects all PhysCal outputs (one-body, direct DC two-body, and factored
two-body) against regressions.

The goal is faithful reproduction of C-mVMC. The factored path mirrors
C's `CalculateGreenFunc()` (`calgrn.c`) exactly: the same index mapping,
the same product formula, and the same output format.

This corresponds to roadmap item P0 "PhysCal を実用計算レベルへ上げる"
(`docs/plans/2026-06-03-julia-mvmc-v0.3-roadmap.md`), scoped to choice (B)
agreed with the user: factored port + PhysCal-wide reference fixtures +
end-to-end gate.

## Goals

- Port the factored two-body Green function (`TwoBodyGEx` / `greentwoex.def`
  → `zvo_cisajscktaltex_*.dat`) faithfully from C-mVMC.
- Resolve factored terms to index pairs into the one-body Green list,
  mirroring C's `CisAjsCktAltIdx` / `iOneBodyGIdx` indexing.
- Add committed C reference fixtures for PhysCal across four non-FSZ
  systems and compare Julia output bit-for-bit (tight-tolerance fallback).
- Add unit/contract tests that pin the parser, index resolution, product
  formula, and output format.
- Update the manual to reflect that the factored path is now supported
  (non-FSZ).

## Non-Goals

- FSZ measurement correctness: wiring the FSZ Green functions
  (`green_func1_fsz` / `green_func2_fsz*`) into the PhysCal measurement
  path. This is a separate, pre-existing gap (TODO at
  `vmc_main_cal.jl:2998`) and is deferred to its own spec. The factored
  path is therefore not verified or claimed for FSZ inputs; instead a
  runtime guard rejects FSZ + factored runs (Review Finding 5) so no wrong
  output is produced in the meantime.
- Full removal of all "experimental" warnings (the broader roadmap item D).
- BackFlow, MPI parallelization, and the post-Lanczos pipeline.
- Re-running optimization inside the gate. Optimization regression is
  already covered by the existing energy/parameter integration tests; the
  new gate consumes the committed `zqp_opt.dat` as a fixed input.

## Background: Current State

C-mVMC computes two distinct two-body Green functions during PhysCal:

| Green function | C keyword / input | Method | Julia status |
|---|---|---|---|
| One-body `⟨c†_i c_j⟩` | `OneBodyG` / greenone.def | `GreenFunc1` | implemented |
| Two-body direct (DC) `⟨c†_i c_j c†_k c_l⟩` | `TwoBodyG` / greentwo.def | `GreenFunc2` (4-operator) | implemented |
| Two-body factored `⟨c†_i c_j⟩·⟨c†_k c_l⟩` | `TwoBodyGEx` / greentwoex.def | product of one-body Greens | **missing** |

Much of the Julia plumbing for the factored path already exists but is
inert:

- `PhysicalQuantities.phys_cis_ajs_ckt_alt::Vector{ComplexF64}` exists but
  is allocated with length 0 (`green_func_calc.jl:19` hardcodes
  `n_cis_ajs_ckt_alt = 0`).
- `local_cis_ajs` (per-sample one-body Greens) is already computed every
  sample in `calculate_green_func!` (`green_func_calc.jl:101`).
- The output writer `zvo_cisajscktaltex_%03d.dat` already exists and uses
  C's value-only format (`data_io.jl:269-278`); it is gated on the array
  being non-empty.
- Weight averaging already normalizes `phys_cis_ajs_ckt_alt`
  (`weight_average.jl`).
- Output basenames already match C: `zvo_cisajs` (one-body),
  `zvo_cisajscktalt` (direct DC, indexed), and `zvo_cisajscktaltex`
  (factored, values only). The C-compatible sampling index
  (`NDataIdxStart`, usually `_001`) and `output_dir` support still need the
  fixes described in Review Finding 6.

What is missing:

1. Parser: `TwoBodyGEx` dispatch and a `greentwoex.def` reader. The
   keyword exists in `constants.jl:39` but there is no `elseif` branch in
   `MVMCExpertModeParsers.jl:516`.
2. Index resolution: converting factored terms into index pairs
   `(idx0, idx1)` into the one-body Green list.
3. Sizing: `n_cis_ajs_ckt_alt` set from the resolved count instead of 0.
4. Accumulation: the empty product loop at `green_func_calc.jl:140-143`.
5. Fixtures, tests, the e2e gate, and docs.

## Design Review Findings and Required Revisions

The review below was done against the current Julia implementation and the
C reference points in `mVMC/src/mVMC/readdef.c`, `calgrn.c`, `initfile.c`,
and `vmcmain.c`. These are required design fixes before implementation, not
optional follow-up cleanup.

### 1. C-compatible one-body canonicalization

Review result: the current plan says a factored term that references a
one-body Green absent from `green_one_terms` should error. This is not C's
behavior. When `TwoBodyGEx` is present, C runs `CountOneBodyGForLanczos()`:
it starts from the explicit `OneBodyG` terms, then appends any missing
constituent one-body terms from `TwoBodyGEx`, de-duplicated by
`iOneBodyGIdx`. `GetInfoOneBodyG()` and `GetInfoTwoBodyGEx()` then fill the
canonical `CisAjsIdx` array in that resolved order.

Required action:

- Build a C-compatible canonical one-body Green list when
  `green_two_ex_terms` is non-empty.
- Preserve this order: explicit `greenone.def` terms first, then for each
  `greentwoex.def` row append the first constituent
  `(x0,x1,x2,x3)` if missing, then the reordered second constituent
  `(x6,x7,x4,x5)` if missing.
- Allocate `local_cis_ajs` / `phys_cis_ajs` and write `zvo_cisajs_*` from
  this canonical list, not only from the raw `green_one_terms`.
- Resolve `cis_ajs_ckt_alt_idx` against this canonical lookup.

Bug if skipped: hand-authored `greentwoex.def` files that C accepts can fail
in Julia, or worse, Julia can produce a `zvo_cisajs_*` file with a different
length/order than C.

### 2. Julia/C index-base boundary

Review result: C stores `CisAjsCktAltIdx` as 0-based indices into
`LocalCisAjs`, while Julia arrays are 1-based. The draft accumulation loop
uses resolved indices directly.

Required action:

- Decide and document one representation:
  - store `cis_ajs_ckt_alt_idx` as Julia 1-based indices, or
  - store C 0-based indices and add `+1` at every Julia array access.
- Prefer storing Julia indices and naming tests around that contract.
- Add a unit test that resolves C index `0` and confirms accumulation reads
  `local_cis_ajs[1]`, preventing an off-by-one regression.

### 3. Count source is the `greentwoex.def` header, not `modpara.def`

Review result: the draft says to compare resolved count against modpara
`NTwoBodyGEx`. In C, `IdxNTwoBodyGEx` is populated by reading the second
line of the `TwoBodyGEx` definition file listed in `namelist.def` via
`ReadBuffInt()`. Normal `modpara.def` files do not carry `NTwoBodyGEx`.

Required action:

- Make `parse_green_two_ex_def` read and validate the `greentwoex.def`
  header count.
- Compare `length(green_two_ex_terms)` and
  `length(cis_ajs_ckt_alt_idx)` against that header count.
- Do not rely on `data.modpara.n_two_body_g_ex` unless a future parser
  explicitly populates it from the Green definition header.

Bug if skipped: the count check can become a false mismatch for normal
fixtures, or silently do nothing if the modpara default remains zero.

### 4. Parser strictness must be explicit

Review result: the draft says malformed `greentwoex.def` lines should be a
parse error "same policy as `parse_green_two_def`", but the existing
`parse_green_two_def` is permissive: short rows can be skipped and parser
exceptions are often downgraded to warnings by `parse_expert_mode_files`.

Required action:

- Define `parse_green_two_ex_def` as stricter than the current
  `parse_green_two_def`: each data row must contain exactly the expected
  eight integer fields after comment stripping, and failures must surface to
  the caller.
- If C compatibility is preferred over strictness, explicitly document the
  chosen difference, because C's `sscanf` reads the first eight integers and
  does not robustly validate conversion count.
- In `parse_expert_mode_files`, a failed `TwoBodyGEx` parse must not
  silently continue into PhysCal with an empty factored list. Either make
  this parse fatal or add an explicit post-parse validation when the namelist
  contains `TwoBodyGEx`.

### 5. FSZ must be rejected until its Green path is wired

Review result: the spec excludes FSZ correctness, but current
`vmc_main_cal_fsz!` still calls the standard `calculate_green_func!` in
PhysCal mode. Once `phys_cis_ajs_ckt_alt` becomes non-empty, FSZ inputs can
produce plausible but wrong factored outputs.

Required action:

- Add an early guard: if `green_two_ex_terms` is non-empty and the run is in
  FSZ/general-orbital mode, return a clear error before sampling.
- Keep the documentation caveat, but do not merely document the limitation;
  enforce it at runtime until `green_func1_fsz` / `green_func2_fsz*` are
  correctly connected to PhysCal measurement.

### 6. Output file numbering and output directory handling

Review result: the output basenames match C, but the full filenames do not
currently match. C PhysCal writes with `idx = ismp + NDataIdxStart`, so the
usual first files are `zvo_cisajs_001.dat`,
`zvo_cisajscktalt_001.dat`, and `zvo_cisajscktaltex_001.dat`.
Julia currently passes `ismp = 0` directly to `output_green_func!`, producing
`*_000.dat`. Also, `vmc_phys_cal!` has no `output_dir` argument, so tests
that run from a reference directory can overwrite committed fixtures.

Required action:

- Make PhysCal output use the C-visible sampling index
  `ismp + data.modpara.n_data_idx_start` for Green files and other
  per-sampling PhysCal files, or make the comparison helper explicitly map
  Julia's current names to C names. Prefer fixing the writer for C
  compatibility.
- Add `output_dir` support to `vmc_phys_cal!` / `output_data_phys!` /
  `output_green_func!`, mirroring the optimization entry point.
- The PhysCal e2e gate must run in a fresh temporary output directory and
  compare against committed references read-only.

### 7. Fixed `zqp_opt.dat` consumption must be specified

Review result: the fixture plan says the gate consumes committed
`zqp_opt.dat`, but the current PhysCal entry point does not define how that
file is loaded. Without an explicit loader path, Julia can accidentally run
with random/default or only `In*.def` parameters.

Required action:

- Add a PhysCal runner/helper, e.g. `run_phys_cal_from_namelist`, that mirrors
  C's initialization order:
  `init_parameter!` -> load fixed opt/initial parameter file ->
  `read_input_parameters!` -> `sync_modified_parameter!` ->
  `init_qp_weight!` -> `vmc_phys_cal!`.
- Reuse `read_initial_def!` only if the committed `zqp_opt.dat` layout is
  verified to match its expected layout. Otherwise add a dedicated
  `read_opt_para_file!` with tests.
- The gate should assert that the fixed parameter file was actually consumed
  and should fail on missing or malformed parameter data.

### 8. Green-file comparison format and tolerances

Review result: the draft says the one-body Green file is "10-column", but C
writes one-body as six columns:
`ri si rj sj real imag`. C writes direct two-body as ten columns:
`ri si rj sj rk sk rl sl real imag`. Factored output is value-only
`real imag` pairs on a single line.

Required action:

- Implement comparison helpers with the correct formats:
  - one-body: indexed, six columns;
  - direct two-body: indexed, ten columns;
  - factored: value-only, two floats per requested factored term.
- Compare exact strings first only after filename/indexing and formatting are
  made C-compatible.
- For numeric fallback, use both `rtol` and `atol` (for example
  `rtol = 1e-10`, `atol = 1e-12`) so near-zero Green values do not produce
  misleading relative-error failures.
- Record any fallback per file and quantity in the test output.

### 9. Fixture design should exercise the C-only behavior

Review result: hand-extending `greenone.def` so all factored constituents are
already a subset is valid for simple fixtures, but it will not test the
important C behavior where `TwoBodyGEx` adds missing one-body measurement
targets.

Required action:

- Keep the committed fixtures small, but include at least one unit or
  contract test where `greentwoex.def` references a one-body constituent not
  present in the explicit `greenone.def`.
- Confirm that Julia extends the canonical one-body list in the same order as
  C and that `zvo_cisajs_*` reflects the extended list.

## Proposed Changes

### 1. Input format (`greentwoex.def` / `TwoBodyGEx`)

Faithful to C's `GetInfoTwoBodyGEx` (`readdef.c`): one line = 8 integers
`x0 x1 x2 x3 x4 x5 x6 x7`.

| Columns | Meaning |
|---|---|
| x0 x1 | first one-body Green: **creation** (site, spin) |
| x2 x3 | first one-body Green: **annihilation** (site, spin) |
| x4 x5 | second one-body Green: **annihilation** (site, spin) |
| x6 x7 | second one-body Green: **creation** (site, spin) |

So the first one-body Green is `⟨c†_{x0,x1} c_{x2,x3}⟩` and the second is
`⟨c†_{x6,x7} c_{x4,x5}⟩`. The reorder of the second Green (creation taken
from x6,x7 and annihilation from x4,x5) reproduces C's
`isite1 = x6 + x7·Nsite`, `isite2 = x4 + x5·Nsite`.

### 2. Data structures (new)

In `MVMCExpertModeParsers.jl`:

```julia
struct GreenTwoExTerm           # two one-body Greens (C reorder absorbed)
    site_i1::Int; spin_i1::Int; site_j1::Int; spin_j1::Int   # ⟨c†_{i1} c_{j1}⟩ = (x0,x1,x2,x3)
    site_i2::Int; spin_i2::Int; site_j2::Int; spin_j2::Int   # ⟨c†_{i2} c_{j2}⟩ = (x6,x7,x4,x5)
end
```

- Add `green_two_ex_terms::Vector{GreenTwoExTerm}` to `ExpertModeData`
  (default empty).
- Add `parse_green_two_ex_def(path)` (structured like `parse_green_two_def`
  but stricter — see Finding 4) and a `TwoBodyGEx` dispatch branch near
  `MVMCExpertModeParsers.jl:516`.

In `MVMCOptimizers.jl` `PhysicalQuantities`:

- Add `cis_ajs_ckt_alt_idx::Vector{Tuple{Int,Int}}` to hold the resolved
  index pairs.

### 3. Canonical one-body list and index resolution (Review Findings 1, 2)

Resolution happens in `initialize_phys_quantities!(state, data)` at init, not
at parse, so it does not depend on the order of files in `namelist.def`; it
runs once after both `green_one_terms` and `green_two_ex_terms` are available,
matching C building the index map in `readdef` after all Green definitions are
read.

1. Build the **canonical one-body Green list**, C-compatible
   (`CountOneBodyGForLanczos` + `GetInfoOneBodyG` / `GetInfoTwoBodyGEx`):
   - start from the explicit `green_one_terms` (greenone.def order);
   - then, for each `GreenTwoExTerm` in file order, append the first
     constituent `(x0,x1,x2,x3)` if not already present, then the reordered
     second constituent `(x6,x7,x4,x5)` if not already present,
     de-duplicated by the key `(site_c, spin_c, site_a, spin_a)`.

   When `green_two_ex_terms` is empty, this list equals `green_one_terms`.
2. Allocate `local_cis_ajs` / `phys_cis_ajs` and emit `zvo_cisajs_*` from this
   **canonical list**, not the raw `green_one_terms`. This is what makes a
   hand-authored `greentwoex.def` whose constituents are not all in
   `greenone.def` behave like C (same `zvo_cisajs_*` length and order).
3. Build a lookup from the canonical list: key
   `(site_c, spin_c, site_a, spin_a)::NTuple{4,Int}` → **Julia 1-based**
   index. Spins are converted Symbol→Int (`:up`→0, `:down`→1).
4. Resolve each `GreenTwoExTerm` to a **1-based** `(idx0, idx1)`:
   - `idx0 = lookup[(site_i1, spin_i1, site_j1, spin_j1)]`
   - `idx1 = lookup[(site_i2, spin_i2, site_j2, spin_j2)]`
5. Set `n_cis_ajs_ckt_alt = length(green_two_ex_terms)` and store the pairs in
   `phys.cis_ajs_ckt_alt_idx`.

`cis_ajs_ckt_alt_idx` stores **Julia 1-based** indices (Finding 2): C's 0-based
`CisAjsCktAltIdx` value `0` must read `local_cis_ajs[1]`. Because the canonical
append guarantees every constituent is present, a lookup miss is an internal
invariant error, not a user-facing contract error.

### 4. Runtime accumulation

Fill the empty loop at `green_func_calc.jl:140-143`, faithful to
`calgrn.c:113-118`:

```julia
for (idx, (idx0, idx1)) in enumerate(phys.cis_ajs_ckt_alt_idx)
    phys.phys_cis_ajs_ckt_alt[idx] +=
        w * phys.local_cis_ajs[idx0] * conj(phys.local_cis_ajs[idx1])
end
```

The one-body Greens are already in `local_cis_ajs` (computed earlier in the
same function), so there is no double computation. The `conj` placement and
the multiplication order match C exactly to preserve bit-level agreement.

### 5. Output: file numbering and output directory (Review Finding 6)

The factored writer body (`data_io.jl:269-278`) already emits C's value-only
`% .18e` format, but two output-path issues must be fixed:

- **File numbering.** C PhysCal writes file index `ismp + NDataIdxStart`, so
  the usual first files are `*_001.dat`. Julia passes `ismp = 0` directly
  (`vmc_phys_cal.jl:169`, `:248`, `:290`), producing `*_000.dat`. Fix the
  writer to use `ismp + data.modpara.n_data_idx_start` for all per-sampling
  PhysCal files (Green files included).
- **Output directory.** Add `output_dir` support to `vmc_phys_cal!`,
  `output_data_phys!`, and `output_green_func!`, mirroring `vmc_para_opt!`.
  The e2e gate runs in a fresh temporary directory and reads committed
  references read-only, so it can never overwrite a fixture.

The factored writer activates once `phys_cis_ajs_ckt_alt` is non-empty.

### 6. Parser strictness and count validation (Review Findings 3, 4)

- `parse_green_two_ex_def` reads the `greentwoex.def` **header count** (the
  count line of the definition file, as C does via `ReadBuffInt`) and is
  **strict**: each data row must contain exactly eight integer fields after
  comment stripping; a malformed or short row fails to the caller rather than
  being silently skipped. This is intentionally stricter than the permissive
  `parse_green_two_def`, and the difference is documented.
- Validate `length(green_two_ex_terms)` against the header count. Do **not**
  use `modpara.n_two_body_g_ex`: normal `modpara.def` files do not carry
  `NTwoBodyGEx` and its default is `0` (Finding 3).
- In `parse_expert_mode_files`, a failed `TwoBodyGEx` parse must not silently
  continue into PhysCal with an empty factored list. When the namelist
  contains `TwoBodyGEx`, either make the parse fatal or add an explicit
  post-parse validation.
- A canonical-list lookup miss during resolution is an internal invariant
  error (it should not happen after the Finding 1 canonical append), not a
  user-facing "missing one-body term" error.

### 7. FSZ runtime guard (Review Finding 5)

FSZ measurement correctness is out of scope, but `vmc_main_cal_fsz!` still
calls the standard `calculate_green_func!` (`vmc_main_cal.jl:2998`). To prevent
plausible-but-wrong factored output, add an **early guard**: if
`green_two_ex_terms` is non-empty and the run is in FSZ / general-orbital mode,
raise a clear error before sampling. The limitation is enforced at runtime
(not merely documented) until the FSZ Green path is wired in a later spec.

### 8. PhysCal runner with fixed parameters (Review Finding 7)

Add `run_phys_cal_from_namelist` (mirroring `run_para_opt_from_namelist`) that
loads a fixed optimized-parameter file and runs PhysCal in C's initialization
order:

```
init_parameter! → load fixed opt parameter file → read_input_parameters!
  → sync_modified_parameter! → init_qp_weight! → vmc_phys_cal!
```

`read_initial_def!` does **not** add perturbations itself, but it was written
for the optimization warm-start `initial.def` path; C's test driver often
creates that `initial.def` from `zqp_opt.dat` plus an external
`random.uniform(-0.01, 0.01)` perturbation. A fixed-parameter PhysCal gate must
load the committed unperturbed opt parameter file. Add a dedicated
`read_opt_para_file!` with tests, or reuse `read_initial_def!` only after
verifying the committed `zqp_opt.dat` layout matches its loader contract. The
gate asserts the fixed parameters were actually consumed and fails on
missing/malformed data.

## Testing

### Unit / contract (subpackage, all CI matrix, no C, ~1s)

1. **Parser unit** — `parse_green_two_ex_def` reads the header count, parses 8
   integer fields, applies the `x6,x7→x4,x5` reorder for the second Green, and
   **rejects** malformed/short rows to the caller (Finding 4). Inline fixtures.
2. **Canonical one-body list contract** (Findings 1, 9) — greenone terms come
   first, then `greentwoex.def` constituents are appended in file order,
   de-duplicated. Includes a case where `greentwoex.def` references a one-body
   constituent **not** present in the explicit `greenone.def`, and asserts the
   extended canonical list (hence `zvo_cisajs_*`) matches C order and length.
3. **Index resolution contract** (Finding 2) — resolves to the correct
   **1-based** `(idx0, idx1)`; a term whose C index is `0` must read
   `local_cis_ajs[1]`, guarding against an off-by-one regression.
4. **Product accumulation contract** — known `local_cis_ajs` values and index
   pairs produce `w·local[idx0]·conj(local[idx1])`, checked against
   hand-computed values. Pins the conj placement and the weight.
5. **Output format/numbering contract** (Finding 6) — the factored writer
   produces C's value-only `% .18e` format (golden string), and per-sampling
   files are numbered `ismp + n_data_idx_start` (first file `*_001.dat` when
   `NDataIdxStart = 1`).
6. **FSZ guard contract** (Finding 5) — an FSZ / general-orbital run with a
   non-empty `green_two_ex_terms` raises a clear error before sampling.

### Integration (root, committed C reference, selected CI matrix, env toggle)

7. **PhysCal end-to-end gate** for four systems
   (`heisenberg_chain_real`, `heisenberg_chain_cmp`, `hubbard_chain_real`,
   `kondo_chain_real`): via `run_phys_cal_from_namelist` (Finding 7), load the
   committed fixed parameters, run PhysCal in a fresh temp directory, produce
   `zvo_cisajs` / `zvo_cisajscktalt` (DC) / `zvo_cisajscktaltex` (factored),
   and compare against the committed C reference. The gate asserts the fixed
   parameters were actually consumed (fails on missing/malformed data).

Using the committed fixed parameter file decouples the Green computation from
optimization and keeps the comparison deterministic. No C binary runs at test
time; committed references are read-only.

### Acceptance criterion (Review Finding 8)

- Assert bit-level (exact string) equality first, only after filename,
  numbering, and formatting are made C-compatible.
- For numeric fallback, use **both** `rtol = 1e-10` and `atol = 1e-12`, so
  near-zero Green values do not produce misleading relative-error failures.
  Record any fallback per file and quantity in the test output.
- Comparison helpers use the correct C formats:
  - one-body: indexed, **six** columns `ri si rj sj real imag`;
  - direct two-body: indexed, **ten** columns
    `ri si rj sj rk sk rl sl real imag`;
  - factored: value-only, two floats per requested factored term.

## Fixture Generation (offline, once, committed)

For each of the four systems:

- Hand-construct `greentwoex.def` (a small set of factored pairs) and add a
  `TwoBodyGEx greentwoex.def` line to `namelist.def`. StdFace does not emit
  `greentwoex.def` automatically, so it is authored by hand and kept small and
  well-defined.
- Run C-mVMC with `OMP_NUM_THREADS=1`: (a) optimize to produce the fixed
  parameter file (`zqp_opt.dat`), then (b) run NVMCCalMode=1 to produce the
  three Green outputs (with C's `*_001.dat` numbering, per Finding 6).
- Commit the C reference outputs and input defs under
  `test/integration/reference/<system>/`.
- Record provenance (mVMC commit, build flags, `OMP_NUM_THREADS=1`) per the
  existing `reference/README.md` convention (CLAUDE.md policy).

Committed fixtures may keep all factored constituents within `greenone.def`
for simplicity. The C-only behavior — `greentwoex.def` adding a one-body
measurement target absent from `greenone.def`, and Julia extending the
canonical list in C order (Findings 1, 9) — is covered by the unit/contract
test 2, which does not require a committed C run.

## Documentation Updates (in scope)

- `docs/manual/04_physics_calc.md` and `MVMCOptimizers.jl/README.md`: move
  the factored / product two-body Green (`cisajscktaltex` / `TwoBodyGEx`)
  from "not yet ported" to "supported (non-FSZ)", with the FSZ caveat
  noting it is deferred to a separate spec.
- `MVMCOptimizers.jl/test_unit/INDEX.md`: register the new unit/contract
  tests.
- `test/integration/reference/README.md`: document the PhysCal fixture
  generation procedure.

## Completion Conditions

- All four systems: `zvo_cisajs`, `zvo_cisajscktalt`, and
  `zvo_cisajscktaltex` match the C reference bit-for-bit (or within
  `rtol = 1e-10`, `atol = 1e-12`, documented per quantity), with
  C-compatible `*_001.dat` filenames.
- The canonical one-body list and `cis_ajs_ckt_alt_idx` are C-compatible
  (Findings 1, 2): explicit-then-appended order, Julia 1-based indices.
- The FSZ guard rejects FSZ + factored runs before sampling (Finding 5).
- `run_phys_cal_from_namelist` consumes the committed fixed parameters
  without perturbation, and the gate asserts they were used (Finding 7).
- Unit/contract tests pass on the full CI matrix; `INDEX.md` matches the
  tests actually included.
- The manual reflects the new state of the factored path.

## Key C Reference Points

- `calgrn.c:72-79` — precompute one-body Greens into `LocalCisAjs`.
- `calgrn.c:113-118` — accumulate
  `PhysCisAjsCktAlt[idx] += w·LocalCisAjs[idx0]·conj(LocalCisAjs[idx1])`.
- `readdef.c` `GetInfoTwoBodyGEx` — parse `greentwoex.def`, build
  `CisAjsCktAltIdx[idx][2]` via the `iOneBodyGIdx` lookup.
- `vmcmain.c:673-678` — write `zvo_cisajscktaltex_*.dat` (values only).
- `initfile.c:99` — `_cisajscktaltex_%03d.dat`; `initfile.c:104` —
  `_cisajscktalt_%03d.dat` (DC).
