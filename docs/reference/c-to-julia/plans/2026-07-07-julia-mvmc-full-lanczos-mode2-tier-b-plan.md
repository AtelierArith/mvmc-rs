---
date: 2026-07-07
datetime: 2026-07-07 09:42 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC Full Lanczos mode2 Tier B TwoBodyGEx implementation plan
updated:
  - datetime: 2026-07-07 10:27 JST
    model: GPT-5 Codex
    note: Reflected the plan review: PR #46 merge status, one-body-list invariant, C normal-output comparison, absolute staging path, whitespace checks, and cwd assumptions.
---

# Julia-mVMC Full Lanczos mode2 Tier B implementation plan

## Goal

PR #46 (`feature/v0.5-full-lanczos-mode2`) merged on 2026-07-07 10:13 JST and
landed the Tier A `NLanczosMode = 2` path for `VMCPhysCal`, serial
`NSplitSize = 1`, sz-conserved real fixtures, and direct `TwoBodyG` Lanczos
Green output.

The next PR should add the Tier B C-reference gate for factored `TwoBodyGEx`
Lanczos Green output:

- `zvo_ls_cisajscktaltex_001.dat`

Use `Julia-mVMC/test/integration/reference/hubbard_chain_real/physcal_ref` as
the first Tier B fixture because it is small, real, non-FSZ, already contains
`TwoBodyGEx`, and already has a normal PhysCal factored-Green C-reference gate.

## Scope

Supported path for this PR:

- `NVMCCalMode = 1`
- `NLanczosMode = 2`
- `NSplitSize = 1`
- `NStore = 1`
- `NSRCG = 0`
- real sz-conserved `hubbard_chain_real/physcal_ref`
- existing `TwoBodyGEx` terms in `greentwoex.def`

Expected committed Lanczos files for the fixture:

- `zvo_ls_out_001.dat`
- `zvo_ls_qqqq_001.dat`
- `zvo_ls_cisajs_001.dat`
- `zvo_ls_cisajscktalt_001.dat`
- `zvo_ls_cisajscktaltex_001.dat`

The first four files are useful regression guards because changing
`hubbard_chain_real` from `NLanczosMode = 0` to `NLanczosMode = 2` should not
leave the R1/direct mode2 path untested for this fixture.

## Non-Goals

- No FSZ/general-orbital Lanczos support.
- No complex mode2 C-reference fixture.
- No `NSplitSize > 1` PhysCal support.
- No MPI reduction changes beyond regression smoke tests.
- No SR-CG split support.
- No BackFlow support.
- No broad refactor of `PhysicalQuantities` or Green-output helpers unless the
  C-reference comparison exposes a correctness issue.

## Starting State

After PR #46:

- `Julia-mVMC/MVMCOptimizers.jl/src/data_io.jl` already writes
  `zvo_ls_cisajscktaltex_XXX.dat` from
  `phys.phys_lanczos_qcisajscktaltq`.
- `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl` already builds the
  canonical one-body list from `greenone.def` plus `TwoBodyGEx` constituents
  and resolves `cis_ajs_ckt_alt_idx`.
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`
  already has writer-level coverage for `zvo_ls_cisajscktaltex`.
- `Julia-mVMC/test/integration/lanczos_equivalent.jl` compares Tier A mode2
  files but does not yet include a factored `TwoBodyGEx` fixture.
- `Julia-mVMC/test/integration/phys_cal_equivalent.jl` already includes
  `hubbard_chain_real` for normal `zvo_cisajscktaltex_001.dat`.

Therefore this PR is primarily a reference-generation and integration-gate PR.
If the C-reference comparison fails, investigate the existing factored
accumulator/output path before expanding scope.

## Baseline Decision

Use the same C baseline as PR #46 for new Lanczos mode2 outputs:

- `mVMC @ 622166afe33c6be3402d7c926db7e9c0003a47c4`
- `OMP_NUM_THREADS=1`
- single MPI rank

Do not silently replace the existing normal PhysCal references in
`hubbard_chain_real/physcal_ref/expected/`. Those existing non-Lanczos files are
documented as generated at C `66f17422968009f8cc70f1dec94b2f52e562d344`.

For this PR, keep the baseline split explicit:

- existing `zvo_cisajs*` files remain on the existing non-Lanczos provenance;
- new `zvo_ls_*` files are generated at `622166a`;
- `phys_cal_equivalent.jl` must still pass for `hubbard_chain_real`, proving the
  existing normal PhysCal references remain compatible after changing the
  committed fixture to `NLanczosMode = 2`.
- staging C outputs from `622166a` must be compared directly against the
  existing committed normal PhysCal references for `zvo_cisajs_001.dat`,
  `zvo_cisajscktalt_001.dat`, and `zvo_cisajscktaltex_001.dat`; record max
  differences in the metadata addendum.

If normal `zvo_cisajs*` outputs no longer pass after the mode2 change, stop and
decide separately whether to regenerate the full fixture at `622166a`. Do not
mix refreshed normal references into the Tier B PR without documenting why.

## Implementation Steps

1. Confirm PR #46 is merged, then update local `develop`.
2. Create a new topic branch, for example
   `feature/v0.5-full-lanczos-mode2-tier-b`, from updated `develop`.
3. Generate C references from a staging copy only:
   - copy `Julia-mVMC/test/integration/reference/hubbard_chain_real/physcal_ref/inputs`
     and `zqp_opt.dat` to a temporary staging tree outside `Julia-mVMC/`;
   - use an absolute staging path to avoid public-repo pollution, for example
     `/Users/takahiromisawa/Dropbox/Projects/Shin-mVMC/tmp/lanczos-mode2-tierb-ref-YYYYMMDD-HHMMSS/`;
   - in the staging copy, change only `NLanczosMode` from `0` to `2`;
   - run C from the staging `inputs` directory with
     `OMP_NUM_THREADS=1 <vmc.out> -e namelist.def ../zqp_opt.dat`;
   - copy raw C outputs from the staging `inputs/` directory into a staging
     `c-output/` directory for inspection;
   - compare the raw C normal PhysCal outputs
     `zvo_cisajs_001.dat`, `zvo_cisajscktalt_001.dat`, and
     `zvo_cisajscktaltex_001.dat` against the committed `expected/` files and
     record max differences;
   - commit only curated `zvo_ls_*_001.dat` files to the Julia reference tree.
4. Change the committed
   `Julia-mVMC/test/integration/reference/hubbard_chain_real/physcal_ref/inputs/modpara.def`
   from `NLanczosMode 0` to `NLanczosMode 2`.
5. Add the five generated `zvo_ls_*_001.dat` files under
   `Julia-mVMC/test/integration/reference/hubbard_chain_real/physcal_ref/expected/`.
6. Append a clearly separated "Full Lanczos mode2 addendum" section to
   `Julia-mVMC/test/integration/reference/hubbard_chain_real/physcal_ref/metadata.txt`.
   The public metadata should include:
   - C commit hash;
   - `OMP_NUM_THREADS=1`, single rank;
   - source fixture name;
   - changed setting `NLanczosMode = 2`;
   - output filenames copied into `expected/`;
   - Julia-vs-C max differences for all five `zvo_ls_*` files;
   - C `622166a` staging-vs-committed max differences for the three normal
     `zvo_cisajs*` files;
   - the fixture-specific one-body-list invariant: after C's `TwoBodyGEx`
     second-constituent swap, all constituents are already in the 12-entry
     `greenone.def` list, so mode0 and mode2 use the same one-body list;
   - a note that existing non-Lanczos `zvo_cisajs*` references retain their
     original provenance and were revalidated by `phys_cal_equivalent.jl`.
7. Update `Julia-mVMC/test/integration/lanczos_equivalent.jl`:
   - add `hubbard_chain_real` to the mode2 fixture list with
     `c_model = "HubbardChain"`, `mode = :real`, and `n_para = 19`;
   - compare `zvo_ls_cisajscktaltex_001.dat` for fixtures that have
     `TwoBodyGEx`;
   - keep Tier A fixtures from requiring `zvo_ls_cisajscktaltex_001.dat`, since
     their C byproduct is empty and intentionally not committed.
8. Keep unit changes minimal. Add unit coverage only if the integration gate
   exposes a missing contract that the existing writer/accumulator tests do not
   cover.

## C/Julia Comparison Policy

Use vector comparison for `zvo_ls_cisajscktaltex_001.dat`, matching its
value-only C output contract.

Expected shape for `hubbard_chain_real`:

- `greentwoex.def` has `NTwoBodyGEx = 3`;
- `zvo_ls_cisajscktaltex_001.dat` should contain six floating-point values
  (`real, imag` for each factored term) concatenated on one line.

The normal one-body Green list is expected to remain unchanged when this
fixture is switched from `NLanczosMode = 0` to `NLanczosMode = 2`. C's
`CountOneBodyGForLanczos` registers the second `TwoBodyGEx` constituent with
the `(x6,x7,x4,x5)` swap; for this fixture, every swapped constituent is already
present in the 12-entry `greenone.def` list. Julia's parser applies the same
swap when constructing `GreenTwoExTerm`, so the Julia canonical list should
match C's unchanged order. If this invariant fails for a future fixture, do not
reuse this Tier B assumption.

Use the same tolerance family as the Tier A Lanczos Green gate unless the
documented max differences show a numerical reason to adjust:

- `LANCZOS_LS_OUT_TOL = 1e-8`
- `LANCZOS_QQQQ_TOL = 1e-10`
- `LANCZOS_GREEN_TOL = 1e-8`

If only `zvo_ls_cisajscktaltex_001.dat` needs a looser tolerance, add a named
constant rather than weakening all Lanczos Green comparisons.

## Acceptance Gates

Run these commands from `Julia-mVMC/`.

Minimum local gates before PR:

- diff whitespace check:
  ```bash
  git diff --check develop
  ```
- strict public local-path gate over public text and reference metadata:
  ```bash
  rg -n "/Users/|Dropbox/|Shin-mVMC/|docs/(reports|reviews)/" README.md docs/manual MVMCOptimizers.jl/README.md test/integration/reference
  ```
- standalone public file-name gate:
  ```bash
  rg -n '(^|[^[:alnum:]_])LOG\.md($|[^[:alnum:]_])' README.md docs/manual MVMCOptimizers.jl/README.md test/integration/reference
  ```
- local benchmark-path gate:
  ```bash
  rg -n '(^|[[:space:]"'"'"'(])benchmark/' README.md docs/manual MVMCOptimizers.jl/README.md test/integration/reference
  ```
- Lanczos integration:
  ```bash
  julia --project=. test/integration/lanczos_equivalent.jl
  ```
- focused normal PhysCal regression for the changed fixture:
  ```bash
  JULIA_MVMC_PHYS_CAL_MODELS=hubbard_chain_real julia --project=. test/integration/phys_cal_equivalent.jl
  ```
- focused unit gates:
  ```bash
  julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl")'
  julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl")'
  JULIA_NUM_THREADS=2 julia --project=. -e 'include("MVMCOptimizers.jl/test_unit/test_unit_threading.jl")'
  ```

Broader gates before merge if runtime is acceptable:

- `julia --project=. test/integration/phys_cal_equivalent.jl`
- `julia --project=. test/integration/runtests.jl`
- `JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl`

## Risks

- **Baseline mixing:** The fixture will contain existing normal references from
  `66f174...` and new Lanczos references from `622166a`. This is acceptable only
  if metadata is explicit and normal PhysCal is revalidated.
- **Output-order mismatch:** `TwoBodyGEx` output is value-only, so row-order
  mistakes can hide as value-order mistakes. Keep the `NTwoBodyGEx = 3` shape
  explicit and compare the full vector in file order.
- **Whitespace churn from C output:** Run `git diff --check develop` after
  copying C outputs. If needed, remove only trailing whitespace and extra final
  blank lines. Verify the stripped committed files still match the raw C files
  with `diff -b` or a token-by-token numeric comparison before committing them;
  `git diff --ignore-space-at-eol --ignore-blank-lines` is not sufficient for
  newly added files.
- **Scope creep:** If complex/FSZ/NSplit issues appear, reject or defer them.
  This PR should only establish the first real sz-conserved factored mode2 gate.

## PR Text Notes

Public PR text must not mention local staging paths or internal review/report
paths. It may mention:

- public fixture paths under `test/integration/reference/`;
- C commit hashes;
- command summaries;
- test commands and pass/fail status.
