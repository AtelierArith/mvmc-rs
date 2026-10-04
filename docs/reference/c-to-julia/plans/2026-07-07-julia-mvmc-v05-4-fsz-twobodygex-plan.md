---
date: 2026-07-07
datetime: 2026-07-07 17:56 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC V05-4 FSZ TwoBodyGEx implementation plan
updated:
  - datetime: 2026-07-07 18:20 JST
    model: GPT-5 Codex
    note: Incorporated plan-review findings on the missing FSZ Green measurement driver, fixture requirements, documentation paths, and PhysCal-nsplit base dependency.
  - datetime: 2026-07-08 11:21 JST
    model: GPT-5 Codex
    note: Added the short pre-implementation checklist after PR #48 landed.
  - datetime: 2026-07-08 11:57 JST
    model: GPT-5 Codex
    note: Implemented the FSZ PhysCal Green driver, enabled FSZ TwoBodyGEx, and added the heisenberg_chain_fsz PhysCal C-reference gate.
target:
  repository: Julia-mVMC
  base: develop after the VMCPhysCal NSplitSize > 1 PR is merged
  scope: VMCPhysCal FSZ/general-orbital factored TwoBodyGEx
---

# Julia-mVMC V05-4 FSZ TwoBodyGEx Implementation Plan

## Decision

Implement the FSZ / general-orbital normal-Green measurement driver for
`VMCPhysCal`, then enable `TwoBodyGEx` on top of that driver.

This milestone is not only a `TwoBodyGEx` switch removal. It also fixes a
latent FSZ PhysCal Green bug: current Julia reaches FSZ physical calculation for
plain `OneBodyG` / `TwoBodyG`, but the measurement path still uses the
sz-conserved Green driver and cannot represent spin-flip one-body terms.

R1 supported scope:

- `VMCPhysCal`
- `i_flg_orbital_general != 0`
- `NLanczosMode = 0`
- `NSplitSize = 1`
- FSZ one-body Green output
- FSZ direct two-body Green output
- `TwoBodyGEx` / factored two-body Green output

Keep these combinations out of scope:

- `NSplitSize > 1` with FSZ / general-orbital PhysCal
- `NLanczosMode > 0` with FSZ / general-orbital PhysCal
- BackFlow
- OptTrans-derived QP split
- changing stochastic sampling or RNG semantics

This keeps the milestone separate from the `VMCPhysCal NSplitSize > 1`
milestone. The latter intentionally supports only sz-conserved normal Green in
R1 and keeps FSZ split rejected.

Base dependency:

- The plan assumes the `VMCPhysCal NSplitSize > 1` PR has landed on `develop`,
  including the data-level PhysCal validators that reject FSZ split and Lanczos
  split combinations.
- PR #48 landed on `develop` as merge commit `f7bfb78989082975a38a7a44beba6b0443a8d429`.

## Pre-Implementation Checklist

Before editing Julia code:

- [x] Update local `Julia-mVMC/develop` to include PR #48, then create a fresh
      feature branch for FSZ `TwoBodyGEx`.
- [x] Pick the smallest FSZ fixture family and define the exact `greenone.def`,
      `greentwo.def`, and `greentwoex.def` entries that exercise same-spin,
      spin-flip, direct two-body, and factored Green paths.
- [x] Generate the C reference with `mVMC @ 622166a` or the selected pinned C
      revision, recording the command, `OMP_NUM_THREADS=1`, input diff, and
      sanity checks in fixture metadata.
- [x] Reconfirm the C dispatch table from `CalculateGreenFunc_fsz`: same-spin
      one-body, spin-flip one-body, direct two-body, and factored
      `LocalCisAjs[idx0] * conj(LocalCisAjs[idx1])`.
- [x] Sketch the Julia call boundary for `calculate_green_func_fsz!` before
      touching kernels, so non-FSZ `calculate_green_func!` remains unchanged.
- [x] Implement in this order: fixture metadata and failing gate, validator
      narrowing, FSZ Green driver, docs, then full verification.
- [x] Keep explicit negative tests for FSZ `NSplitSize > 1` and FSZ
      `NLanczosMode > 0`; this PR must not widen those scopes.

## Background

Current Julia state:

- `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl` builds a canonical
  one-body Green list for `TwoBodyGEx`.
- `accumulate_factored_green!` already accumulates
  `local_cis_ajs[idx0] * conj(local_cis_ajs[idx1])`.
- `validate_factored_green_supported(data)` rejects any
  `TwoBodyGEx` input when `data.i_flg_orbital_general != 0`.
- `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`
  currently pins that FSZ rejection.
- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl` has FSZ Green kernels used
  by the Hamiltonian path (`green_func1_fsz`, `green_func1_fsz2`,
  `green_func2_fsz`, `green_func2_fsz2`), but there is no FSZ measurement
  driver equivalent to C's `CalculateGreenFunc_fsz`.
- The current FSZ physical-calculation path calls the shared
  `calculate_green_func!` driver under a TODO comment, so same-spin terms may
  work accidentally but spin-flip measurement terms are not C-faithful.

C reference points:

- `mVMC/src/mVMC/readdef.c` appends the one-body terms needed by `TwoBodyGEx`
  through `CountOneBodyGForLanczos` / `GetInfoTwoBodyGEx`.
- `mVMC/src/mVMC/calgrn_fsz.c` is the FSZ measurement driver. It dispatches
  one-body terms to `GreenFunc1_fsz` or `GreenFunc1_fsz2`, and direct two-body
  terms to `GreenFunc2_fsz` or `GreenFunc2_fsz2`.
- The FSZ Green kernels are defined in `mVMC/src/mVMC/locgrn_fsz.c` and
  `mVMC/src/mVMC/locgrn_fsz_real.c`.
- `CalculateGreenFunc_fsz` accumulates factored `TwoBodyGEx` as
  `LocalCisAjs[idx0] * conj(LocalCisAjs[idx1])`.
- `mVMC/src/mVMC/vmcmain.c` writes `zvo_cisajscktaltex_*.dat` as value-only
  rows, matching the non-FSZ factored output shape.

Fixture state:

- Julia already has non-FSZ `TwoBodyGEx` physical-calculation fixtures.
- The public C-mVMC test tree does not contain a `TwoBodyGEx` fixture, so the
  FSZ reference output for this milestone must be generated and sanity-checked
  from source rather than copied from an existing C-side test.

## Implementation Tasks

### Task 1: Add A New FSZ `TwoBodyGEx` Reference Fixture

Files:

- Add under `Julia-mVMC/test/integration/reference/`
- Update `Julia-mVMC/test/integration/reference/README.md`

Steps:

- [x] Select a small FSZ/general-orbital fixture with `NVMCCalMode = 1`,
      `NLanczosMode = 0`, `NSplitSize = 1`, and `TwoBodyGEx`.
- [x] Reuse a compact existing FSZ input family if possible, but treat
      `greentwoex.def` and `physcal_ref/` output as new reference material.
- [x] Commit the C reference outputs needed by the gate:
  - `zvo_cisajs_001.dat`
  - `zvo_cisajscktalt_001.dat` if direct two-body Green terms are present
  - `zvo_cisajscktaltex_001.dat`
  - input files and fixture metadata
- [x] Record enough fixture metadata to regenerate it: C source revision,
      command shape, `OMP_NUM_THREADS=1` where applicable, and comparison
      tolerances.
- [x] Include at least one `TwoBodyGEx` entry whose second constituent exercises
      C's reversed `(l, v) -> (k, u)` one-body lookup.
- [x] Include at least one direct `greenone.def` spin-flip entry
      `(i, s, j, t), s != t`, so `zvo_cisajs_001.dat` directly pins the
      `GreenFunc1_fsz2` path.
- [x] Include at least one `TwoBodyGEx` constituent with spin labels that
      differ, so the factored path also exercises `GreenFunc1_fsz2`.
- [x] Add C-reference sanity checks during generation:
  - a `|G(i,s,j,t)|^2`-style factored term is real, non-negative, and bounded
  - an on-site density/doublon-style term is consistent with the corresponding
    direct `TwoBodyG` output when such a pair is included

Acceptance:

- The fixture can be regenerated from committed inputs.
- Reference output order is unambiguous and documented.
- The fixture remains small enough for the existing integration test budget.
- The new C reference is sanity-checked because upstream C has no public
  `TwoBodyGEx` regression fixture for this path.

### Task 2: Narrow The FSZ Rejection

Files:

- Modify `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl`
- Modify `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_physcal_factored_green.jl`
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl` only if the
  entry-point guard needs a clearer error message

Steps:

- [x] Replace the blanket `TwoBodyGEx` FSZ rejection with a narrower contract:
  - accept `TwoBodyGEx` for FSZ `NLanczosMode = 0`, `NSplitSize = 1`
  - continue rejecting unsupported split/lanczos combinations through the
    existing PhysCal validators
- [x] Update the `validate_factored_green_supported` message from "not yet
      implemented" to the actual remaining unsupported scope, if the helper is
      kept.
- [x] Preserve a clear error if an unsupported combination reaches the Green
      initializer.
- [x] Update unit tests that currently expect FSZ `TwoBodyGEx` rejection.
- [x] Add tests that still reject:
  - FSZ `TwoBodyGEx` with `NSplitSize > 1`
  - FSZ `TwoBodyGEx` with `NLanczosMode > 0`

Acceptance:

- The rejected combinations still fail before numerical work.
- The supported FSZ normal-Green combination initializes `PhysicalQuantities`.

### Task 3: Implement The FSZ Green Measurement Driver

Files:

- Modify `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl`
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl` if the FSZ kernel
  helpers need to be exported or moved
- Modify FSZ Green helpers only when a C-parity bug is found

Steps:

- [x] Add a `calculate_green_func_fsz!` / `calculate_green_func_fsz_into!`
      driver equivalent to C's `CalculateGreenFunc_fsz`.
- [x] Dispatch canonical `phys.cis_ajs_idx` terms by all four site/spin fields:
  - same-spin terms use the standard FSZ one-body Green function
  - spin-flip terms use the FSZ2 one-body Green function
- [x] Dispatch direct `green_two_terms` by all eight fields:
  - `s == t && u == v` uses `green_func2_fsz`
  - otherwise use `green_func2_fsz2`
- [x] Ensure `phys.local_cis_ajs[idx]` is populated before
      `accumulate_factored_green!`.
- [x] Call the FSZ driver from `vmc_main_cal_fsz!` / `process_sample_range_fsz!`
      instead of the current shared `calculate_green_func!` TODO path.
- [x] Add unit tests that exercise:
  - same-spin FSZ one-body dispatch
  - spin-flip FSZ one-body dispatch
  - direct FSZ two-body dispatch to both FSZ and FSZ2 kernels
  - factored accumulation from FSZ local one-body values
- [x] Preserve the non-FSZ path unchanged.
- [x] Keep the output layout for factored Green value-only rows unchanged.

Acceptance:

- FSZ one-body, direct two-body, and factored outputs match the new C reference.
- Existing FSZ measurement values may change relative to the current Julia
  implementation; that is expected when the old value came from the
  sz-conserved driver.
- Existing non-FSZ `TwoBodyGEx` outputs do not regress.

### Task 4: Add Integration Gate

Files:

- Modify `Julia-mVMC/test/integration/phys_cal_equivalent.jl` or add a focused
  FSZ physical-calculation gate next to it.
- Update integration helper comparison tolerances only if the new fixture needs
  a documented tolerance class.

Steps:

- [x] Run the new FSZ fixture through Julia `VMCPhysCal`.
- [x] Compare `zvo_cisajs_001.dat` against the C reference.
- [x] Compare `zvo_cisajscktaltex_001.dat` against the C reference.
- [x] Compare direct two-body output if present.
- [x] Use the same value parser and max-diff reporting style as existing
      physical-calculation gates.

Recommended tolerance:

- Start with `rtol = 1e-9`, `atol = 1e-12`.
- Loosen only with a documented numerical reason and a fixture-specific max
  difference.

Acceptance:

- FSZ `TwoBodyGEx` matches the C reference within the documented tolerance.
- Existing non-FSZ `TwoBodyGEx` gates still pass.
- FSZ one-body and direct two-body Green outputs match the new C reference.

### Task 5: Documentation And Compatibility Matrix

Files:

- Modify `Julia-mVMC/README.md`
- Modify `Julia-mVMC/MVMCOptimizers.jl/README.md`
- Modify `Julia-mVMC/docs/manual/04_physics_calc.md`
- Modify `Julia-mVMC/docs/manual/05_compatibility.md`
- Modify `Julia-mVMC/MVMCOptimizers.jl/test_unit/INDEX.md` if the unit-test
  index tracks the new FSZ tests

Steps:

- [x] Mark FSZ/general-orbital `TwoBodyGEx` as supported only for normal
      `VMCPhysCal` with `NSplitSize = 1`.
- [x] Keep FSZ `NSplitSize > 1` PhysCal rejected in the matrix.
- [x] Keep FSZ Lanczos rejected in the matrix.
- [x] Mention the new regression fixture without referencing internal review or
      plan paths.

## Verification Commands

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
julia --project=. test/integration/phys_cal_equivalent.jl
julia --project=. test/integration/runtests.jl
git diff --check
```

If shared validators are touched after the PhysCal split work, also run:

```bash
JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl
```

## Risks And Review Focus

- FSZ one-body canonical terms have four spin/site fields, unlike the
  sz-conserved helper that can rely on `si == sj`.
- C's `GetInfoTwoBodyGEx` intentionally reverses the second one-body
  constituent before indexing. The Julia resolver must match this exactly.
- A fixture with only same-spin terms would miss the FSZ2 path.
- Current Julia FSZ measurement output is not a valid oracle for plain
  one-body/direct two-body Green; the C reference fixture is the oracle.
- It is easy to accidentally widen FSZ split or Lanczos support while removing
  the existing `TwoBodyGEx` rejection. Keep those as explicit negative tests.
- Output comparison must check value order, not only aggregate norms.

## Exit Criteria

- FSZ/general-orbital normal physical calculation uses an FSZ-specific Green
  measurement driver for `NSplitSize = 1`.
- FSZ/general-orbital `TwoBodyGEx` is supported on that driver.
- A committed C-reference fixture gates the feature.
- Unsupported FSZ split and FSZ Lanczos combinations remain rejected with clear
  messages.
- Public docs reflect the narrowed support matrix.
