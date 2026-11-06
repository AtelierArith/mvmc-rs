# Project decisions

This file records owner decisions that shape implementation work in mvmc-rs. Each
entry gives the decision, the date, the reason, and how to apply it, with links to
the issues where the decision was made. AGENTS.md remains the general contributor
guide. When a decision here refines AGENTS.md, this file is the more specific
record; when AGENTS.md is updated to absorb a decision, keep the entry here for
history.

Add new decisions here, not in private notes.

## C reference version

**Decision (2026-11-05):** the C numerical authority moves from mVMC 1.3.0
(`extern/mVMC-1.3.0`, `d73d06bd`) to mVMC 1.4.0 (`v1.4.0`, `e83aa6cb`).
Implementation is tracked in #682. The survey of the differences is in #677
(issuecomment-6648989108). The matching proposal for the Julia reference is
tmisawa/Julia-mVMC#95.

- **Submodule:** the C reference is updated as a submodule, nested submodules
  included. `extern/mVMC-1.3.0` is replaced by `extern/mVMC-1.4.0` at `v1.4.0`.
  The nested pins follow the release, not the upstream heads:
  - `src/StdFace` 6fa4ef1f
  - `src/blis` a32257ee (branch `mvmc`)
  - `src/pfapack` ebaa11b7

  These are identical to the 1.3.0 pins.
- **Status (#682):** the submodule is now `extern/mVMC-1.4.0`. AGENTS.md, the
  docs, scripts, `c_toolbox` and code comments cite it as the authority. C
  `file:line` citations in docs and comments were remapped to 1.4.0 line
  numbers from the diff hunks between `d73d06bd` and `v1.4.0`; citations of
  lines that 1.4.0 changed were checked by hand.
- **Fixtures and artifacts:** historical fixtures, benchmark results under
  `benchmark/**/results/` and other generated evidence stay labelled 1.3.0
  (`d73d06bd`). They are records of the 1.3.0 reference, not claims about
  1.4.0. The test fixtures among them were removed or regenerated in #695 (see
  "Removing 1.3.0 and Julia CPU references" below).
- **1.4.0 contract changes adopted** (#682):
  - SR-CG aborts by default at max iterations and on the new breakdown guards
    (`NSRCGAbortOnFail`, `NSRCGFallback`).
  - `NSRCG=2` means DiagScale-preconditioned CG, with `RescaleSmat` and
    preconditioner fallback.
  - SR-CG rejects a negative `DSROptStepDt`.
  - The `zqp_SRinfo.dat` CG row uses the format `%5d %d`.
  - The Jastrow reader is stricter.
  - `rsbOld = rbOld + t*Nsite` on every two-electron update; the PhysCal timer
    report has the `addPhysTwist` line; `reweight 1` is rejected (not ported).
  - Deviations: an SR-CG step with no active parameter stays a no-op (1.4.0
    aborts and reads `r[0]` out of bounds). The resident GPU CG ports plain and
    DiagScale CG, the fallback and `RescaleSmat` with the abort policy for real
    Opt (#694); complex and FSZ Opt stay direct SR. Details:
    `docs/NUMERICAL_COMPARISONS.md`.
- **Out of scope for #682:** the new 1.4.0 features (reweight, twist, spin
  Jastrow, t-J, `NExUpdatePath=6`, backflow) are tracked separately under the
  C-only features policy below.

## Removing 1.3.0 and Julia CPU references

**Decision (owner, 2026-11-05):** once the move to mVMC 1.4.0 is complete
(#682), C 1.4.0 is the only CPU reference. The 1.3.0-era data may be deleted.
Tracked in #695.

- **Remove:**
  - C 1.3.0 references: native `vmc.out` runs and probes generated from
    `d73d06bd`, and 1.3.0-only goldens.
  - The helpers that map 1.4.0 output back to those references
    (`as_historical_cg_srinfo`, `c14_cg_breakdown`, and the
    `NSRCGAbortOnFail 0` settings that exist only to continue 1.3.0 CG
    trajectories).
  - Historical Julia CPU fixtures and goldens. Tests that compare Rust CPU
    with Julia CPU output are deleted or re-pointed to C 1.4.0 references.
- **Keep:**
  - The Julia GPU fixtures (pin `7314015`, XORWOW contract). They are the
    accepted resident GPU reference (AGENTS.md), not 1.3.0 data.
  - Input files that other tests or the GPU tests still read.
- **Replace** a reference a test still needs with a C 1.4.0-generated one with
  recorded provenance. Never use a Rust-generated expectation.
- **Status (#695, done):** the Julia CPU and 1.3.0 references were removed in
  #702, #704 and #710; the references tests still need were regenerated from
  C 1.4.0 or rerun with unchanged data in #703, #705 and #706. No test reads a
  C 1.3.0 or Julia CPU output. `tests/fixtures/C_REFERENCE_PROVENANCE.md` still
  lists historical records kept on purpose: dated benchmark results,
  investigation notes and tools that pin `d73d06bd` by design.
- **Opt-in ctest gates (#689, 2026-11-06):** the prefix gate
  (`MVMC_RS_CTEST_PREFIX_MODELS`), the 20-step ctest gate (`MVMC_RS_CTEST_MODELS`)
  and the 20-step phase5 gate (`MVMC_RS_PHASE5_20STEP`) are retired with their
  Julia/1.3.0 fixtures (#702); they are not regenerated, because their references
  were Julia CPU trajectories. The upstream-rule gate
  (`MVMC_RS_CTEST_UPSTREAM_MODELS`, C-shipped references, byte-identical in 1.4.0)
  is the only long ctest gate; it passes for all 13 models on main
  (`docs/ISSUE180_UPSTREAM_CTEST_RULE.md`).
- **Retired evidence tooling (#718, 2026-11-06):** the #182/#183 threaded gate
  planner (`scripts/thread_gate_plan_183.py`, its test and
  `docs/THREAD_GATE_PLAN_183.md`) and the #184 matrix generator
  (`scripts/build_issue184_complete_matrix.py`) are deleted. They planned or cited
  tests and fixtures removed in #695, their issues (#182-#185) are closed, and no
  current gate uses them. Dated records they produced stay as history and are not
  regenerated; `scripts/verify_threaded_issue182.sh` remains for the existing
  `threaded_issue182.rs` gate.

## Clear C defects are not reproduced

**Decision (2026-10-05, reaffirmed 2026-11-05):** when C has an evident defect,
Rust keeps the mathematically correct behaviour instead of reproducing the bug.
C stays authoritative for operation order, input contracts and numerical
results everywhere else.

To apply it:

1. Document the defect: the C `file:line` and the evidence.
2. Exclude the affected entries from C comparisons, and record the reason in
   `docs/NUMERICAL_COMPARISONS.md` and the compatibility manual
   (`docs/manual/11-compatibility.md` and `ja/`).
3. Report the defect in English on the Julia-mVMC tracker (tmisawa/Julia-mVMC),
   and cross-link it from the mvmc-rs issue.
4. If it is not obvious that the behaviour is a bug, ask the owner first.

Recorded instances:

| C defect | mvmc-rs | Upstream report | Rust behaviour |
|---|---|---|---|
| Real-mode `calculateOptTransDiff` writes through a complex pointer, dropping derivative 1 (`vmccal.c`) | #370 | Julia-mVMC#55 | correct real-mode slots |
| `-b` varbin writes only half of the complex parameters | (2026-10-06) | — | all parameters, with the C header shape |
| Real-mode samplers and Hamiltonian omit the RBM factor; `RBMDiff` reads an unfilled `RBMCnt` (`vmcmake_real.c`, `calham_real.c`, `vmccal.c:120,214`) | #669, PR #671 | Julia-mVMC#59 | RBM factor in sampling and every Green kernel, following the complex path |
| `GetInfoOptTrans` writes `OptFlag[NProj + NOrbitalIdx + i]`: undoubled index, no RBM offset (`readdef.c:1327-1328, 2814`), so OptTrans weights are never optimized | #676 (replaces the #100 emulation) | Julia-mVMC#94 | OptTrans real components optimized; no stray flags |
| ComplexUHF 1.3.0 ignores `ReadDefFileError` on short definition files (fixed upstream in 1.4.0 #81, which Rust follows; no longer a deviation under 1.4.0) | #680 | — | error before running |
| `getUpdateType` returns `NONE` for `NExUpdatePath` outside the defined paths, so the sampler never moves (`vmcmake.c:797-826`; 1.4.0 defines 4-6, other values still no-op) | #681 | — | values other than 0-3 rejected before any output |

Other C defects reported upstream are Julia-mVMC#56 (NSRCG≥2 with NStore=0
undefined behaviour), #57 (Ne=0 admitted), and #58–#63.

The two OptTrans/RBM defects are still present in 1.4.0 master. 1.4.0 rejects
real-parameter RBM; Rust deliberately keeps the correct factor instead.

## C-only features

**Decision (2026-10-05):** C features that Julia lacks are in scope. They are
tracked as issues (umbrella #185) and implemented following C exactly; there is
no need to wait for Julia. Examples: `useDiagScale`, `RescaleSmat`, `NSRCG≥2`,
BackFlow. For these, C is both the feature reference and the numerical
reference.

## Correctness and performance

**Decision (2026-10-06, updated 2026-11):** numerical correctness on CPU and GPU
comes first and is never traded for speed. Performance work is active again;
the owner asks for speedups where gaps against C or Julia are visible, for
example the single-thread L128 gap (#649).

To apply it:

- Validate every performance change with full-state RNG and output checks
  (`cpu_scaling_contract_586`) and the existing bounds.
- When a correctness bug and a performance task are both pending, fix the bug
  first.

## CPU threading: Rayon only

**Decision (2026-10-07, #479 / PR #484):** inner CPU threading stays on Rayon.
A custom OpenMP-style spin pool that was 2.8x faster on L64 was rejected in
favour of optimizing within Rayon.

To apply it:

- Use fewer `install` calls, hoisted scopes, `broadcast`/`scope` and fixed
  chunking.
- Reject custom thread pools in review.
- Document any remaining gap against C OpenMP instead of leaving Rayon.

## GPU direction

**Resident paths (2026-10-13; legacy removed 2026-11-04):**

- GPU work goes through the resident Opt and PhysCal APIs, under the Julia GPU
  numerical and RNG contract (XORWOW). See the GPU policy in AGENTS.md and
  `docs/design/gpu-resident-*.md`.
- The legacy C-order/SFMT GPU StageBackend route
  (`MVMC_RS_*_BACKEND=cuda`, `CudaSamplerStages`, `ResidentCudaSr`) was
  unmaintained and was removed in #643. Backward compatibility was ignored.
  Do not propose extending that route.
- Model families are added to the resident paths one milestone at a time
  (#572). The support matrix is `docs/design/gpu-resident-model-support.md`.

**tenferro (2026-10-06, #417):**

- Prefer tenferro-rs tensors and einsum behind a backend abstraction for
  tensor-shaped code: the SR S-matrix, batched O and energies, Green
  contractions, batched Pfaffian/inverse.
- The scalar C-order CPU path stays the parity reference.
- File upstream tenferro requests only case by case, with owner approval.
  Otherwise prefer in-repo kernels.

## Development hosts

- **Local GPU host:** 2x NVIDIA GeForce RTX 3060 (sm_86, 12 GB, FP64 about 1/64
  of FP32).
  - The host has only `libcuda.so`, with no CUDA toolkit; build CUDA code in
    Docker with the `nvidia` runtime.
  - Recent resident work used the CUDA 12.9 dev-container image `b129bd583bbe`.
    The default image in `scripts/run_cuda_gate.sh`
    (`tenferro-benchmark-cuda:full-verify-20260822`) is no longer present on
    this host.
  - Check `nvidia-smi` before concluding that no GPU is available.
- **A100:** the owner can run an NVIDIA A100 (strong FP64) on a separate
  server that is not reachable from the development host. Deliver portable
  benchmark scripts (#450) and ask the owner for the result archives. Claims
  about A100 performance need A100 measurements.

## Workflow

- **Merges (2026-10-05):** the owner delegates marking PRs ready and merging
  them, after a review of the diff and green CI on the exact head. Never merge
  with failing tests (AGENTS.md).
- **Backward compatibility (2026-11-04):** it may be ignored, and technical debt
  should be removed actively, including obsolete compatibility paths and dead
  vendored code.
