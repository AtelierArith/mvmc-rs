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
- **Until #682 lands:** AGENTS.md and the docs still cite 1.3.0. Treat 1.4.0 as
  the target for new work.
- **Fixtures:** historical fixtures stay labelled 1.3.0 unless they are
  regenerated with recorded provenance.
- **1.4.0 contract changes to adopt** (#682):
  - SR-CG aborts by default at max iterations.
  - `NSRCG=2` means DiagScale-preconditioned CG, with `RescaleSmat` and
    preconditioner fallback.
  - SR-CG rejects a negative `DSROptStepDt`.
  - The `zqp_SRinfo.dat` CG row uses the format `%5d %d`.
  - The Jastrow reader is stricter.
- **Out of scope for #682:** the new 1.4.0 features (reweight, twist, spin
  Jastrow, t-J, `NExUpdatePath=6`, backflow) are tracked separately under the
  C-only features policy below.

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
| Real-mode samplers and Hamiltonian omit the RBM factor; `RBMDiff` reads an unfilled `RBMCnt` (`vmcmake_real.c`, `calham_real.c`, `vmccal.c:120,205`) | #669, PR #671 | Julia-mVMC#59 | RBM factor in sampling and every Green kernel, following the complex path |
| `GetInfoOptTrans` writes `OptFlag[NProj + NOrbitalIdx + i]`: undoubled index, no RBM offset (`readdef.c:1067-1068, 2400`), so OptTrans weights are never optimized | #676 (replaces the #100 emulation) | Julia-mVMC#94 | OptTrans real components optimized; no stray flags |
| ComplexUHF ignores `ReadDefFileError` on short definition files (fixed upstream in 1.4.0 #81) | #680 | — | error before running |

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
