# #179 bounded prefix/group evidence on committed 69a53195

Related to #179/#184/#185. This is same-implementation RNG/discrete/group/prefix
repeatability, **not independent numerical accuracy or full MPI parity**.
No production code, numerical tolerance, reference source or existing ledger
was changed for this stage. #179 remains open.

## Immutable source, build and runtime

Container `73c57e563c61`, retained root (all paths below are relative to this root):

```text
/home/vscode/.cache/mvmc/issue179-prefix69.FFPASr
```

`snapshot/` is an independent detached clone at
`69a53195a97d66aae0fa6cf55d7ee9c41b7557b4`, not a copy of the shared dirty tree.
The only source overlays are the following three new owner scripts:

| Overlay | SHA-256 |
| --- | --- |
| `scripts/mpi_issue179_prefix_evidence.py` | `621d65cfdb870aacd1d597212b2cb8e2d6627a80f03eaa9605751f0087c92bbc` |
| `scripts/test_mpi_issue179_prefix_evidence.py` | `3ed15a174352f609f10975e7cf8459559c4831c0624ff8922ae3bf99daba1776` |
| `scripts/verify_mpi_issue179_prefixes.sh` | `d80f56cac6bfa89d9cca2097eed4ea216258ae9882557b808eca3fe321808963` |

Reference directories were checked against the actual committed gitlinks:
Julia-mVMC `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, PfaPack.jl
`0dcf52c15caec63516d0703f36bfc8a4bc0e58d0`, mVMC-1.3.0
`d73d06bd529d3b2573f38eb5817c4a5f52971006`. No C or Julia reference executable
was launched. Preparation initially used the wrong directory alias `PfaPack`;
the exact `extern/PfaPack.jl` checkout was completed and verified before build.
`snapshot/preparation-note.txt` retains this preparation correction, not a
failed numerical run relabelled as passing.

Named-volume target: `/home/vscode/.cache/mvmc/target/issue179-prefix69`.
Rust1.99.0 (`b940084d7`, LLVM23.1.1), Linux x86_64; actual MPI launcher is
MPICH/Hydra4.2.0, ch4:ucx, internal PMI1, without external PMIx. These results
are not from the old Open MPI container/image. Runtime API metadata identifies
OpenBLAS0.3.26, Haswell, actual threads1, configuration
`NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`.
`runtime-backend.json` is a **separate metadata process** loading the library
linked by the Rust binary; it is not an observation inside every MPI rank.

## Actual commands and terminal results

Build handle7463 terminal0, 14.90s; exact command in `snapshot/build-command.txt`:

```sh
timeout --kill-after=10s 600s cargo test --locked --profile test-fast \
  -p mvmc-core --features mpi --test mpi_issue179_state \
  --test mpi_issue179_mapping --test mpi_issue179_collectives \
  --test mpi_issue179_literal_operator --no-run --message-format=json
```

Cargo compiler-artifact paths were selected from `snapshot/build.json` and
copied into `bin/` before any MPI launch. No rebuilt target path was substituted
for these frozen binaries. Root `build-association.json` binds actual captured
git revision, build exit0, build command/log/JSON/compiler records, baseline
manifest, full source manifest, binary and all three overlay hashes.

Prepare-only handle29798 terminal0: committed
`scripts/verify_mpi_issue179.sh` with `MPI179_PREPARE_ONLY=1` generated the
inventory; `inventory-generation.log` retains its output and original generated
root `/tmp/mvmc-issue179-evidence.fxbunF`. Twelve direct/store0 identity cells
were selected into `inventory/matrix.tsv`. This is input preparation, **not CLI
execution**. Inputs derive from the pinned Julia real/cmp/FSZ six-site
Heisenberg fixtures: seed1, warmup1, interval1, full chain samples3, identity
QP1, InterAll excluded. The Rust state harness sets both optimization steps
and parameter averaging window to the requested prefix1/2/3; these are three
distinct bounded runs, not a truncated long-run artifact.

Launch handle39176 was harvested once, terminal0; no retries or axis expansion.
First, five existing exact-test MPI launches passed in the owner's execution
and artifact audit. The parent subsequently independently read all five actual
logs and result rows: every rank reports1 PASS/no FAIL, with all launcher exits0.
The parent also verified all four frozen binary hashes against the manifest:

| Test | Worlds | Launcher exits |
| --- | --- | --- |
| `issue179_group_width_endpoints` | 2,4 | 0,0 |
| `issue179_collectives` | 2,4 | 0,0 |
| `issue179_literal_two_rank_cg_products` | 2 | 0 |

The literal commands use `timeout --kill-after=5s 60s mpirun -n WORLD
bin/TARGET --ignored --exact TEST --nocapture`. Their listings, logs and exits
are retained under `literal-gates/`; these are communicator/operator proofs,
not independent full-model sampler parity.

The approved wrapper was run from `snapshot/` with:

```sh
export MPI179_SOURCE_COMMIT=69a53195a97d66aae0fa6cf55d7ee9c41b7557b4
export MPI179_CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-prefix69
export MPI179_TIMEOUT=60
bash scripts/verify_mpi_issue179_prefixes.sh ../bin/mpi_issue179_state \
  ../inventory ../prefix-evidence sources.sha256 ../build-association.json
```

`prefix-evidence/commands.txt` records each input/output directory, prefix and
actual launch command: `mpirun -n 2|4 bin/mpi_issue179_state --ignored --exact
issue179_state --nocapture`, bounded60s with kill grace5s.

Matrix: real/cmp/complex-FSZ × world2/4 × width1/2 = **12 configurations**;
each prefix1/2/3 is run in **two fresh processes**, workers1, production default
threshold (environment override unset), OpenBLAS/OMP/MKL/BLIS thread limits1.
All **72 launcher exits0**, **12 strict checker exits0**, and **216 intact native
summary fragments reporting 1 PASS/0 FAIL/0 ignored**. These shared launcher
logs are not per-rank-tagged logs. There were no native101 failures in this
capture; the driver would preserve any such failure and fail, not accept it as
an expected successful cell. Root `terminal.exit` and `wrapper.exit` are0.

## Strict contracts and limits

The checker requires all prefixes/repeats, nonzero evidence, successful
launcher/worker status, exact group membership and base-plus-group seed offsets,
before-init/initial/final raw624 and valid cursors, initial/final next624,
primitive initial/total/sampling counts and cursor/count consistency. It
validates actual proposal shapes/reject flags, Metropolis decisions with the
immediately preceding consumed word, acceptance metadata, and seven-field
sampling checkpoints including saved five planes, ten counters and next624.
Group comparisons require identical sampler traces and RNG snapshots/counts;
width1 is an independent per-rank chain, not world-wide sampling equality.
Prefix comparisons require unchanged initial RNG records and common sampling
trace prefixes. Checkpoints do not newly expose intermediate raw624/cursors:
each independent prefix's final snapshot supplies that bounded evidence.

The existing fresh-repeat checker additionally compares per-rank records and
root `zvo_out.dat`, `zvo_var.dat`, `zqp_opt.dat`. That same-binary/configuration
repeat comparison does **not** impose cross-implementation bitwise computed
float equality or establish independent output/SR accuracy. Measurement-local
arrays, reduced-counter domains and post-measurement scratch are deliberately
not required to agree across group ranks. Checkpoint planes are structurally
validated and compared, not independently proven to match a C/Julia model
trajectory. Synthetic checker regressions16/16 passed; they are protocol tests,
not sixteen MPI scenarios.

Not covered by this matrix: CG, NStore1, width3/other uneven sampler groups,
standard QP/OptTrans, worker2/4/default-threshold activation, long20, independent
numerical accumulators/SR updates/residuals/root output, or expanded failure
phases. The helper mapping/collective tests cover some uneven/empty operations,
but do not promote them to full-model sampling proof. Further stages require
separate approval and fresh evidence; #179 remains open.

## Artifact inventory and mutation checks

| Location | Content |
| --- | --- |
| `snapshot/baseline.sha256`, `baseline-before-check.txt` | Committed source baseline before owner overlays/build |
| `snapshot/sources.sha256`, `build-source-after-check.txt` | Source/reference/fixture/script closure, plus Cargo inputs |
| `snapshot/build-command.txt`, `build.log`, `build.json`, `build.exit`, `compiler.txt`, `git-rev-parse.txt` | Exact compilation and captured revision |
| `build-association.json`, `build-records.json`, `overlays.json`, `association-preflight.log` | Build/source/binary association |
| `bin/`, `binaries.sha256`, `final-binaries-check.txt`, `final-source-check.txt` | Immutable executables and final mutation checks |
| `inventory/matrix.tsv`, `inventory/CELL/inputs/`, `inventory-generation.log` | Twelve prepared configurations and generator provenance |
| `metadata-preflight/`, `input-closure-preflight.sha256`, `runtime-preflight.log`, `input-preflight.log` | Pre-launch runtime/input preflights |
| `literal-gates/*.list`, `results.tsv`, `*-worldN.log` | Exact selected helper tests and five MPI executions |
| `prefix-evidence/sources-before.sha256`, `source-before-check.txt`, `source-after-check.txt`, `baseline-check.txt`, `production-baseline-diff.txt`, `nonowner-scripts-diff.txt` | Actual revision/baseline and pre/post source checks |
| `prefix-evidence/input-closure-before.sha256`, `input-closure-after-check.txt` | All173 input/inventory files, including namelist reference targets, captured before the first launch |
| `prefix-evidence/executables-checkers.sha256`, `executable-check.txt`, `selected-test-list.txt`, `build-association-check.txt`, `build-association.json` | Binary/checker/selection/build preflight and postchecks |
| `prefix-evidence/libraries-before.sha256`, `library-after-check.txt`, `binary-ldd.txt`, `runtime-backend-ldd.txt`, `runtime-backend.json`, `runtime-backend-check.log` | Linked-library hashes and actual BLAS API metadata |
| `prefix-evidence/environment.txt`, `mpi-version.txt`, `rust-version.txt`, `uv-version.txt`, `commands.txt` | Settings, actual versions and reproduction commands |
| `prefix-evidence/results.tsv`, `checker-results.tsv`, `terminal-summary.txt`, `terminal-status.txt` | 36 prefix-pairs, twelve checker outcomes and terminal0 |
| `prefix-evidence/CELL/prefixP/w1/launchR.log`, `launchR.exit`, `repeatR/`, `inputs.sha256`, `input-check.txt`; `CELL/validation.log` | Per-launch exits, raw records, outputs and strict validation |

All recorded source/input/binary/checker/library before/after checks succeeded;
the production and nonowner-script diffs are empty. Manifest entries describe
this immutable snapshot, not subsequent shared-main edits.

Parent independent audit handle81938 terminal0 re-ran the strict checker on all
twelve retained cells, and independently checked the
source/input closure, linked libraries and binary SHA manifests with exit0.
The parent confirmed the build-association/source/binary/input/results hashes
match those listed here. This was **revalidation of the existing artifacts,
not new MPI launches**. Its results were reported in tool output; it did not
save a new result file. The owner counted216 intact successful summary
fragments from the original72 launch logs, not216 independently rank-tagged
lines. The parent identified stdout interleaving in
`r4-real-s1-cg0-store0/prefix3/w1/launch2.log`: one `test result: ok` line is split
by another rank's output, so an anchored complete-line count incorrectly gives
three instead of four. Four intact passing summary fragments and four exact
test-success messages remain in that log; its launcher exit is0. The original
log is preserved without rewriting or normalization. Parent corrected
fragment recount subsequently completed with terminal0: all72 launcher exits0,
216 intact PASS summaries, and no FAILED/panicked/TIMEOUT markers. The initial
naive anchored-count attempt returned1 and remains preserved as a diagnosed
log-interleaving audit error, not a native MPI failure. Separately,
`repeatR/rank-N.txt` files provide individually identified rank state evidence
and are validated for all expected ranks by the strict checker.

| Artifact | SHA-256 |
| --- | --- |
| `snapshot/sources.sha256` | `70117adee28cc4a945a5ce6de869845d9767d7f5a061acfcc938a7dc4ebd4044` |
| `bin/mpi_issue179_state` | `49a860661603d30fdc6a43ab30c2cdf3560a0a6eb819476300c6da24a9f760d7` |
| `build-association.json` | `3e4ae4ad9cea005e336cb8a8dafeac71fcaf455079f0ee7cc6d8046acf118d65` |
| `inventory/matrix.tsv` | `028afba539f56e99591921efa938c65dec28cc3145d3234de20f1aeeddaff94b` |
| `prefix-evidence/input-closure-before.sha256` | `5680f8fb7eda1903544662c3917c96e1212c9894551e174217da6064e74481f9` |
| `prefix-evidence/results.tsv` | `56847f308aca0f845295ecc69cb01f464e67e6a943457d019a26deb1ba49a24c` |
| `prefix-evidence/checker-results.tsv` | `3a3da19169b53322bba1cbd6320e43fa708fc8058e2bbbcd0bc35ccc8be81294` |
| `prefix-evidence/runtime-backend.json` | `4eefaeb530028ca83705bf44e89493c38c410889cdfff46c237ba433963f9657` |
