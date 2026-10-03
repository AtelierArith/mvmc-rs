# Correct the stale grouped PhysCal smoke boundary

Related to #179/#178/#183; no issue completion claim. Owned scoped change:
`crates/mvmc-core/tests/mpi_physcal.rs`. No production/runner/ledger changes.

The prior ignored smoke expected changing NSplitSize to2 to reject **all**
PhysCal. That contradicts current `validate_grouped_runtime`: normal real/complex
Green, Lanczos0 and no additional OptTrans-derived sectors are supported with
grouped standard projection. The fixture is real Heisenberg, NSPGaussLeg8,
NMPTrans-1, no general orbital, no InterAll, Lanczos0. Those actual properties
are asserted before execution, retaining its nontrivial QP projection rather
than substituting an identity QP case. The gate now uses the existing offline
`tests/fixtures/physcal_181/heisenberg_chain_real` copy instead of depending on
a vendored checkout. Existing fixture provenance remains authoritative.

The old blanket negative is replaced by an independent **grouped Lanczos1
capability rejection**, not deleted/ignored. A caller-owned data/state/RNG
invocation verifies full Debug data/state equality, unchanged actual draw
count/next624, diagnostic NSplitSize+NLanczosMode, and absent output directory.
This is before model initialization/sampling, after communicator initialization;
it is not a pre-MPI.Init recovery claim or arbitrary single-rank fault injection.

## Positive assertion boundary

On actual world2 and world4, each rank executes width1 (world reducer) and
width2 (group reducer), two fresh public preparations/runs per configuration:

- sample3, warmup1, quantity windows1; fixed seed1 plus actual group offset;
  BLAS/OMP1, inner workers1. Not the original sample100/warm10 workload or
  20-step optimizer matrix.
- Parsed normal standard-QP support properties, zero preparation draws and
  exact primitive next624 for expected seed before internal initialization.
- All canonical fixed projection/RBM/Slater/OptTrans parameter values remain
  exactly unchanged by measurement.
- Every saved electron index/configuration/count/projection/spin array,
  counter, actual words_consumed and next624 match the same-config second run.
- Energy/weight/spin accumulators and all three Green arrays are finite and
  repeat with explicit abs1e-12+rel1e-12, appropriate to this same-input,
  same-implementation/configuration check; this is not a reference-accuracy
  budget or cross-configuration equality claim. This is the existing scoped
  S196 PhysCal repeat gate's elementary/BLAS math budget, not a newly loosened
  independent golden or solver forward-error tolerance.
- Global root checks the exact five-file set, matching line/field/blank layout,
  exact indexed labels (4/8 fields for indexed Green files), finite numeric
  payload and same bound on repeated output. Every rank joins result agreement.
  No optimized-parameter output is expected for PhysCal.

Output payload checks are **indexed repeatability**, not independent C schema
validation: they do not separately require six-column energy, ten-column Green
or one-record factored output widths. The same-shape checks must not be used as
evidence of those independent formatting contracts.

The pass message now truthfully says MPI world/grouped normal PhysCal, rather
than claiming a serial run. Feature-disabled selected smoke still explicitly
reports Unsupported and panics; no ignored/zero-selected success is added.

## Frozen actual proof

Initial handle98185 compile terminal101 because the newly used Reducer trait
was not imported. No MPI launch occurred. Retained failed root:
`/home/vscode/.cache/mvmc/issue179-physcal-current.vMvYYR`.

Corrected handle41878 terminal0, separate root:
`/home/vscode/.cache/mvmc/issue179-physcal-fixed.sbGDW2`.
Committed base `886574416777c9321ae8048972ae2705fbee46bf` plus **only this
owned test patch**, not shared dirty production. Exact test SHA:
`2ee8b3b73a243a0bc6179f288fb30d39a57d413263655709d698740e51797b76`.
Source manifest SHA:
`7bb86a769d171b4f7dd03d4aa54f99380230bca0257e64556673bbbb21a873b3`.
Patch SHA:
`ab6a70f7b2c1ba4ec42f38524904b9fd83369d14689e9963b3b001d621678749`.
Environment SHA:
`94fea8fabc717e11af33b4d2414c2064989b0c26193d4339dfdf7739fe0a9e37`.
Actual repaired MPICH4.2 internal Hydra container73c57e563c61, Linux x86_64,
Rust1.99, LP64 OpenBLAS.26; target named cache
`/home/vscode/.cache/mvmc/target/issue179-parallel-literals`.

Feature-enabled binary `mpi_physcal-0ba8d3b065cb65f9` SHA:
`2c9d6d5e836692a8dd1871439afc4524e7941b3f7922979fdf850c0af7ef994b`.
`--ignored --list` explicitly selected the exact smoke; world2 and world4
launches each exited0, each rank reported1 passed0 failed0 ignored.
Build exit0 4.08s; focused all-features clippy exit0 2.48s.
World2 stdout SHA:
`467effb26fa2133671ff85081b8ee473291ef1c3adf7f697d4f4d753237c4bc9`.
World4 stdout SHA:
`f45103dfb06ff812ac128e5e8871266800c6a476ce4b51419129ae2cba539dd4`.
Source and binary postchecks passed. Shared owned file rustfmt/diff checks passed.

Commands from the isolated repo, with the recorded target/LIBCLANG_PATH and
BLAS/OMP/inner worker environment:

```sh
timeout --kill-after=10s 600s cargo test --locked --profile test-fast -p mvmc-core --features mpi --test mpi_physcal --no-run --message-format=json
export MVMC_RS_MPI_PHYSICAL=1 MVMC_RS_INNER_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# MPI179_PHYSCAL_OUTPUT must be a new, absent directory shared by all ranks.
MPI179_PHYSCAL_OUTPUT="$ROOT/world2" timeout --kill-after=5s 180s mpiexec -n 2 "$BINARY" --ignored --exact mpi_physcal_reduces_fixed_parameter_samples --nocapture
MPI179_PHYSCAL_OUTPUT="$ROOT/world4" timeout --kill-after=5s 180s mpiexec -n 4 "$BINARY" --ignored --exact mpi_physcal_reduces_fixed_parameter_samples --nocapture
timeout --kill-after=10s 600s cargo clippy --locked -p mvmc-core --all-features --test mpi_physcal -- -D warnings
```

Retained `results.tsv`, `world2.log`, `world4.log`, four repeat output directories
per world, `selected-tests.txt`, `build.json/log/exit`, `clippy.log/exit`,
`matrix.exit`, `test-overlay.patch`, `source.sha256/source-check.txt`,
`binary.sha256/binary-check.txt` allow independent review. This scoped update
does not map all MPI87 public messages/results, compare to independent C/Julia
Green numerics, cover workers2/4, full Lanczos, or complete grouped20 models.

## Final strengthened raw-state / immutable-bit proof

After41878 terminal, parent review required raw SFMT state/cursor in addition
to next624/count and signed-zero-preserving fixed values. The owned file now
compares `Sfmt19937Rng::state_snapshot()` at preparation, between same-config
repeats, and before/after the rejected grouped Lanczos invocation. Fixed values
are compared as `(real.to_bits(), imag.to_bits())`, including negative zeros.
These are **immutable loaded-copy/discrete RNG** contracts, not bitwise gates
for computed numerical observables. The prior2ee source proof above remains
historical and does not prove these added assertions.

New handle45233 terminal0, new root
`/home/vscode/.cache/mvmc/issue179-physcal-rawstate.FvQewe`, same committed
base886574 plus **only final owned test patch**. Final source SHA:
`d73a329b30b345349c932ff03522f443ea10ceb9060299b88531696606e4824c`.
Final full input/source manifest SHA:
`b82176426bb10558b0c20aa1767c8fb7a8c88be74cc22e6fa3b8f9d022713fad`.
Patch SHA:
`5bd842322aff63794da98c34a51a852a76b150466db501a77a9b73cb651cb1f5`.
Final binary SHA:
`d8b58e87bcdeca60da0e72e1d6812c56032fa660be3cd3916fb9f42536f5d2e1`.
The binary is retained at this root's `bin/mpi_physcal`, verified by
`retained-binary.sha256`; use that copy for reproduction, not an incrementally
reused target path. Earlier target binary hashes identify binaries at their
recorded execution time and are not claims about the currently rebuilt target.

Both world2/4 launches exited0, selected the exact ignored gate nonzero, each
rank reported1 PASS/0 fail/0 ignored. Build exit0 3.52s; all-features focused
clippy exit0 2.19s; source/binary postchecks exit0; shared fmt/diff checks exit0.
World2 stdout SHA:
`4ab0c9e6043a79b3541d553255d1449b0419b00ab6711314364353998fad54d7`.
World4 stdout SHA:
`c537f09b0030ee234641c1688a176470da9da3992a5df035667ebbf8d0917b1d`.
Same commands, backend, worker configuration, input and positive/negative scopes
as above; no wider C/Julia numerical, schema, or full model claim follows.
