# Scoped MPI / inner-worker interaction evidence

Related to #179/#182; not full numerical parity or a current-main milestone.
Container: `73c57e563c61`, Linux x86_64, MPICH4.2.0/internal Hydra PMI,
OpenBLAS0.3.26 LP64. Repository base is committed
`11cc735d1b52ffe314e5d25083e126e7d56b32cf` plus the retained owned recorder and
selection overlays, **not post209 main**. Complete production files are frozen
under the container root below; shared dirty production was not imported.

Container root:
`/home/vscode/.cache/mvmc/issue179-worker-QP-funneled.ZeoPym`.
`source.sha256` records pre-build tracked regular files; `source-check.txt` and
`review-source-check.txt` are post-build/read-only checks from `root/repo`.
Gitlink contents originate from the earlier frozen inventory lineage, not
uncommitted reference edits. `overlay.patch`, `build.json`, `build.log` and
`build.exit` retain the source changes/build result (0, 1.91s).

- Recorder SHA: `ad855c8c51bbcf65439b97d1c93af3ac562afc06da8e0de6b86cd8ef8ff9bdcb`.
- Immutable `bin/state` SHA: `d307cb05627b564fae7016247676faaf5f072d69dac3783a283bcbc012306cd8`.
- Actual v2 generator SHA: `79693e11b0a0c0895d4a09d1c3be273bc214c75e2de1e38f774f97f6a8726584`.
- Actual results SHA: `c1f4eb3d66037911c229d5c8ba5f147bef05cd787aeecdb32fc7bb73a01751af`.

The first `QP-evidence` attempt failed inventory preflight, before MPI launches.
Its `launch-source-binary.sha256` refers to the initial generator (`76a3…`),
not the corrected v2 generator. That old failure is preserved.

Actual successful v2 evidence is `QP-evidence-v2/`:

- `repeats/results.tsv`: six named cells × workers1/2/4 =18 pairs,36 launches,
  all launcher and validator statuses0; same handle60325 terminal0.
- `generator-binary.sha256`, `generator-check.txt`: actual v2 script/binary.
- `repeats/executables-checkers.sha256`, `executable-check.txt`: sourced
  selection/environment, comparison scripts and generated matrix hashes.
- `repeats/configuration.txt`, `mpi-version.txt`, `binary-ldd.txt`: actual
  configuration/backend; BLAS/OMP/MKL/BLIS1, threshold override1, timeout90s.
- `original-inputs.sha256`, `original-input-check.txt`, `inventory/` and each
  pair's `inputs.sha256`/`input-check.txt`: original and derived input lineage.

Exact developer command from `root/repo` (choose a NEW output directory):

```sh
bash scripts/verify_mpi_issue179_qp_workers.sh ../bin/state \
  /tmp/mvmc-issue179-evidence.DFupvq /NEW/exclusive/QP-evidence
```

Inputs are public normal complex Heisenberg namelists copied from the generated
inventory. Only `NSPGaussLeg` is changed to31/32/33: synthetic quadrature scenarios,
not the unchanged original projector or an independent numerical expectation.
Width2/world2/4 use actual grouped sampling; seed1 plus chain offset, direct SR,
store0, sample3/warmup1, effective steps/window1. This is real production sampling
via `vmc_para_opt`, not a synthetic sampler, but initialization is recorder-driven,
not full CLI/root-negative-clock lifecycle. Prepare-only CLI only generates inputs.

Every workers2/4 rank has actual positive parallel QP entries and independently
checked worker IDs. Every wrapped collective asserts the initiating OS thread
and no Rayon worker context. These assertions cover reducer calls through this
harness, not every conceivable direct MPI call outside it. Both fresh same-config
runs compare raw RNG624/cursor, draw/events/proposals/acceptance/configuration,
SR records and root output. This is same-implementation repeatability, not
cross-language bitwise computed-value acceptance. Transfer entries are zero:
**transfer activation is NOT COVERED**. Default threshold32 is NOT COVERED.

Next proposed transfer gate (not implemented here): a public normal Hubbard
input with real hopping, InterAll absent, FSZ disabled and supported QP/width2;
confirm the exact C reader accepts duplicate transfer rows before generating
31/32/33 jobs. Split only a selected coefficient among identical rows, retain
declared order and document any changed floating-point summation. Hash original
and derived files; do not claim unchanged arithmetic or independent parity.
Require actual transfer worker entries, per-rank QP ownership and the same
FUNNELED checks, world2/4 × workers1/2/4 × two fresh runs with bounded launches.
This proposal requires source/input-contract review before implementation.

## Focused source gates after the execution capture

Same frozen `root/repo`, unchanged recorder SHA `ad855…`:

```sh
CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-matrix20-current \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout --kill-after=10s 600s \
  cargo clippy --locked --profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue179_state -- -D warnings
uv run --no-project python -m unittest discover -s scripts -p test_mpi_issue179_selection.py
bash -n scripts/verify_mpi_issue179_qp_workers.sh scripts/mpi_issue179_selection.sh scripts/verify_mpi_issue179_repeat.sh
```

Clippy handle62472 terminal0,3.36s: `clippy-command.txt`, `clippy.log`,
`clippy.exit`, `clippy-source-check.txt`. Selection10 PASS,0.183s:
`selection-tests.log`, `selection-tests.exit`; expected Unsupported stderr is
the missing-feature negative test. `bash-n.exit` is0. A separate host rerun of
the same selection SHA passed10 tests,0.209s, using uv0.12.21.

The separate exact C replay's `review-extraction-check.txt` records48 byte-exact
matches between stored operands and fresh read-only extraction of the original
observer dimension/matrix/RHS records. It does not supply an independent SR
assembly expectation; DPOSV outcomes alone are independently computed.
