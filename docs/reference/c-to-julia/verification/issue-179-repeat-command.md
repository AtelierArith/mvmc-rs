# Frozen MPI repeatability command and evidence

Scope: same Rust implementation/configuration, fixed input/seed, fresh process
repeats. This is not cross-Julia trajectory parity or full numerical acceptance.
Strict primitive RNG algorithm/init/conversion/output/state requirements remain
for the same draw order/count. Numerically dependent branching requires separate
first-cause/threshold analysis.

## Commands actually launched

In container `73c57e563c61`, working directory
`/home/vscode/.cache/mvmc/snapshots/issue179-62b.qAZUvg`:

```bash
MPI179_CELL_REGEX='^r4-(real|cmp)-s1-cg1-store0$' \
MPI179_STEPS_LIST=3 MPI179_EXPECTED_CELLS=2 MPI179_EXPECTED_TOTAL=6 \
MPI179_TIMEOUT=60 bash scripts/verify_mpi_issue179_repeat-current.sh \
  /home/vscode/.cache/mvmc/target/issue179-62b/test-fast/deps/mpi_issue179_state-752fbb41b28c449f \
  /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA \
  /home/vscode/.cache/mvmc/issue179-repeat-current-first-v2

MPI179_EXPECTED_CELLS=55 MPI179_EXPECTED_TOTAL=495 MPI179_TIMEOUT=60 \
bash scripts/verify_mpi_issue179_repeat-current.sh \
  /home/vscode/.cache/mvmc/target/issue179-62b/test-fast/deps/mpi_issue179_state-752fbb41b28c449f \
  /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA \
  /home/vscode/.cache/mvmc/issue179-repeat-current-full
```

First handle **21017**, terminal **0**: six pairs/twelve launches passed.
Full handle **51328** terminated **1**, not passed. It uses prefixes
1/2/3 and workers 1/2/4, two launches per pair, with 60-second per-launch timeout
and five-second kill grace. BLAS/OMP/MKL/BLIS threads are 1; actual inner-worker
threshold is 1. Seed 1 is supplied by the immutable inventory input files.
Every cell retains input hashes for independent verification of the exact values.

## Hashes and source lineage

All SHA-256 values below describe the executed frozen capture, not latest shared
production edits:

| Artifact | SHA-256 |
|---|---|
| Rust binary | `cf5c188b1d858464661f50eafcb7099d9bb23ebd49058401e3f10a6fa580deb5` |
| Snapshot source manifest | `d40a0537f253b7da7b2f56074b474d021afabb195e6c9383261aea67d49149e1` |
| `sr_cg.rs` | `7df70d1074424dbe202276e54e2962574c174bf571b7d621446176f65254caac` |
| `run.rs` | `d2bbeb6ee0579dc886e5949f4021b47b85c662c3ac4f75fb937d00d7142faf05` |
| Executed repeat checker | `3ccc2572dbe92c47b4f5e7b9f90c31fcdf7aff371a2a1d866a38f4a2364f76ed` |
| Actual worker checker | `f1508697ae44e4683f2d84691d9cc41414ddc27abd105de574511682e7ce834c` |
| Inventory `matrix.tsv` | `8a18770f5352d96691c6d982da67fda9f1fa1381b43ffe38b7d024ffbae4005e` |
| Selection helper | `70d04c0914cbacbd298e460b79c886a19e6ceb79fab7946f6935bab7b8bfc54e` |
| Runtime helper | `221e37e01ef4dcb51cb1520768f93c4c9437c9031f879044a646588554de7ac3` |

Snapshot manifest is `snapshot-sources.sha256`; prior source checks for this
binary are retained in `issue179-62b-six.BQxPjU`. Runtime is the repaired
MPICH 4.2.0 container, not the historical Open MPI container.

## Exact evidence files and known checker defect

Both output roots retain `results.tsv`, `executables-checkers.sha256`,
`mpi-version.txt`, and `binary-ldd.txt`. Terminal runs also retain
`executable-check.txt`. Each `<cell>/prefix<N>/w<W>/` contains `inputs.sha256`,
`input-check.txt`, `launch1.log`, `launch2.log`, `workers1.log`, `workers2.log`,
and independent `repeat1/` / `repeat2/` captures. Rank records, CG records and
root output files are compared; actual worker scheduling IDs are checked for
valid activation but are not required to reproduce their scheduling order.

The frozen checker incorrectly requires `zvo_SRinfo.dat` for direct SR. Production
only emits that file for CG. Preserve its validation failures and the complete
original run; do not alter the live checker or retrospectively claim it passed.
The inventory contains 43 direct-SR inputs (387 pairs) and 12 CG inputs (108
pairs). After handle 51328 terminated, only affected direct-SR configurations
were launched with a separately frozen corrected checker and a distinct root.
Other errors, if observed, are not automatically attributed to this known defect.

The original 495-row table has 990 launcher exits 0, 108 repeat validations 0,
and 387 validation failures. Its three executable/checker postchecks are OK.
All snapshot source checks passed; `snapshot-source-check.txt` SHA-256 is
`31489bc0b60a4bedf55742479e40168ab2ba09c5067f85a6181ec0c73863f167`.

Corrected direct-SR run handle **83487** terminated **0**, root
`/home/vscode/.cache/mvmc/issue179-repeat-corrected-direct`. It selects exactly
43 declared-success inventory rows whose CG field is 0; expected 387 pairs.
Executed script is `scripts/verify_mpi_issue179_repeat-corrected.sh` inside the
same frozen snapshot, SHA-256
`b73ef950d3c218a54065237eaab283a1a24f0e6ade2ec589a1bd8a1e03771d23`.
Extracted validator SHA-256 is
`2dac47927ad78a04ecb178475efd37222568f7e9c27686aa441498cbe725ab8c`.
The binary, inventory, prefix/worker axes and timeout are unchanged.
The actual selection regex and thread/timeout settings are retained in
`configuration.txt`; `executables-checkers.sha256` now includes the sourced
helpers, inventory, and extracted validator. The original checker file and
captures were not overwritten.

Actual corrected table: 387 pairs, 774 launcher exits 0, 387 validations 0.
All seven pinned artifact postchecks passed. `results.tsv` SHA-256:
`3be3f415652772faa91fa995f9e42d42fb064ffffb661426ea19f35cbdc8c92d`;
`terminal-summary.txt` SHA-256:
`740014cb03182bb2f436fe5c3b2df75486ec9d4b51e31a1e241cefaba8613c48`.
Together with the original run's 108 passing CG pairs, this covers same-config
repeatability for all 495 declared configurations, not independent numerical
accuracy or a successful original 495-pair command (that command exited 1).

A separate read-only inventory/key audit terminated 0 after harvesting handle
83487. It selected CG rows from the original table and direct-SR rows from the
corrected table, required all selected launcher/validation exits 0, rejected
duplicate keys `(cell,prefix,workers)`, and required exact set equality with the
55 declared-success inputs × three prefixes × three worker settings. Actual
stdout: `COMBINED_REPEATABILITY pairs=495 fresh_launches=990 missing=0
duplicates=0 selected_failures=0 ORIGINAL_FULL_TERMINAL=1
CORRECTED_DIRECT_TERMINAL=0`. No MPI launches were restarted for this audit.

Selection and rerun command (same snapshot working directory):

```bash
export MPI179_CELL_REGEX="^($(awk -F '\t' \
  'NR>1&&$8=="success"&&$5==0{if(n++)printf "|";printf "%s",$1}' \
  /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/matrix.tsv))$"
MPI179_EXPECTED_CELLS=43 MPI179_EXPECTED_TOTAL=387 MPI179_TIMEOUT=60 \
bash scripts/verify_mpi_issue179_repeat-corrected.sh \
  /home/vscode/.cache/mvmc/target/issue179-62b/test-fast/deps/mpi_issue179_state-752fbb41b28c449f \
  /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA \
  /home/vscode/.cache/mvmc/issue179-repeat-corrected-direct
```

The prospective corrected checker also pins the sourced selection/runtime helpers,
inventory and extracted repeat-evidence validator in its hash manifest, records
the effective configuration in `configuration.txt`, and retains a terminal summary.
Six focused repeat-checker tests cover direct SR without SRinfo, missing CG SRinfo,
altered rank/output, missing rank, and nonzero/timeout/killed launcher exits even
with identical retained files. The independent utility suite currently passes
46 tests via `uv run --no-project python -m unittest discover -s scripts
-p 'test_*mpi_issue179*.py'`. These are checker tests, not MPI accuracy evidence.

Supporting independent fixed-Julia-operand C replay and residual results remain
separate: see [C replay provenance](../../../../c_toolbox/mpi_issue179_cg_replay.md).
Their Rust/C/Julia scope is the first common solve only; no arbitrary downstream
CG forward tolerance is adopted.
