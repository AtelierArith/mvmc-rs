# #179 canonical-CG discrete checkpoint

Container `73c57e563c61`; paths below are retained container paths.
Comparison handle **3028**, terminal **0**, actual terminal stdout:

```text
CG_STRICT_DISCRETE cells=108 expected=108 bad=0 evidence=/home/vscode/.cache/mvmc/issue179-cgfix-strict108.YovVvt
```

Coverage is world 2/4 × real/complex/FSZ × NStore 0/1 × prefix 1/2/3 ×
workers 1/2/4, **requested group width 1 only**. Each comparison checks every
rank's exact RNG states, actual draw counts/word streams, proposals, acceptance
decisions, saved configurations, counters and discrete metadata. Independent
C-written flags masks scope otherwise-unwritten metadata; raw captures remain
unchanged. Computed floating-point arrays, SR increments, CG numeric sidecars,
root numerical outputs and justified numerical bounds are **not checked by this
discrete-only command**. This is not full #179 or full numerical CG parity.

Audit root `/home/vscode/.cache/mvmc/issue179-cgfix-strict108.YovVvt`:

- `results.tsv`: header plus 108 rows, all `exact_discrete_exit=0`.
- `<cell>-p<prefix>-w<workers>.log`: actual per-cell comparison stdout/stderr.
- `compare_mpi_issue179.py`, `checker-sha256.txt`, `checker-check.txt`: frozen
  comparator and successful post-run hash check. SHA-256:
  `72670a44a7f871a51a40fd0b7c685a775be5c128c86da6e1edc197d3cb0cfe23`.
- `<cell>-c-flags.txt`: independently read input flag values/written masks and
  reader/enumerator/helper provenance hashes.
- `scope.txt`: exact-discrete-only scope; zero tolerance arguments are unused
  for computed comparisons, not selected portable numerical bounds.

Rust captures use frozen snapshot `issue179-cgfix.PtoNII`, manifest SHA-256
`31d62d4dea86309d9c4e2179ae82db93a7024f5436ca9a63a85d7c6c8edc0432`,
binary SHA-256
`b97917e94ee84b479dceef4c79e6e8113ed3d5e1d4acb748bfb75c5969f1c24a`.
Capture roots under `/home/vscode/.cache/mvmc`:

- World 2 real/complex: `issue179-cgfix-matrix.PYvMcv`.
- World 4 real/complex: `issue179-cgfix-world4.iJRyvh`.
- FSZ worlds 2/4: `issue179-cgfix-fsz.1n78nS`.

Rust state directories are `<root>/<cell>/prefix<prefix>/w<workers>/rust`.
Independent Julia capture handle **21388**, terminal **0**, 36 references,
expected 36, bad 0; each reference is reused across the three worker settings.
Reference directories:
`/home/vscode/.cache/mvmc/issue179-cgref.yqrDc0/matrix/<cell>/prefix<prefix>/julia`.
That root retains `matrix-results.tsv`, `matrix-source-check.txt` and
`source-sha256.txt`; Rust roots retain their source/binary checks separately.

Initial handle **71391**, terminal **1**, failed before any comparison because
the requested frozen helper path was absent. Its root
`/home/vscode/.cache/mvmc/issue179-cgfix-strict108.nm7A4E` has only a TSV header;
it is retained failed setup evidence, not part of the 108 successful cells.
The corrected command used the previously frozen C-flags helper at
`issue179-strict-post.7E1alt/scripts/mpi_issue179_c_flags.jl`.

The original 165-reference job 81717 and pre-CG-repair Rust 495 captures are
separate lineages. Neither this checkpoint nor incidental computed equality in
individual cells upgrades those old captures to current full-matrix acceptance.

## Retained actual-worker audit (same old source lineage)

Handle **30135**, terminal **0**, actual stdout:

```text
RETAINED_OLD_CG_ACTUAL_WORKERS cells=108 expected108 bad=0 root=/home/vscode/.cache/mvmc/issue179-cg108-worker-audit.3QdiFq
```

That root retains `results.tsv` (108 rows), actual per-rank classifications in
`<cell>-p<prefix>-w<workers>.log`, copied `checker.py` and post-run hash check.
Checker SHA-256:
`f1508697ae44e4683f2d84691d9cc41414ddc27abd105de574511682e7ce834c`.
This checks requested/configured pool and actual kernel calls/items/worker
entries/IDs, with separate small/empty serial-domain labels. It does not require
every requested worker ID to be scheduled, and it does not substitute pool
capacity for kernel activation. It is retained old-PtoNII worker evidence only,
not current C-kernel verification or numerical MPI parity.
