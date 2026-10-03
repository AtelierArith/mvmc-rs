# Fresh C-recurrence MPI diagnostic checkpoint

Container `73c57e563c61`, snapshot
`/home/vscode/.cache/mvmc/snapshots/issue179-C-cg.xKpf6t`.
Snapshot creation handle 4484 terminal0 verified identical source hashes before
and after copying, and every copied file. Manifest SHA-256:
`92e32ee63c6987ab0b4b55f7ec6f57dcbdd5d2827f67b0d4a370f89f1c498302`.
Captured CG kernel SHA-256:
`905d3fecb6b3e6e8c5b5574c469ae955c9ae6b22d45305d3ba32ead7239d39b2`.
Captured runner SHA-256:
`a7b2cf1902e04a05e5250ca277e5766ecb8fc1ad47abfeb8d120244fbd5a63a3`.
Later shared-owner changes are not part of this snapshot.

Bounded 600-second build handle **29585**, terminal **0**, Finished test-fast
17.26s; isolated named-volume target `target/issue179-C-cg`.
Logs under `/home/vscode/.cache/mvmc`: `issue179-C-cg-build.json`,
`issue179-C-cg-build.log`, `issue179-C-cg-build-source-check.txt`.

Actual capture handle **72496**, terminal **0**, stdout:

```text
C_KERNEL_DIAGNOSTIC cells=6 expected6 sampler_bad=0 root=/home/vscode/.cache/mvmc/issue179-C-cg-first.TBh6at
```

Six cells: world4/width1/store0, real prefix3 and complex prefix2, each workers
1/2/4. Launches use timeout60s/kill-after5s, BLAS/OMP1, actual worker threshold1.
Root `results.tsv` has six rows with launch and exact-sampler comparison exits0.
Per-cell `prefixN/wW/rust` retains actual states, CG sidecars, launch logs,
sampler comparison stdout, and numeric diagnostic JSON/stderr per rank.
`source-check.txt` has zero non-OK lines;
`binary-checker-check.txt` reports all recorded hashes OK.

Binary SHA-256:
`94f747a30a7dba9c634c5fd809c966f84fda5f66c6b85f5c61924e6c1f1a0841`.
Frozen sampler comparator SHA-256:
`72670a44a7f871a51a40fd0b7c685a775be5c128c86da6e1edc197d3cb0cfe23`.
CG diagnostic checker SHA-256:
`6fa168d1d892b38b8789866d0e1665233a5ca4402b25478ba55a389aa96ecf36`.
Exact paths are retained in `binary-checker-sha256.txt`.

The Julia reference is still the independent **old recurrence** capture in
`issue179-cgref.yqrDc0/matrix`; it is not silently changed to the new C algorithm.
CG diagnostics are non-gating output, with errors retained; terminal0 means
the six launches and sampler exact comparisons succeeded, not numerical PASS.
Complex rank0 finishes 12/12 iterations (the old Rust capture finished 12/13).
Real rank0 finishes 10/10/7; first numerical product difference is still phase2
global sum, event9, about 1.0842e-19. Later amplification remains unaudited for
acceptance. No tolerance has been selected.

Independent residual audit outputs, when available, are retained separately
as `real-independent-residual-audit.stdout`/`.stderr`, with copied-snapshot
auditor hashes in `residual-auditor-sha256.txt` and its check file. Audit terminal
status and results must be reported separately; diagnostic completion is not
solver acceptance. The old PtoNII/32750 failures remain unmodified evidence.

Independent real residual audit handle **28508**, terminal **0**. Stdout SHA-256:
`de067ac05dd16bc78346df45c808a3122689b29a21686dc88d1cab5982679eaf`.
Frozen auditor/helper hashes are respectively
`1a9b3f64ffb49450100be422200320f284961c08b50884774a040e60e3e63931` and
`440cbcb61b58663775fb7e69c52f71464bdd7ee3f4fddce7ebdeb79fe43eb7c1`;
post-run hash checks OK. Actual final CG solution differences by solve are
4.69636e-9, 9.52187e-4 and 1.04710e-4 (not parameter increments).
Solve2 Rust backward error is approximately 1.22896e-5 and actual residual
infinity norm 2.03104e-7; old Julia values are 7.81660e-6 and 1.29324e-7.
Solve1 reconstructed operator/gradient agree; solve2 operator difference is
1.02126e-10, solve3 1.25009e-5. These are retained conditioning/residual
diagnostics, **not acceptance bounds or a numerical PASS**. The named real
auditor is not bypassed or relabelled to cover complex inputs.

Separate complex auditor execution **35520**, terminal **0**, uses actual
Orbital ComplexType1 and Gutzwiller/Jastrow ComplexType0, mandatory real/imag
sample planes and four ranks/global Wc12. Initial execution **19771** failed an
incorrect all-complex header assumption; it is not coverage. Frozen stdout:
`evidence/issue-180-cg-complex-audit-35520.stdout.txt`, SHA-256
`4189a62cfc8e5c7041608b00255c13b9c595c038122dfc650de5c646f7e0d9c5`.
Executed auditor archive SHA-256:
`aa4893a92262ff1aa7dea9bef36954c64f4bdbb175b2bf51694abe230f285f2c`.
Actual dimension20, iterations12/12, final CG solution differences
4.01155e-10 and 1.57485e-9. Step2 backward errors Julia/Rust are approximately
2.89188e-8/3.31359e-8. This is independently retained complex residual
diagnosis against old Julia recurrence, **not C-algorithm numerical parity**.
