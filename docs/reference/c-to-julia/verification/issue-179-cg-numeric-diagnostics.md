# #179 actual CG sidecar diagnostic checkpoint — not acceptance

Handle **32750**, terminal **1**, container `73c57e563c61`, actual stdout:

```text
CG_DIAGNOSTIC_ONLY ranks=324 expected324 bad=1 root=/home/vscode/.cache/mvmc/issue179-cg-numeric-diagnostics.GlAmGY
```

Root retains 324 per-rank comparison rows in `results.tsv`, paired-comparison
JSON/stdout and `.stderr` per cell, and copied `checker.py` with its hash/check.
Checker SHA-256:
`6fa168d1d892b38b8789866d0e1665233a5ca4402b25478ba55a389aa96ecf36`.
Owned source: `scripts/mpi_issue179_cg_diagnostics.py`. It never selects a
numerical tolerance; its exit checks record presence, shapes, keys and discrete
CG events. The 108 cells/rank sources are those in `issue-179-cg108-checkpoint.md`.

There are **48 structural comparison failures**, preserved, not ignored as
floating noise. First failing cell: world4 complex, width1, CG, NStore0,
prefix2, worker1, rank0. Rust has 131 events and solve iteration counts 12/13;
Julia has 126 events and counts 12/12. Actual evidence:
`r4-cmp-s1-cg1-store0-p2-w1-r0.stderr` and original paired `cg-rank-0.txt` files.
The previously passing sampler exact-discrete comparison excludes these CG
solver sidecars; it does not establish equal CG stopping/iterations.

World4 real, width1, NStore0, prefix3, worker1, rank0 has equal event structure
but first numeric difference at `n:event-000009-product`, index2:
Rust 0.00032703927060699655 versus Julia 0.00032703927060699644;
absolute difference 1.0842021724855044e-19, product scale 0.003537216365179344.
This is observed phase2/global sum; the preceding local product and search
records match. Later solution differences reach approximately 0.0199230266.
This amplification is **not numerically accepted**. Actual operator products,
refresh/stopping behavior, conditioning and independently calculated backward
residuals require investigation with the solver owner before selecting bounds.

No fresh whole 495-cell run is started before pending production corrections
and owner readiness. Original captures remain unchanged.
