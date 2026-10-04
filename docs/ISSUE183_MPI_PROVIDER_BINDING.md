# Optional MPI provider/startup binding

Related to #183/#185. Publication SOURCE base main
`4347fa88917f3a714a8f30424e75bddb3851dd12`. Synthetic controls were executed
on frozen d24787339804e0a32767bc6b648f6118b8e0d006 plus this reviewed patch;
the four-file main delta is parser-only and does not overlap these seven paths.
No installer, numerical gate or optional dispatch has run for this patch.

Only the MPI family in optional-gates.yml reuses the existing reviewed
issue234 installer before Rust MPI compilation. It builds job-local MPICH4.2.0
with ch4:ucx, Hydra, PMI1 and without PMIx, using the unchanged pinned release
archive SHA and recipe. The same installer already proved real worlds2/4 at
PR245 exact4a9/run37166158453; that is historical evidence for the installer,
not evidence this new optional family has executed.

Driver preflight validates complete raw installer/startup receipts, every
world2/4 rank, configured prefix and actual compiler/launcher resolution. It
verifies all live installer-bound files (including headers and actual provider
contents), copies the raw proof, then invokes the unchanged MPI Rust gate.
After selected execution it verifies provider/receipt/tool immutability and
that the selected ELF resolves exactly one libmpi in the bound prefix, with no
PMIx or missing-library linkage. The seal includes the existing complete
artifact closure; aggregate additionally checks the MPI receipt schema,
startup ranks/status, before/after closure, header/tools and selected binary
hash. Failed installer receipts are retained as a non-PASS sidecar even if
the driver never started.

Normal Cargo tests never install/run this C startup program or a numerical
C/Julia/toolbox oracle. Non-MPI families, worker/rank assertions, model scopes,
sample counts, expected values and numerical tolerances are unchanged.

Focused controls are synthetic only: capture failure before Cargo compilation
(existing `cargo nextest --version` discovery is permitted exactly once);
failed installer sidecar; rehashed missing receipt, failed terminal,
duplicate/wrong startup rank/world, native failure, tampered receipt, changed
runtime, wrong library/linkage, wrong selected ELF, launcher or header. The
existing four-family synthetic positive package now contains a bounded fake
MPI receipt; it is still not native model execution.

Actual light verification, each bounded by timeout -k 5s 60s:

```
uv run --offline --no-project --no-python-downloads --python /usr/bin/python3 python -B scripts/test_optional_gates_183.py
uv run --offline --no-project --no-python-downloads --python /usr/bin/python3 python -B scripts/test_optional_gate_aggregation_183.py
```

Receipt `/tmp/issue183-mpi-provider-light-closure.Nnj4RJ`, owner session77591:
driver19/19 and aggregation12/12 PASS, aggregate0; source/tools and all1,452
fixture files pre/post0. uv0.12.21, Python3.12.3. Fixtures are exact d247 main
blobs and d247's Julia gitlink c0788c34a6a5753c611633a97cd1ea233203320c,
not a Julia execution. Earlier d2DP7i and 03nL6R terminal1 receipts are retained:
first overly broad Cargo assertion plus missing Manifest, then missing static
fixture directory. Complete closure was hydrated before the successful run.
This publication reconciliation is SOURCE-only, not a new4347 test execution.

No heavy Cargo/build/model/dispatch authorization is implied. All-four-family
actual dispatch/seal/aggregate evidence and wider #185 scenario coverage remain
pending; this patch does not complete #183.
