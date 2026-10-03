# Optional #179 grouped stored-derivative call-chain probe

Sources: `mpi_issue179_store.c` and `mpi_issue179_store_upstream.inc`.
This explicit developer command is not invoked by Cargo. It is not a full C
sampling run or an independent CG solve. It probes absolute sample-column
assignment followed by the authoritative real `calculateOO_Store_real` helper
called with the base pointer and local sample count.

Retained container `73c57e563c61`, handle **79945**, terminal **0**:
`/home/vscode/.cache/mvmc/issue179-C-store.dYQDn0`.
`r2-w1.txt` through `r4-w4.txt` retain actual stdout for worlds 2/4 and requested
widths 1/2/3/4; `build.log`, `source-binary-sha256.txt`, and `source-check.txt`
retain compiler warnings and successful provenance checks.

SHA-256 at this checkpoint:

- Driver: `2ceb4352cf06e8919c8b30d4fbec639d0aac802f1687bff7e13f86080ead9673`.
- Extracted helper: `34e0337a9f6b13d9d5999e714d3b0f308d5457768689dfa27862fdc06ecc464e`.
- Binary: `c730fc2f9455497bc1cd2acf5c236c4c3404d590acaccb0409dde7c0425f6630`.
- Upstream `vmccal.c`: `c2db5fd32c5c83be189ffd9fbb89684f0696bab7fff5f7d36abaa370274d9d2f`.
- Upstream `splitloop.c`: `3734f9deadd12f1c5af0fbf4927dabc3e77c80bdb512c3d0160a809ff0117ff8`.
- Upstream `setmemory.c`: `573d1fb995de33386e7452f5e2fa34aeb05efe14b5bf4a904eb54d2ac7cf1b9e`.

The extracted helper is bounded by `calculateOO_Store_real` and the following
complex helper in `extern/mVMC-1.3.0/src/mVMC/vmccal.c`. See the include's origin
header. The driver reproduces the assignment at line 241 and base-pointer call
at line 314. Native storage uses malloc (`setmemory.c`, lines 419/421); neither
explicit diagnostic fill, 0 or 17, claims its otherwise-unwritten contents.

Reproduce with the repaired MPICH 4.2.0 internal-PMI/Hydra runtime:

```bash
mpicc -std=c11 -O0 -ffp-contract=off -Wall \
  -Iextern/mVMC-1.3.0/src/mVMC -Ic_toolbox \
  c_toolbox/mpi_issue179_store.c -o "$probe179"
for ranks in 2 4; do
  for width in 1 2 3 4; do
    timeout --kill-after=5s 30s mpiexec -n "$ranks" "$probe179" "$width" </dev/null
  done
done
```

Set `probe179` to a new explicit output path before compiling. The unused BLAS
branch aborts if reached; NSRCG=1 selects the scalar helper. Ignored upstream
OpenMP/vendor pragmas are recorded in `build.log`; no threaded claim is made.

World 2, width 2, five samples assigns ranges 0..2 and 2..5. Assigned global
sum is 15; actual prefix helper sum is 6 with fill 0 and 40 with fill 17.
Width 1 controls read only assigned columns. This demonstrates dependence on
unwritten storage in the grouped C call chain, **not invalid C input**, and
does not establish a portable numerical expectation for that storage. Grouped
CG supported Rust design remains a separate pending contract decision; a
historical Julia feature guard must not be relabelled as C input rejection.
