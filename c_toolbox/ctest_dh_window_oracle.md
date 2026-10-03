# DH C-window history/probe provenance

The separate Julia observer includes only the setup/read-only observation
portion of `ctest_prefix_oracle.jl`, ending before its canonical model loop.
It runs the same original reference runner on the exact DH runtime-test inputs,
with steps=window=3 and the original seeds. It captures full declared parameters
after SR/synchronization. The original vendored source is never edited.
FSZ requires `MVMC_CTEST_NATIVE_FSZ_BRIDGE` pointing to the independently verified
bridge documented by `scripts/reference_native_fsz_energy.jl`.

`ctest_dh_opt_window.c` includes the existing finite-input standalone adapter,
renaming its `main`, and supplies the input's orbital general/AP/P counts.
It executes the identical extracted `avevar.c` functions documented in
[C-window probe](ctest_opt_window.md): copyright lines 1–27 and full function
bodies lines 34–260, upstream SHA-256
`509a573944a5eabde93864346902674faaff6c23d0c88f89867263f8372dd51a`.
There is no C numerical-body change. AP/P auxiliaries use C lines 224–245;
the main window layout does not depend on this filename split.

Reproduce externally, from the repository root:

```sh
stage=$(mktemp -d /tmp/mvmc-dh-c-window.XXXXXX)
cc -O0 -ffp-contract=off c_toolbox/ctest_dh_opt_window.c -lm -o "$stage/dh-probe"
MVMC_CTEST_WINDOW_PROBE="$stage/dh-probe" \
MVMC_CTEST_NATIVE_FSZ_BRIDGE=/absolute/path/to/verified-bridge \
  julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/ctest_dh_window_oracle.jl "$stage" \
  dh2_real,dh2_cmp,dh2_fsz,dh4_real,dh4_cmp,dh4_fsz,dh24_real,dh24_cmp,dh24_fsz 3
# To aggregate already captured histories with the actual C orbital branches:
julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/ctest_dh_reaggregate.jl "$stage" "$stage/dh-probe"
```

Actual reference environment: native Linux x86_64, GCC 13.3.0,
`-O0 -ffp-contract=off -lm`, C linked to libm/libc without BLAS; Julia 1.13.1
with pinned Manifest-v1.13.toml and one-thread ILP64 OpenBLAS. The per-case
provenance pins each generated input, native executable, wrapper/excerpt,
observer and C source. No Rust result enters either history or expected output.
The authoritative body retains compiler filename-size warnings; the adapter
bounds output heads. Standalone aggregation is distinct from full C/MPI runs.
