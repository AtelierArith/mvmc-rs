# Issue #342 native negative-INFO probe

Optional developer tool. Cargo builds and tests never compile, run or read this
directory; they read the checked-in fixtures in
`tests/fixtures/issue342_negative_info/` (see its `PROVENANCE.txt`).

Scope: standalone native kernel check, NOT a full `vmc.out`/MPI trajectory.
`probe.c` links the UNMODIFIED original `matrix.c` (`CalculateMAll_real/fcmp`),
`projection.c`, `sfmt/SFMT.c`, the extracted original `makeInitialSample`
(`vmcmake.c`, SHA-256 `8431b58e...`) and the real pfapack Fortran `dsktrf`/`zsktrf`
with OpenBLAS (whose `XERBLA` reports and returns). A real/complex x `Ne=0`/`Ne=1`
matrix is run; `Ne=0` yields native INFO -5.

Reproduction (Dev Container image, Linux x86_64):

```bash
# 1. CMake-build extern/mVMC-1.3.0 (target vmc.out) in $BUILD:
#    cmake <repo>/extern/mVMC-1.3.0 -DCMAKE_BUILD_TYPE=Release \
#      -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF -DMPI_C_COMPILER=/opt/mpich/bin/mpicc
#    make -j8 vmc.out
# 2. Run the probe into an EMPTY output directory:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  c_toolbox/issue342_negative_info/run_probe.sh <repo> <empty-out-dir> $BUILD
```

Outputs `{real,complex}.{ne0,ne1}.stdout` (the fixtures), `sha256.txt` with the
source/tool hashes, and the extracted `initializer-complex.inc` /
`source-provenance.txt`. Fixture regeneration is explicit and optional; do not fill
fixtures from Rust output.

The real sampler calls the same complex `makeInitialSample` (`vmcmake_real.c:69`)
and then separately `CalculateMAll_real` (`vmcmake_real.c:102`); `probe.c` mirrors
that order. `makeInitialSample_real` is BackFlow-only and is not used.

Whole-program observation (also in
`docs/reference/c-to-julia/verification/issue-342-negative-info.md`): the same
`Ne=0` input with `NVMCWarmUp=NVMCSample=0` run through the unmodified `vmc.out`
(`mpiexec -n 1`) prints the XERBLA argument-5 messages and completes (exit 0).
