#!/usr/bin/env bash
# Optional developer reproduction (Dev Container, Linux x86_64). Never run by Cargo.
# usage: run_probe.sh <repo-root> <exclusive-output-dir> [mVMC build dir containing src/ltl2inv, src/pfapack]
# The build dir is a normal CMake build of extern/mVMC-1.3.0 (target vmc.out), see README.md.
set -euo pipefail
root=$(realpath "${1:?repo root}")
out=$(realpath "${2:?exclusive output dir}")
build=$(realpath "${3:?mVMC cmake build dir}")
src=$root/extern/mVMC-1.3.0/src
tb=$root/c_toolbox/issue342_negative_info
test -d "$out" && test -z "$(ls -A "$out")"
perl "$tb/extract_initializers.pl" "$src/mVMC/vmcmake.c" "$out"
MPICC=${MPICC:-/opt/mpich/bin/mpicc}
"$MPICC" -O2 -fopenmp -DBLAS_EXTERNAL -DMEXP=19937 -D_lapack -D_mVMC -D_mpi_use \
  -I"$out" -I"$src/mVMC" -I"$src/mVMC/include" -I"$src/common" -I"$src" -I"$src/StdFace/src" -I"$build/include/blis" \
  "$tb/probe.c" "$src/sfmt/SFMT.c" \
  "$build/src/pfapack/fortran/libpfapack.a" "$build/src/ltl2inv/libltl2inv.a" "$build/lib/libblis.a" -lpthread \
  -lopenblas -lgfortran -lquadmath -lm -o "$out/probe"
for mode in real complex; do
  for c in ne0 ne1; do
    "$out/probe" $mode $c > "$out/$mode.$c.stdout" 2> "$out/$mode.$c.stderr"
  done
done
sha256sum "$tb/probe.c" "$tb/extract_initializers.pl" "$tb/run_probe.sh" "$out/probe" \
  "$src/mVMC/matrix.c" "$src/mVMC/projection.c" "$src/sfmt/SFMT.c" \
  "$src/pfapack/fortran/dsktrf.f" "$src/pfapack/fortran/zsktrf.f" > "$out/sha256.txt"
