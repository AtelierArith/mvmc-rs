#!/usr/bin/env bash
# Optional developer reproduction (Linux x86_64). Never run by Cargo.
# usage: MPI_PREFIX=<dir with include/mpi.h; a single-rank shim replaces libmpi> run_probe.sh <repo-root> <empty-out-dir>
# MPICC may instead name an MPI compiler wrapper (e.g. /opt/mpich/bin/mpicc).
# Builds unchanged vendored C/Fortran/C++ routines (no CMake, no BLIS), then runs the
# matrix cases on the literal operands of tests/fixtures/real_fsz/setup.txt and the five
# initializer cases. Outputs matrix.stdout, init.stdout, sha256.txt, versions.txt.
set -euo pipefail
root=$(realpath "${1:?repo root}")
out=$(realpath "${2:?exclusive output dir}")
src=$root/extern/mVMC-1.3.0/src
tb=$root/c_toolbox/issue176_real_fsz
test -d "$out" && test -z "$(ls -A "$out")"
perl "$tb/extract_initializer.pl" "$src/mVMC/vmcmake_fsz_real.c" "$out"
{ gcc --version | head -1; gfortran --version | head -1; c++ --version | head -1; } > "$out/versions.txt"
CFLAGS="-O0 -ffp-contract=off -fopenmp"
# Source list of src/pfapack/fortran/CMakeLists.txt without USE_GEMMT.
for n in dlasktrd dskmv dskr2 dsktd2 dsktrd zlasktrd zskmv zskr2 zsktd2 zsktrd \
         dlasktrf dskpfa dsktf2 dsktrf zlasktrf zskpfa zsktf2 zsktrf dskr2k zskr2k; do
  gfortran -O0 -ffp-contract=off -c "$src/pfapack/fortran/$n.f" -o "$out/$n.o"
done
c++ -std=c++11 -O0 -ffp-contract=off -DBLAS_EXTERNAL \
  -I"$src/common" -I"$src/common/deps" -I"$src/ltl2inv" -c "$src/ltl2inv/ltl2inv.cc" -o "$out/ltl2inv.o"
c++ -std=c++11 -O0 -ffp-contract=off -DBLAS_EXTERNAL \
  -I"$src/common" -I"$src/common/deps" -I"$src/ltl2inv" -c "$src/ltl2inv/ilaenv_lauum.cc" -o "$out/ilaenv_lauum.o"
gfortran -O0 -ffp-contract=off -J"$out" -c "$src/ltl2inv/ilaenv_wrap.f90" -o "$out/ilaenv_wrap.o"
if [ -n "${MPICC:-}" ]; then
  mpi_cc=("$MPICC"); mpi_flags=(); mpi_link=()
else
  prefix=${MPI_PREFIX:?set MPI_PREFIX or MPICC}
  mpi_cc=(gcc); mpi_flags=(-I"$prefix/include"); mpi_link=("$out/MPI_stub.o")
  gcc -O0 -I"$prefix/include" -c "$tb/mpi_single_rank_stub.c" -o "$out/MPI_stub.o"
fi
"${mpi_cc[@]}" $CFLAGS -DBLAS_EXTERNAL -DMEXP=19937 -D_lapack -D_mVMC -D_mpi_use \
  -I"$out" -I"$src/mVMC" -I"$src/mVMC/include" -I"$src/common" -I"$src" -I"$src/StdFace/src" -I"$src/common/deps" \
  "${mpi_flags[@]}" "$tb/probe.c" "$src/sfmt/SFMT.c" "$out"/[a-z]*.o \
  "${mpi_link[@]}" -lopenblas -lstdc++ -lgfortran -lquadmath -lm -o "$out/probe"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$out/probe" matrix "$root/tests/fixtures/real_fsz/setup.txt" > "$out/matrix.stdout"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$out/probe" init > "$out/init.stdout"
ldd "$out/probe" > "$out/libraries.txt"
sha256sum "$tb/probe.c" "$tb/extract_initializer.pl" "$tb/run_probe.sh" "$out/probe" \
  "$root/tests/fixtures/real_fsz/setup.txt" \
  "$src/mVMC/matrix.c" "$src/mVMC/projection.c" "$src/mVMC/vmcmake_fsz_real.c" "$src/sfmt/SFMT.c" \
  "$src"/pfapack/fortran/dsktrf.f "$src"/pfapack/fortran/dsktf2.f \
  "$src"/ltl2inv/{ltl2inv.cc,invert.tcc,pfaffian.tcc,trmmt.tcc,ilaenv_lauum.cc,ilaenv_wrap.f90} > "$out/sha256.txt"
