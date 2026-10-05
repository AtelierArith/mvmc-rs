#!/bin/bash
# Build the rank-state-dumping copy of the authoritative C vmc.out (issue #179).
# Usage (inside the Linux x86_64 Dev Container, from the repository root):
#   c_toolbox/mpi_matrix_179/build.sh /tmp/mvmc-179-c
# The vendored extern/mVMC-1.3.0 tree is copied, never modified. Two patches are
# applied in order: sr_operand_dump (number formats of upstream's own SR operand
# dumps) and rank_state.patch (read-only per-rank dump of the sampler state:
# Counter[], EleIdx[] and the SFMT state, after each optimization step).
set -euo pipefail
out=${1:?output directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$out"
mkdir -p "$out/build"
cp -r "$root/extern/mVMC-1.3.0" "$out/src"
(cd "$out/src" &&
  patch -p1 < "$root/c_toolbox/sr_operand_dump/dump_sr_operands.patch" &&
  patch -p1 < "$root/c_toolbox/mpi_matrix_179/rank_state.patch")
cd "$out/build"
export CC=gcc CXX=g++ FC=gfortran
cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF \
  -DMPI_C_COMPILER=/opt/mpich/bin/mpicc \
  -DCMAKE_C_FLAGS="-D_DEBUG_DUMP_SROPTOO -D_MVMC179_RANKSTATE" > cmake.log 2>&1
make -j4 vmc.out > make.log 2>&1
sha256sum src/mVMC/vmc.out
sha256sum "$root/c_toolbox/sr_operand_dump/dump_sr_operands.patch" "$root/c_toolbox/mpi_matrix_179/rank_state.patch"
