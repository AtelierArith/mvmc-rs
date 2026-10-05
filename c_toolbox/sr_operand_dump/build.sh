#!/bin/bash
# Build an operand-dumping copy of the authoritative C vmc.out (issue #358).
# Usage (inside the Linux x86_64 Dev Container, from the repository root):
#   c_toolbox/sr_operand_dump/build.sh /tmp/mvmc-sr-dump
# The vendored extern/mVMC-1.3.0 tree is copied, never modified. Only the
# number formats of upstream's own debug dumps change (see the patch).
set -euo pipefail
out=${1:?output directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$out"
mkdir -p "$out"
cp -r "$root/extern/mVMC-1.3.0" "$out/src"
(cd "$out/src" && patch -p1 < "$root/c_toolbox/sr_operand_dump/dump_sr_operands.patch")
mkdir -p "$out/build"
cd "$out/build"
export CC=gcc CXX=g++ FC=gfortran
cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF \
  -DMPI_C_COMPILER=/opt/mpich/bin/mpicc \
  -DCMAKE_C_FLAGS="-D_DEBUG_DUMP_SROPTOO -D_DEBUG_DUMP_SROPTO_STORE"
make -j8 vmc.out
sha256sum src/mVMC/vmc.out
