#!/usr/bin/env bash
# Optional developer probe. Never invoked by Cargo or ordinary Rust tests.
set -euo pipefail
out=${1:?new isolated build directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
test ! -e "$out" || { echo "requires a new output directory: $out" >&2; exit 1; }
mkdir -p "$out"
cp -r "$root/extern/mVMC-1.3.0" "$out/src"
patch -d "$out/src" -p1 < "$root/c_toolbox/sr_operand_dump/dump_sr_operands.patch"
patch -d "$out/src" -p1 < "$root/c_toolbox/physcal_native/state_dump.patch"
uv run --no-project python "$root/c_toolbox/native_comparison_492/add_probe.py" "$out/src"
CC=gcc CXX=g++ FC=gfortran cmake -S "$out/src" -B "$out/build" \
  -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF \
  -DMPI_C_COMPILER=/opt/mpich/bin/mpicc \
  -DCMAKE_C_FLAGS="-D_DEBUG_DUMP_SROPTOO -D_DEBUG_DUMP_SROPTO_STORE"
# Upstream has two BLIS extraction targets sharing output paths. Serial build
# avoids their concurrent archive extraction without changing vendored CMake.
cmake --build "$out/build" --target vmc.out -j1
sha256sum "$out/build/src/mVMC/vmc.out"
