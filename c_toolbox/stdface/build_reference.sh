#!/bin/sh
# Build the C StdFace dry-run driver for the mVMC solver (mvmc_dry.out equivalent).
# Usage: c_toolbox/stdface/build_reference.sh <output-dir>
# Mirrors extern/mVMC-1.3.0/src/StdFace/src/CMakeLists.txt (MVMC=ON):
#   sources = SOURCES_StdFace + dry.c, -D_mVMC -DMEXP=19937, link -lm.
# Override the optimisation level with OPT (default: the CMake Release flags `-O3 -DNDEBUG`).
# No MPI is needed: StdFace_exit only calls MPI_Abort when compiled with -DMPI.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}
CC=${CC:-gcc}
OPT=${OPT:--O3 -DNDEBUG}
S="$ROOT/extern/mVMC-1.3.0/src/StdFace/src"
mkdir -p "$OUT"
"$CC" $OPT -ffp-contract=off -w -D_mVMC -DMEXP=19937 -I"$S" -o "$OUT/mvmc_dry.out" \
  "$S/dry.c" "$S/ChainLattice.c" "$S/HoneycombLattice.c" "$S/SquareLattice.c" \
  "$S/StdFace_main.c" "$S/StdFace_ModelUtil.c" "$S/TriangularLattice.c" \
  "$S/Ladder.c" "$S/Kagome.c" "$S/Orthorhombic.c" "$S/Pyrochlore.c" \
  "$S/Wannier90.c" "$S/FCOrtho.c" "$S/setmemory.c" "$S/export_wannier90.c" -lm
echo "built $OUT/mvmc_dry.out"
