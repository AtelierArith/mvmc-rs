#!/bin/sh
# Build the C StdFace dry-run driver for the mVMC solver (mvmc_dry.out equivalent).
# Usage: c_toolbox/stdface/build_reference.sh <output-dir> [--fixed]
# Mirrors extern/mVMC-1.3.0/src/StdFace/src/CMakeLists.txt (MVMC=ON):
#   sources = SOURCES_StdFace + dry.c, -D_mVMC -DMEXP=19937, link -lm.
# Override the optimisation level with OPT (default: the CMake Release flags `-O3 -DNDEBUG`).
# With --fixed the sources are first copied to a temporary directory and
# c_toolbox/stdface/lattice_defects.patch (the corrections of issue #404) is applied; the binary
# is named mvmc_dry_fixed.out. The vendored sources in extern/ are never modified.
# No MPI is needed: StdFace_exit only calls MPI_Abort when compiled with -DMPI.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}
MODE=${2:-}
CC=${CC:-gcc}
OPT=${OPT:--O3 -DNDEBUG}
S="$ROOT/extern/mVMC-1.3.0/src/StdFace/src"
NAME=mvmc_dry.out
if [ "$MODE" = "--fixed" ]; then
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  cp "$S"/*.c "$S"/*.h "$TMP"/
  patch -d "$TMP" -p1 < "$ROOT/c_toolbox/stdface/lattice_defects.patch" > /dev/null
  S="$TMP"
  NAME=mvmc_dry_fixed.out
fi
mkdir -p "$OUT"
"$CC" $OPT -ffp-contract=off -w -D_mVMC -DMEXP=19937 -I"$S" -o "$OUT/$NAME" \
  "$S/dry.c" "$S/ChainLattice.c" "$S/HoneycombLattice.c" "$S/SquareLattice.c" \
  "$S/StdFace_main.c" "$S/StdFace_ModelUtil.c" "$S/TriangularLattice.c" \
  "$S/Ladder.c" "$S/Kagome.c" "$S/Orthorhombic.c" "$S/Pyrochlore.c" \
  "$S/Wannier90.c" "$S/FCOrtho.c" "$S/setmemory.c" "$S/export_wannier90.c" -lm
echo "built $OUT/$NAME"
