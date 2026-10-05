#!/bin/sh
# Build the C StdFace dry-run driver with the 3D-lattice defect fixes applied (issue #356).
# Usage: c_toolbox/stdface/build_reference_3d_fixed.sh <output-dir>
# Copies extern/mVMC-1.3.0/src/StdFace/src to <output-dir>/src, applies 3d_defects.patch (the
# vendored tree is never modified) and builds <output-dir>/mvmc_dry_3d_fixed.out with the same
# flags as build_reference.sh. The fixtures' corrected expectation comes from this binary; the
# unmodified build_reference.sh binary gives the historical C behaviour.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?output directory}
CC=${CC:-gcc}
OPT=${OPT:--O3 -DNDEBUG}
rm -rf "$OUT/tree"
mkdir -p "$OUT/tree/src/StdFace"
cp -r "$ROOT/extern/mVMC-1.3.0/src/StdFace/src" "$OUT/tree/src/StdFace/src"
(cd "$OUT/tree" && patch -p1 < "$ROOT/c_toolbox/stdface/3d_defects.patch")
S="$OUT/tree/src/StdFace/src"
"$CC" $OPT -ffp-contract=off -w -D_mVMC -DMEXP=19937 -I"$S" -o "$OUT/mvmc_dry_3d_fixed.out" \
  "$S/dry.c" "$S/ChainLattice.c" "$S/HoneycombLattice.c" "$S/SquareLattice.c" \
  "$S/StdFace_main.c" "$S/StdFace_ModelUtil.c" "$S/TriangularLattice.c" \
  "$S/Ladder.c" "$S/Kagome.c" "$S/Orthorhombic.c" "$S/Pyrochlore.c" \
  "$S/Wannier90.c" "$S/FCOrtho.c" "$S/setmemory.c" "$S/export_wannier90.c" -lm
echo "built $OUT/mvmc_dry_3d_fixed.out"
