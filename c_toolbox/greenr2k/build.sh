#!/bin/bash
# Build the authoritative Fortran greenr2k and the C StdFace driver (issue #351).
# Usage (Linux, from the repository root): c_toolbox/greenr2k/build.sh <out-dir>
# The vendored extern/mVMC-1.3.0 tree is only read, never modified.
set -euo pipefail
out=${1:?output directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
src="$root/extern/mVMC-1.3.0"
# greenr2k.F90 exactly as upstream (tool/CMakeLists.txt links LAPACK).
gfortran -O2 -J "$out" "$src/tool/greenr2k.F90" -llapack -lblas -o "$out/greenr2k"
# StdFace dry-run driver (mvmc_dry.out) used to generate geometry.dat and the
# greenone/greentwo index files exactly as C does.
sf="$src/src/StdFace/src"
gcc -O2 -D_mVMC -DMEXP=19937 -I"$sf" -I"$src/src/include" -I"$src/src/common" \
  "$sf"/dry.c "$sf"/StdFace_main.c "$sf"/StdFace_ModelUtil.c "$sf"/ChainLattice.c \
  "$sf"/SquareLattice.c "$sf"/TriangularLattice.c "$sf"/HoneycombLattice.c \
  "$sf"/Ladder.c "$sf"/Kagome.c "$sf"/Orthorhombic.c "$sf"/Pyrochlore.c \
  "$sf"/Wannier90.c "$sf"/FCOrtho.c "$sf"/setmemory.c "$sf"/export_wannier90.c \
  -lm -o "$out/mvmc_dry.out"
sha256sum "$src/tool/greenr2k.F90" "$out/greenr2k" "$out/mvmc_dry.out"
gfortran --version | head -1
gcc --version | head -1
