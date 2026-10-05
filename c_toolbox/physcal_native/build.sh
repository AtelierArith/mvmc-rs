#!/bin/bash
# Build the UNMODIFIED authoritative C vmc.out for PhysCal/Lanczos references (issue #181).
# Usage (inside the Linux x86_64 Dev Container, from the repository root):
#   c_toolbox/physcal_native/build.sh /tmp/mvmc-physcal-native
# extern/mVMC-1.3.0 is copied, never modified; no patch is applied.
# With a second argument `dump`, state_dump.patch (additive printing of saved
# configurations, counters, RNG draw count and next 624 words after each
# sample; no numerical code changes) is applied to the copy.
set -euo pipefail
out=${1:?output directory}
variant=${2:-plain}
root=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$out"
mkdir -p "$out"
cp -r "$root/extern/mVMC-1.3.0" "$out/src"
if [ "$variant" = dump ]; then
  (cd "$out/src" && patch -p1 < "$root/c_toolbox/physcal_native/state_dump.patch")
fi
mkdir -p "$out/build"
cd "$out/build"
export CC=gcc CXX=g++ FC=gfortran
cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF \
  -DMPI_C_COMPILER=/opt/mpich/bin/mpicc
make -j"$(nproc)" vmc.out
sha256sum src/mVMC/vmc.out
