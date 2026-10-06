#!/usr/bin/env bash
# Build the unpatched native C vmc.out in the Dev Container image (issue #448 timing baseline).
set -euo pipefail
repo=/home/terasaki/work/atelierarith/mvmc-rs/.claude/worktrees/agent-a5c265a75cba6a4a8
out=/tmp/claude-1000/c448
mkdir -p "$out"
docker run --rm -v "$repo:/repo:ro" -v "$out:/out" \
  vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32-uid:latest bash -lc '
set -euo pipefail
rm -rf /out/src /out/build
cp -r /repo/extern/mVMC-1.3.0 /out/src
mkdir -p /out/build && cd /out/build
export CC=gcc CXX=g++ FC=gfortran
cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF -DMPI_C_COMPILER=/opt/mpich/bin/mpicc > cmake.log 2>&1
make -j4 vmc.out > make.log 2>&1
ls -la src/mVMC/vmc.out; sha256sum src/mVMC/vmc.out; gcc --version | head -1; grep -E "CMAKE_C_FLAGS_RELEASE|CMAKE_BUILD_TYPE" CMakeCache.txt | head -3
'
