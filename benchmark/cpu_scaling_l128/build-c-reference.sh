#!/usr/bin/env bash
# Optional developer reference build; never invoked by Cargo or Rust tests.
set -euo pipefail
bench_repo=$(realpath "${1:?repository root with initialized C submodule}")
bench_image=${2:?development image}
bench_out=$(realpath -m "${3:?fresh reference directory}")
[[ ! -e "$bench_out" ]]
mkdir -p "$bench_out"
docker run --rm \
 -v "$bench_repo/extern/mVMC-1.4.0:/source:ro" -v "$bench_out:/reference" \
 "$bench_image" bash -lc '
 set -euo pipefail
 cp -a /source /reference/src
 mkdir /reference/build
 cd /reference/build
 export CC=gcc CXX=g++ FC=gfortran
 cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF \
  -DTesting=OFF -DMPI_C_COMPILER=/opt/mpich/bin/mpicc
 cmake --build . --target vmc.out --parallel "$(nproc)"
 sha256sum src/mVMC/vmc.out
 gcc --version
 '
