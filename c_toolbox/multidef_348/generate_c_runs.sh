#!/bin/bash
# Generate the native-C `vmc.out -m N` reference fixtures of issue #348.
#
# Explicit developer command; never invoked by Cargo or by Rust tests. Run inside the
# Linux x86_64 Dev Container (MPICH under /opt/mpich, OpenBLAS, GCC) from the repo root:
#
#   SOURCE_COMMIT=$(git -C extern/mVMC-1.3.0 rev-parse HEAD) \
#     c_toolbox/multidef_348/generate_c_runs.sh /tmp/mvmc-348-c
#
# The vendored extern/mVMC-1.3.0 tree is copied and built unmodified (stock vmc.out).
# Two directories with different inputs, `a` (physcal_181/heisenberg_chain_real) and `b`
# (physcal_181/hubbard_chain_real), are listed in dirs.txt. `mpiexec -n W vmc.out -m 2
# dirs.txt namelist.def zqp_opt.dat` runs at W = 2, 3, 4 (2x1, 2+1 with the C load
# imbalance warning, 2x2 ranks). Numerical outputs go to
# tests/fixtures/multidef_348/c/r<W>/{a,b}/; the stderr of the successful runs and the
# first stderr line plus exit status of the error cases go to
# tests/fixtures/multidef_348/c_messages/.
set -euo pipefail
work=${1:?scratch directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
fx=$root/tests/fixtures/multidef_348
inputs=$root/tests/fixtures/physcal_181
export CC=gcc CXX=g++ FC=gfortran OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rm -rf "$work"
mkdir -p "$work/build"
cp -r "$root/extern/mVMC-1.3.0" "$work/src"
(cd "$work/build" &&
  cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF \
    -DMPI_C_COMPILER=/opt/mpich/bin/mpicc > cmake.log 2>&1 &&
  make -j4 vmc.out > make.log 2>&1)
vmc=$work/build/src/mVMC/vmc.out
mpiexec=/opt/mpich/bin/mpiexec

setup() { # setup <run dir>
  local run=$1
  mkdir -p "$run"
  for pair in a:heisenberg_chain_real b:hubbard_chain_real; do
    local d=${pair%%:*} model=${pair##*:}
    mkdir -p "$run/$d"
    cp "$inputs/$model"/inputs/*.def "$run/$d/"
    cp "$inputs/$model/zqp_opt.dat" "$run/$d/"
  done
  printf 'a\nb\n' > "$run/dirs.txt"
}

rm -rf "$fx/c" "$fx/c_messages"
mkdir -p "$fx/c_messages"
for w in 2 3 4; do
  run=$work/runs/r$w
  setup "$run"
  (cd "$run" && timeout 900 "$mpiexec" -n "$w" "$vmc" -m 2 dirs.txt namelist.def zqp_opt.dat \
    </dev/null > stdout.log 2> stderr.log)
  for d in a b; do
    dest=$fx/c/r$w/$d
    mkdir -p "$dest"
    for f in "$run/$d"/output/zvo_out_001.dat "$run/$d"/output/zvo_var_001.dat \
      "$run/$d"/output/zvo_cisajs_001.dat "$run/$d"/output/zvo_cisajscktalt_001.dat \
      "$run/$d"/output/zvo_cisajscktaltex_001.dat; do
      [ -f "$f" ] && cp "$f" "$dest/"
    done
  done
  cp "$run/stderr.log" "$fx/c_messages/r${w}_stderr.txt"
done

# Error cases: one rank, so the message order is deterministic. After the first message C
# calls MPI_Abort, which is asynchronous in MPICH: the process keeps running and prints
# unrelated follow-up errors (and at world sizes > 1 other ranks scatter uninitialized
# memory, see tests/fixtures/multidef_348/README.md). Only the first stderr line (C's own
# message) and the exit status are kept.
msg=$fx/c_messages
err() { # err <name> <list file content, or - for no list file> <vmc.out args...>
  local name=$1 content=$2
  shift 2
  local run=$work/runs/err_$name
  setup "$run"
  [ "$content" = "-" ] || printf "$content" > "$run/list.txt"
  set +e
  (cd "$run" && timeout 120 "$mpiexec" -n 1 "$vmc" "$@" </dev/null > stdout.log 2> stderr.log)
  echo "$?" > "$msg/$name.status"
  set -e
  head -1 "$run/stderr.log" > "$msg/$name.first_stderr_line"
}
err size_lt_n 'a\nb\n' -m 2 list.txt namelist.def zqp_opt.dat
err missing_list - -m 1 list.txt namelist.def zqp_opt.dat
err incomplete_list '' -m 1 list.txt namelist.def zqp_opt.dat
err bad_dir 'nope\n' -m 1 list.txt namelist.def zqp_opt.dat

{
  echo "generator=c_toolbox/multidef_348/generate_c_runs.sh"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source=extern/mVMC-1.3.0 (unmodified copy), submodule commit ${SOURCE_COMMIT:-unknown}"
  echo "vmcmain.c_sha256=$(sha256sum "$root/extern/mVMC-1.3.0/src/mVMC/vmcmain.c" | cut -d' ' -f1)"
  echo "vmc_out_sha256=$(sha256sum "$vmc" | cut -d' ' -f1)"
  echo "arch=$(uname -m) kernel=$(uname -sr)"
  echo "gcc=$(gcc --version | head -1)"
  echo "mpich=$(/opt/mpich/bin/mpichversion | head -1)"
  echo "openblas=$(pkg-config --modversion openblas 2>/dev/null || echo unknown)"
  echo "cmake_flags=-DCMAKE_BUILD_TYPE=Release -DTesting=OFF (CMake default C flags; no extra flags)"
  echo "threads=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
  echo "command=mpiexec -n W vmc.out -m 2 dirs.txt namelist.def zqp_opt.dat, W in 2 3 4"
  echo "inputs: a=tests/fixtures/physcal_181/heisenberg_chain_real, b=tests/fixtures/physcal_181/hubbard_chain_real (unmodified)"
} > "$fx/PROVENANCE.txt"
echo "wrote $fx"
