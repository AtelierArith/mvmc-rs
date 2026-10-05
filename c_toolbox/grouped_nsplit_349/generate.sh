#!/bin/bash
# Generate the native-C grouped (NSplitSize) reference fixtures of issue #349.
#
# Explicit developer command; never invoked by Cargo or by Rust tests.
# Run inside the Linux x86_64 Dev Container (MPICH 4.2.0 under /opt/mpich,
# OpenBLAS, GCC) from the repository root:
#
#   c_toolbox/grouped_nsplit_349/generate.sh /tmp/mvmc-349-c
#
# The vendored extern/mVMC-1.3.0 tree is copied and built unmodified (no
# patch); only the stock vmc.out is used. For every case in
# tests/fixtures/grouped_nsplit_349/cases.txt it runs the C executable on the
# physcal_181 inputs with the case's modpara overrides at world sizes 1/2/4 and
# NSplitSize 1/2/4 and stores the numerical output files under
# tests/fixtures/grouped_nsplit_349/<case>/c/r<world>s<split>/.
set -euo pipefail
work=${1:?scratch directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
fx=$root/tests/fixtures/grouped_nsplit_349
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

configs="1:1 2:1 2:2 4:1 4:2 4:4"
for dir in "$fx"/*/; do
  rm -rf "${dir}c"
done
while read -r id model mode opttrans compare_c; do
  case $id in '#'* | '') continue ;; esac
  opts=""
  [ "$opttrans" = 1 ] && opts="-o"
  for cfg in $configs; do
    ranks=${cfg%%:*}
    split=${cfg##*:}
    run=$work/runs/$id/r${ranks}s${split}
    mkdir -p "$run"
    cp "$inputs/$model"/inputs/*.def "$run/"
    cp "$inputs/$model/zqp_opt.dat" "$run/"
    (
      cd "$run"
      sed -i -E "s/^NSplitSize.*/NSplitSize $split/" modpara.def
      while read -r key value; do
        [ -z "${key:-}" ] && continue
        grep -q "^$key[[:space:]]" modpara.def || {
          echo "override key $key missing from modpara.def" >&2
          exit 1
        }
        sed -i -E "s/^$key[[:space:]].*/$key $value/" modpara.def
      done < "$fx/$id/overrides.txt"
      timeout 900 /opt/mpich/bin/mpiexec -n "$ranks" "$vmc" $opts -e namelist.def zqp_opt.dat </dev/null \
        > stdout.log 2> stderr.log
    )
    dest=$fx/$id/c/r${ranks}s${split}
    mkdir -p "$dest"
    for f in "$run"/output/zvo_out_001.dat "$run"/output/zvo_var_001.dat "$run"/output/zqp_opt.dat \
      "$run"/output/zvo_cisajs_001.dat "$run"/output/zvo_cisajscktalt_001.dat \
      "$run"/output/zvo_cisajscktaltex_001.dat "$run"/output/zvo_ls_*_001.dat; do
      [ -f "$f" ] && cp "$f" "$dest/"
    done
    # PhysCal runs do not write zqp_opt.dat; optimization runs have no Green files.
  done
done < "$fx/cases.txt"

{
  echo "generator=c_toolbox/grouped_nsplit_349/generate.sh"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source=extern/mVMC-1.3.0 (unmodified copy), submodule commit ${SOURCE_COMMIT:-unknown}"
  echo "vmc_out_sha256=$(sha256sum "$vmc" | cut -d' ' -f1)"
  echo "arch=$(uname -m) kernel=$(uname -sr)"
  echo "gcc=$(gcc --version | head -1)"
  echo "mpich=$(/opt/mpich/bin/mpichversion | head -1)"
  echo "openblas=$(pkg-config --modversion openblas 2>/dev/null || echo unknown)"
  echo "cmake_flags=-DCMAKE_BUILD_TYPE=Release -DTesting=OFF (CMake default C flags; no extra flags)"
  echo "threads=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
  echo "configs(world:split)=$configs"
  echo "rust_input_sha256:"
  (cd "$fx" && sha256sum cases.txt */overrides.txt)
} > "$fx/PROVENANCE.txt"
echo "wrote $fx"
