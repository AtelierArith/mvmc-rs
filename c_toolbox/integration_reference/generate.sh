#!/bin/bash
# Regenerate, from C mVMC 1.4.0, the native C outputs that the Rust tests used to read from
# the C 1.3.0-era files under extern/Julia-mVMC/test/integration/reference (#695).
#
# Usage (inside the Linux x86_64 Dev Container, from the repository root):
#   c_toolbox/physcal_native/build.sh /tmp/mvmc-plain           # unmodified vmc.out
#   c_toolbox/integration_reference/generate.sh /tmp/mvmc-plain/build/src/mVMC/vmc.out \
#     tests/fixtures/c_integration_reference
#
# The inputs are read unchanged from extern/Julia-mVMC (only copied into a staging
# directory). One MPI rank, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.
#
# - PhysCal (`<model>/physcal_ref/expected/`): `vmc.out -e namelist.def zqp_opt.dat` on
#   `<model>/physcal_ref/inputs` with its fixed `zqp_opt.dat`.
# - First optimization rows (`<model>/zvo_out_first10.dat`, `zvo_out_first1.dat`):
#   `vmc.out -e namelist.def [initial.def]` on the input directory the Rust test reads,
#   passing `initial.def` when the directory has one (the Rust runner loads it by default),
#   with `NSROptItrStep` set to the number of rows kept (the rows of earlier steps do not
#   depend on the total step count).
set -euo pipefail
vmc=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
out=${2:?output directory}
root=$(cd "$(dirname "$0")/../.." && pwd)
ref="$root/extern/Julia-mVMC/test/integration/reference"
examples="$root/extern/Julia-mVMC/examples/inputs"
mkdir -p "$out"
out=$(cd "$out" && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_c() { # dir, args...
  local dir=$1
  shift
  (cd "$dir" && mpiexec -n 1 "$vmc" "$@" > stdout.txt 2>&1)
}

for model in heisenberg_chain_real heisenberg_chain_cmp heisenberg_chain_fsz kondo_chain_real \
  hubbard_chain_real hubbard_chain_dh_real hubbard_chain_lanczos spin_chain_lanczos; do
  work="$stage/$model-physcal"
  cp -r "$ref/$model/physcal_ref/inputs" "$work"
  cp "$ref/$model/physcal_ref/zqp_opt.dat" "$work/zqp_opt.dat"
  run_c "$work" -e namelist.def zqp_opt.dat
  dest="$out/$model/physcal_ref/expected"
  rm -rf "$dest"
  mkdir -p "$dest"
  for f in "$work"/output/zvo_*; do
    case $(basename "$f") in
      *_time_*|*CalcTimer*|zvo_out_*|zvo_var_*) ;;
      *) cp "$f" "$dest/" ;;
    esac
  done
done

first_rows() { # model input_dir rows file
  local model=$1 input=$2 rows=$3 file=$4
  work="$stage/$model-opt$rows"
  cp -r "$input" "$work"
  sed -i -E "s/^(NSROptItrStep[[:space:]]+).*/\1$rows/" "$work/modpara.def"
  if [ -f "$work/initial.def" ]; then
    run_c "$work" -e namelist.def initial.def
  else
    run_c "$work" -e namelist.def
  fi
  mkdir -p "$out/$model"
  head -n "$rows" "$work/output/zvo_out_001.dat" > "$out/$model/$file"
}
for model in heisenberg_chain_real heisenberg_chain_cmp heisenberg_chain_fsz hubbard_chain_real; do
  first_rows "$model" "$examples/$model" 10 zvo_out_first10.dat
done
for model in hubbard_chain_pairhop_real hubbard_chain_pairhop_fsz; do
  first_rows "$model" "$ref/$model/inputs" 1 zvo_out_first1.dat
done

{
  echo "generator=c_toolbox/integration_reference/generate.sh"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source=extern/mVMC-1.4.0 (unmodified copy) commit ${SOURCE_COMMIT:-unknown}"
  echo "vmc_out_sha256=$(sha256sum "$vmc" | cut -d' ' -f1)"
  echo "inputs=extern/Julia-mVMC ${JULIA_MVMC_COMMIT:-unknown} (test/integration/reference, examples/inputs)"
  echo "arch=$(uname -m) kernel=$(uname -sr)"
  echo "gcc=$(gcc --version | head -1)"
  echo "mpich=$(mpichversion 2>/dev/null | head -1 | tr -s ' ')"
  echo "openblas=$(dpkg-query -W -f='${Version}' libopenblas0-pthread 2>/dev/null || echo unknown)"
  echo "cmake_flags=-DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF (c_toolbox/physcal_native/build.sh plain)"
  echo "threads=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1, mpiexec -n 1"
  echo "image=${DEVCONTAINER_IMAGE:-unknown}"
} > "$out/provenance.txt"
