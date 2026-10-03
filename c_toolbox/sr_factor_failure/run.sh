#!/usr/bin/env bash
# Explicit optional developer command. Never called by Cargo or Rust tests.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$source_dir/../.." && pwd)
out=${1:?usage: bash c_toolbox/sr_factor_failure/run.sh new-evidence-directory}
[[ ! -e $out ]] || { echo 'evidence directory already exists' >&2; exit 2; }
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
compiler=${CC:-cc}
command -v "$compiler" >/dev/null
command -v timeout >/dev/null
pkg-config --exists openblas
read -r -a blas_cflags <<< "$(pkg-config --cflags openblas)"
read -r -a blas_libs <<< "$(pkg-config --libs openblas)"
flags=(-std=c11 -O0 -ffp-contract=off -Wall -Wextra -Werror)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
"$compiler" --version > "$out/compiler.txt"
uname -a > "$out/platform.txt"
pkg-config --modversion openblas > "$out/blas.txt"
printf 'BLAS/OMP/MKL/BLIS threads=1; ABI=LP64; timeout=10s; kill-after=2s\n' > "$out/environment.txt"
sha256sum "$source_dir/zero_sr_boundary.c" "$source_dir/step14_sr_boundary.c" "$source_dir/run.sh" "$source_dir/README.md" > "$out/local-sources.sha256"
sha256sum "$repo/extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c" "$repo/extern/mVMC-1.3.0/src/mVMC/stcopt.c" > "$out/upstream-sources.sha256"
printf 'probe\tlaunch_exit\n' > "$out/results.tsv"
bad=0
for name in zero_sr_boundary step14_sr_boundary; do
    command=("$compiler" "${flags[@]}" "${blas_cflags[@]}" "$source_dir/$name.c" "${blas_libs[@]}" -lm -o "$out/$name")
    printf '%q ' "${command[@]}" >> "$out/commands.txt"
    printf '\n' >> "$out/commands.txt"
    "${command[@]}" > "$out/$name.build.log" 2>&1
    sha256sum "$out/$name" >> "$out/binaries.sha256"
    ldd "$out/$name" > "$out/$name.ldd.txt"
    rc=0
    timeout --kill-after=2s 10s "$out/$name" > "$out/$name.stdout.txt" 2> "$out/$name.stderr.txt" || rc=$?
    printf '%s\t%s\n' "$name" "$rc" | tee -a "$out/results.tsv"
    if (( rc != 0 )); then bad=1; fi
done
sha256sum -c "$out/local-sources.sha256" > "$out/local-source-check.txt" || bad=1
sha256sum -c "$out/upstream-sources.sha256" > "$out/upstream-source-check.txt" || bad=1
sha256sum -c "$out/binaries.sha256" > "$out/binary-check.txt" || bad=1
printf 'FIXED_OPERAND_SR_FAILURE bad=%s scope=DPOSV-boundary-not-native-sampler\n' "$bad" | tee "$out/terminal-summary.txt"
exit "$bad"
