#!/usr/bin/env bash
# Batched Pfaffian + inverse benchmark (issue #423): pfapack (1 thread, rayon), tenferro
# tensor-native, tenferro ExtensionOp and the CUDA kernel, f64 and c64 (plus an f32 timing
# variant). Medians of up to 7 runs after a warm-up; CUDA timings include upload/download.
#
# Usage: scripts/run_pfaffian_bench.sh [native|docker] [bench args, e.g. --quick --dtype f64]
#
# Output: CSV in $MVMC_RS_PFAFFIAN_BENCH_OUT (default
# benchmark/gpu_pfaffian/results/pfaffian_batched.csv). Record `uptime`, nvidia-smi and the
# CUDA toolkit version next to the CSV (the script prints them); run on a quiet host, CPU rows
# are sensitive to other load.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
[ -n "$mode" ] && shift || true
if [ -z "$mode" ]; then
  if command -v docker >/dev/null 2>&1; then mode=docker; else mode=native; fi
fi
out="${MVMC_RS_PFAFFIAN_BENCH_OUT:-$root/benchmark/gpu_pfaffian/results/pfaffian_batched.csv}"
mkdir -p "$(dirname "$out")"
rel="${out#"$root"/}"

echo "# host: $(uname -srm); load: $(cut -d' ' -f1-3 /proc/loadavg)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

inner="cd gpu/mvmc-gpu-cuda && nvcc --version | tail -1; cargo run --release --locked --example bench_pfaffian_batched -- --out /work/$rel $*"
case "$mode" in
  native) cd "$root" && bash -c "$inner" ;;
  docker)
    image="${MVMC_RS_CUDA_IMAGE:-tenferro-benchmark-cuda:full-verify-20260822}"
    docker run --rm --gpus all \
      -e HOME="$HOME" -v "$HOME/.rustup:$HOME/.rustup" -v "$HOME/.cargo:$HOME/.cargo" \
      -e CARGO_TARGET_DIR=/work/gpu/mvmc-gpu-cuda/target \
      -v "$root:/work" -w /work "$image" \
      bash -lc "apt-get update -qq >/dev/null && apt-get install -y -qq libopenblas-dev liblapack-dev >/dev/null \
        && exec setpriv --reuid=$(id -u) --regid=$(id -g) --clear-groups bash -c \
        'export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; $inner'"
    ;;
  *) echo "usage: $0 [native|docker] [bench args]" >&2; exit 2 ;;
esac
echo "csv: $out"
