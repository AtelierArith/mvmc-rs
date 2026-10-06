#!/usr/bin/env bash
# Device-resident SR step vs OpenBLAS (issue #447). Usage: scripts/run_sr_device_bench.sh [native|docker] [bench args, e.g. --sizes 3000x3000 --reps 3 --cg-iters 50]
# Output: CSV in $MVMC_RS_SR_BENCH_OUT (default benchmark/gpu_sr_device/results/sr_device.csv). Run on a quiet host: the CPU rows are load sensitive.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
[ -n "$mode" ] && shift || true
if [ -z "$mode" ]; then
  if command -v docker >/dev/null 2>&1; then mode=docker; else mode=native; fi
fi
out="${MVMC_RS_SR_BENCH_OUT:-$root/benchmark/gpu_sr_device/results/sr_device.csv}"
mkdir -p "$(dirname "$out")"
rel="${out#"$root"/}"

echo "# host: $(uname -srm); load: $(cut -d' ' -f1-3 /proc/loadavg)"
inner="cd gpu/mvmc-gpu-cuda && nvcc --version | tail -1; cargo run --release --locked --example bench_sr_device -- --out /work/$rel $*"
case "$mode" in
  native) cd "$root" && bash -c "$inner" ;;
  docker)
    image="${MVMC_RS_CUDA_IMAGE:-tenferro-benchmark-cuda:full-verify-20260822}"
    cpuset=()
    [ -n "${MVMC_RS_SAMPLER_CORES:-}" ] && cpuset=(--cpuset-cpus "$MVMC_RS_SAMPLER_CORES")
    docker run --rm --gpus all "${cpuset[@]}" \
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
