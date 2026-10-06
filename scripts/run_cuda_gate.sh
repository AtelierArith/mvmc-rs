#!/usr/bin/env bash
# Optional CUDA gate (issue #420): build and run the tenferro CUDA gate on a GPU host.
#
# Usage:
#   scripts/run_cuda_gate.sh [native|docker] [extra cargo-test args]
#
#   native  Build and run on this host (needs the CUDA toolkit libraries: cuBLAS, cuSOLVER,
#           NVRTC, plus the driver's libcuda).
#   docker  (default when `docker` exists) Run in an NVIDIA CUDA toolkit image with
#           `--gpus all`. The host rustup/cargo directories are mounted so the image needs no
#           Rust toolchain. Override the image with MVMC_RS_CUDA_IMAGE.
#
# The gate is requested (MVMC_RS_CUDA_GATE=1): no device is a hard failure. Results
# (device/driver/CUDA/cuBLAS/cuSOLVER versions plus the CPU-vs-CUDA micro-benchmark) are
# written to $MVMC_RS_CUDA_GATE_OUT (default: gpu/mvmc-gpu-cuda/results/cuda-gate.md).
#
# Not run by default CI: hosted runners have no GPU, so the optional-gates `cuda` job is
# NotRun unless dispatched on a self-hosted GPU runner.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
[ -n "$mode" ] && shift || true
if [ -z "$mode" ]; then
  if command -v docker >/dev/null 2>&1; then mode=docker; else mode=native; fi
fi
out="${MVMC_RS_CUDA_GATE_OUT:-$root/gpu/mvmc-gpu-cuda/results/cuda-gate.md}"
mkdir -p "$(dirname "$out")"

inner='cd gpu/mvmc-gpu-cuda && cargo test --profile test --locked --test cuda_gate -- --ignored --nocapture'

case "$mode" in
  native)
    cd "$root"
    MVMC_RS_CUDA_GATE=1 MVMC_RS_CUDA_GATE_OUT="$out" bash -c "$inner $*"
    ;;
  docker)
    image="${MVMC_RS_CUDA_IMAGE:-tenferro-benchmark-cuda:full-verify-20260822}"
    rel="${out#"$root"/}"
    # The image has no OpenBLAS/LAPACK (mvmc-core links them), so install them as root and
    # then drop to the invoking user for the build so target/ files keep host ownership.
    docker run --rm --gpus all \
      -e HOME="$HOME" -e MVMC_RS_CUDA_GATE=1 \
      -e MVMC_RS_CUDA_GATE_OUT="/work/$rel" \
      -e MVMC_RS_CUDA_GATE_SIZES -e MVMC_RS_CUDA_GATE_REPS \
      -e CARGO_TARGET_DIR=/work/gpu/mvmc-gpu-cuda/target \
      -v "$HOME/.rustup:$HOME/.rustup" -v "$HOME/.cargo:$HOME/.cargo" \
      -v "$root:/work" -w /work "$image" \
      bash -lc "apt-get update -qq >/dev/null && apt-get install -y -qq libopenblas-dev liblapack-dev >/dev/null \
        && exec setpriv --reuid=$(id -u) --regid=$(id -g) --clear-groups bash -c \
        'export PATH=$HOME/.cargo/bin:\$PATH; $inner $*'"
    ;;
  *) echo "usage: $0 [native|docker]" >&2; exit 2 ;;
esac
echo "report: $out"
