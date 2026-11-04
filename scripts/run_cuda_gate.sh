#!/usr/bin/env bash
# Optional CUDA gate for the device-resident GPU Opt/PhysCal runners (issues #574, #643).
#
# Builds the standalone `gpu/mvmc-gpu-cuda` workspace and runs its CUDA-only (ignored) tests:
# the resident Opt/PhysCal library tests (Julia fixtures, 20-step repeatability, transfer and
# synchronization contracts), the resident interval sampler tests and the interval kernel gates.
# The L128 sampling benchmark runs for its exact decisions and conditioned Pfaffian bounds
# (#656). The throughput-only `interval_bench` and the two-device context test are not part
# of the gate.
#
# Usage:
#   scripts/run_cuda_gate.sh [native|docker] [extra cargo-test args]
#
#   native  Build and run on this host (needs the CUDA toolkit libraries: cuBLAS, cuSOLVER,
#           cuRAND, NVRTC, plus the driver's libcuda) on LD_LIBRARY_PATH.
#   docker  (default when `docker` exists) Run in an NVIDIA CUDA toolkit image with
#           `--gpus all`. The host rustup/cargo directories are mounted so the image needs no
#           Rust toolchain. Override the image with MVMC_RS_CUDA_IMAGE.
#
# A missing device is a hard failure (`nvidia-smi -L` must list one) before any test runs. The
# test log is written to $MVMC_RS_CUDA_GATE_OUT (default:
# gpu/mvmc-gpu-cuda/results/cuda-gate.log).
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
out="${MVMC_RS_CUDA_GATE_OUT:-$root/gpu/mvmc-gpu-cuda/results/cuda-gate.log}"
mkdir -p "$(dirname "$out")"

skips='--skip device_rng_rejects_a_different_device_context'
inner="nvidia-smi -L && cd gpu/mvmc-gpu-cuda && cargo test --profile test --locked --lib --bins --test interval_gate --test interval_parity --test resident_interval -- --include-ignored --test-threads=1 $skips"

case "$mode" in
  native)
    cd "$root"
    bash -c "$inner $*" 2>&1 | tee "$out"
    ;;
  docker)
    image="${MVMC_RS_CUDA_IMAGE:-tenferro-benchmark-cuda:full-verify-20260822}"
    # The image has no OpenBLAS/LAPACK (mvmc-core links them), so install them as root and
    # then drop to the invoking user for the build so target/ files keep host ownership.
    docker run --rm --gpus all \
      -e HOME="$HOME" \
      -e CARGO_TARGET_DIR=/work/gpu/mvmc-gpu-cuda/target \
      -v "$HOME/.rustup:$HOME/.rustup" -v "$HOME/.cargo:$HOME/.cargo" \
      -v "$root:/work" -w /work "$image" \
      bash -lc "apt-get update -qq >/dev/null && apt-get install -y -qq libopenblas-dev liblapack-dev >/dev/null \
        && exec setpriv --reuid=$(id -u) --regid=$(id -g) --clear-groups bash -c \
        'export PATH=$HOME/.cargo/bin:\$PATH; $inner $*'" 2>&1 | tee "$out"
    ;;
  *) echo "usage: $0 [native|docker]" >&2; exit 2 ;;
esac
echo "gate log: $out"
