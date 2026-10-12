#!/usr/bin/env bash
# Run all three E2E cases sharing one result directory and print the summary.
# Issue #544.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$here/env.sh"
export RESULT_DIR
rel=${RESULT_DIR#"$BENCH_DIR"/}

echo "== result directory: $RESULT_DIR"
echo "== [1/3] Rust GPU (host)"
bash "$here/run_rust_gpu.sh"

echo "== [2/3] Julia GPU (host)"
bash "$here/run_julia_gpu.sh"

echo "== [3/3] Rust MPI16 (Dev Container)"
devcontainer exec --workspace-folder "$ROOT" \
    bash -lc "RESULT_DIR='$rel' bash benchmark/host_gpu_comparison/run_rust_mpi16.sh"

echo "== summary"
bash "$here/summarize.sh"
