#!/usr/bin/env bash
# Common configuration for the host E2E comparison (issue #544).
# Source this file; every value can be overridden from the environment.

: "${MODEL:=128}"
: "${STEPS:=20}"
: "${SAMPLES:=320}"
: "${WARMUP_SWEEPS:=1}"
: "${INTERVAL_SWEEPS:=1}"
: "${SEED:=79}"
: "${NPART:=16}"
: "${REPS:=3}"
: "${WARMUPS:=1}"
: "${PHYSCAL_RUNS:=1}"

BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$BENCH_DIR/../.." && pwd)

INPUT_SRC=$ROOT/benchmark/hubbard_chain/inputs/hubbard_chain_L${MODEL}
JULIA_SRC=$ROOT/extern/Julia-mVMC
JULIA_BIN=${JULIA_BIN:-julia}
RUST_GPU_BIN=${RUST_GPU_BIN:-$ROOT/gpu/mvmc-gpu-cuda/target/release/mvmc-cuda}
MPI_BIN=${MPI_BIN:-${CARGO_TARGET_DIR:-$ROOT/target}/release/mvmc}
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib}

# One result directory per invocation. A relative value is resolved against the
# benchmark directory so the same path works on the host and in the Dev
# Container (the repository is mounted at a different absolute path there).
RESULT_DIR=${RESULT_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)}
case "$RESULT_DIR" in
    /*) ;;
    *) RESULT_DIR="$BENCH_DIR/$RESULT_DIR" ;;
esac
mkdir -p "$RESULT_DIR"
