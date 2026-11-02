#!/usr/bin/env bash
set -euo pipefail
# Adapt these library and target paths to the local host; use source.txt's revision.
export LD_LIBRARY_PATH=/tmp/mvmc-570-cuda-env/lib/python3.12/site-packages/cutensor/lib:/home/terasaki/.julia/artifacts/369f5472daa95aec77a8de76aa620f1b8f13293f/lib:/home/terasaki/.julia/artifacts/6b0fef1d16bb8733a19f60c6efec768f99e5f90d/lib
export CARGO_TARGET_DIR=/tmp/mvmc-583-budget-target
export MVMC_RS_CUDA_GATE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
cargo nextest run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --locked --run-ignored all \
 --test pfaffian_extension_gate --test pfaffian_gate --test transfer_gate --test sampler_gate \
 --test measurement_gate --test sr_device_gate --test sr_routing_gate --test interval_parity \
 -E 'not test(microbenchmark)' --test-threads 1 --no-fail-fast --retries 0
cargo nextest run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --locked --run-ignored all \
 --lib -E 'test(opt::tests::resident_opt)' --test-threads 1 --no-fail-fast --retries 0
export CUDA_VISIBLE_DEVICES=0,1
uv run --no-project python - <<'PY'
import ctypes
cuda = ctypes.CDLL('libcuda.so.1')
assert cuda.cuInit(0) == 0
count = ctypes.c_int()
assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0
print('CUDA driver visible devices:', count.value)
assert count.value == 2
PY
cargo nextest run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --locked --run-ignored all \
 --test resident_interval -E 'test(=device_rng_rejects_a_different_device_context)' \
 --test-threads 1 --no-fail-fast --retries 0
