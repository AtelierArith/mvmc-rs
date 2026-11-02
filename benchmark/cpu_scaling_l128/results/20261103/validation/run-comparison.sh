#!/usr/bin/env bash
set -euo pipefail
bench_common=(
 --image vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
 --target-volume mvmc-583-budget-mpi-target
 --c-dir /tmp/mvmc-576-c-reference
 --opt-input /tmp/mvmc-586-baseline-20261103/inputs/opt
 --phys-input /tmp/mvmc-586-baseline-20261103/inputs/phys
)
uv run --no-project python /tmp/mvmc-586-baseline-plan/measure.py \
 "${bench_common[@]}" --repo /tmp/mvmc-583-budget \
 --output /tmp/mvmc-586-after-c-order-20261103 \
 > /tmp/mvmc-586-baseline-plan/after-c-order.log 2>&1
uv run --no-project python /tmp/mvmc-586-baseline-plan/measure.py \
 "${bench_common[@]}" --repo /tmp/mvmc-586-contract-before \
 --pf-backend calc-m-all --only-rust \
 --rust-binary /tmp/mvmc-586-profile-before-20261103/before-mvmc \
 --output /tmp/mvmc-586-before-default-20261103 \
 > /tmp/mvmc-586-baseline-plan/before-default.log 2>&1
uv run --no-project python /tmp/mvmc-586-baseline-plan/measure.py \
 "${bench_common[@]}" --repo /tmp/mvmc-583-budget \
 --pf-backend calc-m-all --only-rust \
 --output /tmp/mvmc-586-after-default-20261103 \
 > /tmp/mvmc-586-baseline-plan/after-default.log 2>&1
