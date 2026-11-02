#!/usr/bin/env bash
set -euo pipefail
docker run --rm --name mvmc-586-candidate-tests \
 -v /tmp/mvmc-583-budget:/work \
 -v /home/terasaki/work/atelierarith/mvmc-rs/extern/Julia-mVMC:/home/terasaki/work/atelierarith/mvmc-rs/extern/Julia-mVMC:ro \
 -v mvmc-583-budget-mpi-target:/target \
 -v mvmc-cargo-1sfb79cckrsg96djc0mmplgg7itrvqo1hnckps4d2slo0q6vroku:/home/vscode/.cargo \
 -v mvmc-linux-cache-1sfb79cckrsg96djc0mmplgg7itrvqo1hnckps4d2slo0q6vroku:/home/vscode/.cache \
 -e CARGO_TARGET_DIR=/target -e "MVMC_586_TEST_SCOPE=${1:-full}" -w /work \
 vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest \
 bash -lc 'set -euo pipefail
 if [[ $MVMC_586_TEST_SCOPE == affected ]]; then
 cargo nextest run -p mvmc-core --locked --cargo-profile test-fast \
  --test c_order_pfaffian_threading --no-fail-fast --retries 0
 else
 cargo nextest run -p mvmc-core --locked --cargo-profile test-fast \
  --test c_order_pfaffian_threading --test stage_backend_437 \
  --test measurement_batch_422 --test measurement_stage_519 \
  --test saved_sample_team_regression --test native274_boundaries \
  --test threaded_issue182 --no-fail-fast --retries 0
 fi
 cargo clippy --workspace --all-targets --locked -- -D warnings
 cargo fmt --all --check
 '
