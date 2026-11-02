#!/usr/bin/env bash
set -euo pipefail
docker run --rm --name mvmc-586-workspace-check \
 -v /tmp/mvmc-583-budget:/work \
 -v /home/terasaki/work/atelierarith/mvmc-rs/extern/Julia-mVMC:/home/terasaki/work/atelierarith/mvmc-rs/extern/Julia-mVMC:ro \
 -v /home/terasaki/work/atelierarith/mvmc-rs/.git:/home/terasaki/work/atelierarith/mvmc-rs/.git:ro \
 -v mvmc-583-budget-mpi-target:/target \
 -v mvmc-cargo-1sfb79cckrsg96djc0mmplgg7itrvqo1hnckps4d2slo0q6vroku:/home/vscode/.cargo \
 -v mvmc-linux-cache-1sfb79cckrsg96djc0mmplgg7itrvqo1hnckps4d2slo0q6vroku:/home/vscode/.cache \
 -e CARGO_TARGET_DIR=/target -w /work \
 vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest \
 bash -lc 'cargo nextest run --workspace --locked --cargo-profile test-fast --no-fail-fast --retries 0'
