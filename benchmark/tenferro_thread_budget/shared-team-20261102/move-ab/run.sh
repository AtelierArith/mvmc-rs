#!/usr/bin/env bash
set -euo pipefail
cargo nextest list -p mvmc-core --test threaded_issue182 --locked --cargo-profile test-fast --message-format json > /tmp/list.json
binary=$(jq -r '."rust-suites" | to_entries[] | .value."binary-path"' /tmp/list.json | head -1)
test -x "$binary"
for mode in off on; do
  if [[ $mode == on ]]; then shared=1; assertion=1; else shared=0; assertion=0; fi
  ISSUE182_CHILD=runners MVMC_RS_INNER_THREADS=4 MVMC_RS_INNER_THRESHOLD=1 MVMC_RS_SR_BACKEND=tenferro MVMC_RS_TENFERRO_THREADS=4 MVMC_RS_THREADED_182=1 MVMC_EXPERIMENT_SHARED_CPU_TEAM="$shared" MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE=1 MVMC_EXPERIMENT_ASSERT_SHARED="$assertion" MVMC_EXPERIMENT_TRACE_MOVES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 "$binary" --ignored --exact issue182_child --nocapture --test-threads=1 > "/results/$mode.stdout" 2> "/results/$mode.stderr"
  rg '^D\|.*sampling-events\|' "/results/$mode.stdout" > "/results/$mode.events"
  test -s "/results/$mode.events"
done
cmp /results/off.events /results/on.events
sha256sum /results/off.events /results/on.events
