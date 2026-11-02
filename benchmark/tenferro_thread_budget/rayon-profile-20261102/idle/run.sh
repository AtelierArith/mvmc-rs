#!/usr/bin/env bash
set -euo pipefail
# Run in the Linux dev container on the intended CPU affinity. Write to a fresh log.
binary=$(cargo test -p mvmc-core --test rayon_idle_probe_581 --locked --profile test-fast --no-run --message-format=json | jq -r 'select(.profile.test == true and .executable != null) | .executable' | tail -1)
test -x "$binary"
getconf CLK_TCK
for rep in 1 2 3; do
  for inside in 0 1; do
    for threads in 1 4; do
      for gap in 0 20 60 1000; do
        echo "repetition=$rep"
        PROBE_THREADS=$threads PROBE_GAP_US=$gap PROBE_INSIDE=$inside "$binary" --ignored --exact rayon_idle_probe --nocapture
      done
    done
  done
done
