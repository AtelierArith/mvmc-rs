#!/usr/bin/env bash
set -euo pipefail
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
out=/tmp/mvmc-581-dispatch-results
for rep in 0 1 2 3; do
 if ((rep % 2 == 0)); then counts='1 4'; else counts='4 1'; fi
 for workers in $counts; do
  docker run --rm --cpuset-cpus=0-3 -v mvmc-581-dispatch-target:/target:ro -e "MVMC_DISPATCH_THREADS=$workers" -e MVMC_DISPATCH_ITERATIONS=1000 --entrypoint bash "$image" -lc '
   set -euo pipefail
   binary=$(ls -t /target/test-fast/deps/rayon_dispatch_479-* | grep -v "[.]d$" | head -1)
   sha256sum "$binary"
   TIMEFORMAT="wall=%R user=%U sys=%S"
   time timeout --kill-after=5s 180s "$binary" --ignored --exact rayon_dispatch_variants --nocapture
  ' > "$out/threads-${workers}-rep-${rep}.log" 2>&1
 done
done
