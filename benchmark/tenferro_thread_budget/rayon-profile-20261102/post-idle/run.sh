#!/usr/bin/env bash
set -euo pipefail
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
out=/tmp/mvmc-581-post-idle
mkdir -p "$out"; [[ ! -e "$out/config.env" ]]
printf 'cpus=0-3 workers=1,4 caller=inside,outside reps=5 warmups=1 binary_sha256=006a5bd0c3d32323eeee1f4ae02b3875e1fd903401ce22d466223d84a167e137\n' > "$out/config.env"
docker run --rm --cpuset-cpus=0-3 -v "$out:/out" -v mvmc-581-idle-only-target:/target:ro --entrypoint bash "$image" -lc '
 set -euo pipefail
 getconf CLK_TCK > /out/ticks.txt
 for rep in 0 1 2 3 4 5; do
  if ((rep % 2)); then combos="4:1 1:1 4:0 1:0"; else combos="1:0 4:0 1:1 4:1"; fi
  for combo in $combos; do
   threads=${combo%%:*}; inside=${combo#*:}
   PROBE_THREADS=$threads PROBE_INSIDE=$inside /target/test-fast/deps/rayon_idle_probe_581-db4458d3d5262432 --ignored --exact rayon_post_broadcast_idle_probe --nocapture > "/out/t${threads}-i${inside}-r${rep}.log" 2>&1
  done
 done
'
