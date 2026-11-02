#!/usr/bin/env bash
set -euo pipefail
root=/tmp/mvmc-583-team-wait
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
mkdir -p "$root"; [[ ! -e "$root/config.env" ]]
printf 'cpus=0-3 team=4 callers=1,4 iterations=5000 gap_us=0,60 spin=default,0 binary_sha256=ddc43c1f6c8162fed107181924a88c8b36180fa773c4d191041b50e14c332af2\n' > "$root/config.env"
for callers in 1 4; do
 for gap in 0 60; do
  for rep in 0 1 2 3; do
   if ((rep % 2)); then settings='0 default'; else settings='default 0'; fi
   for spin in $settings; do
    label="c${callers}-g${gap}-p${spin}-r${rep}"
    docker run --rm --cpuset-cpus=0-3 --shm-size=1g -v "$root:/out" -v mvmc-583-walkers-target:/target:ro -e "BENCH_CALLERS=$callers" -e "BENCH_GAP=$gap" -e "BENCH_SPIN=$spin" -e "BENCH_LABEL=$label" --entrypoint bash "$image" -lc '
     set -euo pipefail
     TIMEFORMAT="wall=%R user=%U sys=%S"
     { time timeout --kill-after=5s 120s /target/release/examples/bench_team_wait_583 "$BENCH_CALLERS" 5000 "$BENCH_GAP" "$BENCH_SPIN" > "/out/$BENCH_LABEL.stdout" 2> "/out/$BENCH_LABEL.stderr"; } 2> "/out/$BENCH_LABEL.time"
    '
   done
  done
 done
done
