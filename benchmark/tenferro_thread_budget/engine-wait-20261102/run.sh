#!/usr/bin/env bash
set -euo pipefail
image=${1:?development image}
volume=${2:?release target volume}
out=${3:?fresh output directory}
[[ ! -e $out ]]
mkdir -p "$out"
for callers in 1 4; do
 for gap in 0 60; do
  for rep in 0 1 2 3; do
   if ((rep % 2)); then spins='0 default'; else spins='default 0'; fi
   for spin in $spins; do
    label="c${callers}-g${gap}-s${spin}-r${rep}"
    docker run --rm --cpuset-cpus=0-3 -v "$volume:/target:ro" \
      -e "CALLERS=$callers" -e "GAP=$gap" -e "SPIN=$spin" \
      --entrypoint bash "$image" -lc '
        set -euo pipefail
        TIMEFORMAT="wall=%R user=%U sys=%S"
        time timeout 60 /target/release/examples/bench_engine_wait_583 "$CALLERS" 1000 "$GAP" "$SPIN"
      ' > "$out/$label.log" 2>&1
   done
  done
 done
done
