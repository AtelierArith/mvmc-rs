#!/usr/bin/env bash
set -euo pipefail
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
root=/tmp/mvmc-581-gates-serial
inputs=/tmp/mvmc-581-profile/inputs
mkdir -p "$root"
[[ ! -e "$root/config.env" ]]
printf 'cpus=0-3 workers=1 rayon=4 blas=1 binary_sha256=5ee7c581f626932bb88a73cf02f77667e8314f74843f768182f376c8654359cf\n' > "$root/config.env"
sha256sum "$inputs"/{l32-opt,l32-phys,l64-opt}/* > "$root/input-sha256.txt"
for case in l32-opt l32-phys l64-opt; do
 for rep in 0 1 2 3; do
  if ((rep % 2)); then gates='160000 20000'; else gates='20000 160000'; fi
  for gate in $gates; do
   docker run --rm --cpuset-cpus=0-3 --shm-size=1g -v "$inputs:/inputs:ro" -v "$root:/out" -v mvmc-581-dispatch-target:/target:ro -w "/inputs/$case" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 -e MVMC_RS_INNER_THREADS=1 -e "MVMC_RS_INNER_MIN_WORK_NS=$gate" -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_CASE=$case" -e "BENCH_DEST=/out/${case}-g${gate}-r${rep}-output" --entrypoint bash "$image" -lc '
    set -euo pipefail
    unset MVMC_RS_TENFERRO_THREADS MVMC_RS_TENFERRO_SPIN_POLLS MVMC_RS_INNER_THRESHOLD MVMC_RS_INNER_PROFILE MVMC_RS_INNER_GAP_PROFILE
    TIMEFORMAT="wall=%R user=%U sys=%S"
    if [[ "$BENCH_CASE" = l32-phys ]]; then
      time timeout --kill-after=5s 180s /target/release/mvmc namelist.def --physcal zqp_opt.dat --out-dir "$BENCH_DEST"
    else
      time timeout --kill-after=5s 180s /target/release/mvmc namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
    fi
   ' > "$root/${case}-g${gate}-r${rep}.log" 2>&1
  done
 done
done
