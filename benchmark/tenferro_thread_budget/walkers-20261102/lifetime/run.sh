#!/usr/bin/env bash
set -euo pipefail
root=/tmp/mvmc-583-walker-lifetime
image=vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest
inputs=/tmp/mvmc-583-default/inputs
mkdir -p "$root"
[[ ! -e "$root/config.env" ]]
printf 'cpus=0-3 walkers=4 inner=1 rayon=4 blas=1 tenferro=4 lifetime=held,stage binary_sha256=41169114f94d55414e013cb6c350ed2b9cb3f41cb634143e1070cc88f30e6e81\n' > "$root/config.env"
sha256sum "$inputs"/{opt,phys}/* > "$root/input-sha256.txt"
for rep in 0 1 2 3; do
 if ((rep % 2)); then lifetimes='stage held'; else lifetimes='held stage'; fi
 for lifetime in $lifetimes; do
  label="opt-${lifetime}-r${rep}"
  docker run --rm --cpuset-cpus=0-3 --shm-size=1g -v "$inputs:/inputs:ro" -v "$root:/out" -v mvmc-583-walkers-target:/target:ro -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 -e MVMC_RS_INNER_THREADS=1 -e MVMC_RS_TENFERRO_THREADS=4 -e MVMC_RS_SR_BACKEND=tenferro -e "MVMC_RS_TENFERRO_TEAM_LIFETIME=$lifetime" -e "BENCH_LABEL=$label" --entrypoint bash "$image" -lc '
   set -euo pipefail
   unset MVMC_RS_MEASURE_PF_BACKEND MVMC_RS_TENFERRO_SPIN_POLLS MVMC_RS_INNER_PROFILE MVMC_RS_INNER_GAP_PROFILE MVMC_EXPERIMENT_SHARED_CPU_TEAM MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE
   TIMEFORMAT="wall=%R user=%U sys=%S"
   { time timeout --kill-after=5s 240s /target/release/examples/bench_walkers_583 opt /inputs/opt/namelist.def /inputs/phys/zqp_opt.dat 4 "/out/$BENCH_LABEL-output" > "/out/$BENCH_LABEL.stdout" 2> "/out/$BENCH_LABEL.stderr"; } 2> "/out/$BENCH_LABEL.time"
  '
 done
done
