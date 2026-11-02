#!/usr/bin/env bash
set -euo pipefail
image=${1:?development image}
volume=${2:?release target volume}
inputs=${3:?directory containing opt and phys inputs}
out=${4:?fresh output directory}
[[ ! -e "$out" ]]
mkdir -p "$out"
printf 'cpus=0-3 walkers=1,4 inner=1 rayon=4 blas=1 tenferro=4 budgets=unset,0,3 warmups=1 measured=3\n' > "$out/config.env"
sha256sum "$inputs"/{opt,phys}/* > "$out/input-sha256.txt"
for mode in opt phys; do
 for walkers in 1 4; do
  for rep in 0 1 2 3; do
   if ((rep % 2)); then budgets='3 0 unset'; else budgets='unset 0 3'; fi
   for budget in $budgets; do
    label="${mode}-w${walkers}-b${budget}-r${rep}"
    docker run --rm --cpuset-cpus=0-3 --shm-size=1g \
      -v "$inputs:/inputs:ro" -v "$out:/out" -v "$volume:/target:ro" \
      -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e RAYON_NUM_THREADS=4 \
      -e MVMC_RS_INNER_THREADS=1 -e MVMC_RS_TENFERRO_THREADS=4 \
      -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_MODE=$mode" \
      -e "BENCH_WALKERS=$walkers" -e "BENCH_LABEL=$label" -e "BENCH_BUDGET=$budget" \
      --entrypoint bash "$image" -lc '
        set -euo pipefail
        unset MVMC_RS_TENFERRO_WORKER_BUDGET MVMC_RS_MEASURE_PF_BACKEND MVMC_RS_TENFERRO_SPIN_POLLS MVMC_RS_TENFERRO_TEAM_LIFETIME MVMC_RS_INNER_PROFILE MVMC_RS_INNER_GAP_PROFILE MVMC_RS_INNER_GAP_SITE_PROFILE MVMC_EXPERIMENT_SHARED_CPU_TEAM MVMC_EXPERIMENT_SHARED_CPU_TEAM_PROFILE
        if [[ $BENCH_BUDGET != unset ]]; then export MVMC_RS_TENFERRO_WORKER_BUDGET="$BENCH_BUDGET"; fi
        TIMEFORMAT="wall=%R user=%U sys=%S"
        { time timeout --kill-after=5s 240s /target/release/examples/bench_walkers_583 "$BENCH_MODE" "/inputs/$BENCH_MODE/namelist.def" /inputs/phys/zqp_opt.dat "$BENCH_WALKERS" "/out/$BENCH_LABEL-output" > "/out/$BENCH_LABEL.stdout" 2> "/out/$BENCH_LABEL.stderr"; } 2> "/out/$BENCH_LABEL.time"
      '
   done
  done
 done
done
