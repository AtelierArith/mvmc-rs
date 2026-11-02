#!/usr/bin/env bash
set -euo pipefail
for placement in shared disjoint partial; do
 for ranks in 2 4; do
  if [[ $placement == partial ]]; then requests=1; else requests='1 2'; fi
  for count in $requests; do
   for bad in 0 1; do
    docker run --rm --cpuset-cpus=0-7 -v mvmc-583-budget-mpi-target:/target:ro \
      -e MVMC_RS_INNER_THREADS=2 -e MVMC_PLACEMENT_EXISTING_POOL=1 -e "PLACEMENT=$placement" -e "MPI_RANKS=$ranks" -e "MVMC_PLACEMENT_REQUEST=$count" -e "BAD=$bad" \
      --entrypoint bash vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest -lc '
       set -euo pipefail
       if [[ $BAD == 1 ]]; then export MVMC_PLACEMENT_BAD_LAST=1; else unset MVMC_PLACEMENT_BAD_LAST; fi
       timeout 30 /opt/mpich/bin/mpirun -n "$MPI_RANKS" bash -lc '\''
        rank=${PMI_RANK:?}
        case $PLACEMENT in
         shared) mask=0-7 ;;
         disjoint) mask="$((rank * 2))-$((rank * 2 + 1))" ;;
         partial) mask="$rank-$((rank + 1))" ;;
        esac
        exec taskset -c "$mask" /target/test-fast/deps/mpi_cpu_placement_583-bb203e16fc09db63 --ignored --exact node_cpu_placement_applies_restores_and_recovers --nocapture
       '\''
      ' > "/tmp/mvmc-583-existing-pool-verified/$placement-r$ranks-c$count-bad$bad.log" 2>&1
   done
  done
 done
done
