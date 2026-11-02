#!/usr/bin/env bash
set -euo pipefail
for placement in shared disjoint partial; do
 for count in 1 2; do
  docker run --rm --cpuset-cpus=0-7 -v mvmc-583-budget-mpi-target:/target:ro \
   -e "PLACEMENT=$placement" -e "PLAN_COUNT=$count" \
   --entrypoint bash vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest -lc '
    set -euo pipefail
    timeout 30 /opt/mpich/bin/mpirun -n 4 bash -lc '\''
     rank=${PMI_RANK:?}
     case $PLACEMENT in
      shared) mask=0-3 ;;
      disjoint) mask="$((rank * 2))-$((rank * 2 + 1))" ;;
      partial) mask="$rank-$((rank + 1))" ;;
     esac
     exec taskset -c "$mask" /target/test-fast/examples/mpi_affinity_audit "--plan=$PLAN_COUNT"
    '\''
   ' > "/tmp/mvmc-583-placement-evidence/$placement-$count.log" 2>&1
 done
done
