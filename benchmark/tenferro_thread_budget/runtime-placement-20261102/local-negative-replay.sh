#!/usr/bin/env bash
set -euo pipefail
for ranks in 2 4; do
 for case in invalid mismatch capacity budget-over budget-missing budget-asym; do
  expected=2; [[ $case != mismatch ]] || expected=1
  set +e
  docker run --rm --cpuset-cpus=0-7 -v mvmc-583-budget-mpi-target:/target:ro \
   -v /tmp/mvmc-583-runtime-cli-harness:/harness:ro -v /tmp/mvmc-583-runtime-cli-negative:/evidence \
   -e "CASE=$case" -e "MPI_RANKS=$ranks" --entrypoint bash \
   vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest -lc \
   'timeout 60 /opt/mpich/bin/mpirun -n "$MPI_RANKS" bash /harness/wrapper.sh' \
   > "/tmp/mvmc-583-runtime-cli-negative/$case-r$ranks-launcher.log" 2>&1
  rc=$?
  set -e
  [[ $rc == "$expected" ]]
  prefix="/tmp/mvmc-583-runtime-cli-negative/$case-r$ranks"
  [[ ! -e $prefix-output ]]
  for ((rank=0;rank<ranks;rank++)); do
   [[ $(cat "$prefix-rank$rank.status") == "$expected" ]]
   rg -q '^error:' "$prefix-rank$rank.stderr"
   ! rg -q 'panicked|failed to read|No such file' "$prefix-rank$rank.stderr"
  done
  printf '%s ranks=%s rc=%s verified\n' "$case" "$ranks" "$rc"
 done
done
