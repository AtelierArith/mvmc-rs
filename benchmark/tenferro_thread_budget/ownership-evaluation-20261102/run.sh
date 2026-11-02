#!/usr/bin/env bash
set -euo pipefail
bash /tmp/mvmc-583-final-policy-plan/walkers.sh \
 vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest \
 mvmc-583-budget-mpi-target /tmp/mvmc-583-default/inputs /tmp/mvmc-583-final-policy-walkers
bash /tmp/mvmc-583-final-policy-plan/mpi.sh
