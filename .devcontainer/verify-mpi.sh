#!/usr/bin/env bash
# Process-manager health only; no numerical oracle or Cargo dependency.
set -euo pipefail
workspace_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
probe_dir=$(mktemp -d)
trap 'rm -rf -- "$probe_dir"' EXIT
"${MPICC:-mpicc}" -std=c11 -O2 -Wall -Wextra -Werror \
    "$workspace_root/scripts/issue234/native/mpi-startup.c" -o "$probe_dir/world"
for ranks in 2 4; do
    timeout --kill-after=5s 30s mpiexec -n "$ranks" "$probe_dir/world" "$ranks" \
        > "$probe_dir/world-$ranks.log" 2>&1
    cat "$probe_dir/world-$ranks.log"
    awk -v n="$ranks" '
        /^MPI_STARTUP / {
            rank=$2; sub(/^rank=/,"",rank)
            size=$3; sub(/^size=/,"",size)
            if (NF!=7 || size!=n || rank<0 || rank>=n || seen[rank]++ || $7!="ok=1") bad=1
            count++
        }
        END {exit bad || count!=n}
    ' "$probe_dir/world-$ranks.log"
done
