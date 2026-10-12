#!/usr/bin/env bash
# Sum the recorded wall times per case/phase (issue #544).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$here/env.sh"

shopt -s nullglob
files=("$RESULT_DIR"/*/*.csv)
if [ ${#files[@]} -eq 0 ]; then
    echo "no CSVs under $RESULT_DIR" >&2
    exit 1
fi

awk -F, '
    FNR == 1 { next }
    $2 ~ /-measure$/ {
        phase = $2; sub(/-measure$/, "", phase);
        key = $1 "\t" phase;
        n[key]++;
        wall[key, n[key]] = $4;
        all[key,  n[key]] = $5;
    }
    END {
        printf "%-11s %-8s %4s  %12s  %12s\n", "case", "phase", "n", "median_wall", "median_all";
        for (k in n) {
            m = n[k];
            for (i = 1; i <= m; i++)
                for (j = i + 1; j <= m; j++)
                    if (wall[k, j] < wall[k, i]) {
                        t = wall[k, i]; wall[k, i] = wall[k, j]; wall[k, j] = t;
                        t = all[k, i];  all[k, i]  = all[k, j];  all[k, j]  = t;
                    }
            if (m % 2) { medw = wall[k, (m + 1) / 2]; meda = all[k, (m + 1) / 2]; }
            else { medw = (wall[k, m / 2] + wall[k, m / 2 + 1]) / 2;
                   meda = (all[k,  m / 2] + all[k,  m / 2 + 1]) / 2; }
            split(k, a, "\t");
            printf "%-11s %-8s %4d  %12.4f  %12s\n", a[1], a[2], m, medw, (meda == "" ? "-" : sprintf("%.4f", meda));
        }
    }
' "${files[@]}"

echo
echo "result directory: $RESULT_DIR"
