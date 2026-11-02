#!/usr/bin/env bash
# Compare instrumented 0.7.1 and 0.8 SR examples. Use an otherwise idle host.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: $0 BEFORE_BINARY AFTER_BINARY EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
before=$(realpath "$1")
after=$(realpath "$2")
out=$3
mkdir -p "$out"
if [[ -e "$out/binary-sha256.txt" ]]; then
    echo "output already contains a run: $out" >&2
    exit 2
fi
sha256sum "$before" "$after" > "$out/binary-sha256.txt"
uname -a > "$out/environment.txt"
rustc -Vv >> "$out/environment.txt"
for threads in 1 4; do
    for rep in 0 1 2 3; do
        if (( rep % 2 == 0 )); then versions='before after'; else versions='after before'; fi
        for version in $versions; do
            if [[ $version == before ]]; then binary=$before; else binary=$after; fi
            RAYON_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
                /usr/bin/time -v "$binary" 64 128 > "$out/${version}-t${threads}-r${rep}.txt" 2>&1
done
done
done
