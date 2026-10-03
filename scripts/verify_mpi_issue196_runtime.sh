#!/usr/bin/env bash
# Explicit optional developer command; Cargo tests never run the C toolbox.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MPI196_IMAGE_ID:?set from docker inspect .Image for this actual container}"
command -v jq >/dev/null || { echo 'Unsupported: jq is required for artifact selection.' >&2; exit 2; }
export CARGO_TARGET_DIR=${MPI196_TARGET_DIR:-${CARGO_TARGET_DIR:?named-volume target required}}
case "$CARGO_TARGET_DIR" in
    /home/vscode/.cache/mvmc/target/issue196-*) ;;
    *) echo 'Use a named-volume issue196 target in the current devcontainer.' >&2; exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
out=$(mktemp -d /home/vscode/.cache/mvmc/issue196-world.XXXXXX)
echo "Evidence: $out"
{
    date -u
    printf 'Actual image: %s\n' "$MPI196_IMAGE_ID"
    uname -a
    git -c safe.directory="$PWD" rev-parse HEAD
    git -c safe.directory="$PWD" submodule status --recursive
    rustc -Vv
    cargo -V
    "${MPICC:-mpicc}" -show
    mpichversion
    mpiexec --version
    gcc --version
    pkg-config --modversion openblas
    command -v jq
    jq --version
    printf 'C command: %s -std=c11 -O2 -Wall -Wextra -Werror c_toolbox/mpi_issue196_world.c -o %s/world\n' "${MPICC:-mpicc}" "$out"
} > "$out/provenance.txt" 2>&1
git -c safe.directory="$PWD" ls-files --cached --others --exclude-standard -z -- crates scripts .devcontainer c_toolbox Cargo.toml Cargo.lock rustfmt.toml |
    sort -zu | xargs -0 sha256sum > "$out/source-sha256.txt"
sha256sum "$out/source-sha256.txt" >> "$out/provenance.txt"
"${MPICC:-mpicc}" -std=c11 -O2 -Wall -Wextra -Werror c_toolbox/mpi_issue196_world.c -o "$out/world"
ldd "$out/world" >> "$out/provenance.txt"
check_world() {
    awk -v n="$1" '$1=="WORLD" { if (NF!=5 || $2<0 || $2>=n || seen[$2]++ || $3!=n || $4!=n*(n+1)/2 || $5!=196) bad=1; count++ } END { exit bad || count!=n }' "$2"
}
for ranks in 2 4; do
    timeout --kill-after=5s 30s mpiexec -n "$ranks" "$out/world" "$ranks" </dev/null > "$out/c-$ranks.txt" 2>&1
    check_world "$ranks" "$out/c-$ranks.txt"
done
timeout --kill-after=10s 1800s cargo test --locked --profile test-fast -p mvmc-core --features mpi \
    --test mpi_issue196_world --no-run --message-format=json > "$out/build.json" 2> "$out/build.log"
binary=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue196_world" and .executable!=null) | .executable' "$out/build.json" | tail -n 1)
test -n "$binary" && test -x "$binary"
listing=$("$binary" --list)
[[ $listing == *'issue196_genuine_world: test'* ]] || {
    echo 'Unsupported: MPI feature/test is missing; zero tests cannot pass this gate.' >&2; exit 2;
}
sha256sum "$binary" >> "$out/provenance.txt"
ldd "$binary" >> "$out/provenance.txt"
for ranks in 2 4; do
    MPI196_EXPECT_RANKS="$ranks" timeout --kill-after=5s 30s mpiexec -n "$ranks" "$binary" \
        --ignored --exact issue196_genuine_world --nocapture </dev/null > "$out/rust-$ranks.txt" 2>&1
    check_world "$ranks" "$out/rust-$ranks.txt"
done
sha256sum -c "$out/source-sha256.txt" > "$out/source-check.txt"
echo "PASS: independent C and Rust genuine 2/4-rank worlds, exact integer sums and nonzero-root broadcast. Runtime-only, not numerical parity. Evidence: $out"
