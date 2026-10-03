#!/usr/bin/env bash
# Explicit developer diagnostic. Cargo/Rust tests never invoke the C toolbox.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/mpi_issue179_environment.sh
: "${MPI179_IMAGE_ID:?set MPI179_IMAGE_ID from docker inspect .Image for this actual container}"
case "$CARGO_TARGET_DIR" in
  /home/vscode/.cache/mvmc/target/issue179-*) ;;
  *) echo 'Run this preflight in the current devcontainer with a named-volume issue179 target.' >&2; exit 2 ;;
esac
out=$(mktemp -d /home/vscode/.cache/mvmc/issue196-preflight.XXXXXX)
echo "Evidence: $out"
{
  date -u
  printf 'Actual image: %s\n' "$MPI179_IMAGE_ID"
  uname -a
  git rev-parse HEAD
  rustc -Vv
  cargo -V
  "${MPICC:-mpicc}" -show
  mpichversion
  mpirun --version
  gcc --version
  sha256sum .devcontainer/Dockerfile c_toolbox/mpi_issue196_world.c "$0"
  printf 'C compiler command: %s -std=c11 -O2 -Wall -Wextra -Werror c_toolbox/mpi_issue196_world.c -o %s/world\n' "${MPICC:-mpicc}" "$out"
} > "$out/provenance.txt" 2>&1
git ls-files --cached --others --exclude-standard -z -- crates scripts .devcontainer c_toolbox Cargo.toml Cargo.lock rustfmt.toml |
  sort -z | xargs -0 sha256sum > "$out/source-sha256.txt"
"${MPICC:-mpicc}" -std=c11 -O2 -Wall -Wextra -Werror c_toolbox/mpi_issue196_world.c -o "$out/world"
ldd "$out/world" >> "$out/provenance.txt"
for ranks in 2 4; do
  timeout --kill-after=5s 30s "${mpi179_mpirun[@]}" -n "$ranks" "$out/world" "$ranks" </dev/null > "$out/c-$ranks.txt" 2>&1
  awk -v n="$ranks" '$1=="WORLD" { if (NF!=5 || $2<0 || $2>=n || seen[$2]++ || $3!=n || $4!=n*(n+1)/2 || $5!=196) bad=1; count++ } END { exit bad || count!=n }' "$out/c-$ranks.txt"
done
timeout --kill-after=10s 600s cargo test --locked --profile test-fast -p mvmc-core --features mpi \
  --test mpi_issue179_mapping --no-run --message-format=json > "$out/build.json" 2> "$out/build.log"
binary=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_mapping" and .executable!=null) | .executable' "$out/build.json" | tail -n 1)
test -n "$binary" && test -x "$binary"
source scripts/mpi_issue179_selection.sh
mpi179_require_test "$binary" issue179_group_width_endpoints
sha256sum "$binary" >> "$out/provenance.txt"
ldd "$binary" >> "$out/provenance.txt"
for ranks in 2 4; do
  timeout --kill-after=5s 30s env MPI179_EXPECT_RANKS="$ranks" \
    "${mpi179_mpirun[@]}" -n "$ranks" "$binary" --ignored --exact issue179_group_width_endpoints --nocapture \
    </dev/null > "$out/rust-$ranks.txt" 2>&1
done
sha256sum -c "$out/source-sha256.txt" > "$out/source-check.txt"
printf 'PASS: independent C and Rust genuine 2/4-rank worlds, group widths including uneven groups, QP ownership, global broadcasts and counters. Evidence: %s\n' "$out"
