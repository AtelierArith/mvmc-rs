#!/usr/bin/env bash
# Regenerate crates/mvmc-core/tests/fixtures/normalization_c_literals.tsv from mVMC 1.4.0
# (explicit developer command; Cargo never runs it). Linux x86_64, GCC.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
src="$root/extern/mVMC-1.4.0/src/mVMC/average.c"
expected=ccdaaa27bd0393e9bd73341459227555da798928fd1ee25b3e1151b6aecc54be
[ "$(sha256sum "$src" | cut -d' ' -f1)" = "$expected" ] || { echo "average.c pin mismatch" >&2; exit 1; }
out=${1:-$root/crates/mvmc-core/tests/fixtures/normalization_c_literals.tsv}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# weightAverageReduce_fcmp and weightAverageReduce_real, verbatim.
sed -n 277,333p "$src" > "$tmp/average_helpers.inc"
head -1 "$tmp/average_helpers.inc" | grep -q '^void weightAverageReduce_fcmp(' || { echo "boundary" >&2; exit 1; }
gcc -std=c11 -O3 -Wall -Wextra -Wno-unknown-pragmas -I"$tmp" "$here/driver.c" -o "$tmp/driver" -lm
"$tmp/driver" > "$out"
echo "wrote $out"
