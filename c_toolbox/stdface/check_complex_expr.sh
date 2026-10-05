#!/bin/sh
# Build c_toolbox/stdface/complex_expr.c at -O0, -O2 and -O3, check that all three agree and
# compare the per-family digests with tests/fixtures/stdface/complex_expr.digests.
# Usage: c_toolbox/stdface/check_complex_expr.sh [--update]   (--update rewrites the fixture)
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CC=${CC:-gcc}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
for opt in -O0 -O2 -O3; do
  "$CC" $opt -DNDEBUG -ffp-contract=off -o "$TMP/probe$opt" "$ROOT/c_toolbox/stdface/complex_expr.c" -lm
  "$TMP/probe$opt" > "$TMP/digests$opt"
done
cmp "$TMP/digests-O0" "$TMP/digests-O2"
cmp "$TMP/digests-O2" "$TMP/digests-O3"
FIXTURE="$ROOT/tests/fixtures/stdface/complex_expr.digests"
if [ "${1:-}" = "--update" ]; then
  cp "$TMP/digests-O3" "$FIXTURE"
  echo "updated $FIXTURE"
else
  cmp "$TMP/digests-O3" "$FIXTURE"
  echo "complex_expr digests match $FIXTURE"
fi
