#!/usr/bin/env bash
# Regenerate tests/fixtures/pfaffian_cg/two_hop_bilinear_c.txt (explicit developer command;
# ordinary Rust tests never run this). Builds the probe with and without FMA contraction and
# checks the sequential C reduction is identical in both (the loop is a serial dependency chain,
# so contraction of `tmp += a*b` into one fma is the only possible difference; it only happens
# with -mfma / -march=native).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here/../.."
out="${1:-$root/tests/fixtures/pfaffian_cg/two_hop_bilinear_c.txt}"
tmp="$(mktemp -d)"
gcc -O2 -ffp-contract=off "$here/probe.c" -o "$tmp/probe_nofma"
gcc -O2 "$here/probe.c" -o "$tmp/probe_default"
input="$root/tests/fixtures/pfaffian_cg/two_hop_bilinear.txt"
"$tmp/probe_nofma" < "$input" > "$tmp/nofma.txt"
"$tmp/probe_default" < "$input" > "$tmp/default.txt"
cmp "$tmp/nofma.txt" "$tmp/default.txt"
{
  echo "# C two-hop bilinear form bMa (pfupdate_two_real.c:167-177), sequential, no FMA."
  echo "# One '<n> <hex bits>' line per case of two_hop_bilinear.txt, same order."
  echo "# Provenance: see c_toolbox/two_hop_bilinear_449/README.md"
  cat "$tmp/nofma.txt"
} > "$out"
echo "wrote $out"
rm -rf "$tmp"
