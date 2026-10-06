#!/usr/bin/env bash
# Regenerate tests/fixtures/glibc_complex_470/probe.txt (explicit developer command; ordinary
# Rust tests never run this). Records the C library actually used.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here/../.."
out="${1:-$root/tests/fixtures/glibc_complex_470/probe.txt}"
tmp="$(mktemp -d)"
gcc -O1 -fno-builtin -ffp-contract=off "$here/probe.c" -o "$tmp/probe" -lm
python3 "$here/gen_inputs.py" > "$tmp/inputs.txt"
{
  echo "# glibc complex probe (issue #470): re im | cexp | clog | ccosh | ctanh, hex binary64."
  echo "# gcc: $(gcc --version | head -1)"
  echo "# libc: $(ldd --version | head -1)"
  echo "# CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
  echo "# Provenance: c_toolbox/glibc_complex_470/README.md"
  "$tmp/probe" < "$tmp/inputs.txt"
} > "$out"
echo "wrote $out ($(grep -vc '^#' "$out") cases)"
rm -rf "$tmp"
