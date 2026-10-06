#!/bin/bash
# Regenerate the nodal-configuration native-C Lanczos references (issue #481).
# Run on the host from the repository root; the C build and the generator run inside the
# Linux x86_64 Dev Container (see docs/DEV_CONTAINER.md). Cargo never runs this script.
#
#   c_toolbox/nodal_481/regenerate.sh [IMAGE]
#
# 1. python3 c_toolbox/nodal_481/make_sources.py writes the hand-made inputs and fixed
#    parameters to tests/fixtures/native_c_physcal_181/_sources/nodal_neel.
# 2. c_toolbox/physcal_native/build.sh builds the unmodified vmc.out (plain) and the additive
#    state-dump probe build (dump), and generate.py runs the `nodal_neel_*` rows of
#    tests/fixtures/native_c_physcal_181/scenarios.tsv.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
image=${1:-$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^vsc-mvmc-rs' | head -1)}
commit=$(git -C "$root" rev-parse HEAD:extern/mVMC-1.3.0)
python3 "$root/c_toolbox/nodal_481/make_sources.py" "$root/tests/fixtures/native_c_physcal_181/_sources/nodal_neel"
docker run --rm -v "$root":/repo -w /repo -u "$(id -u):$(id -g)" "$image" bash -lc "
  set -euo pipefail
  c_toolbox/physcal_native/build.sh /tmp/pn
  c_toolbox/physcal_native/build.sh /tmp/pnd dump
  python3 c_toolbox/physcal_native/generate.py \
    --vmc /tmp/pn/build/src/mVMC/vmc.out --vmc-dump /tmp/pnd/build/src/mVMC/vmc.out \
    --mvmc-commit $commit --out tests/fixtures/native_c_physcal_181 \
    --only nodal_neel_lanczos1 nodal_neel_lanczos2
"
