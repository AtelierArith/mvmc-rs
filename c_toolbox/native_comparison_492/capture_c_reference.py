#!/usr/bin/env python3
"""Regenerate the C-only #492 fixtures from the probe built by build_probe.sh.

Writes `rbm_sr_system.json` (the C `direct_S`/`direct_g`/`direct_map` dump before the
first SR solve) and `rbm_state.txt` (the first `state_dump_000.txt`) for the
`opt_hubbard_rbm_opttrans` workload, with the same input preparation as
`scripts/capture_native_c.py` (`NSROptItrStep 1`, `NSROptItrSmp 1`), but without the
Rust comparison. Optional developer command, run inside the Linux Dev Container;
Cargo never invokes it.

    python3 c_toolbox/native_comparison_492/capture_c_reference.py \
        --c BUILD/build/src/mVMC/vmc.out --work NEW_DIR --out tests/fixtures/native_comparison_492
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))
import bench_cpu_round as b  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--c", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
name = "opt_hubbard_rbm_opttrans"
work = args.work.resolve()
work.mkdir(parents=True, exist_ok=False)
dest, flags, _ = b.prepare(name, work)
b.patch_modpara(dest / "modpara.def", {"NSROptItrStep": "1", "NSROptItrSmp": "1"})
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BLIS_NUM_THREADS="1")
with (work / "c.log").open("w") as f:
    subprocess.run(["/opt/mpich/bin/mpirun", "-np", "1", str(args.c.resolve()), *flags,
                    "namelist.def"], cwd=dest, env=env, stdout=f, stderr=subprocess.STDOUT,
                   check=True)
c = {}
for line in (work / "c.log").open():
    m = re.match(r"DEBUG: (direct_\w+)\[(\d+)\]=([^ ]+)", line)
    if m:
        key, _, val = m.groups()
        c.setdefault(key, []).append(float(val))
out = args.out.resolve()
(out / "rbm_sr_system.json").write_text(json.dumps(c, indent=2))
shutil.copyfile(dest / "state_dump_000.txt", out / "rbm_state.txt")
sources = [root / "extern/mVMC-1.4.0/src/mVMC" / n
           for n in ["vmcmain.c", "vmccal.c", "stcopt_dposv.c", "readdef.c"]]
sources += [args.c.resolve(), Path(__file__).resolve()]
(out / "full_probe_provenance.json").write_text(json.dumps(
    {str(p.relative_to(root)) if p.is_relative_to(root) else p.name:
     hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2) + "\n")
print("wrote", out)
