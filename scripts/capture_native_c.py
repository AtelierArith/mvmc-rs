#!/usr/bin/env python3
"""Optional full native SR/RNG capture (#492); run through uv in Linux container.

Requires explicitly built C toolbox probe and Rust native_c_diagnostics example.
No Cargo build or test invokes this command.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import bench_cpu_round as b

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--c", type=Path, required=True)
parser.add_argument("--rust", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument(
    "--workloads", default="opt_heisenberg_real,opt_hubbard_rbm_opttrans"
)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
for name in args.workloads.split(","):
    if name not in b.WORKLOADS or not name.startswith("opt_"):
        parser.error(f"requires an optimization workload: {name}")
base = args.out.resolve()
base.mkdir(parents=True, exist_ok=False)
sources = [
    root / "extern/mVMC-1.3.0/src/mVMC" / name
    for name in ["vmcmain.c", "vmccal.c", "stcopt_dposv.c", "readdef.c"]
]
sources.extend([args.c.resolve(), args.rust.resolve(), Path(__file__)])
(base / "provenance.json").write_text(
    json.dumps(
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2
    )
)
env = dict(
    os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BLIS_NUM_THREADS="1"
)
for name in args.workloads.split(","):
    dest, flags, para = b.prepare(name, base)
    b.patch_modpara(dest / "modpara.def", {"NSROptItrStep": "1", "NSROptItrSmp": "1"})
    (base / f"{name}-input-hashes.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in dest.iterdir()
                if p.is_file()
            },
            indent=2,
        )
    )
    with (base / f"{name}-c.log").open("w") as f:
        subprocess.run(
            [
                "/opt/mpich/bin/mpirun",
                "-np",
                "1",
                str(args.c.resolve()),
                *flags,
                "namelist.def",
            ],
            cwd=dest,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    subprocess.run(
        [
            str(args.rust.resolve()),
            str(dest / "namelist.def"),
            str(base / f"{name}-rust"),
            *flags,
        ],
        env=env,
        check=True,
    )
    rust = json.load(open(base / f"{name}-rust/diagnostics.json"))
    c = {}
    for line in (base / f"{name}-c.log").open():
        m = re.match(r"DEBUG: (SROpt\w+)\[(\d+)\]=([^ ]+) \+I\*([^ ]+)", line)
        if m:
            key, idx, reval, imval = m.groups()
            c.setdefault(key, []).append([float(reval), float(imval)])
        m = re.match(r"DEBUG: (direct_\w+)\[(\d+)\]=([^ ]+)", line)
        if m:
            key, idx, val = m.groups()
            c.setdefault(key, []).append(float(val))
    n = rust["normalized"][0]
    checks = {}
    for ck, rk in [
        ("SROptOO_real", "oo_real"),
        ("SROptHO_real", "ho_real"),
        ("SROptOO", "oo"),
        ("SROptHO", "ho"),
    ]:
        if ck not in c:
            continue
        cv = c[ck]
        rv = n[rk]
        if rk.endswith("_real"):
            rv = [[v, 0] for v in rv]
        # C's Hermitian complex OO dump uses the opposite storage orientation.
        # Compare its conjugate; the real symmetric direct SR system below needs
        # no transformation and independently validates the arithmetic operand.
        if rk == "oo":
            rv = [[a, -b] for a, b in rv]
        diffs = [
            (abs(a - b), i, j, a, b)
            for i, (cr, rr) in enumerate(zip(cv, rv))
            for j, (a, b) in enumerate(zip(cr, rr))
        ]
        print(name, ck, "shape", len(cv), len(rv), "max", max(diffs), flush=True)
        checks[ck] = dict(
            c_length=len(cv),
            rust_length=len(rv),
            max_absolute=max(diffs)[0],
            common_prefix=True,
        )
    state = {
        line.split()[0]: [int(v) for v in line.split()[1:]]
        for line in (dest / "state_dump_000.txt").read_text().splitlines()
    }
    for ckey, rkey in [
        ("draws", "rng_words_consumed"),
        ("next624", "rng_next624"),
        ("ele_idx", "ele_idx"),
        ("ele_cfg", "ele_cfg"),
        ("counter", "counter"),
    ]:
        cv = state[ckey][0] if ckey == "draws" else state[ckey]
        if ckey == "counter":
            cv = cv
            rv = rust[rkey][: len(cv)]
        else:
            rv = rust[rkey]
        print(name, ckey, cv == rv, flush=True)
        checks[ckey] = cv == rv
    system = rust["systems"][0]
    print(
        name,
        "active map",
        c["direct_map"] == system["active_indices"],
        len(c["direct_map"]),
        len(system["active_indices"]),
        flush=True,
    )
    checks["active_map"] = c["direct_map"] == system["active_indices"]
    for ck, rk in [("direct_S", "matrix"), ("direct_g", "rhs")]:
        cv, rv = c[ck], system[rk]
        print(
            name,
            ck,
            "shape",
            len(cv),
            len(rv),
            "max",
            max(abs(a - b) for a, b in zip(cv, rv)),
            flush=True,
        )
    (base / f"{name}-c-system.json").write_text(
        json.dumps({k: v for k, v in c.items() if k.startswith("direct_")}, indent=2)
    )
    (base / f"{name}-checks.json").write_text(json.dumps(checks, indent=2))
    if not all(
        checks[key]
        for key in ["draws", "next624", "ele_idx", "ele_cfg", "counter", "active_map"]
    ):
        raise SystemExit(f"{name}: discrete/RNG mismatch; inspect retained capture")
