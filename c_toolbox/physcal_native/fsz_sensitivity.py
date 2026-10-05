#!/usr/bin/env python3
"""Input-sensitivity probe for the FSZ PhysCal energy (issue #181).

Runs the Rust CLI on a native-C scenario directory and then on copies of its fixed
parameter file in which every value is multiplied by (1 + eps*u), u uniform in
[-1, 1]. With eps = 1e-16 (one ulp) the first-sample FSZ energy moves by up to
~1e-11, i.e. the Rust-versus-C energy difference of the one-configuration FSZ
scenarios (1e-10) is the size of an input roundoff perturbation amplified by the
conditioning of the Pfaffian/inverse chain, not an algorithmic difference. The same
probe on a complex non-FSZ scenario moves the energy by < 3e-15.

Usage (repository root, after `cargo build --profile test-fast -p mvmc-cli`):
    python3 c_toolbox/physcal_native/fsz_sensitivity.py fsz_warm0_sample1 fsz 1e-16
    python3 c_toolbox/physcal_native/fsz_sensitivity.py heisenberg_chain_cmp cmp 1e-16
"""
import os, random, shutil, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
scenario, mode = sys.argv[1], sys.argv[2]
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-16
fixture = os.path.join(ROOT, "tests/fixtures/native_c_physcal_181", scenario)
binary = os.path.join(ROOT, "target/test-fast/mvmc")
tokens = open(os.path.join(fixture, "zqp_opt.dat")).read().split()
head, body = tokens[:6], tokens[6:]


def run(values):
    work = tempfile.mkdtemp(prefix="sens-")
    param = os.path.join(work, "zqp.dat")
    open(param, "w").write("  ".join(head + values) + "\n")
    result = subprocess.run(
        [binary, os.path.join(fixture, "inputs/namelist.def"), "--physcal", param,
         "--seed", "1", "--mode", mode, "--out-dir", os.path.join(work, "out")],
        capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert result.returncode == 0, result.stderr
    first = sorted(f for f in os.listdir(os.path.join(work, "out")) if f.startswith("zvo_out_"))[0]
    energies = open(os.path.join(work, "out", first)).read().split()
    shutil.rmtree(work)
    return energies


base = run(body)
print("base energy, variance-like, squared:", base[:3])
random.seed(3)
for _ in range(4):
    values = list(body)
    for k in range(0, len(values), 3):
        values[k] = "%.18e" % (float(values[k]) * (1 + eps * (random.random() * 2 - 1)))
    out = run(values)
    print("perturbed - base:", ["%.3e" % (float(a) - float(b)) for a, b in zip(out[:3], base[:3])])
