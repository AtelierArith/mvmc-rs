#!/usr/bin/env python3
"""First-divergence analysis of Lanczos single-step outputs (issue #181).

Transcribes C `physcal_lanczos.c` CalculateEne/CalculateEneByAlpha in IEEE double
(same operation order; pow(x, 2) is x*x) and evaluates them on the moments
`zvo_ls_qqqq_*.dat` written by native C and by Rust. It prints, per file:

* whether the transcription reproduces each implementation's own `zvo_ls_out`
  (energy, relative variance, alpha) exactly, i.e. the Rust and C outputs differ
  only through their moments;
* the relative moment difference (the first divergence, from sample accumulation
  order) and the resulting relative output difference, whose ratio is the
  condition number of alpha with respect to the moments.

Usage: lanczos_sensitivity.py <native scenario dir> <Rust output dir>
Run with plain python3; no Cargo, C or Julia is involved.
"""
import glob, math, os, sys


def calc(h1, h21, h22, h3, h4):
    aa = h21 * (h21 + h22) - 2 * h1 * h3
    bb = -h1 * h21 + h3
    cc = (h21 * (h21 + h22) ** 2 - h1**2 * h21 * (h21 + 2.0 * h22) + 4 * h1**3 * h3
          - 2.0 * h1 * (2 * h21 + h22) * h3 + h3 * h3)
    if cc < 0:
        return None
    out = []
    for sign in (1.0, -1.0):
        alpha = (bb + sign * math.sqrt(cc)) / aa
        tene = h1 + alpha * (h21 + h22) + alpha * alpha * h3
        dnorm = 1.0 + 2 * alpha * h1 + alpha * alpha * h21
        tv = h21 + 2 * alpha * h3 + alpha * alpha * h4
        ev = ((tv / dnorm) - (tene / dnorm) ** 2) / (tene / dnorm) ** 2
        out.append((tene / dnorm, ev, alpha))
    return out[0] if out[0][0] <= out[1][0] else out[1]


def moments(path):
    q = [float(x) for x in open(path).read().split()]
    return q[2], q[3], q[10], q[11], q[15]


def main():
    scenario, rust = sys.argv[1:3]
    for path in sorted(glob.glob(os.path.join(scenario, "expected", "zvo_ls_qqqq_*.dat"))):
        name = os.path.basename(path)
        tag = name[len("zvo_ls_qqqq_"):]
        mc, mr = moments(path), moments(os.path.join(rust, name))
        outc = [float(x) for x in open(os.path.join(scenario, "expected", "zvo_ls_out_" + tag)).read().split()]
        outr = [float(x) for x in open(os.path.join(rust, "zvo_ls_out_" + tag)).read().split()]
        ec, er = calc(*mc), calc(*mr)
        if ec is None or er is None or len(outc) != 3 or len(outr) != 3:
            print(tag, "no Lanczos value (illegal alpha or empty file)")
            continue
        dm = max(abs(a - b) / max(abs(a), abs(b)) for a, b in zip(mc, mr))
        do = max(abs(a - b) / max(abs(a), abs(b), 1e-300) for a, b in zip(outc, outr))
        recon_c = max(abs(a - b) / max(abs(a), 1e-300) for a, b in zip(ec, outc))
        recon_r = max(abs(a - b) / max(abs(a), 1e-300) for a, b in zip(er, outr))
        print(f"{tag}: moment rel diff {dm:.2e}; output rel diff {do:.2e}; "
              f"amplification {do / dm if dm else float('inf'):.2e}; "
              f"transcription vs own output rel {recon_c:.1e} (C) {recon_r:.1e} (Rust)")


main()
