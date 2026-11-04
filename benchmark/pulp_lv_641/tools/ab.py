"""Alternating A/B timing of mvmc binaries on the L128 inputs (native, BLAS 1).

usage: ab.py MODE THREADS FIRST_CPU REPS(after warmup) NAME=BIN [NAME=BIN ...]
Prints per-rep process wall and selected C timers; medians at the end.
"""
import os, shutil, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path.cwd()  # per-run work directories
REPO = Path(__file__).resolve().parents[3]
mode, n, first, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
bins = dict(arg.split('=', 1) for arg in sys.argv[5:])
keys = ['All', 'VMCMakeSample', 'UpdateMAll', 'recal PfM', 'VMCMainCal', 'CalculateMAll', 'LocEnergyCal',
        'CalculateNewPfM2', 'StochasticOpt']
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           BLIS_NUM_THREADS='1', MVMC_RS_INNER_THREADS=str(n), RAYON_NUM_THREADS=str(n),
           MVMC_RS_SR_BACKEND='c-order', MVMC_RS_MEASURE_PF_BACKEND='c-order', MVMC_C_TIMER='1')
for k in [k for k in env if k.startswith('MVMC_') and k not in (
        'MVMC_RS_INNER_THREADS', 'MVMC_RS_SR_BACKEND', 'MVMC_RS_MEASURE_PF_BACKEND', 'MVMC_C_TIMER')]:
    del env[k]
rows = {name: [] for name in bins}
cpus = f'{first}-{first + n - 1}'
for rep in range(reps + 1):
    order = list(bins) if rep % 2 == 0 else list(reversed(bins))
    for name in order:
        work = HERE / f'ab-{mode}-t{n}-{name}'
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(REPO / 'benchmark/cpu_scaling_l128/inputs' / mode, work)
        t = time.perf_counter()
        extra = ['zqp_opt.dat'] if mode == 'phys' else []
        subprocess.run(['taskset', '-c', cpus, bins[name], 'namelist.def', *extra], cwd=work, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wall = time.perf_counter() - t
        timers = {}
        for line in (work / 'output/zvo_CalcTimer.dat').read_text().splitlines():
            for k in keys:
                if line.strip().startswith(k) and k not in timers:
                    timers[k] = float(line.split()[-1])
        print(f'rep{rep} {name}: wall={wall:.3f} ' + ' '.join(f'{k}={v:.3f}' for k, v in timers.items()), flush=True)
        if rep > 0:
            rows[name].append((wall, timers))
for name, rs in rows.items():
    med = statistics.median(w for w, _ in rs)
    tm = {k: statistics.median(t.get(k, 0) for _, t in rs) for k in keys}
    print(f'MEDIAN {name}: wall={med:.3f} [{", ".join(f"{w:.3f}" for w, _ in rs)}] ' +
          ' '.join(f'{k}={v:.3f}' for k, v in tm.items()))
