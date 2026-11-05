"""Recompute the A/B table from raw `tools/ab.py` outputs (#698).

usage: ab_summary.py DIR [A=main] [B=pulp]

Reads DIR/ab-{opt,phys}-t{1,8}.txt. Repetition 0 is the warmup and is excluded, as in
`ab.py`. Prints, per case, the number of measured repetitions, the median and range of the
process wall time for A and B, the median change, whether the ranges overlap, and the
median of every C timer for A and B with its change. The medians printed by `ab.py` itself
(`MEDIAN` lines) are checked against the recomputed values.
"""
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
a_name = sys.argv[2] if len(sys.argv) > 2 else 'main'
b_name = sys.argv[3] if len(sys.argv) > 3 else 'pulp'
rep = re.compile(r'^rep(\d+) (\S+): (.*)$')
median_line = re.compile(r'^MEDIAN (\S+): wall=([0-9.]+) ')


def fields(text):
    """`key=value` pairs; timer names may contain spaces (`recal PfM=...`)."""
    return {k.strip(): float(v) for k, v in re.findall(r'([A-Za-z][A-Za-z0-9 ]*?)=([0-9.]+)', text)}


cases = []
for mode in ('phys', 'opt'):
    for t in (1, 8):
        f = out / f'ab-{mode}-t{t}.txt'
        if f.exists():
            cases.append((f'{"PhysCal" if mode == "phys" else "Opt"} t{t}', f))

print(f'| case | reps | {a_name} median (range) s | {b_name} median (range) s | change | ranges overlap |')
print('|---|---:|---:|---:|---:|---|')
timers = []
for label, f in cases:
    runs = {a_name: [], b_name: []}
    printed = {}
    for line in f.read_text().splitlines():
        m = rep.match(line)
        if m and int(m.group(1)) > 0 and m.group(2) in runs:
            runs[m.group(2)].append(fields(m.group(3)))
        m = median_line.match(line)
        if m:
            printed[m.group(1)] = float(m.group(2))
    wall = {k: [r['wall'] for r in v] for k, v in runs.items()}
    med = {k: statistics.median(v) for k, v in wall.items()}
    for k, v in printed.items():
        # ab.py prints medians with three decimals.
        assert abs(v - med[k]) <= 5e-4, f'{f}: printed median {v} != recomputed {med[k]}'
    assert len(wall[a_name]) == len(wall[b_name]) > 0, f'{f}: unequal repetitions'
    change = 100 * (med[b_name] / med[a_name] - 1)
    overlap = min(wall[b_name]) <= max(wall[a_name]) and min(wall[a_name]) <= max(wall[b_name])
    rng = {k: f'{min(v):.3f}-{max(v):.3f}' for k, v in wall.items()}
    print(f'| {label} | {len(wall[a_name])} | {med[a_name]:.3f} ({rng[a_name]}) | '
          f'{med[b_name]:.3f} ({rng[b_name]}) | {change:+.1f} % | {"yes" if overlap else "no"} |')
    keys = [k for k in runs[a_name][0] if k != 'wall']
    for k in keys:
        ta = statistics.median(r.get(k, 0.0) for r in runs[a_name])
        tb = statistics.median(r.get(k, 0.0) for r in runs[b_name])
        timers.append((label, k, ta, tb))

print()
print(f'| case | C timer | {a_name} median s | {b_name} median s | change |')
print('|---|---|---:|---:|---:|')
for label, k, ta, tb in timers:
    change = f'{100 * (tb / ta - 1):+.1f} %' if ta > 0 else 'n/a'
    print(f'| {label} | {k} | {ta:.3f} | {tb:.3f} | {change} |')
