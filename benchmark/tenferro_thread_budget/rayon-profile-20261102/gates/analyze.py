import json, math, re, statistics, sys
from pathlib import Path
root = Path(sys.argv[1]); summary = {}
def numbers(p):
    vals = [float(x) for line in p.read_text().splitlines() if not line.lstrip().startswith('#') for x in line.split()]
    if not all(map(math.isfinite, vals)): raise ValueError(f'non-finite {p}')
    return vals
for case in ('l32-opt', 'l32-phys', 'l64-opt'):
    summary[case] = {'gates': {}}
    dirs = [root / f'{case}-g{g}-r{r}-output' for g in (20000,40000,80000,160000) for r in (1,2,3)]
    names = sorted(p.name for p in dirs[0].iterdir() if p.is_file() and '_time_' not in p.name)
    if not names: raise ValueError(f'{case}: no outputs')
    if any(sorted(p.name for p in d.iterdir() if p.is_file() and '_time_' not in p.name) != names for d in dirs): raise ValueError(f'{case}: file mismatch')
    maximum = 0.0
    for name in names:
        ref = numbers(dirs[0] / name)
        for directory in dirs[1:]:
            vals = numbers(directory / name)
            if len(vals) != len(ref): raise ValueError(f'{directory / name}: count mismatch')
            maximum = max(maximum, max((abs(a-b) for a,b in zip(ref, vals)), default=0.0))
    summary[case].update(files_compared=len(names), max_abs_from_first=maximum)
    for gate in (20000,40000,80000,160000):
        samples=[]
        for rep in (1,2,3):
            match=re.search(r'wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)', (root / f'{case}-g{gate}-r{rep}.log').read_text())
            if not match: raise ValueError(f'{case} {gate} {rep}: missing timing')
            wall,user,system=map(float,match.groups()); samples.append({'wall':wall,'user':user,'system':system})
        summary[case]['gates'][gate]={'samples':samples, **{f'{k}_median':statistics.median(v[k] for v in samples) for k in ('wall','user','system')}}
print(json.dumps(summary,indent=2))
