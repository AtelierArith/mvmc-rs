#!/usr/bin/env python3
"""Summarize matched MPI spin-policy timings and formatted output values."""
import json
import math
from pathlib import Path
import re
import statistics
import sys
root = Path(sys.argv[1])
summary = {}
for mode in ('opt', 'phys'):
    conditions = ('off-default', 'on-default', 'off-zero', 'on-zero')
    dirs = [root/'results'/f'{mode}-{condition}-r{rep}' for condition in conditions for rep in (1, 2, 3)]
    names = sorted(p.name for p in dirs[0].iterdir() if p.is_file() and '_time_' not in p.name)
    assert names and all(sorted(p.name for p in d.iterdir() if p.is_file() and '_time_' not in p.name) == names for d in dirs)
    maximum = 0.0
    for name in names:
        def values(p):
            xs = [float(x) for line in p.read_text().splitlines() if not line.lstrip().startswith('#') for x in line.split()]
            assert all(map(math.isfinite, xs))
            return xs
        ref = values(dirs[0]/name)
        for d in dirs[1:]:
            other = values(d/name)
            assert len(ref) == len(other)
            maximum = max(maximum, max((abs(x-y) for x,y in zip(ref,other)), default=0.0))
    entry = {'files_compared':len(names), 'max_abs_from_first':maximum, 'timings':{}}
    for condition in conditions:
        samples = []
        for rep in (1,2,3):
            s=(root/f'{mode}-{condition}-r{rep}.log').read_text()
            match=re.search(r'wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)',s)
            assert match
            w,u,system=map(float,match.groups())
            samples.append({'wall':w,'user':u,'system':system})
        entry['timings'][condition]={'samples':samples, **{f'{k}_median':statistics.median(v[k] for v in samples) for k in ('wall','user','system')}}
    summary[mode]=entry
print(json.dumps(summary, indent=2))
