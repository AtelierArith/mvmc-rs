#!/usr/bin/env python3
"""Compare same-binary L64 timing samples and finite formatted outputs."""
import json
import math
from pathlib import Path
import re
import statistics
import sys
root=Path(sys.argv[1])
out={}
for n in (1,4,16):
    dirs=[root/'results'/f'{mode}-i{n}-r{rep}' for mode in ('off','on') for rep in (1,2,3)]
    names=sorted(p.name for p in dirs[0].iterdir() if p.is_file() and '_time_' not in p.name)
    assert names and all(sorted(p.name for p in d.iterdir() if p.is_file() and '_time_' not in p.name)==names for d in dirs)
    maximum=0.0
    for name in names:
        def numbers(p):
            values=[float(x) for line in p.read_text().splitlines() if not line.lstrip().startswith('#') for x in line.split()]
            assert all(map(math.isfinite,values))
            return values
        reference=numbers(dirs[0]/name)
        for directory in dirs[1:]:
            actual=numbers(directory/name)
            assert len(actual)==len(reference)
            maximum=max(maximum,max((abs(a-b) for a,b in zip(actual,reference)),default=0.0))
    entry={'files_compared':len(names),'max_abs_from_first':maximum,'timings':{}}
    for mode in ('off','on'):
        samples=[]
        for rep in (1,2,3):
            log=(root/f'{mode}-i{n}-r{rep}.log').read_text()
            match=re.search(r'wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)',log)
            assert match
            wall,user,system=map(float,match.groups())
            samples.append({'wall':wall,'user':user,'system':system})
        entry['timings'][mode]={'samples':samples,**{f'{key}_median':statistics.median(x[key] for x in samples) for key in ('wall','user','system')}}
    out[str(n)]=entry
print(json.dumps(out,indent=2))
