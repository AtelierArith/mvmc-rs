import ast
import json
import math
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
summary = {}
for case in ('l32-opt', 'l32-phys', 'l64-opt'):
    summary[case] = {}
    for shared in (0, 1):
        text = (root / f'{case}-w4-g40000-s{shared}-rdiagnostic-p1.stderr').read_text()
        rows = []
        sites = []
        aggregates = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) == 7 and fields[1] in ('pool', 'serial'):
                rows.append(dict(site=fields[0], mode=fields[1], calls=int(fields[2]),
                                 items=int(fields[3]), total_ms=float(fields[4]),
                                 us_per_call=float(fields[5]), ns_per_item=float(fields[6])))
            match = re.fullmatch(r'\s+(\S+:\d+) (hot|cold) (\[.*\])', line)
            if match:
                sites.append(dict(site=match[1], hot=match[2] == 'hot',
                                  counts=ast.literal_eval(match[3])))
            match = re.fullmatch(r'\s+(hot|cold): (\[.*\])', line)
            if match:
                aggregates[match[1]] = ast.literal_eval(match[2])
        assert rows and sites and set(aggregates) == {'hot', 'cold'}, case
        for kind in ('hot', 'cold'):
            assert [sum(s['counts'][i] for s in sites if s['hot'] == (kind == 'hot'))
                    for i in range(7)] == aggregates[kind]
        counters = re.findall(r'shared-team-probe: attempts=(\d+) successes=(\d+) busy=(\d+) unavailable=(\d+) participants=(\d+)', text)
        assert len(counters) == 1, (case, shared)
        attempts, successes, busy, unavailable, participants = map(int, counters[0])
        assert attempts == successes + busy + unavailable
        if shared:
            assert successes > 0 and participants >= 2
        diagnostic = root / f'{case}-w4-g40000-s{shared}-rdiagnostic-p1-output'
        reference = root / f'{case}-w4-g40000-s{shared}-r1-p0-output'
        names = lambda d: {p.name for p in d.iterdir() if p.is_file() and '_time_' not in p.name}
        assert names(diagnostic) and names(diagnostic) == names(reference)
        maximum = 0.0
        for name in names(reference):
            def values(directory):
                vals = [float(v) for line in (directory / name).read_text().splitlines()
                        if not line.lstrip().startswith('#') for v in line.split()]
                assert all(math.isfinite(v) for v in vals)
                return vals
            a, b = values(diagnostic), values(reference)
            assert len(a) == len(b)
            maximum = max(maximum, max((abs(x-y) for x,y in zip(a,b)), default=0.0))
        summary[case][shared] = dict(scopes=rows, gap_sites=sites, gaps=aggregates,
                                    output_max_absolute_difference=maximum,
                                    dispatch=dict(attempts=attempts, successes=successes,
                                                  busy=busy, unavailable=unavailable,
                                                  participants=participants))
print(json.dumps(summary, indent=2))
