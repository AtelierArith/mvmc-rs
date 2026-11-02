import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
summary = {}


def numbers(path):
    vals = [float(v) for line in path.read_text().splitlines()
            if not line.lstrip().startswith('#') for v in line.split()]
    assert all(math.isfinite(v) for v in vals), path
    return vals


for mode in ('opt', 'phys'):
    summary[mode] = {}
    for ranks in (1, 4):
        prefix = f'{mode}-r{ranks}'
        reference = root / f'{prefix}-s0-spindefault-rep1-output'
        names = {p.name for p in reference.iterdir() if p.is_file() and '_time_' not in p.name}
        assert names
        baseline = {n: numbers(reference / n) for n in names}
        maximum = 0.0
        groups = []
        for shared in (0, 1):
            for spin in ('default', '0'):
                samples = []
                for rep in (1, 2, 3):
                    label = f'{prefix}-s{shared}-spin{spin}-rep{rep}'
                    directory = root / f'{label}-output'
                    assert {p.name for p in directory.iterdir() if p.is_file() and '_time_' not in p.name} == names
                    for name in names:
                        a, b = baseline[name], numbers(directory / name)
                        assert len(a) == len(b)
                        maximum = max(maximum, max((abs(x-y) for x,y in zip(a,b)), default=0.0))
                    times = re.findall(r'^wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)$', (root / f'{label}.time').read_text(), re.M)
                    assert len(times) == 1, label
                    samples.append(dict(zip(('wall', 'user', 'system'), map(float, times[0]))))
                groups.append(dict(shared=shared, spin=spin, samples=samples,
                                   medians={k: statistics.median(s[k] for s in samples)
                                            for k in ('wall', 'user', 'system')}))
        result = dict(files_compared=len(names), max_absolute_difference=maximum, groups=groups)
        if ranks == 4:
            diagnostics = []
            for spin in ('default', '0'):
                for rank in range(4):
                    text = (root / f'{prefix}-s1-spin{spin}-repdiagnostic-rank{rank}.stderr').read_text()
                    records = re.findall(r'shared-team-probe: attempts=(\d+) successes=(\d+) busy=(\d+) unavailable=(\d+) participants=(\d+)', text)
                    assert len(records) == 1
                    attempts, successes, busy, unavailable, participants = map(int, records[0])
                    assert attempts == successes + busy + unavailable
                    assert successes > 0 and participants == 4
                    diagnostics.append(dict(spin=spin, rank=rank, attempts=attempts,
                                            successes=successes, busy=busy, unavailable=unavailable,
                                            participants=participants))
            result['diagnostics'] = diagnostics
        summary[mode][ranks] = result
print(json.dumps(summary, indent=2))
