import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
result = {}


def files(directory):
    return {p.name for p in directory.iterdir()
            if p.is_file() and '_time_' not in p.name}


def values(path):
    numbers = [float(token) for line in path.read_text().splitlines()
               if not line.lstrip().startswith('#') for token in line.split()]
    assert all(math.isfinite(n) for n in numbers), path
    return numbers


for case in ('l32-opt', 'l32-phys', 'l64-opt'):
    reference = root / f'{case}-w1-g20000-s0-r1-p0-output'
    names = files(reference)
    assert names
    expected = {name: values(reference / name) for name in names}
    maximum = 0.0
    groups = []
    for workers in (1, 4):
        for gate in (20000, 40000, 80000):
            for shared in (0, 1):
                samples = []
                for rep in (1, 2, 3):
                    label = f'{case}-w{workers}-g{gate}-s{shared}-r{rep}-p0'
                    directory = root / f'{label}-output'
                    assert files(directory) == names, directory
                    for name in names:
                        a, b = expected[name], values(directory / name)
                        assert len(a) == len(b), directory / name
                        maximum = max(maximum, max((abs(x-y) for x, y in zip(a, b)), default=0.0))
                    text = (root / f'{label}.stderr').read_text()
                    matches = re.findall(r'^wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)$', text, re.M)
                    assert len(matches) == 1, label
                    samples.append(dict(zip(('wall', 'user', 'system'), map(float, matches[0]))))
                groups.append(dict(workers=workers, gate_ns=gate, shared=shared,
                                   samples=samples,
                                   medians={k: statistics.median(s[k] for s in samples)
                                            for k in ('wall', 'user', 'system')}))
    result[case] = dict(files_compared=len(names), max_absolute_difference=maximum,
                        groups=groups)
print(json.dumps(result, indent=2))
