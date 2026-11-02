"""Analyze explicit same-binary runs; discrete records must agree exactly."""
import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
report = []
for mode in ('opt', 'phys'):
    for walkers in (1, 4):
        reference_discrete = None
        reference_numeric = None
        for budget in ('unset', '0', '3'):
            timings = []
            maximum_difference = 0.0
            for rep in range(4):
                stem = f'{mode}-w{walkers}-b{budget}-r{rep}'
                lines = (root / f'{stem}.stdout').read_text().splitlines()
                discrete = sorted(line for line in lines if line.startswith('D|'))
                assert len(discrete) == walkers, stem
                if reference_discrete is None:
                    reference_discrete = discrete
                assert discrete == reference_discrete, f'discrete divergence: {stem}'
                numeric = {}
                for line in lines:
                    if not line.startswith('N|'):
                        continue
                    _, walker, payload = line.split('|')
                    values = tuple(map(float, payload.split()))
                    if mode == 'opt' and int(walker) != 0:
                        assert len(values) == 1 and math.isnan(values[0]), stem
                        continue  # Only the root optimizer reports a final energy.
                    assert all(math.isfinite(x) for x in values), stem
                    numeric[int(walker)] = values
                assert len(numeric) == (1 if mode == 'opt' else walkers), stem
                if reference_numeric is None:
                    reference_numeric = numeric
                assert numeric.keys() == reference_numeric.keys(), stem
                for walker, values in numeric.items():
                    expected = reference_numeric[walker]
                    assert len(values) == len(expected), stem
                    maximum_difference = max(maximum_difference, *(abs(a-b) for a,b in zip(values, expected)))
                timing = dict((k, float(v)) for k,v in re.findall(r'(wall|user|sys)=([0-9.]+)', (root / f'{stem}.time').read_text()))
                assert set(timing) == {'wall', 'user', 'sys'}, stem
                if rep:
                    timings.append(timing)
            report.append(dict(mode=mode, walkers=walkers, budget=budget,
                median={k:statistics.median(t[k] for t in timings) for k in timings[0]},
                samples=timings, max_absolute_numeric_difference=maximum_difference,
                discrete_records_match=True))
print(json.dumps(report, indent=2))
