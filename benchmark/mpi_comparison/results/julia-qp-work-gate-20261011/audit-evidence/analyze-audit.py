"""Require all planned audits; compare discrete states exactly and floats numerically."""
from pathlib import Path
import json
import math

root = Path(__file__).parent
cases = []
rank_pairs = 0
layouts = [(1,16,20), (1,16,300), (4,4,300)]
for ranks, threads, steps in layouts:
    for size in (32,64):
        paths = [root/'audit'/f'{version}-L{size}-r{ranks}-t{threads}-s{steps}'
                 for version in ('baseline','candidate')]
        for rank in range(ranks):
            for kind in ('rng','state'):
                # These snapshots contain integer states/counters and provider
                # identifiers only; never compare computed floats bitwise.
                assert (paths[0]/f'{kind}-rank-{rank}.txt').read_bytes() == (paths[1]/f'{kind}-rank-{rank}.txt').read_bytes()
            rank_pairs += 1
        arrays = []
        for path in paths:
            rows = [list(map(float,line.split())) for line in
                    (path/'production/zvo_out.dat').read_text().splitlines() if line.strip()]
            assert len(rows)==steps
            assert all(len(row)==6 and all(map(math.isfinite,row)) for row in rows)
            arrays.append(rows)
        maximum = 0.0
        for row, reference in zip(*arrays):
            for actual, expected in zip(row,reference):
                delta = abs(actual-expected)
                maximum = max(maximum,delta)
                # Repository long-run repeatability budget; unchanged per-matrix
                # arithmetic is also checked with independent inverse residuals.
                assert delta <= 1e-11 + 1e-11*max(abs(actual),abs(expected))
        cases.append(dict(sites=size,ranks=ranks,threads=threads,steps=steps,
                          exact_rng_and_integer_state=True,output_max_abs_observed=maximum))
assert len(cases)==6 and rank_pairs==12
result = dict(success=True,cases=cases,exact_rank_pairs=rank_pairs,
              float_absolute_tolerance=1e-11,float_relative_tolerance=1e-11)
(root/'full-audit-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
