"""Validate all refreshed output sets; report historical differences separately."""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results', type=Path)
p.add_argument('--historical', type=Path)
a = p.parse_args()
def read(directory):
    arrays = {}
    for f in sorted(directory.glob('zvo_ls_*.dat')):
        lines = [line.split() for line in f.read_text().splitlines() if line.strip()]
        assert lines, f
        values = [float(x) for line in lines for x in line]
        assert all(math.isfinite(x) for x in values), f
        arrays[f.name] = ([len(line) for line in lines], values)
    assert len([f for f in arrays if f.startswith('zvo_ls_qqqq_')]) == 20, directory
    assert arrays, directory
    return arrays
def compare(left, right):
    assert left.keys() == right.keys(), (left.keys(), right.keys())
    maxima = {}
    for name, (shape, values) in left.items():
        other_shape, other_values = right[name]
        assert shape == other_shape, name
        group = name.rsplit('_', 1)[0]
        maxima[group] = max(maxima.get(group, 0), max(abs(x-y) for x,y in zip(values, other_values)))
    return maxima
results = []
for case in ['phys_lanczos', 'phys_interall_lanczos1', 'phys_interall_lanczos2']:
    reference = {}
    for impl in ['rust', 'julia']:
        reference[impl] = read(a.results / '1threads' / case / (f'{impl}-1/output' if impl == 'rust' else f'{impl}-1'))
        for n in [1,4,8,16]:
            thread_path = a.results / f'{n}threads' / case / f'{impl}-1'
            thread_reference = read(thread_path / 'output' if impl == 'rust' else thread_path)
            for rep in range(4):
                path = a.results / f'{n}threads' / case / f'{impl}-{rep}'
                arrays = read(path / 'output' if impl == 'rust' else path)
                differences = compare(arrays, reference[impl])
                fixed_thread_differences = compare(arrays, thread_reference)
                # Same-implementation repeatability; this is not an oracle for cross-language correctness.
                assert max(differences.values()) <= 1e-11, (case, impl, n, rep, differences)
                results.append(dict(case=case, impl=impl, threads=n, rep=rep, files=len(arrays), repeat_max_abs=differences, fixed_thread_repeat_max_abs=fixed_thread_differences))
        if a.historical:
            old = read(a.historical / '1threads/outputs' / case / f'{impl}-1')
            results.append(dict(case=case, impl=impl, historical_max_abs=compare(reference[impl], old)))
    results.append(dict(case=case, julia_rust_max_abs=compare(reference['julia'], reference['rust'])))
(a.results / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps([r for r in results if 'historical_max_abs' in r or 'julia_rust_max_abs' in r], indent=2))
