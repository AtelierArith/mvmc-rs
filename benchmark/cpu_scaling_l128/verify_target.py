"""Audit #629 paired wall medians and numerical outputs; no reference runtimes."""
import argparse
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
args = parser.parse_args()
base = args.results
rows = json.loads((base / 'timings.json').read_text())
assert len(rows) == 32
ratios, checks = [], []
for mode in ('opt', 'phys'):
    for threads in (8, 16):
        medians = {}
        for kind in ('c', 'rust'):
            group = [r for r in rows if (r['mode'], r['threads'], r['kind']) == (mode, threads, kind)]
            assert sorted(r['rep'] for r in group) == [0, 1, 2, 3]
            assert all(math.isfinite(r['wall']) and r['wall'] > 0 for r in group)
            medians[kind] = statistics.median(r['wall'] for r in group if r['rep'] > 0)
        ratio = medians['c'] / medians['rust']
        ratios.append(dict(mode=mode, threads=threads, **medians, c_over_rust=ratio, meets_target=ratio >= 1.1))
        for rep in range(4):
            c = base / f'{mode}-c-t{threads}-r{rep}/output'
            rust = base / f'{mode}-rust-t{threads}-r{rep}/output'
            # Existing Opt writer naming differs; compare the same output family.
            pairs = [('zvo_out_001.dat', 'zvo_out.dat' if mode == 'opt' else 'zvo_out_001.dat'),
                     ('zvo_var_001.dat', 'zvo_var.dat' if mode == 'opt' else 'zvo_var_001.dat')]
            pairs += [('zqp_opt.dat', 'zqp_opt.dat')] if mode == 'opt' else [
                ('zvo_cisajs_001.dat', 'zvo_cisajs_001.dat'),
                ('zvo_cisajscktalt_001.dat', 'zvo_cisajscktalt_001.dat')]
            for cn, rn in pairs:
                x = [line.split() for line in (c / cn).read_text().splitlines()]
                y = [line.split() for line in (rust / rn).read_text().splitlines()]
                assert [len(row) for row in x] == [len(row) for row in y], (cn, rn)
                maximum, count, scaled = 0.0, 0, 0.0
                for a, b in zip(x, y):
                    for col, (u, v) in enumerate(zip(a, b)):
                        if cn.startswith('zvo_cisajs') and col < len(a) - 2:
                            assert int(u) == int(v), (cn, col, u, v)
                        u, v = float(u), float(v)
                        assert math.isfinite(u) and math.isfinite(v)
                        error = abs(u - v)
                        bound = 1e-11 + 1e-11 * max(abs(u), abs(v))
                        assert error <= bound, (mode, threads, rep, cn, u, v, error)
                        maximum = max(maximum, error)
                        scaled = max(scaled, error / bound)
                        count += 1
                checks.append(dict(mode=mode, threads=threads, rep=rep, c_file=cn,
                                   rust_file=rn, fields=count,
                                   max_absolute_difference=maximum, max_fraction_of_bound=scaled))
(base / 'target-results.json').write_text(json.dumps(ratios, indent=2) + '\n')
(base / 'c-output-verification.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(ratios, indent=2))
print(f'{len(checks)} C/Rust file pairs passed; max absolute difference '
      f'{max(r["max_absolute_difference"] for r in checks):.17g}; '
      f'max fraction of bound {max(r["max_fraction_of_bound"] for r in checks):.6g}')
assert all(r['meets_target'] for r in ratios), 'C/Rust must be >= 1.1 in all four conditions'
