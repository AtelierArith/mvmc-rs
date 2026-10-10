from pathlib import Path
import json
import statistics

root = Path(__file__).resolve().parent
pairs = [(1, 1), (1, 16), (2, 8), (4, 4), (8, 2), (16, 1)]
output = [
    '# C / Julia / Rust rank and thread comparison',
    '',
    'Periodic half-filled Hubbard chain, t=1, U=4. Opt300 / PhysCal100; total320 samples, BLAS1; warmup1, repetitions3; medians in seconds.',
    '',
    'The 1×1 case is a baseline. The remaining five cases have ranks×threads=16. C uses rank-zero internal All; Julia/Rust use warmed maximum-rank production API time. Startup/JIT/setup are excluded; timing boundaries differ.',
    '',
    'PhysCal uses a common C-generated optimized parameter file for the three implementations within each size/configuration. Different rank configurations generate their own parameter files and RNG trajectories, so this is an end-to-end workload comparison rather than a fixed-trajectory scaling experiment.',
    '',
]
all_rows = []
for ranks, threads in pairs:
    cell = root / f'r{ranks}-t{threads}'
    assert (cell / 'report.md').is_file(), f'incomplete: {cell}'
    rows = json.loads((cell / 'measurements.json').read_text())
    assert len(rows) == 36
    metadata = json.loads((cell / 'environment.json').read_text())['arguments']
    assert (metadata['ranks'], metadata['threads'], metadata['samples']) == (ranks, threads, 320)
    output += [f'## {ranks} rank × {threads} thread per rank', '',
               '| 計算 | サイト数 | C | Julia | Rust |',
               '|---|---:|---:|---:|---:|']
    for workload in ('Opt', 'PhysCal'):
        for size in (32, 64):
            medians = []
            for impl in ('C', 'Julia', 'Rust'):
                values = [r['seconds'] for r in rows if r['workload'] == workload and r['sites'] == size and r['implementation'] == impl]
                assert len(values) == 3
                medians.append(statistics.median(values))
            output.append(f'| {workload} | {size} | ' + ' | '.join(f'{v:.3f}' for v in medians) + ' |')
    output.append('')
    all_rows.extend(dict(r, ranks=ranks, threads=threads) for r in rows)
(root / 'report.md').write_text('\n'.join(output).rstrip() + '\n')
(root / 'measurements.json').write_text(json.dumps(all_rows, indent=2) + '\n')
print('\n'.join(output))
