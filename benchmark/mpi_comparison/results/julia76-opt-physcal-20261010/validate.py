from pathlib import Path
import hashlib
import json
import math
import statistics
from collections import Counter

root = Path(__file__).resolve().parent
rows = json.loads((root / 'measurements.json').read_text())
assert len(rows) == 36
counts = Counter((r['workload'], r['sites'], r['implementation']) for r in rows)
assert len(counts) == 12 and all(n == 3 for n in counts.values())
assert all(math.isfinite(r['seconds']) and r['seconds'] > 0 for r in rows)
hashes = json.loads((root / 'sha256.json').read_text())
for name, expected in hashes.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
outputs = []
for size in (32, 64):
    for mode in ('Opt', 'PhysCal'):
        for impl in ('C', 'Julia', 'Rust'):
            directory = root / f'L{size}' / f'{mode}-{impl}'
            files = list(directory.rglob('zvo_out*.dat'))
            if mode == 'PhysCal':
                assert len(files) == (500 if impl == 'Julia' else 400), (directory, len(files))
            else:
                assert len(files) == (4 if impl == 'C' else 5), (directory, len(files))
            for file in files:
                values = [[float(x) for x in line.split()] for line in file.read_text().splitlines()]
                expected_rows = 20 if mode == 'Opt' and 'run-observe' in file.parts else (300 if mode == 'Opt' else 1)
                assert len(values) == expected_rows, file
                assert all(len(line) == 6 and all(math.isfinite(x) for x in line) for line in values), file
            outputs.append(dict(sites=size, workload=mode, implementation=impl, files=len(files)))
summary = dict(measurements=len(rows), source_hashes_verified=len(hashes), outputs=outputs)
(root / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
for mode in ('Opt', 'PhysCal'):
    for size in (32, 64):
        print(mode, size, [statistics.median(r['seconds'] for r in rows if r['workload'] == mode and r['sites'] == size and r['implementation'] == impl) for impl in ('C', 'Julia', 'Rust')])
