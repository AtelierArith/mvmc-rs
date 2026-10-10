from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument('cell', type=Path)
args = parser.parse_args()
root = args.cell.resolve()
metadata = json.loads((root / 'environment.json').read_text())['arguments']
assert metadata['samples'] == 320 and metadata['warmups'] == 1 and metadata['reps'] == 3
assert metadata['steps'] == 300 and metadata['groups'] == 100
rows = json.loads((root / 'measurements.json').read_text())
assert (root / 'report.md').is_file() and len(rows) == 36
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
            expected = (500 if impl == 'Julia' else 400) if mode == 'PhysCal' else (4 if impl == 'C' else 5)
            assert len(files) == expected, (directory, len(files), expected)
            for file in files:
                values = [[float(x) for x in line.split()] for line in file.read_text().splitlines()]
                expected_rows = 20 if mode == 'Opt' and 'run-observe' in file.parts else (300 if mode == 'Opt' else 1)
                assert len(values) == expected_rows, file
                assert all(len(line) == 6 and all(math.isfinite(x) for x in line) for line in values), file
            outputs.append(dict(sites=size, workload=mode, implementation=impl, files=len(files)))
    opt = (root / f'L{size}' / 'opt-inputs' / 'modpara.def').read_text()
    assert any(line.split() == ['NVMCSample', str(320 // metadata['ranks'])] for line in opt.splitlines())
summary = dict(ranks=metadata['ranks'], threads=metadata['threads'], total_samples=320,
               measurements=len(rows), verified_source_hashes=len(hashes), outputs=outputs)
(root / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
