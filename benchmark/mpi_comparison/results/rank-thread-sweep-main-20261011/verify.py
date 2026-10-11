"""Verify the exported evidence without invoking C, Julia or Rust."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import tarfile

root = Path(__file__).resolve().parent
pairs = ((1, 1), (1, 16), (2, 8), (4, 4), (8, 2), (16, 1))
files_checked = 0
for ranks, threads in pairs:
    cell = root / f'r{ranks}-t{threads}'
    hashes = json.loads((cell / 'artifact-sha256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((cell / name).read_bytes()).hexdigest() == expected, name
    assert gzip.decompress((cell / 'source.log.gz').read_bytes()).decode().strip() == '0f86e42e3881c68b97cad9fb5a2979f7959de89d'
    assert not gzip.decompress((cell / 'dirty.log.gz').read_bytes()).strip()
    validation = json.loads((cell / 'validation.json').read_text())
    assert (validation['ranks'], validation['threads'], validation['measurements']) == (ranks, threads, 36)
    for sites in (32, 64):
        for workload in ('Opt', 'PhysCal'):
            for implementation in ('C', 'Julia', 'Rust'):
                archive_path = cell / f'L{sites}' / f'{workload}-{implementation}' / 'outputs.tar.gz'
                count = 0
                with tarfile.open(archive_path) as archive:
                    for member in archive.getmembers():
                        if not Path(member.name).name.startswith('zvo_out'):
                            continue
                        rows = [list(map(float, line.split())) for line in archive.extractfile(member).read().decode().splitlines()]
                        assert len(rows) == (300 if workload == 'Opt' else 1), member.name
                        assert all(len(row) == 6 and all(math.isfinite(x) for x in row) for row in rows), member.name
                        count += 1
                assert count == (3 if workload == 'Opt' else 300), archive_path
                files_checked += count
print(f'All six layouts: checksums, source provenance and {files_checked} measured output files verified.')
