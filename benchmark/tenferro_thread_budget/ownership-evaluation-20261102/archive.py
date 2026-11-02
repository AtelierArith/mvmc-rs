"""Copy raw outputs/inputs and deduplicate identical contents using tar hardlinks."""
import hashlib
import os
from pathlib import Path
import shutil
import sys

walkers, mpi, inputs, destination = map(Path, sys.argv[1:])
destination.mkdir(exist_ok=False)
for source, name in ((walkers, 'walkers'), (mpi, 'mpi'), (inputs, 'inputs')):
    shutil.copytree(source, destination / name)
seen = {}
for path in sorted(destination.rglob('*')):
    if not path.is_file():
        continue
    key = hashlib.sha256(path.read_bytes()).digest()
    if key in seen:
        # This comparison protects archival identity, not floating-point parity.
        assert path.read_bytes() == seen[key].read_bytes()
        path.unlink()
        os.link(seen[key], path)
    else:
        seen[key] = path
