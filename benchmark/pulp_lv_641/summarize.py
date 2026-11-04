"""Summarize run_kernels.sh outputs: per kernel/variant, the median over processes of the
per-process median ns/call, its min..max range, and the net time (minus copy-only)."""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
rows = defaultdict(list)  # (kernel, impl) -> [median_ns per process]
for f in sorted(out.glob('*-p*.txt')):
    src = f.name.rsplit('-p', 1)[0]  # rust-default / rust-align64 / julia
    for line in f.read_text().splitlines():
        m = re.match(r'^(K\d [^\t]*)\t([^\t]+)\tmedian_ns=([0-9.]+)', line)
        if m:
            kernel = m.group(1).split(' ')[0]
            rows[(kernel, f'{src}:{m.group(2)}')].append(float(m.group(3)))

print('| kernel | build:variant | procs | median ns | range | net of copy |')
print('|---|---|---:|---:|---:|---:|')
for (kernel, impl), vals in sorted(rows.items()):
    med = statistics.median(vals)
    src = impl.split(':')[0]
    copy = rows.get((kernel, f'{src}:copy-only'))
    net = f'{med - statistics.median(copy):.0f}' if copy and not impl.endswith('copy-only') else ''
    print(f'| {kernel} | {impl} | {len(vals)} | {med:.0f} | {min(vals):.0f}..{max(vals):.0f} | {net} |')
