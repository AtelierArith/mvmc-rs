"""Summarize run_kernels.sh outputs: per kernel/variant, the median over processes of the
per-process median ns/call, its min..max range, and the net time (minus copy-only).

Numerical acceptance (#697): every process output must end with `correctness\tPASS` and
contain no `status=FAIL` check. Otherwise the summary lists the failures and exits with
status 1, so a failing or truncated run cannot produce a successful report."""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
rows = defaultdict(list)  # (kernel, impl) -> [median_ns per process]
verdicts = []  # (file, verdict, failed checks)
for f in sorted(out.glob('*-p*.txt')):
    src = f.name.rsplit('-p', 1)[0]  # rust-default / rust-align64 / julia
    text = f.read_text()
    failed = [line.split('\t')[1] for line in text.splitlines()
              if line.startswith('check\t') and line.endswith('status=FAIL')]
    verdict = re.search(r'^correctness\t(PASS|FAIL)\tchecks=(\d+)', text, re.M)
    verdicts.append((f.name, verdict.group(1) if verdict else 'MISSING',
                     verdict.group(2) if verdict else '0', failed))
    for line in text.splitlines():
        m = re.match(r'^(K\d [^\t]*)\t([^\t]+)\tmedian_ns=([0-9.]+)', line)
        if m:
            kernel = m.group(1).split(' ')[0]
            rows[(kernel, f'{src}:{m.group(2)}')].append(float(m.group(3)))

print('| process output | correctness | checks | failed checks |')
print('|---|---|---:|---|')
for name, verdict, checks, failed in verdicts:
    print(f'| {name} | {verdict} | {checks} | {", ".join(failed)} |')
print()
print('| kernel | build:variant | procs | median ns | range | net of copy |')
print('|---|---|---:|---:|---:|---:|')
for (kernel, impl), vals in sorted(rows.items()):
    med = statistics.median(vals)
    src = impl.split(':')[0]
    copy = rows.get((kernel, f'{src}:copy-only'))
    net = f'{med - statistics.median(copy):.0f}' if copy and not impl.endswith('copy-only') else ''
    print(f'| {kernel} | {impl} | {len(vals)} | {med:.0f} | {min(vals):.0f}..{max(vals):.0f} | {net} |')
if not verdicts or any(v != 'PASS' or failed for _, v, _, failed in verdicts):
    print('\nnumerical acceptance FAILED', file=sys.stderr)
    sys.exit(1)
