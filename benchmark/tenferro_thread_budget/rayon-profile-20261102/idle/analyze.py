import collections
import re
import statistics
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    hz = next(int(line) for line in stream if line.strip().isdigit())
    rows = collections.defaultdict(list)
    for line in stream:
        if not line.startswith("threads="):
            continue
        values = dict(re.findall(r"(\w+)=([\w.]+)", line))
        key = (int(values["threads"]), int(values["requested_sleep_us"]), values["inside"])
        wall = float(values["wall_s"])
        rows[key].append((int(values["other_ticks"]) / (hz * wall), int(values["calls"]) / wall))
for key, samples in sorted(rows.items()):
    print(*key, len(samples), *(round(statistics.median(v[i] for v in samples), 5) for i in (0, 1)))
