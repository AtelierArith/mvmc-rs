from pathlib import Path
import json
import math
import re
import statistics

root = Path(__file__).parent
cases = []
for folder in sorted(root.glob('physcal-paired-L*-r*-t*-g100')):
    size, ranks, threads = map(int, re.fullmatch(r'physcal-paired-L(\d+)-r(\d+)-t(\d+)-g100', folder.name).groups())
    records = re.findall(r'^PAIRED (baseline|candidate) ([123]) ([0-9.e+-]+) 100$', (folder/'run.log').read_text(), re.M)
    assert len(records) == 6, folder
    times = {name: {} for name in ('baseline', 'candidate')}
    outputs = []
    filenames = None
    for name, rep, seconds in records:
        assert rep not in times[name]
        seconds = float(seconds)
        assert math.isfinite(seconds) and seconds > 0
        times[name][rep] = seconds
        files = sorted((folder/f'{name}-{rep}').glob('zvo_out_*.dat'))
        assert len(files) == 100, (folder, name, rep, len(files))
        names = [path.name for path in files]
        if filenames is None:
            filenames = names
        assert names == filenames
        rows = []
        for path in files:
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            assert len(lines) == 1
            row = list(map(float, lines[0].split()))
            assert len(row) == 6 and all(map(math.isfinite, row))
            rows.append(row)
        outputs.append(rows)
    maximum = 0.0
    for rows in outputs:
        for row, reference in zip(rows, outputs[0]):
            for actual, expected in zip(row, reference):
                difference = abs(actual-expected)
                maximum = max(maximum, difference)
                # Existing CLI numerical-output bound; no arithmetic/reduction reordering.
                assert difference <= 1e-12 + 1e-12*max(abs(actual), abs(expected)), (folder, actual, expected)
    medians = {name: statistics.median(values.values()) for name, values in times.items()}
    cases.append(dict(sites=size, ranks=ranks, threads=threads, groups=100, total_samples=320,
        times=times, medians=medians, candidate_percent_change=100*(medians['candidate']/medians['baseline']-1),
        pair_wins=sum(times['candidate'][str(i)] < times['baseline'][str(i)] for i in range(1,4)),
        output_max_abs_observed=maximum))
assert cases
(root/'physcal-paired-summary.json').write_text(json.dumps({'cases': cases, 'complete': True}, indent=2)+'\n')
print(json.dumps(cases, indent=2))
