import collections, gzip, json, sys
for path in sys.argv[1:]:
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    samples = data['samples']
    names = collections.Counter(s['name'] for s in samples)
    waits = collections.Counter(s['wchan'] for s in samples)
    totals = [sum(s['delta'][i] for s in samples if s['delta'] is not None) for i in range(3)]
    print(json.dumps(dict(lifetime=data['lifetime'], spin=data['spin'], thread_samples=len(samples), names=names, wait_channels=waits, observed_delta_ticks_vol_invol=totals)))
